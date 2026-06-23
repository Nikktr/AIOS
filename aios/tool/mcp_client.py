"""
MCP Client — connects to external MCP servers using the official MCP Python SDK.
Supports stdio (subprocess) and HTTP (Streamable HTTP / SSE) transports.

The SDK handles JSON-RPC framing, process lifecycle (Job Objects on Windows),
SSE parsing, session management, and reconnection — replacing the previous
hand-rolled implementation.

MCPManager runs an internal asyncio event loop on a background thread and
exposes synchronous methods so callers (orchestrator, tool manager, etc.)
can use it without async.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

CALL_TIMEOUT = 60


class _MCPConnection:
    """Holds one live MCP server connection (transport + session)."""

    def __init__(self, name: str):
        self.name = name
        self.session: ClientSession | None = None
        self._tools_cache: list[dict] | None = None

    async def list_tools(self) -> list[dict]:
        if self._tools_cache is not None:
            return self._tools_cache
        if not self.session:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        result = await self.session.list_tools()
        self._tools_cache = [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
            }
            for t in result.tools
        ]
        return self._tools_cache

    async def call_tool(self, tool_name: str, arguments: dict | None = None) -> Any:
        if not self.session:
            raise RuntimeError(f"MCP server '{self.name}' not connected")
        result = await self.session.call_tool(tool_name, arguments or {})
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(block.data)
            else:
                parts.append(str(block))
        combined = "\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, TypeError):
            return combined

    def invalidate_cache(self):
        self._tools_cache = None


class MCPManager:
    def __init__(self):
        self._configs: dict[str, dict] = {}
        self._connections: dict[str, _MCPConnection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stack: AsyncExitStack | None = None
        self._started = False

    # ---- Config loading (sync, before start) ----

    def load_from_config(self, mcp_config: dict):
        for name, cfg in (mcp_config or {}).items():
            if not isinstance(cfg, dict):
                continue
            transport = cfg.get("type", "stdio")
            if transport == "http":
                if not cfg.get("url"):
                    logger.warning("MCP server '%s' (http) has no url, skipping", name)
                    continue
            else:
                if not cfg.get("command"):
                    logger.warning("MCP server '%s' has no command, skipping", name)
                    continue
            self._configs[name] = cfg
            logger.info("MCP server '%s' registered (%s)", name, transport)

    # ---- Lifecycle (sync wrappers) ----

    def start_all(self):
        if self._started:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-event-loop")
        self._thread.start()
        self._run_sync(self._start_all_async())
        self._started = True

    def stop_all(self):
        if not self._started:
            return
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._connections.clear()
        self._started = False
        self._loop = None
        self._thread = None

    # ---- Public sync API (same interface as before) ----

    def list_all_tools(self) -> dict[str, list[dict]]:
        result = {}
        for name in self._configs:
            try:
                result[name] = self._run_sync(self._list_tools_async(name))
            except Exception as e:
                result[name] = [{"error": str(e)}]
        return result

    def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> Any:
        if server_name not in self._configs:
            raise KeyError(f"MCP server '{server_name}' not configured")
        return self._run_sync(self._call_tool_async(server_name, tool_name, arguments))

    def call_tools_parallel(self, calls: list[dict]) -> list[dict]:
        return self._run_sync(self._call_tools_parallel_async(calls))

    def get_tools_as_openai_functions(self, servers: list[str] | None = None) -> list[dict]:
        functions: list[dict] = []
        target = servers or list(self._configs.keys())
        for srv_name in target:
            if srv_name not in self._configs:
                continue
            try:
                tools = self._run_sync(self._list_tools_async(srv_name))
            except Exception:
                continue
            for t in tools:
                if "error" in t:
                    continue
                name = t.get("name", "")
                if not name:
                    continue
                fn_name = f"mcp__{srv_name}__{name}"
                schema = dict(t.get("inputSchema", {"type": "object", "properties": {}}))
                schema.pop("$schema", None)
                schema.pop("additionalProperties", None)
                functions.append({
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "description": (t.get("description") or "")[:1024],
                        "parameters": schema,
                    },
                })
        return functions

    def execute_tool_call(self, fn_name: str, arguments: dict | None = None) -> Any:
        parts = fn_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Not an MCP tool call: {fn_name}")
        server_name, tool_name = parts[1], parts[2]
        return self.call_tool(server_name, tool_name, arguments)

    @staticmethod
    def is_mcp_tool_call(fn_name: str) -> bool:
        return fn_name.startswith("mcp__") and fn_name.count("__") >= 2

    @property
    def server_names(self) -> list[str]:
        return list(self._configs.keys())

    # ---- Async internals ----

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro) -> Any:
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("MCP event loop not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=CALL_TIMEOUT)

    async def _start_all_async(self):
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, cfg in self._configs.items():
            try:
                await self._connect_server(name, cfg)
                logger.info("MCP server '%s' connected", name)
            except Exception as e:
                logger.error("Failed to connect MCP server '%s': %s", name, e)

    async def _stop_all_async(self):
        if self._stack:
            await self._stack.aclose()
            self._stack = None
        self._connections.clear()

    async def _connect_server(self, name: str, cfg: dict):
        transport = cfg.get("type", "stdio")
        conn = _MCPConnection(name)

        if transport == "http":
            url = cfg["url"]
            headers = cfg.get("headers", {})
            read_stream, write_stream, _get_sid = await self._stack.enter_async_context(
                streamablehttp_client(url=url, headers=headers)
            )
        else:
            env_extra = cfg.get("env", {})
            full_env = {**os.environ, **env_extra} if env_extra else None
            server_params = StdioServerParameters(
                command=cfg["command"],
                args=cfg.get("args", []),
                env=full_env,
            )
            read_stream, write_stream = await self._stack.enter_async_context(
                stdio_client(server_params)
            )

        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        conn.session = session
        self._connections[name] = conn

    async def _list_tools_async(self, name: str) -> list[dict]:
        if name not in self._connections:
            if name in self._configs:
                await self._connect_server(name, self._configs[name])
            else:
                raise KeyError(f"MCP server '{name}' not configured")
        return await self._connections[name].list_tools()

    async def _call_tool_async(self, server_name: str, tool_name: str, arguments: dict | None = None) -> Any:
        if server_name not in self._connections:
            if server_name in self._configs:
                await self._connect_server(server_name, self._configs[server_name])
            else:
                raise KeyError(f"MCP server '{server_name}' not configured")
        return await self._connections[server_name].call_tool(tool_name, arguments)

    async def _call_tools_parallel_async(self, calls: list[dict]) -> list[dict]:
        async def _do(call):
            try:
                result = await self._call_tool_async(call["server"], call["tool"], call.get("arguments"))
                return {"server": call["server"], "tool": call["tool"], "result": result}
            except Exception as e:
                return {"server": call["server"], "tool": call["tool"], "error": str(e)}

        tasks = [asyncio.create_task(_do(c)) for c in calls]
        return list(await asyncio.gather(*tasks))
