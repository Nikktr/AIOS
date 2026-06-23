"""
MCP Client — connects to external MCP servers via stdio or HTTP transport.
Stdio: subprocess with JSON-RPC over stdin/stdout.
HTTP: POST JSON-RPC to a URL endpoint (Streamable HTTP).
"""

import subprocess
import json
import threading
import logging
import os
from queue import Queue, Empty
from typing import Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"


class BaseMCPConnection(ABC):
    def __init__(self, name: str):
        self.name = name
        self._request_id = 0
        self._tools: list[dict] | None = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def _send_request(self, method: str, params: dict | None = None) -> dict: ...

    def list_tools(self) -> list[dict]:
        if self._tools is not None:
            return self._tools
        result = self._send_request("tools/list")
        self._tools = result.get("tools", [])
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> Any:
        params = {"name": tool_name, "arguments": arguments or {}}
        return self._send_request("tools/call", params)

    def is_running(self) -> bool:
        return True


class StdioMCPConnection(BaseMCPConnection):
    REQUEST_TIMEOUT = 60

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        super().__init__(name)
        self.command = command
        self.args = args
        self.env = env or {}
        self._process: subprocess.Popen | None = None
        self._write_lock = threading.Lock()
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, dict] = {}
        self._reader_thread: threading.Thread | None = None

    def start(self):
        if self._process and self._process.poll() is None:
            return
        full_env = {**os.environ, **self.env}
        self._process = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            encoding="utf-8",
            errors="replace",
        )
        self._pending.clear()
        self._results.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._initialize()

    def _reader_loop(self):
        """Background thread: reads stdout and dispatches responses by id."""
        while self._process and self._process.poll() is None:
            try:
                line = self._process.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = msg.get("id")
                if rid is not None and rid in self._pending:
                    self._results[rid] = msg
                    self._pending[rid].set()
            except Exception:
                break
        for evt in self._pending.values():
            evt.set()

    def _ensure_running(self):
        if self._process and self._process.poll() is None:
            return
        logger.warning("MCP server '%s' died (rc=%s), restarting...",
                        self.name, self._process.returncode if self._process else "N/A")
        self._tools = None
        self.start()

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        self._ensure_running()

        req_id = self._next_id()
        request = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "method": method}
        if params is not None:
            request["params"] = params

        evt = threading.Event()
        self._pending[req_id] = evt

        line = json.dumps(request) + "\n"
        with self._write_lock:
            self._process.stdin.write(line)
            self._process.stdin.flush()

        if not evt.wait(timeout=self.REQUEST_TIMEOUT):
            self._pending.pop(req_id, None)
            raise TimeoutError(f"MCP server '{self.name}' did not respond within {self.REQUEST_TIMEOUT}s")

        self._pending.pop(req_id, None)
        response = self._results.pop(req_id, None)

        if response is None:
            stderr = self._process.stderr.read() if self._process and self._process.stderr else ""
            raise RuntimeError(f"MCP server '{self.name}' closed: {stderr[:500]}")

        if "error" in response:
            err = response["error"]
            raise RuntimeError(f"MCP error ({err.get('code')}): {err.get('message')}")
        return response.get("result", {})

    def _initialize(self):
        self._send_request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "aios-kernel", "version": "1.0.0"},
        })
        self._send_notification("notifications/initialized")

    def _send_notification(self, method: str, params: dict | None = None):
        if not self._process or self._process.poll() is not None:
            return
        notif = {"jsonrpc": JSONRPC_VERSION, "method": method}
        if params:
            notif["params"] = params
        line = json.dumps(notif) + "\n"
        with self._write_lock:
            self._process.stdin.write(line)
            self._process.stdin.flush()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self):
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
        self._process = None
        self._tools = None
        self._pending.clear()
        self._results.clear()


class HttpMCPConnection(BaseMCPConnection):
    def __init__(self, name: str, url: str, headers: dict[str, str] | None = None):
        super().__init__(name)
        self.url = url
        self.headers = headers or {}
        self._session_id: str | None = None
        self._initialized = False

    def start(self):
        if self._initialized:
            return
        import requests as req_lib
        self._req_lib = req_lib
        self._initialize()
        self._initialized = True

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        req_id = self._next_id()
        payload = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params

        hdrs = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self._session_id:
            hdrs["Mcp-Session-Id"] = self._session_id

        resp = self._req_lib.post(self.url, json=payload, headers=hdrs, timeout=30)
        resp.raise_for_status()

        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers["mcp-session-id"]

        content_type = resp.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            return self._parse_sse(resp.text, req_id)

        data = resp.json()
        if "error" in data:
            err = data["error"]
            raise RuntimeError(f"MCP error ({err.get('code')}): {err.get('message')}")
        return data.get("result", {})

    def _parse_sse(self, text: str, req_id: int) -> dict:
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("id") == req_id:
                        if "error" in data:
                            err = data["error"]
                            raise RuntimeError(f"MCP error ({err.get('code')}): {err.get('message')}")
                        return data.get("result", {})
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(f"No matching response for request {req_id} in SSE stream")

    def _initialize(self):
        self._send_request("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "aios-kernel", "version": "1.0.0"},
        })

    def is_running(self) -> bool:
        return self._initialized

    def stop(self):
        self._initialized = False
        self._session_id = None
        self._tools = None


class MCPManager:
    def __init__(self):
        self._servers: dict[str, BaseMCPConnection] = {}

    def load_from_config(self, mcp_config: dict):
        for name, cfg in (mcp_config or {}).items():
            if not isinstance(cfg, dict):
                continue

            transport = cfg.get("type", "stdio")

            if transport == "http":
                url = cfg.get("url")
                if not url:
                    logger.warning(f"MCP server '{name}' (http) has no url, skipping")
                    continue
                headers = cfg.get("headers", {})
                self._servers[name] = HttpMCPConnection(name, url, headers)
                logger.info(f"MCP server '{name}' registered (http): {url}")

            else:
                command = cfg.get("command")
                if not command:
                    logger.warning(f"MCP server '{name}' has no command, skipping")
                    continue
                args = cfg.get("args", [])
                env = cfg.get("env", {})
                self._servers[name] = StdioMCPConnection(name, command, args, env)
                logger.info(f"MCP server '{name}' registered (stdio): {command} {' '.join(args)}")

    def start_all(self):
        for name, server in self._servers.items():
            try:
                server.start()
                logger.info(f"MCP server '{name}' started")
            except Exception as e:
                logger.error(f"Failed to start MCP server '{name}': {e}")

    def start_server(self, name: str):
        if name not in self._servers:
            raise KeyError(f"MCP server '{name}' not configured")
        self._servers[name].start()

    def list_all_tools(self) -> dict[str, list[dict]]:
        result = {}
        for name, server in self._servers.items():
            try:
                if not server.is_running():
                    server.start()
                result[name] = server.list_tools()
            except Exception as e:
                result[name] = [{"error": str(e)}]
        return result

    def call_tool(self, server_name: str, tool_name: str, arguments: dict | None = None) -> Any:
        if server_name not in self._servers:
            raise KeyError(f"MCP server '{server_name}' not configured")
        server = self._servers[server_name]
        if not server.is_running():
            server.start()
        return server.call_tool(tool_name, arguments)

    def call_tools_parallel(self, calls: list[dict]) -> list[dict]:
        """Execute multiple tool calls in parallel.

        Args:
            calls: List of dicts with keys: server, tool, arguments.

        Returns:
            List of dicts with keys: server, tool, result or error.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _do(call):
            try:
                result = self.call_tool(call["server"], call["tool"], call.get("arguments"))
                return {"server": call["server"], "tool": call["tool"], "result": result}
            except Exception as e:
                return {"server": call["server"], "tool": call["tool"], "error": str(e)}

        results = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
            futures = {pool.submit(_do, c): i for i, c in enumerate(calls)}
            for f in as_completed(futures):
                results[futures[f]] = f.result()
        return results

    def get_tools_as_openai_functions(self, servers: list[str] | None = None) -> list[dict]:
        """Convert MCP tool schemas to OpenAI function-calling format.

        Tool names are namespaced as ``mcp__<server>__<tool_name>`` so the
        caller can route tool_calls back through :meth:`execute_tool_call`.

        Args:
            servers: Restrict to these server names. ``None`` = all servers.

        Returns:
            List of dicts ready for ``litellm.completion(tools=...)``.
        """
        functions: list[dict] = []
        target = servers or list(self._servers.keys())
        for srv_name in target:
            if srv_name not in self._servers:
                continue
            server = self._servers[srv_name]
            try:
                if not server.is_running():
                    server.start()
                tools = server.list_tools()
            except Exception:
                continue
            for t in tools:
                if "error" in t:
                    continue
                name = t.get("name", "")
                if not name:
                    continue
                fn_name = f"mcp__{srv_name}__{name}"
                schema = t.get("inputSchema", {"type": "object", "properties": {}})
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
        """Route an OpenAI-style tool_call back to the right MCP server.

        Args:
            fn_name: Function name in ``mcp__<server>__<tool>`` format.
            arguments: Tool arguments dict.

        Returns:
            MCP tool call result.

        Raises:
            ValueError: If *fn_name* doesn't match the expected format.
            KeyError: If the server is not configured.
        """
        parts = fn_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Not an MCP tool call: {fn_name}")
        server_name, tool_name = parts[1], parts[2]
        return self.call_tool(server_name, tool_name, arguments)

    @staticmethod
    def is_mcp_tool_call(fn_name: str) -> bool:
        """Check if a function name is an MCP-namespaced tool call."""
        return fn_name.startswith("mcp__") and fn_name.count("__") >= 2

    def stop_all(self):
        for server in self._servers.values():
            server.stop()

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())
