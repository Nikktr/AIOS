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
    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None):
        super().__init__(name)
        self.command = command
        self.args = args
        self.env = env or {}
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

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
        self._initialize()

    def _send_request(self, method: str, params: dict | None = None) -> dict:
        if not self._process or self._process.poll() is not None:
            raise RuntimeError(f"MCP server '{self.name}' is not running")

        req_id = self._next_id()
        request = {"jsonrpc": JSONRPC_VERSION, "id": req_id, "method": method}
        if params is not None:
            request["params"] = params

        line = json.dumps(request) + "\n"
        with self._lock:
            self._process.stdin.write(line)
            self._process.stdin.flush()
            while True:
                response_line = self._process.stdout.readline()
                if not response_line:
                    stderr = self._process.stderr.read() if self._process.stderr else ""
                    raise RuntimeError(f"MCP server '{self.name}' closed: {stderr[:500]}")
                response_line = response_line.strip()
                if not response_line:
                    continue
                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    continue
                if "id" in response and response["id"] == req_id:
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
        with self._lock:
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

    def stop_all(self):
        for server in self._servers.values():
            server.stop()

    @property
    def server_names(self) -> list[str]:
        return list(self._servers.keys())
