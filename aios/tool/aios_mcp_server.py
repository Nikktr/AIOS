"""
AIOS MCP Server — exposes AIOS kernel capabilities as MCP tools.
Hermes (or any MCP client) connects via stdio and gets full control
over agents, orchestrator, LLM, MCP proxying, memory, and system ops.

Communicates with the kernel via HTTP (localhost:8000).
The kernel must be running before this server is used.
"""

import json
import uuid
import threading
import time
import requests
from mcp.server.fastmcp import FastMCP

KERNEL_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120

mcp = FastMCP("aios")

# ===== Background task system =====
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()


def _submit_background(task_type: str, func, *args, **kwargs) -> str:
    task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "running",
            "type": task_type,
            "started": time.time(),
            "result": None,
        }

    def _worker():
        try:
            result = func(*args, **kwargs)
            with _tasks_lock:
                _tasks[task_id]["status"] = "completed"
                _tasks[task_id]["result"] = result
                _tasks[task_id]["finished"] = time.time()
        except Exception as e:
            with _tasks_lock:
                _tasks[task_id]["status"] = "error"
                _tasks[task_id]["result"] = str(e)
                _tasks[task_id]["finished"] = time.time()

    threading.Thread(target=_worker, daemon=True).start()
    return task_id


def _get(path: str, timeout: int = REQUEST_TIMEOUT) -> dict:
    resp = requests.get(f"{KERNEL_URL}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, body: dict | None = None, timeout: int = REQUEST_TIMEOUT) -> dict:
    resp = requests.post(f"{KERNEL_URL}{path}", json=body or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ===== System tools =====

@mcp.tool(description="Get AIOS kernel status — shows which components are active (llms, storage, memory, tool, scheduler, factory)")
async def aios_status() -> str:
    try:
        result = _get("/core/status")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"
    except Exception as e:
        return f"ERROR: {e}"


# ===== Agent tools =====

@mcp.tool(description="List all registered agents with their capabilities and strengths")
async def aios_list_agents() -> str:
    try:
        result = _get("/agents/roles")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool(description=(
    "Submit a task to a specific agent. Returns task_id immediately — use aios_task_result(task_id) to poll. "
    "agent_id examples: 'aios_local/claude_code_agent', 'aios_local/codex_agent', 'aios_local/hermes_agent'. "
    "task: the task description. project_id and project_path are optional."
))
async def aios_submit_agent(
    agent_id: str,
    task: str,
    project_id: str = "global",
    project_path: str = "",
) -> str:
    try:
        requests.get(f"{KERNEL_URL}/core/status", timeout=5)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"

    def _run():
        submit = _post("/agents/submit", {
            "agent_id": agent_id,
            "agent_config": {
                "task": task,
                "project_id": project_id,
                "project_path": project_path,
            },
        })
        execution_id = submit.get("execution_id")
        if not execution_id:
            return submit
        while True:
            time.sleep(3)
            status = _get(f"/agents/{execution_id}/status")
            st = status.get("status", "")
            if st in ("completed", "failed", "error"):
                return status
            if st not in ("running", "pending", "submitted"):
                return status

    task_id = _submit_background("agent", _run)
    return json.dumps({
        "task_id": task_id,
        "status": "running",
        "agent_id": agent_id,
        "hint": "Use aios_task_result to check progress",
    })


# ===== MCP proxy tools =====

@mcp.tool(description="List all connected MCP servers and their tools (Linear, GitHub, etc.)")
async def aios_mcp_list() -> str:
    try:
        result = _get("/mcp/list")
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool(description=(
    "Call a tool on a connected MCP server. "
    "Example: server='linear', tool='list_teams', arguments={}"
))
async def aios_mcp_call(server: str, tool: str, arguments: dict | None = None) -> str:
    try:
        result = _post("/mcp/call", {
            "server": server,
            "tool": tool,
            "arguments": arguments or {},
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"
    except Exception as e:
        return f"ERROR: {e}"


# ===== Task management tools =====

@mcp.tool(description=(
    "Check the result of a background task by task_id. "
    "Returns status ('running', 'completed', 'error') and result when done. "
    "Use this to poll long-running operations (orchestrate, llm_chat, agent submit)."
))
async def aios_task_result(task_id: str) -> str:
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        return json.dumps({"error": f"Task '{task_id}' not found"})
    info = {"task_id": task_id, "status": task["status"], "type": task["type"]}
    elapsed = (task.get("finished") or time.time()) - task["started"]
    info["elapsed_seconds"] = round(elapsed, 1)
    if task["status"] != "running":
        info["result"] = task["result"]
    return json.dumps(info, ensure_ascii=False, indent=2)


@mcp.tool(description="List all background tasks with their status (running/completed/error).")
async def aios_task_list() -> str:
    with _tasks_lock:
        items = []
        for tid, t in _tasks.items():
            elapsed = (t.get("finished") or time.time()) - t["started"]
            items.append({
                "task_id": tid,
                "status": t["status"],
                "type": t["type"],
                "elapsed_seconds": round(elapsed, 1),
            })
    return json.dumps(items, ensure_ascii=False, indent=2)


# ===== Orchestrator tools =====

@mcp.tool(description=(
    "Create an execution plan for a high-level task WITHOUT executing it. "
    "Returns a structured plan with steps. Useful for reviewing before execution. "
    "task: what needs to be done. context: optional conversation/background context."
))
async def aios_plan(
    task: str,
    project_id: str = "global",
    project_path: str = "",
    context: str = "",
) -> str:
    try:
        result = _post("/orchestrator/plan", {
            "task": task,
            "project_id": project_id,
            "project_path": project_path,
            "context": context,
        })
        return json.dumps(result, ensure_ascii=False, indent=2)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"
    except Exception as e:
        return f"ERROR: {e}"


@mcp.tool(description=(
    "Plan AND execute a high-level task end-to-end via the orchestrator pipeline. "
    "Returns task_id immediately — use aios_task_result(task_id) to poll for completion. "
    "This is a long-running operation (minutes)."
))
async def aios_orchestrate(
    task: str,
    project_id: str = "global",
    project_path: str = "",
    context: str = "",
) -> str:
    try:
        requests.get(f"{KERNEL_URL}/core/status", timeout=5)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"

    def _run():
        return _post("/orchestrator/execute", {
            "task": task,
            "project_id": project_id,
            "project_path": project_path,
            "context": context,
        }, timeout=600)

    task_id = _submit_background("orchestrate", _run)
    return json.dumps({"task_id": task_id, "status": "running", "hint": "Use aios_task_result to check progress"})


# ===== LLM tools =====

@mcp.tool(description=(
    "Send a message to the LLM configured in AIOS and get a response. "
    "Returns task_id — use aios_task_result(task_id) to get the answer. "
    "use_tools=true enables MCP tool usage by the LLM (Linear, GitHub, etc.)."
))
async def aios_llm_chat(
    message: str,
    system_prompt: str = "",
    use_tools: bool = False,
) -> str:
    try:
        requests.get(f"{KERNEL_URL}/core/status", timeout=5)
    except requests.ConnectionError:
        return "ERROR: Kernel is not running on localhost:8000"

    def _run():
        return _post("/llm/chat", {
            "message": message,
            "system_prompt": system_prompt,
            "use_tools": use_tools,
        }, timeout=300)

    task_id = _submit_background("llm_chat", _run)
    return json.dumps({"task_id": task_id, "status": "running", "hint": "Use aios_task_result to get the response"})


if __name__ == "__main__":
    mcp.run(transport="stdio")
