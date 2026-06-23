"""
AIOS Orchestrator — Task Planner and Executor.
Takes a high-level task, breaks it into subtasks using LLM,
assigns agents by capabilities, and executes the pipeline.

Uses direct function calls instead of HTTP to avoid self-deadlock.
"""

import json
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

STEP_CONTEXT_LIMIT = 4000

PLANNER_SYSTEM_PROMPT = """You are the AIOS Task Planner. You break down a user's high-level task into concrete subtasks and assign each to the best available agent.

Available agents:
{agents_block}
{mcp_block}
Special agent — "orchestrator/mcp":
- Use agent_id "orchestrator/mcp" for steps that call external tools (Linear, GitHub, etc.).
- The orchestrator executes these steps directly — no agent is involved.
- The "subtask" field MUST contain a JSON object with "server", "tool", and "arguments". Example:
  {{"server": "linear", "tool": "save_issue", "arguments": {{"title": "Fix bug", "team": "Nikita"}}}}
- Use this for creating/reading issues, listing teams, fetching data, etc.
- Gather data via orchestrator/mcp FIRST, then pass results to agents that need them.

Rules:
- Output ONLY valid JSON, no markdown, no explanation.
- Each step has: "step" (number), "agent_id", "subtask" (clear instruction or MCP JSON), "depends_on" (list of step numbers whose results this step needs, or []).
- Assign agents based on their capabilities. Match the subtask type to agent strengths.
- Use the minimum number of steps needed. Don't over-split.
- Steps with empty depends_on can run in parallel.
- If the task is simple and needs only one MCP call, return a single orchestrator/mcp step.

Output format:
{{"steps": [{{"step": 1, "agent_id": "orchestrator/mcp", "subtask": "{{\\"server\\": \\"linear\\", \\"tool\\": \\"save_issue\\", \\"arguments\\": {{\\"title\\": \\"My task\\", \\"team\\": \\"Nikita\\"}}}}", "depends_on": []}}]}}"""


# --- Kernel callbacks (set by launch.py at startup) ---
_kernel_execute_request: Optional[Callable] = None
_kernel_submit_agent: Optional[Callable] = None
_kernel_await_execution: Optional[Callable] = None
_kernel_get_agents: Optional[Callable] = None


def bind_kernel(execute_request, submit_agent, await_execution, get_agents):
    global _kernel_execute_request, _kernel_submit_agent
    global _kernel_await_execution, _kernel_get_agents
    _kernel_execute_request = execute_request
    _kernel_submit_agent = submit_agent
    _kernel_await_execution = await_execution
    _kernel_get_agents = get_agents


def _format_agents_block(agents: list[dict]) -> str:
    lines = []
    for a in agents:
        caps = ", ".join(a.get("capabilities", []))
        lines.append(f'- {a["agent_id"]}: capabilities=[{caps}]. {a.get("strengths", "")}')
    return "\n".join(lines)


_mcp_manager_ref = None


def set_mcp_manager(mgr):
    global _mcp_manager_ref
    _mcp_manager_ref = mgr


def _format_mcp_block() -> str:
    if not _mcp_manager_ref:
        return ""
    try:
        tools = _mcp_manager_ref.list_all_tools()
    except Exception:
        return ""
    if not tools:
        return ""
    lines = ["Available MCP tools (agents can use these for external data):"]
    for srv, tool_list in tools.items():
        if not tool_list or (len(tool_list) == 1 and "error" in tool_list[0]):
            continue
        lines.append(f"  Server: {srv}")
        for t in tool_list[:10]:
            name = t.get("name", "?")
            desc = (t.get("description") or "")[:80]
            lines.append(f"    - {name}: {desc}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _call_llm_direct(messages: list[dict], max_tokens: int = 4000) -> str:
    from cerebrum.llm.apis import LLMQuery
    query = LLMQuery(
        messages=messages,
        action_type="chat",
        message_return_type="text",
        max_new_tokens=max_tokens,
    )
    result = _kernel_execute_request("orchestrator", query)
    response = result.get("response", result) if isinstance(result, dict) else result
    # Handle litellm ModelResponse objects (check first — most common path)
    if hasattr(response, "choices") and response.choices:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if content:
            return content
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            logger.warning("LLM returned reasoning_content but no content (likely truncated)")
            return ""
    if hasattr(response, "response_message") and response.response_message:
        return response.response_message
    if isinstance(response, dict):
        return response.get("response_message", "") or response.get("content", "") or str(response)
    return str(response)


def create_plan(task: str, project_id: str = "global", project_path: str = "",
                model: str = None, provider: str = None,
                custom_system_prompt: str = None, soul: str = None) -> dict:
    agents = _kernel_get_agents() if _kernel_get_agents else []
    if not agents:
        return {"error": "No agents available"}

    agents_block = _format_agents_block(agents)

    mcp_block = _format_mcp_block()

    if custom_system_prompt:
        system_msg = custom_system_prompt.replace("{agents_block}", agents_block)
    else:
        system_msg = PLANNER_SYSTEM_PROMPT.format(
            agents_block=agents_block,
            mcp_block=("\n" + mcp_block + "\n") if mcp_block else ""
        )

    user_parts = []
    if soul:
        user_parts.append(f"[Orchestrator instructions]\n{soul}")
    user_parts.append(f"[Task]\n{task}")
    if project_path:
        user_parts.append(f"[Project directory]\n{project_path}")
    user_msg = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    llm_response = _call_llm_direct(messages)

    try:
        clean = llm_response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        plan = json.loads(clean)
        plan["task"] = task
        plan["project_id"] = project_id
        plan["project_path"] = project_path
        plan["status"] = "planned"
        return plan
    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON", "raw": llm_response, "task": task}


def _resolve_mcp_arguments(mcp_call: dict, results: dict, depends_on: list) -> dict:
    """Resolve placeholder arguments using context from dependent steps.
    Uses litellm directly to avoid kernel scheduler deadlock."""
    context_parts = []
    for dep in depends_on:
        if dep in results:
            context_parts.append(f"[Step {dep} result]:\n{results[dep][:STEP_CONTEXT_LIMIT]}")
    if not context_parts:
        return mcp_call.get("arguments", {})

    context_text = "\n\n".join(context_parts)
    original_args = json.dumps(mcp_call.get("arguments", {}), ensure_ascii=False)

    messages = [
        {"role": "system", "content": (
            "You fill in MCP tool arguments using data from previous pipeline steps.\n"
            "Output ONLY a valid JSON object with the final arguments. No markdown, no explanation.\n"
            "Rules:\n"
            "- Replace any placeholder (<...>, __...__) with actual values from context.\n"
            "- For Linear save_issue: 'team' must be the team NAME (e.g. 'Nikita'), NOT a UUID.\n"
            "- For 'description' fields: compose a clear summary from context if not already set.\n"
            "- Keep explicitly set values unchanged unless they are placeholders."
        )},
        {"role": "user", "content": (
            f"MCP tool: {mcp_call.get('server')}/{mcp_call.get('tool')}\n\n"
            f"Original arguments:\n{original_args}\n\n"
            f"Context from previous steps:\n{context_text}\n\n"
            "Return the final arguments as JSON:"
        )},
    ]

    try:
        import os, litellm
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        kwargs = {"model": "openai/moonshotai/kimi-k2.7-code", "messages": messages, "max_tokens": 1000}
        if base_url:
            kwargs["api_base"] = base_url
        resp = litellm.completion(**kwargs)
        raw = resp.choices[0].message.content or ""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            clean = clean.rsplit("```", 1)[0]
        resolved = json.loads(clean)
        if isinstance(resolved, dict):
            logger.info("Resolved MCP arguments: %s", json.dumps(resolved, ensure_ascii=False)[:200])
            return resolved
    except Exception as e:
        logger.warning("Failed to resolve MCP arguments: %s", e)

    return mcp_call.get("arguments", {})


def _run_mcp_step(step: dict, results: dict) -> None:
    """Execute an orchestrator/mcp step directly via mcp_manager."""
    step_num = step["step"]
    subtask = step["subtask"]
    depends_on = step.get("depends_on", [])

    logger.info("Orchestrator: MCP step %d: %s", step_num, subtask[:80])

    try:
        try:
            mcp_call = json.loads(subtask) if isinstance(subtask, str) else subtask
        except json.JSONDecodeError:
            step["result"] = f"Invalid MCP call JSON: {subtask[:200]}"
            step["status"] = "error"
            results[step_num] = step["result"]
            return

        server = mcp_call.get("server", "")
        tool = mcp_call.get("tool", "")
        arguments = mcp_call.get("arguments", {})

        if not server or not tool:
            step["result"] = f"MCP call missing server/tool: {mcp_call}"
            step["status"] = "error"
            results[step_num] = step["result"]
            return

        # Resolve arguments using LLM when step has dependencies
        if depends_on:
            has_dep_results = any(dep in results for dep in depends_on)
            if has_dep_results:
                arguments = _resolve_mcp_arguments(mcp_call, results, depends_on)

        result = _mcp_manager_ref.call_tool(server, tool, arguments)
        result_str = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result

        step["result"] = result_str
        step["status"] = "completed"
        results[step_num] = result_str

    except Exception as e:
        step["result"] = f"MCP error: {e}"
        step["status"] = "error"
        results[step_num] = step["result"]


def _run_step(step: dict, results: dict, project_id: str, project_path: str) -> None:
    """Execute a single step: submit agent, poll until done."""
    step_num = step["step"]
    agent_id = step["agent_id"]
    subtask = step["subtask"]
    depends_on = step.get("depends_on", [])

    if agent_id == "orchestrator/mcp":
        return _run_mcp_step(step, results)

    context_parts = []
    for dep in depends_on:
        if dep in results:
            context_parts.append(f"[Step {dep} result]: {results[dep][:STEP_CONTEXT_LIMIT]}")

    full_task = subtask
    if context_parts:
        full_task = "Previous results:\n" + "\n".join(context_parts) + "\n\nYour task: " + subtask

    logger.info("Orchestrator: step %d -> %s: %s", step_num, agent_id, subtask[:80])

    try:
        task_input = {
            "task": full_task,
            "project_id": project_id,
            "project_path": project_path,
        }
        execution_id = _kernel_submit_agent(
            agent_name=agent_id, task_input=task_input,
        )
        step["execution_id"] = execution_id
        step["status"] = "running"

        timeout = 600
        poll_interval = 1.0
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = _kernel_await_execution(execution_id)
            except ValueError:
                step["result"] = "Process not found"
                step["status"] = "error"
                results[step_num] = step["result"]
                return

            if result is not None:
                result_text = result.get("result", str(result)) if isinstance(result, dict) else str(result)
                step["result"] = result_text
                step["status"] = "completed"
                results[step_num] = result_text
                return

            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 10.0)

        step["result"] = f"Timeout after {timeout}s"
        step["status"] = "timeout"
        results[step_num] = step["result"]

    except Exception as e:
        step["result"] = f"Exception: {e}"
        step["status"] = "error"
        results[step_num] = step["result"]


def execute_plan(plan: dict) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    steps = plan.get("steps", [])
    if not steps:
        return {"error": "Empty plan", "plan": plan}

    project_id = plan.get("project_id", "global")
    project_path = plan.get("project_path", "")
    results = {}
    plan["status"] = "running"

    step_map = {s["step"]: s for s in steps}
    completed_steps = set()

    with ThreadPoolExecutor(max_workers=4) as pool:
        while len(completed_steps) < len(steps):
            ready = []
            for s in steps:
                sn = s["step"]
                if sn in completed_steps:
                    continue
                deps = set(s.get("depends_on", []))
                if deps.issubset(completed_steps):
                    ready.append(s)

            if not ready:
                logger.error("Orchestrator: deadlock — no runnable steps left")
                break

            futures = {}
            for s in ready:
                f = pool.submit(_run_step, s, results, project_id, project_path)
                futures[f] = s["step"]

            for f in as_completed(futures):
                completed_steps.add(futures[f])

    all_done = all(s.get("status") == "completed" for s in steps)
    plan["status"] = "completed" if all_done else "partial"
    return plan
