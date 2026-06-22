"""
AIOS Orchestrator — Task Planner and Executor.
Takes a high-level task, breaks it into subtasks using LLM,
assigns agents by capabilities, and executes the pipeline.
"""

import json
import logging
import requests
import time
from typing import Any

logger = logging.getLogger(__name__)

KERNEL_URL = "http://localhost:8000"

PLANNER_SYSTEM_PROMPT = """You are the AIOS Task Planner. You break down a user's high-level task into concrete subtasks and assign each to the best available agent.

Available agents:
{agents_block}

Rules:
- Output ONLY valid JSON, no markdown, no explanation.
- Each step has: "step" (number), "agent_id", "subtask" (clear instruction), "depends_on" (list of step numbers whose results this step needs, or []).
- Assign agents based on their capabilities. Match the subtask type to agent strengths.
- Use the minimum number of steps needed. Don't over-split.
- Steps with empty depends_on can run in parallel (but we run sequentially for now).
- If the task is simple and fits one agent, return a single step.

Output format:
{{"steps": [{{"step": 1, "agent_id": "aios_local/xxx_agent", "subtask": "...", "depends_on": []}}]}}"""


def _get_agents() -> list[dict]:
    try:
        r = requests.get(f"{KERNEL_URL}/agents/roles", timeout=5)
        return r.json().get("agents", [])
    except Exception:
        return []


def _format_agents_block(agents: list[dict]) -> str:
    lines = []
    for a in agents:
        caps = ", ".join(a.get("capabilities", []))
        lines.append(f'- {a["agent_id"]}: capabilities=[{caps}]. {a.get("strengths", "")}')
    return "\n".join(lines)


def _call_llm(messages: list[dict], model: str = None, provider: str = None) -> str:
    """Call LLM through kernel /query endpoint, optionally with specific model."""
    payload = {
        "agent_name": "orchestrator",
        "query_type": "llm",
        "query_data": {
            "messages": messages,
            "action_type": "chat",
            "message_return_type": "text",
        }
    }
    if model and provider:
        payload["query_data"]["llms"] = [{"name": model, "provider": provider}]

    r = requests.post(f"{KERNEL_URL}/query", json=payload, timeout=120)
    data = r.json()
    response = data.get("response", data)
    if isinstance(response, dict):
        msg = response.get("response_message", "")
        if msg:
            return msg
        content = response.get("content", "")
        if content:
            return content
    return str(response)


def create_plan(task: str, project_id: str = "global", project_path: str = "",
                model: str = None, provider: str = None,
                custom_system_prompt: str = None) -> dict:
    """Use LLM to create an execution plan for the task."""
    agents = _get_agents()
    if not agents:
        return {"error": "No agents available"}

    agents_block = _format_agents_block(agents)

    if custom_system_prompt:
        system_msg = custom_system_prompt.replace("{agents_block}", agents_block)
    else:
        system_msg = PLANNER_SYSTEM_PROMPT.format(agents_block=agents_block)

    user_msg = f"Task: {task}"
    if project_path:
        user_msg += f"\nProject directory: {project_path}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    llm_response = _call_llm(messages, model=model, provider=provider)

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


def execute_plan(plan: dict) -> dict:
    """Execute a plan step by step, passing results forward."""
    steps = plan.get("steps", [])
    if not steps:
        return {"error": "Empty plan", "plan": plan}

    project_id = plan.get("project_id", "global")
    project_path = plan.get("project_path", "")
    results = {}
    plan["status"] = "running"

    for step in steps:
        step_num = step["step"]
        agent_id = step["agent_id"]
        subtask = step["subtask"]
        depends_on = step.get("depends_on", [])

        context_parts = []
        for dep in depends_on:
            if dep in results:
                context_parts.append(f"[Step {dep} result]: {results[dep][:1000]}")

        full_task = subtask
        if context_parts:
            full_task = "Previous results:\n" + "\n".join(context_parts) + "\n\nYour task: " + subtask

        logger.info(f"Orchestrator: step {step_num} -> {agent_id}: {subtask[:80]}")

        try:
            submit_resp = requests.post(f"{KERNEL_URL}/agents/submit", json={
                "agent_id": agent_id,
                "agent_config": {
                    "task": full_task,
                    "project_id": project_id,
                    "project_path": project_path,
                }
            }, timeout=10)
            submit_data = submit_resp.json()

            if submit_data.get("status") != "success":
                step["result"] = f"Submit failed: {submit_data}"
                step["status"] = "error"
                results[step_num] = step["result"]
                continue

            execution_id = submit_data["execution_id"]
            step["execution_id"] = execution_id
            step["status"] = "running"

            timeout = 600
            start = time.time()
            while time.time() - start < timeout:
                status_resp = requests.get(f"{KERNEL_URL}/agents/{execution_id}/status", timeout=10)
                status_data = status_resp.json()

                if status_data.get("status") == "completed":
                    result = status_data.get("result", {})
                    result_text = result.get("result", str(result)) if isinstance(result, dict) else str(result)
                    step["result"] = result_text
                    step["status"] = "completed"
                    results[step_num] = result_text
                    break
                elif status_data.get("status") == "error":
                    step["result"] = f"Agent error: {status_data}"
                    step["status"] = "error"
                    results[step_num] = step["result"]
                    break

                time.sleep(3)
            else:
                step["result"] = f"Timeout after {timeout}s"
                step["status"] = "timeout"
                results[step_num] = step["result"]

        except Exception as e:
            step["result"] = f"Exception: {e}"
            step["status"] = "error"
            results[step_num] = step["result"]

    all_done = all(s.get("status") == "completed" for s in steps)
    plan["status"] = "completed" if all_done else "partial"
    return plan
