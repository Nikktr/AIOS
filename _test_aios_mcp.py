"""Test AIOS MCP Server: start as subprocess, send JSON-RPC, check response."""
import subprocess
import json
import sys
import time

SERVER_CMD = [
    sys.executable,
    r"C:\AIOS\AIOS-repo\aios\tool\aios_mcp_server.py",
]

def send_and_receive(proc, method, params=None, req_id=1):
    request = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        request["params"] = params
    line = json.dumps(request) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    while True:
        resp_line = proc.stdout.readline().strip()
        if not resp_line:
            continue
        try:
            resp = json.loads(resp_line)
            if resp.get("id") == req_id:
                return resp
        except json.JSONDecodeError:
            continue

proc = subprocess.Popen(
    SERVER_CMD,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    encoding="utf-8",
    errors="replace",
)

try:
    # 1. Initialize
    print("1. Sending initialize...")
    resp = send_and_receive(proc, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0.0"},
    }, req_id=1)
    print(f"   Server: {resp.get('result', {}).get('serverInfo', {})}")

    # 2. List tools
    print("2. Listing tools...")
    resp = send_and_receive(proc, "tools/list", {}, req_id=2)
    tools = resp.get("result", {}).get("tools", [])
    print(f"   Found {len(tools)} tools:")
    for t in tools:
        print(f"     - {t['name']}: {t.get('description', '')[:60]}")

    tool_names = [t["name"] for t in tools]

    # 3. Call aios_status
    print("3. Calling aios_status...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_status",
        "arguments": {},
    }, req_id=3)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    print(f"   Result: {text[:200]}")

    # 4. Call aios_mcp_list
    print("4. Calling aios_mcp_list...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_mcp_list",
        "arguments": {},
    }, req_id=4)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    servers = data.get("servers", {})
    for srv, tlist in servers.items():
        print(f"   {srv}: {len(tlist)} tools")

    # 5. Call aios_mcp_call (linear/list_teams)
    print("5. Calling aios_mcp_call (linear/list_teams)...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_mcp_call",
        "arguments": {"server": "linear", "tool": "list_teams", "arguments": {}},
    }, req_id=5)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    print(f"   Result: {text[:150]}")

    # 6. Call aios_list_agents
    print("6. Calling aios_list_agents...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_list_agents",
        "arguments": {},
    }, req_id=6)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    agents = data.get("agents", [])
    for a in agents:
        print(f"   {a.get('agent_id', '?')}: {a.get('capabilities', [])}")

    # 7. Call aios_plan (sync, returns plan directly)
    print("7. Calling aios_plan...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_plan",
        "arguments": {"task": "List all files in the current directory"},
    }, req_id=7)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    print(f"   Result: {text[:300]}")

    # 8. Call aios_orchestrate (async — returns task_id)
    print("8. Calling aios_orchestrate (async)...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_orchestrate",
        "arguments": {"task": "Say hello"},
    }, req_id=8)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    orch_task_id = data.get("task_id", "")
    print(f"   Got task_id: {orch_task_id}, status: {data.get('status')}")
    assert orch_task_id, "No task_id returned from aios_orchestrate"
    assert data.get("status") == "running"

    # 9. Call aios_task_list — should show the orchestrate task
    print("9. Calling aios_task_list...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_task_list",
        "arguments": {},
    }, req_id=9)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    tasks = json.loads(text) if text else []
    print(f"   Tasks: {len(tasks)}")
    for t in tasks:
        print(f"     - {t['task_id']}: {t['status']} ({t['type']})")

    # 10. Call aios_task_result — check orchestrate task status
    print("10. Calling aios_task_result...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_task_result",
        "arguments": {"task_id": orch_task_id},
    }, req_id=10)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    print(f"   Status: {data.get('status')}, type: {data.get('type')}")

    # 11. Call aios_llm_chat (async — returns task_id)
    print("11. Calling aios_llm_chat (async)...")
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_llm_chat",
        "arguments": {"message": "Say hello in one word"},
    }, req_id=11)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    llm_task_id = data.get("task_id", "")
    print(f"   Got task_id: {llm_task_id}, status: {data.get('status')}")
    assert llm_task_id, "No task_id returned from aios_llm_chat"

    # 12. Wait a bit and check llm_chat result
    print("12. Waiting 15s then checking aios_llm_chat result...")
    time.sleep(15)
    resp = send_and_receive(proc, "tools/call", {
        "name": "aios_task_result",
        "arguments": {"task_id": llm_task_id},
    }, req_id=12)
    content = resp.get("result", {}).get("content", [])
    text = content[0].get("text", "") if content else ""
    data = json.loads(text) if text else {}
    print(f"   Status: {data.get('status')}")
    if data.get("status") == "completed":
        result = data.get("result", {})
        print(f"   LLM response: {result.get('response', '')[:200]}")
    else:
        print(f"   Still running or error: {text[:200]}")

    # 13. Verify all expected tools are registered
    print("13. Checking all tools are registered...")
    expected = [
        "aios_status", "aios_list_agents", "aios_submit_agent",
        "aios_mcp_list", "aios_mcp_call",
        "aios_task_result", "aios_task_list",
        "aios_plan", "aios_orchestrate",
        "aios_llm_chat",
    ]
    for name in expected:
        assert name in tool_names, f"{name} not found in tools"
        print(f"   OK: {name}")

    print(f"\n[ALL {len(expected)} TOOLS VERIFIED - TESTS PASSED]")

finally:
    proc.terminate()
    proc.wait(timeout=5)
