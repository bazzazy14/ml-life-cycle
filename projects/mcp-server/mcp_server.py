# Cleaned from the completed Break Through Tech MCP notebook.

import uuid
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("todo-list")
TASKS = {}

@mcp.tool()
def add_task(task: str) -> dict:
    """Add a task to the todo list."""
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"task": task, "done": False}
    return {"id": task_id, "task": task, "done": False}

@mcp.tool()
def list_tasks() -> dict:
    """List all tasks."""
    return {"tasks": [{"id": tid, **data} for tid, data in TASKS.items()]}

@mcp.tool()
def complete_task(id: str) -> dict:
    """Mark a task as completed."""
    if id in TASKS:
        TASKS[id]["done"] = True
        return {"ok": True, "task": TASKS[id]["task"]}
    return {"ok": False, "task": None}

if __name__ == "__main__":
    mcp.run()

# Example OpenAI Agents SDK client used in the notebook:
#
# from agents import Agent, Runner
# from agents.mcp import MCPServerStdio
#
# async def demo():
#     server = MCPServerStdio(
#         params={"command": "python", "args": ["mcp_server.py"]},
#         client_session_timeout_seconds=60,
#     )
#     async with server:
#         agent = Agent(
#             name="Todo Assistant",
#             instructions="Use the available tools to manage the user's todo list.",
#             model="gpt-4.1",
#             mcp_servers=[server],
#         )
#         result = await Runner.run(agent, "Add buy groceries to my todo list")
#         print(result.final_output)
