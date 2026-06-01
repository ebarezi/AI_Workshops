import argparse
import asyncio
import logging
import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "ollama:llama3.2")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_llm():
    """
    Initialize the LLM from environment configuration.

    Uses init_chat_model() which accepts a "provider:model" string and
    optional kwargs. The base_url kwarg is passed through to the underlying
    provider, enabling connections to custom endpoints like Open WebUI,
    vLLM, or LM Studio.
    """
    
    kwargs = {"temperature": 0}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return init_chat_model(LLM_MODEL, **kwargs)


def build_graph(tools):
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


SYSTEM_PROMPT = """
You are the Boilermaker Autonomous TA for a Purdue course. Your job is to help students and instructors manage course planning using the tools available.

Tools:
- search_knowledge_base(query): search the course knowledge base for syllabus details, policies, and study guidance.
- get_academic_calendar(query): read the academic calendar or schedule details from the course calendar.
- create_notification(subject, body): write a formatted notification announcement to the announcements file.

Always use the tools when you need facts from the knowledge base or calendar. Use create_notification only when you have a final announcement or message to write.
"""


DEFAULT_QUESTION = (
    "A professor wants to schedule a CS course review session that does not conflict with holidays or exams. "
    "Summarize the plan and write a notification announcement for students."
)


async def run_once(agent, question: str) -> None:
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    human_message = HumanMessage(content=question)

    log.info("Starting agent invocation...")

    result = await agent.ainvoke({
        "messages": [system_message, human_message],
    })

    final_message = result["messages"][-1]
    log.info("Agent report:\n%s", final_message.content)

    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            print("Agent reply:\n", msg.content)


async def main(question: str):
    log.info("Boilermaker TA agent starting up")
    log.info("MCP server:  %s", MCP_SERVER_URL)
    log.info("LLM model:   %s", LLM_MODEL)

    try:
        client = MultiServerMCPClient({
            "boiler_ta": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            },
        })
        tools = await client.get_tools()
    except Exception as e:
        log.error("Could not connect to MCP server at %s", MCP_SERVER_URL)
        log.error("Is step2_boiler_ta_server.py running?")
        log.error("Detail: %s", e)
        return

    log.info("Loaded %d tools: %s", len(tools), [t.name for t in tools])

    agent = build_graph(tools)
    await run_once(agent, question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boilermaker TA Agent")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask the agent")
    args = parser.parse_args()

    print("Starting Boilermaker TA agent. Ensure step2_boiler_ta_server.py is running first.")
    asyncio.run(main(args.question))
