"""LangGraph agent demo — the 5-minute Glassbox experience.

This builds a simple ReAct agent with tools, wraps it with observe(),
and runs a multi-step scenario. Every LLM call within every node
automatically produces a ContextPack.

Usage:
    pip install glassbox[langgraph,server] langchain-openai
    export OPENAI_API_KEY=sk-...
    python examples/demo_langgraph.py
"""

import threading
from glassbox import observe, serve

# Start viewer
threading.Thread(target=serve, daemon=True).start()

# --- Build a simple agent with tools ---

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "Cloudy, 12°C, light rain expected",
        "tokyo": "Sunny, 24°C, clear skies",
        "new york": "Partly cloudy, 18°C, mild breeze",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


@tool
def recommend_outfit(weather: str) -> str:
    """Recommend an outfit based on weather conditions."""
    if "rain" in weather.lower():
        return "Waterproof jacket, umbrella, and boots"
    elif "sunny" in weather.lower() or "clear" in weather.lower():
        return "Light clothing, sunglasses, and sunscreen"
    else:
        return "Layered outfit with a light jacket"


tools = [get_weather, recommend_outfit]


def build_agent():
    """Build a LangGraph ReAct agent."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": "__end__"})
    graph.add_edge("tools", "agent")

    return graph.compile()


# --- Run the agent with Glassbox observation ---

print("Building agent...")
graph = build_agent()

print("Wrapping with observe()...")
observed = observe(graph, agent_name="weather-agent", app_name="glassbox-langgraph-demo")

print("Running agent...\n")
result = observed.invoke({
    "messages": [
        HumanMessage(content="I'm traveling to London tomorrow. What's the weather like and what should I wear?")
    ]
})

# Print the final response
final_message = result["messages"][-1]
print(f"Agent response:\n{final_message.content}\n")
print("  Open http://localhost:4100 to see ContextPacks for every LLM call")
print("  You'll see: agent → tool call → tool result → agent (multi-step)")
print("  Press Ctrl+C to stop\n")

try:
    threading.Event().wait()
except KeyboardInterrupt:
    pass
