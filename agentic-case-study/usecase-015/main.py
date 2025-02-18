"""
Use Case 015: Tool Use in LangGraph
"""

import sys
import json
import random
import datetime
from typing import TypedDict, List, Dict, Optional, Callable, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define tools
class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# Tool implementations (simulated)
def get_weather(location: str, date: Optional[str] = None) -> str:
    """Simulated weather data retrieval"""
    weather_conditions = [
        "sunny",
        "cloudy",
        "rainy",
        "stormy",
        "windy",
        "snowy",
        "foggy",
    ]
    temps = {
        "sunny": (70, 85),
        "cloudy": (60, 75),
        "rainy": (50, 65),
        "stormy": (45, 60),
        "windy": (55, 70),
        "snowy": (20, 35),
        "foggy": (50, 65),
    }

    # Simulate weather based on location
    weather = random.choice(weather_conditions)
    temp_range = temps[weather]
    temperature = random.randint(temp_range[0], temp_range[1])

    if date:
        return (
            f"Weather in {location} on {date}: {weather.capitalize()}, {temperature}°F"
        )
    else:
        return f"Current weather in {location}: {weather.capitalize()}, {temperature}°F"


def wiki_lookup(topic: str) -> str:
    """Simulated Wikipedia lookup"""
    common_topics = {
        "atmospheric pressure": (
            "Atmospheric pressure, also known as barometric pressure, is the pressure within the atmosphere of Earth. "
            "The standard atmosphere is a unit of pressure defined as 101,325 Pa, which is equivalent to 1013.25 mbar, "
            "760 mm Hg, 29.92 inches Hg, or 14.7 psi. Atmospheric pressure decreases with increasing altitude."
        ),
        "rainfall": (
            "Rainfall is a type of precipitation in which water drops from atmospheric water vapor condense and fall under gravity. "
            "Rainfall is measured using rain gauges, and global precipitation amounts to approximately 505,000 km³ of water per year."
        ),
        "climate": (
            "Climate is the long-term average of weather patterns in a specific region. Factors affecting climate include latitude, "
            "altitude, terrain, nearby water bodies, and ocean currents. Climate change refers to significant changes in global temperature, "
            "precipitation, wind patterns, and other measures of climate that occur over several decades or longer."
        ),
    }

    # Check for exact matches first
    if topic.lower() in common_topics:
        return common_topics[topic.lower()]

    # Check for partial matches
    for key, value in common_topics.items():
        if key in topic.lower() or topic.lower() in key:
            return value

    # Default response
    return f"Information about '{topic}' could not be found. Please try a different search term."


def calculate(expression: str) -> str:
    """Simple calculator for basic math operations"""
    try:
        # Use eval with some safe guards for simple calculations
        # In a real application, you would use a safer evaluation method
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic arithmetic operations are supported"

        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


def get_date_info(query: str) -> str:
    """Get date-related information"""
    today = datetime.datetime.now()

    if "today" in query.lower():
        return f"Today is {today.strftime('%A, %B %d, %Y')}"
    elif "tomorrow" in query.lower():
        tomorrow = today + datetime.timedelta(days=1)
        return f"Tomorrow will be {tomorrow.strftime('%A, %B %d, %Y')}"
    elif "yesterday" in query.lower():
        yesterday = today - datetime.timedelta(days=1)
        return f"Yesterday was {yesterday.strftime('%A, %B %d, %Y')}"
    elif "current month" in query.lower():
        return f"The current month is {today.strftime('%B %Y')}"
    elif "current year" in query.lower():
        return f"The current year is {today.year}"
    else:
        return f"Date query '{query}' not understood. Try asking about today, tomorrow, yesterday, current month, or current year."


# Create tool instances
TOOLS = [
    Tool(
        name="weather_tool",
        description="Get weather information for a location. Args: location (required), date (optional)",
        func=get_weather,
    ),
    Tool(
        name="wiki_tool",
        description="Look up information on a topic. Args: topic (required)",
        func=wiki_lookup,
    ),
    Tool(
        name="calculator",
        description="Perform mathematical calculations. Args: expression (required)",
        func=calculate,
    ),
    Tool(
        name="date_tool",
        description="Get date-related information. Args: query (required)",
        func=get_date_info,
    ),
]

# Create a lookup dictionary for tools
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


# Define the state for our graph
class ToolUseState(TypedDict):
    query: str
    thoughts: List[str]
    tools_to_use: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    final_answer: str


# Define the nodes for our graph
def analyze_query(state: ToolUseState) -> ToolUseState:
    """Analyzes the query and determines what tools might be needed"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]

    # Create tool descriptions for the prompt
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in TOOLS]
    )

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful research assistant with access to several tools. "
                "Analyze the query and determine which tools would be helpful to answer it. "
                "Respond with a JSON object that includes your thoughts and a list of tools to use.\n\n"
                f"Available tools:\n{tool_descriptions}\n\n"
                "Your response should be formatted as JSON with these fields:\n"
                "- thoughts: your reasoning about the query and what information is needed\n"
                "- tools_to_use: a list of objects, each with 'name' (must match an available tool), "
                "'args' (parameters to pass to the tool), and 'reason' (why this tool is needed)",
            },
            {"role": "user", "content": f"Analyze this query: {query}"},
        ]
    )

    try:
        # Parse the response to extract structured data
        result_text = response.content
        # Find JSON content if embedded in explanatory text
        if "```json" in result_text:
            json_content = result_text.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_content)
        else:
            result = json.loads(result_text)

        thoughts = result.get("thoughts", ["No explicit thoughts provided"])
        if isinstance(thoughts, str):
            thoughts = [thoughts]

        tools_to_use = result.get("tools_to_use", [])
        if not isinstance(tools_to_use, list):
            tools_to_use = [tools_to_use]
    except:
        # Fallback if parsing fails
        thoughts = ["Failed to parse structured analysis"]
        tools_to_use = []

    return {
        **state,
        "thoughts": thoughts,
        "tools_to_use": tools_to_use,
        "tool_results": [],
    }


def execute_tools(state: ToolUseState) -> ToolUseState:
    """Executes the selected tools and collects results"""
    tools_to_use = state["tools_to_use"]
    tool_results = []

    for tool_request in tools_to_use:
        tool_name = tool_request.get("name")
        tool_args = tool_request.get("args", {})

        if tool_name in TOOLS_BY_NAME:
            tool = TOOLS_BY_NAME[tool_name]
            try:
                # Execute the tool with provided arguments
                if isinstance(tool_args, dict):
                    result = tool(**tool_args)
                else:
                    # If args is a string or other type, pass it directly
                    result = tool(tool_args)

                # Record the successful result
                tool_results.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "status": "success",
                        "result": result,
                    }
                )
            except Exception as e:
                # Record the error
                tool_results.append(
                    {
                        "tool": tool_name,
                        "args": tool_args,
                        "status": "error",
                        "result": f"Error: {str(e)}",
                    }
                )
        else:
            # Tool not found
            tool_results.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "error",
                    "result": f"Error: Tool '{tool_name}' not found",
                }
            )

    return {**state, "tool_results": tool_results}


def analyze_tool_results(state: ToolUseState) -> ToolUseState:
    """Analyzes the tool results and creates a final answer"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]
    thoughts = state["thoughts"]
    tool_requests = state["tools_to_use"]
    tool_results = state["tool_results"]

    # Format tool requests and results for the prompt
    tool_info = []
    for i, (request, result) in enumerate(zip(tool_requests, tool_results)):
        tool_info.append(
            f"Tool {i+1}: {request['name']}\n"
            f"Arguments: {json.dumps(request['args'])}\n"
            f"Reason for use: {request.get('reason', 'No reason provided')}\n"
            f"Status: {result['status']}\n"
            f"Result: {result['result']}\n"
        )

    tool_info_text = "\n".join(tool_info)

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful research assistant. You've used various tools to gather information "
                "in response to a query. Now, synthesize all the tool results into a comprehensive, "
                "well-structured answer. Be sure to cite which tool provided which information.",
            },
            {
                "role": "user",
                "content": f"Original query: {query}\n\n"
                f"Your initial thoughts: {json.dumps(thoughts)}\n\n"
                f"TOOL RESULTS:\n{tool_info_text}\n\n"
                f"Based on these results, provide a comprehensive answer to the original query. "
                f"If the tools didn't provide adequate information, acknowledge the limitations.",
            },
        ]
    )

    return {**state, "final_answer": response.content}


# Create the graph
def create_tool_use_graph() -> StateGraph:
    """Creates the LangGraph for tool use"""
    # Initialize the graph
    graph = StateGraph(ToolUseState)

    # Add nodes to the graph
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("analyze_results", analyze_tool_results)

    # Define edges
    graph.add_edge("analyze_query", "execute_tools")
    graph.add_edge("execute_tools", "analyze_results")
    graph.add_edge("analyze_results", END)

    # Set the entry point
    graph.set_entry_point("analyze_query")

    return graph


def main():
    """Run the LangGraph agent with tool use"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<query>"')
        sys.exit(1)

    query = sys.argv[1]

    # Initialize the state
    initial_state = {
        "query": query,
        "thoughts": [],
        "tools_to_use": [],
        "tool_results": [],
        "final_answer": "",
    }

    # Create and compile the graph
    graph = create_tool_use_graph().compile()

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output results
    print("\n=== TOOL-ENHANCED RESEARCH RESULTS ===\n")

    print("--- Query ---")
    print(result["query"])

    print("\n--- Initial Analysis ---")
    for thought in result["thoughts"]:
        print(f"• {thought}")

    print("\n--- Tools Used ---")
    for i, tool_result in enumerate(result["tool_results"], 1):
        print(f"{i}. {tool_result['tool']}")
        print(f"   Args: {json.dumps(tool_result['args'])}")
        print(f"   Status: {tool_result['status']}")
        print(f"   Result: {tool_result['result']}")
        print()

    print("--- Final Answer ---")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
