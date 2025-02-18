"""
Use Case 012: Conditional Branching with LangGraph
"""

import sys
import json
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our graph
class AgentState(TypedDict):
    problem: str
    complexity: Literal["simple", "complex"]
    steps: list[str]
    solution: str
    final_answer: str


# Define the nodes for our graph
def analyze_complexity(state: AgentState) -> AgentState:
    """Analyzes the problem and determines its complexity"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]

    response = llm.invoke(
        f"Analyze this problem and determine if it's 'simple' or 'complex': '{problem}'. "
        f"Consider a problem simple if it can be solved in one or two straightforward steps. "
        f"Consider it complex if it requires multiple steps or advanced reasoning. "
        f"Return only the word 'simple' or 'complex'."
    )

    # Extract complexity assessment
    complexity_text = response.content.strip().lower()
    complexity = "simple" if "simple" in complexity_text else "complex"

    return {
        "problem": problem,
        "complexity": complexity,
        "steps": [],
        "solution": "",
        "final_answer": "",
    }


def solve_simple_problem(state: AgentState) -> AgentState:
    """Solves a simple problem with direct computation"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]

    response = llm.invoke(
        f"Solve this simple problem directly: '{problem}'. "
        f"Provide a straightforward calculation or reasoning."
    )

    return {**state, "steps": ["Direct calculation"], "solution": response.content}


def break_down_complex_problem(state: AgentState) -> AgentState:
    """Breaks down a complex problem into steps"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]

    response = llm.invoke(
        f"Break down this complex problem into clear steps: '{problem}'. "
        f"Format your response as a JSON list of steps."
    )

    try:
        steps = json.loads(response.content)
        if not isinstance(steps, list):
            steps = [
                "Understand the problem",
                "Break into sub-problems",
                "Solve each part",
                "Combine results",
            ]
    except:
        steps = [
            "Understand the problem",
            "Break into sub-problems",
            "Solve each part",
            "Combine results",
        ]

    return {**state, "steps": steps}


def solve_complex_problem(state: AgentState) -> AgentState:
    """Solves a complex problem by following the breakdown steps"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]
    steps = state["steps"]

    response = llm.invoke(
        f"Solve this complex problem: '{problem}' by following these steps: {steps}. "
        f"Show your work clearly for each step, explaining your reasoning process."
    )

    return {**state, "solution": response.content}


def create_final_answer(state: AgentState) -> AgentState:
    """Creates a concise final answer"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    solution = state["solution"]

    response = llm.invoke(
        f"Based on this solution: '{solution}', "
        f"provide a concise final answer to the original problem."
    )

    return {**state, "final_answer": response.content}


# Define routing function
def route_by_complexity(state: AgentState) -> str:
    """Routes to different nodes based on problem complexity"""
    if state["complexity"] == "simple":
        return "solve_simple"
    else:
        return "break_down_complex"


# Create the graph
def create_agent_graph() -> StateGraph:
    """Creates the LangGraph for the reasoning agent with conditional branching"""
    # Initialize the graph
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("analyze", analyze_complexity)
    graph.add_node("solve_simple", solve_simple_problem)
    graph.add_node("break_down_complex", break_down_complex_problem)
    graph.add_node("solve_complex", solve_complex_problem)
    graph.add_node("finalize", create_final_answer)

    # Define conditional edges
    graph.add_conditional_edges(
        "analyze",
        route_by_complexity,
        {"solve_simple": "solve_simple", "break_down_complex": "break_down_complex"},
    )

    # Define remaining edges
    graph.add_edge("solve_simple", "finalize")
    graph.add_edge("break_down_complex", "solve_complex")
    graph.add_edge("solve_complex", "finalize")
    graph.add_edge("finalize", END)

    # Set the entry point
    graph.set_entry_point("analyze")

    return graph


def main():
    """Run the LangGraph agent with conditional branching"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<problem_statement>"')
        sys.exit(1)

    problem = sys.argv[1]

    # Create and compile the graph
    graph = create_agent_graph().compile()

    # Execute the graph
    result = graph.invoke({"problem": problem})

    # Output results
    print("\n--- Problem ---")
    print(result["problem"])

    print("\n--- Complexity Assessment ---")
    print(result["complexity"].capitalize())

    print("\n--- Solution Steps ---")
    for i, step in enumerate(result["steps"], 1):
        print(f"{i}. {step}")

    print("\n--- Detailed Solution ---")
    print(result["solution"])

    print("\n--- Final Answer ---")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
