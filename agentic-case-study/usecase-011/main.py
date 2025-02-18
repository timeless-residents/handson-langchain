"""
Use Case 011: Simple LangGraph Introduction
"""

import sys
import json
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our graph
class AgentState(TypedDict):
    problem: str
    steps: list[str]
    solution: str
    final_answer: str


# Define the nodes for our graph
def problem_breakdown(state: AgentState) -> AgentState:
    """Breaks down the problem into steps"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]

    response = llm.invoke(
        f"Break down this problem into clear steps: '{problem}'. "
        f"Format your response as a JSON list of steps."
    )

    try:
        steps = json.loads(response.content)
        if not isinstance(steps, list):
            steps = ["Understand the problem", "Solve systematically", "Verify answer"]
    except:
        steps = ["Understand the problem", "Solve systematically", "Verify answer"]

    return {"problem": problem, "steps": steps, "solution": "", "final_answer": ""}


def solve_problem(state: AgentState) -> AgentState:
    """Solves the problem based on the breakdown steps"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]
    steps = state["steps"]

    response = llm.invoke(
        f"Solve this problem: '{problem}' by following these steps: {steps}. "
        f"Show your work clearly, explaining each step."
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


# Create the graph
def create_agent_graph() -> StateGraph:
    """Creates the LangGraph for the reasoning agent"""
    # Initialize the graph
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("breakdown", problem_breakdown)
    graph.add_node("solve", solve_problem)
    graph.add_node("finalize", create_final_answer)

    # Define edges
    graph.add_edge("breakdown", "solve")
    graph.add_edge("solve", "finalize")
    graph.add_edge("finalize", END)

    # Set the entry point
    graph.set_entry_point("breakdown")

    return graph


def main():
    """Run the LangGraph agent"""
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

    print("\n--- Solution Steps ---")
    for i, step in enumerate(result["steps"], 1):
        print(f"{i}. {step}")

    print("\n--- Detailed Solution ---")
    print(result["solution"])

    print("\n--- Final Answer ---")
    print(result["final_answer"])


if __name__ == "__main__":
    main()
