"""
Use Case 013: Iterative Refinement with LangGraph
"""

import sys
import json
from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our graph
class AgentState(TypedDict):
    problem: str
    current_solution: str
    quality_assessment: str
    quality_score: float  # 0-10 scale
    improvement_suggestions: List[str]
    iteration_count: int
    final_solution: str


# Define the nodes for our graph
def initial_solution(state: AgentState) -> AgentState:
    """Creates an initial solution to the problem"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]

    response = llm.invoke(
        f"Create an initial solution to this problem: '{problem}'. "
        f"Focus on correctness first, optimization second."
    )

    return {
        "problem": problem,
        "current_solution": response.content,
        "quality_assessment": "",
        "quality_score": 0.0,
        "improvement_suggestions": [],
        "iteration_count": 0,
        "final_solution": "",
    }


def assess_solution(state: AgentState) -> AgentState:
    """Evaluates the current solution and suggests improvements"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]
    current_solution = state["current_solution"]
    iteration = state["iteration_count"]

    response = llm.invoke(
        f"Assess this solution to the problem: '{problem}'\n\n"
        f"SOLUTION:\n{current_solution}\n\n"
        f"Provide: \n"
        f"1. A quality score from 0-10 (where 10 is perfect)\n"
        f"2. An assessment of its strengths and weaknesses\n"
        f"3. A list of specific suggestions for improvement\n\n"
        f"Format your response as a JSON object with keys: 'score', 'assessment', 'suggestions'"
    )

    try:
        result = json.loads(response.content)
        quality_score = float(result.get("score", 5.0))
        quality_assessment = result.get("assessment", "No assessment provided")
        improvement_suggestions = result.get("suggestions", ["No suggestions provided"])

        if not isinstance(improvement_suggestions, list):
            improvement_suggestions = [improvement_suggestions]
    except:
        # Fallback if JSON parsing fails
        quality_score = 5.0
        quality_assessment = "Unable to parse assessment"
        improvement_suggestions = ["Improve overall solution"]

    return {
        **state,
        "quality_assessment": quality_assessment,
        "quality_score": quality_score,
        "improvement_suggestions": improvement_suggestions,
        "iteration_count": iteration + 1,
    }


def refine_solution(state: AgentState) -> AgentState:
    """Refines the solution based on assessment and suggestions"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]
    current_solution = state["current_solution"]
    suggestions = state["improvement_suggestions"]
    quality_assessment = state["quality_assessment"]
    iteration = state["iteration_count"]

    response = llm.invoke(
        f"Refine this solution for iteration {iteration}:\n\n"
        f"PROBLEM: {problem}\n\n"
        f"CURRENT SOLUTION:\n{current_solution}\n\n"
        f"ASSESSMENT: {quality_assessment}\n\n"
        f"IMPROVEMENT SUGGESTIONS:\n"
        + "\n".join([f"- {s}" for s in suggestions])
        + "\n\n"
        f"Please provide an improved version that addresses these suggestions."
    )

    return {**state, "current_solution": response.content}


def finalize_solution(state: AgentState) -> AgentState:
    """Creates the final polished solution"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    problem = state["problem"]
    current_solution = state["current_solution"]

    response = llm.invoke(
        f"Create a final, polished solution to this problem: '{problem}'\n\n"
        f"Based on the current solution:\n{current_solution}\n\n"
        f"Make sure it's well-formatted, optimized, and includes any necessary explanations."
    )

    return {**state, "final_solution": response.content}


# Define routing function
def should_continue_refining(state: AgentState) -> str:
    """Determines whether to continue refining or finalize the solution"""
    # Check if reached quality threshold or max iterations
    if state["quality_score"] >= 8.0 or state["iteration_count"] >= 3:
        return "finalize"
    else:
        return "refine"


# Create the graph
def create_agent_graph() -> StateGraph:
    """Creates the LangGraph for iterative solution refinement"""
    # Initialize the graph
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("initial", initial_solution)
    graph.add_node("assess", assess_solution)
    graph.add_node("refine", refine_solution)
    graph.add_node("finalize", finalize_solution)

    # Define edges for the iterative loop
    graph.add_edge("initial", "assess")

    # Conditional branching based on assessment
    graph.add_conditional_edges(
        "assess", should_continue_refining, {"refine": "refine", "finalize": "finalize"}
    )

    # Complete the loop
    graph.add_edge("refine", "assess")
    graph.add_edge("finalize", END)

    # Set the entry point
    graph.set_entry_point("initial")

    return graph


def main():
    """Run the LangGraph agent with iterative refinement"""
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

    print(f"\n--- Final Solution (after {result['iteration_count']} iterations) ---")
    print(f"Quality Score: {result['quality_score']}/10")
    print("\n" + result["final_solution"])

    print("\n--- Refinement Process ---")
    print(f"Initial quality assessment: {result['quality_assessment']}")
    print("\nImprovement suggestions implemented:")
    for suggestion in result["improvement_suggestions"]:
        print(f"- {suggestion}")


if __name__ == "__main__":
    main()
