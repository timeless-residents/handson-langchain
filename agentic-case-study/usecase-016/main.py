"""
Use Case 016: Simple Error Handling in LangGraph
"""

import sys
from typing import TypedDict, Optional, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our graph
class ProcessingState(TypedDict):
    input_text: str
    processed_result: Optional[str]
    error: Optional[str]
    status: Literal["success", "error", "recovered"]


# Node functions
def validate_input(state: ProcessingState) -> ProcessingState:
    """Validates the input text"""
    input_text = state["input_text"]

    # Check if input is valid
    if not input_text or len(input_text.strip()) < 3:
        return {**state, "error": "Input text is too short or empty", "status": "error"}

    return {**state, "status": "success"}


def process_text(state: ProcessingState) -> ProcessingState:
    """Processes the text normally"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    input_text = state["input_text"]

    response = llm.invoke(f"Summarize this text in one sentence: '{input_text}'")

    return {**state, "processed_result": response.content, "status": "success"}


def handle_error(state: ProcessingState) -> ProcessingState:
    """Handles the error by providing a default response"""
    return {
        **state,
        "processed_result": "Could not process the input due to validation error. Please provide longer text.",
        "status": "recovered",
    }


def create_final_output(state: ProcessingState) -> ProcessingState:
    """Creates the final output based on processing status"""
    status = state["status"]
    result = state["processed_result"]

    if status == "recovered":
        prefix = "[RECOVERED] "
    elif status == "error":
        prefix = "[ERROR] "
    else:
        prefix = ""

    if result:
        final_output = f"{prefix}{result}"
    else:
        final_output = f"{prefix}No result available"

    return {**state, "processed_result": final_output}


# Routing function
def route_based_on_validation(state: ProcessingState) -> str:
    """Routes to error handling or normal processing based on validation"""
    if state["status"] == "error":
        return "handle_error"
    else:
        return "process_text"


# Create the graph
def create_error_handling_graph() -> StateGraph:
    """Creates the LangGraph with error handling"""
    graph = StateGraph(ProcessingState)

    # Add nodes
    graph.add_node("validate", validate_input)
    graph.add_node("process_text", process_text)
    graph.add_node("handle_error", handle_error)
    graph.add_node("finalize", create_final_output)

    # Add conditional edge from validation
    graph.add_conditional_edges(
        "validate",
        route_based_on_validation,
        {"process_text": "process_text", "handle_error": "handle_error"},
    )

    # Add remaining edges
    graph.add_edge("process_text", "finalize")
    graph.add_edge("handle_error", "finalize")
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("validate")

    return graph


def main():
    """Run the error-handling LangGraph"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<input_text>"')
        sys.exit(1)

    input_text = sys.argv[1]

    # Initialize the state
    initial_state = {
        "input_text": input_text,
        "processed_result": None,
        "error": None,
        "status": "success",
    }

    # Create and compile the graph
    graph = create_error_handling_graph().compile()

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output results
    print("\n=== TEXT PROCESSING RESULTS ===\n")

    print("--- Input Text ---")
    print(f"'{result['input_text']}'")

    print("\n--- Status ---")
    print(result["status"].upper())

    if result["error"]:
        print("\n--- Error ---")
        print(result["error"])

    print("\n--- Result ---")
    print(result["processed_result"])


if __name__ == "__main__":
    main()
