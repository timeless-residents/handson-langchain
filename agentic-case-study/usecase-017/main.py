"""
Use Case 017: Human-in-the-Loop with LangGraph
"""

import sys
import json
from typing import TypedDict, Optional, Literal, List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our graph
class ContentState(TypedDict):
    prompt: str
    outline: Optional[List[str]]
    draft_content: Optional[str]
    human_feedback: Optional[Dict[str, Any]]
    final_content: Optional[str]
    status: Literal["draft_needed", "awaiting_feedback", "revising", "complete"]


# Node functions
def create_outline(state: ContentState) -> ContentState:
    """Creates an initial content outline"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = state["prompt"]

    response = llm.invoke(
        f"Create a brief outline for content with this prompt: '{prompt}'. "
        f"Return a JSON array of 3-5 main points to cover."
    )

    try:
        # Extract outline from response
        content = response.content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            outline = json.loads(json_content)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            outline = json.loads(content)
        else:
            outline = content.split("\n")
            outline = [line.strip("- ").strip() for line in outline if line.strip()]
    except:
        # Fallback if parsing fails
        outline = [
            "Introduction to the topic",
            "Main point 1",
            "Main point 2",
            "Conclusion and key takeaways",
        ]

    return {**state, "outline": outline, "status": "draft_needed"}


def draft_content(state: ContentState) -> ContentState:
    """Creates draft content based on the outline"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = state["prompt"]
    outline = state["outline"]

    response = llm.invoke(
        f"Write content based on this prompt: '{prompt}' "
        f"Follow this outline: {outline}. "
        f"Create engaging, informative content for a general audience."
    )

    return {**state, "draft_content": response.content, "status": "awaiting_feedback"}


def get_human_feedback() -> ContentState:
    """
    This is a special function that pauses execution and waits for human input.
    In a real implementation, this would connect to a user interface.
    For this example, we simulate by getting input from the console.
    """
    # This is a placeholder for the state - in a real system,
    # this function wouldn't take or return state directly
    print("\n=== HUMAN FEEDBACK REQUIRED ===")
    print("Review the generated content and provide feedback:")
    print("1. Type 'approve' to accept as-is")
    print("2. Type 'reject' to request a complete rewrite")
    print("3. Or provide specific feedback for revision")

    feedback = input("\nYour feedback: ")

    if feedback.lower().strip() == "approve":
        feedback_type = "approve"
        feedback_content = "Content approved as-is."
    elif feedback.lower().strip() == "reject":
        feedback_type = "reject"
        feedback_content = "Content rejected. Please rewrite completely."
    else:
        feedback_type = "revise"
        feedback_content = feedback

    # This simulates the human's contribution to state
    # In a real system, this would be handled by the LangGraph framework
    return {
        "human_feedback": {
            "type": feedback_type,
            "content": feedback_content,
            "timestamp": "simulated_timestamp",
        }
    }


def process_human_feedback(state: ContentState) -> ContentState:
    """Processes the human feedback and updates the state"""
    # In a real implementation, the human feedback would be already
    # incorporated into the state by the framework

    # For demo purposes, we'll update the state with our simulated feedback
    feedback = get_human_feedback()

    combined_state = {**state, **feedback}

    feedback_type = combined_state["human_feedback"]["type"]

    if feedback_type == "approve":
        return {
            **combined_state,
            "final_content": combined_state["draft_content"],
            "status": "complete",
        }
    else:
        return {**combined_state, "status": "revising"}


def revise_content(state: ContentState) -> ContentState:
    """Revises content based on human feedback"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    draft = state["draft_content"]
    feedback_type = state["human_feedback"]["type"]
    feedback_content = state["human_feedback"]["content"]

    if feedback_type == "reject":
        # Complete rewrite
        response = llm.invoke(
            f"The previous draft was rejected. Create a completely new version addressing: '{feedback_content}'. "
            f"Original prompt: '{state['prompt']}'"
        )
    else:
        # Targeted revision
        response = llm.invoke(
            f"Revise this content based on the following feedback:\n\n"
            f"ORIGINAL CONTENT:\n{draft}\n\n"
            f"FEEDBACK: {feedback_content}\n\n"
            f"Provide a revised version that addresses all feedback points."
        )

    return {**state, "draft_content": response.content, "status": "awaiting_feedback"}


# Create the graph
def create_human_in_loop_graph() -> StateGraph:
    """Creates the LangGraph with human-in-the-loop workflow"""
    graph = StateGraph(ContentState)

    # ノードを追加（ノード名を変更）
    graph.add_node("create_outline", create_outline)
    graph.add_node("create_draft", draft_content)  # ノード名を `create_draft` に変更
    graph.add_node("process_feedback", process_human_feedback)
    graph.add_node("revise_content", revise_content)

    # エッジの定義も更新
    graph.add_edge("create_outline", "create_draft")
    graph.add_edge("create_draft", "process_feedback")

    # 条件付きエッジの定義
    def route_after_feedback(state: ContentState) -> str:
        if state["status"] == "complete":
            return END
        else:
            return "revise_content"

    graph.add_conditional_edges(
        "process_feedback",
        route_after_feedback,
        {"revise_content": "revise_content", END: END},
    )

    # ループバック
    graph.add_edge("revise_content", "process_feedback")

    # エントリーポイントを設定
    graph.set_entry_point("create_outline")

    return graph


def main():
    """Run the human-in-the-loop LangGraph"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<content_prompt>"')
        sys.exit(1)

    prompt = sys.argv[1]

    # Initialize the state
    initial_state = {
        "prompt": prompt,
        "outline": None,
        "draft_content": None,
        "human_feedback": None,
        "final_content": None,
        "status": "draft_needed",
    }

    # Create and compile the graph
    graph = create_human_in_loop_graph().compile()

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output final results
    print("\n=== FINAL CONTENT ===\n")
    print(result["final_content"])

    print("\n=== PROCESS SUMMARY ===")
    print(f"Original Prompt: {result['prompt']}")
    print(f"Outline Points: {len(result['outline'])}")
    feedback_type = result["human_feedback"]["type"]
    print(f"Human Feedback Type: {feedback_type}")
    print(f"Final Status: {result['status']}")


if __name__ == "__main__":
    main()
