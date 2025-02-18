"""
Use Case 018: Parallel Processing with LangGraph
"""

import sys
import json
import time
import concurrent.futures
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# Define the states for our graphs
class MainState(TypedDict):
    query: str
    sub_questions: List[str]
    research_results: Dict[str, Dict[str, Any]]
    synthesis: Optional[str]
    execution_stats: Dict[str, Any]


class ResearchState(TypedDict):
    sub_question: str
    findings: Optional[str]
    key_points: Optional[List[str]]
    sources: Optional[List[str]]
    execution_time: Optional[float]


# Main graph nodes
def break_down_question(state: MainState) -> MainState:
    """Breaks down the main query into sub-questions for parallel research"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]

    response = llm.invoke(
        f"Break down this research question into 3-5 focused sub-questions that can be researched independently: '{query}'. "
        f"Return the sub-questions as a JSON array of strings. "
        f"Each sub-question should be specific, focused, and contribute to answering the main question."
    )

    try:
        # Extract sub-questions from response
        content = response.content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            sub_questions = json.loads(json_content)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            sub_questions = json.loads(content)
        else:
            # Parse line by line if not JSON format
            lines = content.split("\n")
            sub_questions = []
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("*") or line[0].isdigit()
                ):
                    sub_questions.append(line.strip("- *0123456789.").strip())

        # Ensure sub_questions is a list
        if not isinstance(sub_questions, list):
            sub_questions = [sub_questions]

        # Limit to 5 questions maximum
        sub_questions = sub_questions[:5]

    except:
        # Fallback if parsing fails
        sub_questions = [
            f"What are the key aspects of {query}?",
            f"What challenges exist regarding {query}?",
            f"What are current solutions related to {query}?",
        ]

    # Initialize research_results dictionary with empty entries for each sub-question
    research_results = {q: {} for q in sub_questions}

    return {
        **state,
        "sub_questions": sub_questions,
        "research_results": research_results,
        "execution_stats": {"start_time": time.time()},
    }


def run_parallel_research(state: MainState) -> MainState:
    """Executes research for each sub-question in parallel"""
    sub_questions = state["sub_questions"]

    # Store the original state to merge results back
    original_state = state

    # Create and compile the research graph
    research_graph = create_research_graph().compile()

    # Function to research one sub-question
    def research_task(sub_question):
        start_time = time.time()

        # Initialize research state
        research_state = {
            "sub_question": sub_question,
            "findings": None,
            "key_points": None,
            "sources": None,
            "execution_time": None,
        }

        # Run the research graph
        result = research_graph.invoke(research_state)

        # Calculate execution time
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time

        return sub_question, result

    # Use ThreadPoolExecutor to run tasks in parallel
    results = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(sub_questions)
    ) as executor:
        # Submit all research tasks
        future_to_question = {
            executor.submit(research_task, q): q for q in sub_questions
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_question):
            question, result = future.result()
            results[question] = result

    # Update the main state with research results
    return {**original_state, "research_results": results}


def synthesize_findings(state: MainState) -> MainState:
    """Synthesizes the findings from parallel research into a cohesive answer"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]
    research_results = state["research_results"]

    # Format research results for the prompt
    formatted_results = []
    for question, results in research_results.items():
        formatted_results.append(
            f"SUB-QUESTION: {question}\n"
            f"KEY POINTS: {json.dumps(results.get('key_points', []))}\n"
            f"DETAILED FINDINGS: {results.get('findings', 'No findings available')}\n"
        )

    all_results = "\n\n".join(formatted_results)

    response = llm.invoke(
        f"Synthesize these research findings into a comprehensive answer to the original question: '{query}'\n\n"
        f"RESEARCH FINDINGS:\n{all_results}\n\n"
        f"Provide a well-structured, cohesive response that integrates all the research. "
        f"Include key insights from each sub-question and highlight any interconnections."
    )

    # Calculate total execution time
    start_time = state["execution_stats"]["start_time"]
    total_time = time.time() - start_time

    # Gather execution statistics
    execution_stats = {
        "start_time": start_time,
        "end_time": time.time(),
        "total_time": total_time,
        "sub_question_times": {
            q: results.get("execution_time", 0)
            for q, results in research_results.items()
        },
    }

    return {**state, "synthesis": response.content, "execution_stats": execution_stats}


# Research graph nodes (for parallel execution)
def research_sub_question(state: ResearchState) -> ResearchState:
    """Researches a specific sub-question"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    sub_question = state["sub_question"]

    response = llm.invoke(
        f"Research this question thoroughly: '{sub_question}'. "
        f"Provide detailed findings with supporting evidence. "
        f"Include various perspectives and cite potential sources where applicable."
    )

    return {**state, "findings": response.content}


def extract_key_points(state: ResearchState) -> ResearchState:
    """Extracts key points from the research findings"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    findings = state["findings"]

    response = llm.invoke(
        f"Extract the 3-5 most important key points from these research findings. "
        f"Format your response as a JSON array of strings.\n\n"
        f"FINDINGS:\n{findings}"
    )

    try:
        # Extract key points from response
        content = response.content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            key_points = json.loads(json_content)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            key_points = json.loads(content)
        else:
            # Parse line by line if not JSON format
            lines = content.split("\n")
            key_points = []
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("*") or line[0].isdigit()
                ):
                    key_points.append(line.strip("- *0123456789.").strip())
    except:
        # Fallback if parsing fails
        key_points = ["Unable to parse key points"]

    # Ensure key_points is a list
    if not isinstance(key_points, list):
        key_points = [key_points]

    return {**state, "key_points": key_points}


def identify_sources(state: ResearchState) -> ResearchState:
    """Identifies potential sources for the research"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    sub_question = state["sub_question"]
    findings = state["findings"]

    response = llm.invoke(
        f"Based on this research about '{sub_question}', suggest 3-5 credible sources that might "
        f"provide this information. These can be academic journals, organizations, government agencies, "
        f"or reputable publications. Format as a JSON array of strings.\n\n"
        f"FINDINGS SUMMARY:\n{findings[:500]}..."  # Truncate for brevity
    )

    try:
        # Extract sources from response
        content = response.content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            sources = json.loads(json_content)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            sources = json.loads(content)
        else:
            # Parse line by line if not JSON format
            lines = content.split("\n")
            sources = []
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("*") or line[0].isdigit()
                ):
                    sources.append(line.strip("- *0123456789.").strip())
    except:
        # Fallback if parsing fails
        sources = ["Journal of relevant research", "Government report"]

    # Ensure sources is a list
    if not isinstance(sources, list):
        sources = [sources]

    return {**state, "sources": sources}


# Create the research graph (to be executed in parallel)
def create_research_graph() -> StateGraph:
    """Creates the research graph for processing a sub-question"""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("research", research_sub_question)
    graph.add_node("extract_key_points", extract_key_points)
    graph.add_node("identify_sources", identify_sources)

    # Add edges
    graph.add_edge("research", "extract_key_points")
    graph.add_edge("extract_key_points", "identify_sources")
    graph.add_edge("identify_sources", END)

    # Set entry point
    graph.set_entry_point("research")

    return graph


# Create the main graph
def create_main_graph() -> StateGraph:
    """Creates the main control flow graph with parallel processing"""
    graph = StateGraph(MainState)

    # Add nodes
    graph.add_node("break_down", break_down_question)
    graph.add_node("parallel_research", run_parallel_research)
    graph.add_node("synthesize", synthesize_findings)

    # Add edges for sequential flow
    graph.add_edge("break_down", "parallel_research")
    graph.add_edge("parallel_research", "synthesize")
    graph.add_edge("synthesize", END)

    # Set entry point
    graph.set_entry_point("break_down")

    return graph


def main():
    """Run the parallel processing LangGraph"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<research_query>"')
        sys.exit(1)

    query = sys.argv[1]

    # Initialize the state
    initial_state = {
        "query": query,
        "sub_questions": [],
        "research_results": {},
        "synthesis": None,
        "execution_stats": {},
    }

    # Create and compile the graph
    graph = create_main_graph().compile()

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output results
    print("\n=== PARALLEL RESEARCH RESULTS ===\n")

    print("--- Original Query ---")
    print(result["query"])

    print("\n--- Sub-Questions Researched in Parallel ---")
    for i, question in enumerate(result["sub_questions"], 1):
        print(f"{i}. {question}")
        sub_result = result["research_results"][question]
        execution_time = sub_result.get("execution_time", "unknown")
        print(
            f"   Time: {execution_time:.2f}s"
            if isinstance(execution_time, (int, float))
            else f"   Time: {execution_time}"
        )

    print("\n--- Performance Statistics ---")
    stats = result["execution_stats"]
    total_time = stats.get("total_time", 0)
    print(f"Total execution time: {total_time:.2f} seconds")

    if "sub_question_times" in stats:
        avg_time = sum(stats["sub_question_times"].values()) / len(
            stats["sub_question_times"]
        )
        print(f"Average time per sub-question: {avg_time:.2f} seconds")
        print(
            f"Parallelization speedup: {(avg_time * len(stats['sub_question_times'])) / total_time:.2f}x"
        )

    print("\n--- Synthesized Answer ---")
    print(result["synthesis"])


if __name__ == "__main__":
    main()
