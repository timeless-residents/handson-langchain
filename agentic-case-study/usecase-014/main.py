"""
Use Case 014: Multi-Agent Collaboration with LangGraph
"""

import sys
import json
from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define the state for our multi-agent system
class CollaborationState(TypedDict):
    query: str
    research_findings: List[Dict[str, str]]
    analysis_results: Dict[str, Any]
    critique: Dict[str, Any]
    final_synthesis: str
    messages: List[Dict[str, str]]  # For inter-agent communication


# Define the agent nodes
def researcher_agent(state: CollaborationState) -> CollaborationState:
    """Researches information relevant to the query"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]

    # Prepare system message for the researcher role
    researcher_prompt = (
        "You are a meticulous researcher. Your job is to gather and organize information "
        "relevant to the query. Focus on finding diverse perspectives, key facts, and "
        "identifying important sub-topics. Format your findings as a JSON array of objects, "
        "each with 'topic', 'key_points', and 'relevance' fields."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": researcher_prompt},
            {"role": "user", "content": f"Research this topic thoroughly: {query}"},
        ]
    )

    try:
        # Parse the response to extract structured findings
        findings_text = response.content
        # Find JSON content if embedded in explanatory text
        if "```json" in findings_text:
            json_content = findings_text.split("```json")[1].split("```")[0].strip()
            findings = json.loads(json_content)
        else:
            findings = json.loads(findings_text)

        if not isinstance(findings, list):
            findings = [findings]
    except:
        # Fallback if parsing fails
        findings = [
            {
                "topic": "General information",
                "key_points": ["Unable to parse structured findings"],
                "relevance": "Directly related to query",
            }
        ]

    # Add a message to the communication log
    message = {
        "from": "Researcher",
        "content": f"I've gathered information on {len(findings)} topics related to '{query}'.",
    }

    messages = state.get("messages", [])
    messages.append(message)

    return {**state, "research_findings": findings, "messages": messages}


def analyst_agent(state: CollaborationState) -> CollaborationState:
    """Analyzes the research findings and identifies patterns/insights"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]
    findings = state["research_findings"]

    # Prepare system message for the analyst role
    analyst_prompt = (
        "You are an insightful analyst. Your job is to process research findings, "
        "identify patterns, draw connections between topics, and extract meaningful insights. "
        "Organize your analysis as a JSON object with keys for 'main_insights', 'patterns', "
        "'controversies', and 'knowledge_gaps'."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": analyst_prompt},
            {
                "role": "user",
                "content": f"Analyze these research findings related to: {query}\n\n"
                + f"FINDINGS: {json.dumps(findings, indent=2)}",
            },
        ]
    )

    try:
        # Parse the response to extract structured analysis
        analysis_text = response.content
        # Find JSON content if embedded in explanatory text
        if "```json" in analysis_text:
            json_content = analysis_text.split("```json")[1].split("```")[0].strip()
            analysis = json.loads(json_content)
        else:
            analysis = json.loads(analysis_text)
    except:
        # Fallback if parsing fails
        analysis = {
            "main_insights": ["Analysis could not be structured properly"],
            "patterns": [],
            "controversies": [],
            "knowledge_gaps": [],
        }

    # Add a message to the communication log
    main_insights_count = len(analysis.get("main_insights", []))
    message = {
        "from": "Analyst",
        "content": f"I've analyzed the research and identified {main_insights_count} key insights.",
    }

    messages = state.get("messages", [])
    messages.append(message)

    return {**state, "analysis_results": analysis, "messages": messages}


def critic_agent(state: CollaborationState) -> CollaborationState:
    """Critically evaluates the analysis and identifies weaknesses"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]
    findings = state["research_findings"]
    analysis = state["analysis_results"]

    # Prepare system message for the critic role
    critic_prompt = (
        "You are a constructive critic. Your job is to evaluate the research and analysis, "
        "identify weaknesses, spot potential biases, and suggest improvements. Format your "
        "critique as a JSON object with keys for 'strengths', 'weaknesses', 'potential_biases', "
        "and 'improvement_suggestions'."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": critic_prompt},
            {
                "role": "user",
                "content": f"Critically evaluate this research and analysis on: {query}\n\n"
                + f"RESEARCH FINDINGS: {json.dumps(findings, indent=2)}\n\n"
                + f"ANALYSIS: {json.dumps(analysis, indent=2)}",
            },
        ]
    )

    try:
        # Parse the response to extract structured critique
        critique_text = response.content
        # Find JSON content if embedded in explanatory text
        if "```json" in critique_text:
            json_content = critique_text.split("```json")[1].split("```")[0].strip()
            critique = json.loads(json_content)
        else:
            critique = json.loads(critique_text)
    except:
        # Fallback if parsing fails
        critique = {
            "strengths": ["Some valuable information was gathered"],
            "weaknesses": ["Critique could not be structured properly"],
            "potential_biases": [],
            "improvement_suggestions": ["Consider gathering more diverse perspectives"],
        }

    # Add a message to the communication log
    weakness_count = len(critique.get("weaknesses", []))
    suggestion_count = len(critique.get("improvement_suggestions", []))
    message = {
        "from": "Critic",
        "content": f"I've identified {weakness_count} weaknesses and have {suggestion_count} suggestions for improvement.",
    }

    messages = state.get("messages", [])
    messages.append(message)

    return {**state, "critique": critique, "messages": messages}


def synthesizer_agent(state: CollaborationState) -> CollaborationState:
    """Synthesizes all information into a cohesive final answer"""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    query = state["query"]
    findings = state["research_findings"]
    analysis = state["analysis_results"]
    critique = state["critique"]

    # Prepare system message for the synthesizer role
    synthesizer_prompt = (
        "You are an expert synthesizer. Your job is to integrate research findings, analysis, "
        "and critique into a comprehensive, balanced, and insightful response. Create a well-structured "
        "answer that addresses the original query while acknowledging different perspectives and limitations."
    )

    response = llm.invoke(
        [
            {"role": "system", "content": synthesizer_prompt},
            {
                "role": "user",
                "content": f"Synthesize a comprehensive answer to: {query}\n\n"
                + f"RESEARCH FINDINGS: {json.dumps(findings, indent=2)}\n\n"
                + f"ANALYSIS: {json.dumps(analysis, indent=2)}\n\n"
                + f"CRITIQUE: {json.dumps(critique, indent=2)}\n\n"
                + f"Create a well-structured, balanced response that incorporates all perspectives and acknowledges limitations.",
            },
        ]
    )

    # Add a message to the communication log
    message = {
        "from": "Synthesizer",
        "content": "I've created a comprehensive synthesis incorporating all perspectives and addressing the critique.",
    }

    messages = state.get("messages", [])
    messages.append(message)

    return {**state, "final_synthesis": response.content, "messages": messages}


# Create the multi-agent graph
def create_collaboration_graph() -> StateGraph:
    """Creates the LangGraph for multi-agent collaboration"""
    # Initialize the graph
    graph = StateGraph(CollaborationState)

    # Add agent nodes to the graph
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    # Define the workflow as a linear process for simplicity
    # In more complex scenarios, conditional routing could be added
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", "synthesizer")
    graph.add_edge("synthesizer", END)

    # Set the entry point
    graph.set_entry_point("researcher")

    return graph


def main():
    """Run the multi-agent collaboration system"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<query>"')
        sys.exit(1)

    query = sys.argv[1]

    # Initialize the state
    initial_state = {
        "query": query,
        "research_findings": [],
        "analysis_results": {},
        "critique": {},
        "final_synthesis": "",
        "messages": [],
    }

    # Create and compile the graph
    graph = create_collaboration_graph().compile()

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output results
    print("\n=== MULTI-AGENT COLLABORATION RESULTS ===\n")

    print("--- Query ---")
    print(result["query"])

    print("\n--- Agent Communication Log ---")
    for i, message in enumerate(result["messages"], 1):
        print(f"{i}. {message['from']}: {message['content']}")

    print("\n--- Final Synthesis ---")
    print(result["final_synthesis"])


if __name__ == "__main__":
    main()
