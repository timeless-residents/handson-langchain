"""
LangChain Agent: Web Search
===============================

This module implements a LangChain agent that performs web searches.
The agent can:
- Search for information using DuckDuckGo
- Process and summarize search results
- Answer questions based on the latest information available online

This is useful for getting up-to-date information beyond the LLM's training data.
"""

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()


def summarize_search(query):
    """
    Search for information and provide a summarized response.

    Args:
        query (str): The search query

    Returns:
        str: Summarized search results
    """
    search_results = search.run(query)

    # In a real implementation, you might process/filter the results here
    # For now, we'll return the raw results

    return search_results


# Create tool instances
tools = [
    Tool(
        name="WebSearch",
        func=search.run,
        description="Useful for searching the web for specific information. Input should be a search query.",
    ),
    Tool(
        name="SummarizedSearch",
        func=summarize_search,
        description="Useful for getting summarized information from the web. Input should be a search query.",
    ),
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Test queries
queries = [
    "Who won the most recent Nobel Prize in Literature?",
    "What were the major headlines yesterday?",
    "What is the current population of Tokyo?",
    "What are the latest developments in quantum computing?",
]


def main():
    print("Testing Web Search Agent:")
    print("-" * 50)

    for query in queries:
        print(f"\nQuestion: {query}")
        try:
            response = agent.invoke(query)
            print(f"Answer: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
