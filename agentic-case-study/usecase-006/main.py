"""
LangChain Agent: Document QA Agent
===============================

This module implements a LangChain agent for document question answering.
The agent can:
- Load and process PDF, TXT, or other document types
- Split documents into chunks
- Create vector embeddings for semantic search
- Answer questions based on document content
- Cite sources with page/paragraph references

This is useful for creating assistants that can answer questions about specific documents.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

# Set up document paths - in a real app, these would be user-provided
DOCUMENTS_DIR = "documents"
DOCUMENT_PATHS = [
    os.path.join(DOCUMENTS_DIR, "sample_document.txt"),
]

# Make sure documents directory exists
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Create sample document if it doesn't exist
sample_content = """
# Sample Company Financial Report 2023

## Executive Summary

Our company has experienced significant growth in the fiscal year 2023. Revenue increased by 15% compared to the previous year, reaching $15.2 million. Our net profit margin improved from 12% to 14%, resulting in a total profit of $2.1 million.

## Financial Highlights

- Total Revenue: $15.2 million
- Operating Expenses: $11.7 million
- Net Profit: $2.1 million
- Earnings Per Share (EPS): $3.27
- Return on Investment (ROI): 12.8%

## Revenue Breakdown

Revenue sources:
1. Product A: $7.3 million (48%)
2. Product B: $4.5 million (30%)
3. Product C: $2.1 million (14%)
4. Services: $1.3 million (8%)

## Market Analysis

The market for our products has grown by approximately 7% this year. We've increased our market share from 23% to 26%, primarily due to the success of Product A and our expanded services offering.

## Future Outlook

We project continued growth in the next fiscal year, with an estimated revenue increase of 10-12%. New product lines currently in development are expected to contribute an additional $2-3 million in revenue by Q3 2024.

## Risks and Challenges

Potential challenges include:
- Increasing competition in the Product B segment
- Rising supply chain costs, potentially affecting our gross margin
- Regulatory changes in our primary markets

## Conclusion

The management team is confident in our strategy and our ability to maintain growth in the coming year. We remain committed to innovation and expanding our market presence.
"""

if not os.path.exists(DOCUMENT_PATHS[0]):
    with open(DOCUMENT_PATHS[0], "w") as f:
        f.write(sample_content)


def load_and_process_documents(document_paths: List[str]):
    """
    Load documents, split them into chunks, and create a vector store.

    Args:
        document_paths: List of paths to documents

    Returns:
        Chroma: Vector store with document chunks
    """
    # Load documents
    documents = []
    for path in document_paths:
        if path.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())
        # Add support for other document types as needed

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    return vector_store


# Load and process the documents
vector_store = load_and_process_documents(DOCUMENT_PATHS)

# Set up retrieval QA
llm = ChatOpenAI(temperature=0)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)


def answer_from_documents(query: str) -> str:
    """
    Answer questions based on document content.

    Args:
        query: The question to answer

    Returns:
        str: Answer with citations
    """
    result = qa_chain({"query": query})

    # Get the answer and source documents
    answer = result["result"]
    source_documents = result.get("source_documents", [])

    # Add citations if source documents are available
    if source_documents:
        answer += "\n\nSources:"
        for i, doc in enumerate(source_documents, 1):
            # Extract source information - in real implementation, you might extract page numbers, etc.
            source_info = doc.metadata.get("source", f"Document {i}")
            answer += f"\n{i}. {source_info}"

    return answer


# Create tool instances
tools = [
    Tool(
        name="DocumentQA",
        func=answer_from_documents,
        description="Useful for answering questions based on specific documents that have been loaded. Input should be a clear question about the document content.",
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
    "What was the company's total revenue in 2023?",
    "What is the breakdown of revenue by product?",
    "What are the main risks mentioned in the report?",
    "What is the projected growth for the next year?",
]


def main():
    print("Testing Document QA Agent:")
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
