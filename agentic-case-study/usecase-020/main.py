"""
Use Case 020: Advanced State Management in LangGraph
"""

import sys
import json
import uuid
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.graph import END


# Define Pydantic models for structured state components
class User(BaseModel):
    """Model for user information"""

    id: str
    name: str
    role: str


class EditOperation(BaseModel):
    """Model for tracking document edits"""

    operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    operation_type: Literal["create", "edit", "format", "review", "approve"]
    user_id: str
    description: str


class Section(BaseModel):
    """Model for document sections"""

    section_id: str
    title: str
    content: str
    order: int
    last_modified: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_modified_by: Optional[str] = None


class DocumentVersion(BaseModel):
    """Model for document versions"""

    version_id: str
    version_number: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    sections: List[Section]
    commit_message: str


class DocumentState(BaseModel):
    """Top-level model for document state"""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    current_version: int = 1
    versions: List[DocumentVersion] = []
    edit_history: List[EditOperation] = []
    active_users: List[User] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: Literal["draft", "review", "approved", "published"] = "draft"


# Define the state for our graph
class EditorState(TypedDict):
    prompt: str
    document: Optional[DocumentState]
    current_user: Optional[User]
    current_operation: Optional[EditOperation]
    message_log: List[Dict[str, str]]


# Helper functions for state management
def create_initial_document(prompt: str, user: User) -> DocumentState:
    """Creates an initial document from a prompt"""
    doc_id = str(uuid.uuid4())

    # Create initial document
    document = DocumentState(
        document_id=doc_id,
        title=f"Document based on: {prompt[:50]}...",
        description=prompt,
        active_users=[user],
    )

    # Create initial version with empty sections
    initial_version = DocumentVersion(
        version_id=str(uuid.uuid4()),
        version_number=1,
        sections=[],
        commit_message="Initial document creation",
    )

    # Add the version to document
    document.versions.append(initial_version)

    return document


def add_message_to_log(
    state: EditorState, sender: str, message: str
) -> List[Dict[str, str]]:
    """Adds a message to the state's message log"""
    messages = state.get("message_log", [])
    messages.append(
        {"timestamp": datetime.now().isoformat(), "sender": sender, "message": message}
    )
    return messages


def create_document_snapshot(document: DocumentState) -> Dict[str, Any]:
    """Creates a simplified snapshot of the document for display"""
    current_version = None
    for version in document.versions:
        if version.version_number == document.current_version:
            current_version = version
            break

    if not current_version:
        return {"error": "Could not find current version"}

    section_count = len(current_version.sections)
    word_count = sum(
        len(section.content.split()) for section in current_version.sections
    )
    edit_count = len(document.edit_history)

    return {
        "id": document.document_id,
        "title": document.title,
        "version": document.current_version,
        "status": document.status,
        "sections": section_count,
        "word_count": word_count,
        "edits": edit_count,
        "created_at": document.created_at,
    }


# Node functions
def initialize_document(state: EditorState) -> EditorState:
    """Initializes the document state"""
    prompt = state["prompt"]

    # Create a simulated user
    user = User(id=str(uuid.uuid4()), name="AI Editor", role="author")

    # Create the initial document
    document = create_initial_document(prompt, user)

    # Initialize message log
    message_log = [
        {
            "timestamp": datetime.now().isoformat(),
            "sender": "System",
            "message": f"Document initialized with prompt: '{prompt[:50]}...'",
        }
    ]

    return {
        **state,
        "document": document,
        "current_user": user,
        "message_log": message_log,
    }


def generate_document_content(state: EditorState) -> EditorState:
    """Generates initial document content"""
    document = state["document"]
    user = state["current_user"]
    prompt = state["prompt"]

    # Start an edit operation
    operation = EditOperation(
        operation_type="create",
        user_id=user.id,
        description="Generate initial document content",
    )

    # Generate document content with LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a document creation assistant. Generate a well-structured document "
                "with multiple sections based on the given prompt. Create a JSON structure with "
                "an array of sections, each with 'title', 'content', and 'order'.",
            },
            {
                "role": "user",
                "content": f"Create a structured document for this prompt: '{prompt}'. "
                f"Include 3-5 well-organized sections.",
            },
        ]
    )

    try:
        # Extract sections from response
        content = response.content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
            sections_data = json.loads(json_content)
        elif content.strip().startswith("[") and content.strip().endswith("]"):
            sections_data = json.loads(content)
        else:
            # Try to extract sections from text
            sections_data = []
            lines = content.split("\n")
            current_section = None
            section_content = []

            for line in lines:
                if line.strip().startswith("#") or line.strip().startswith("Section"):
                    # Save previous section if exists
                    if current_section:
                        sections_data.append(
                            {
                                "title": current_section,
                                "content": "\n".join(section_content),
                                "order": len(sections_data) + 1,
                            }
                        )
                        section_content = []

                    # Start new section
                    current_section = line.strip().lstrip("#").strip()
                else:
                    # Add to current section content
                    if current_section:
                        section_content.append(line)

            # Add the last section if exists
            if current_section and section_content:
                sections_data.append(
                    {
                        "title": current_section,
                        "content": "\n".join(section_content),
                        "order": len(sections_data) + 1,
                    }
                )

        # Ensure sections_data is a list
        if not isinstance(sections_data, list):
            sections_data = [sections_data]

        # Create Section objects
        sections = []
        for i, section_data in enumerate(sections_data):
            section = Section(
                section_id=str(uuid.uuid4()),
                title=section_data.get("title", f"Section {i+1}"),
                content=section_data.get("content", "No content provided"),
                order=section_data.get("order", i + 1),
                last_modified_by=user.id,
            )
            sections.append(section)

    except Exception:
        # Create fallback sections if parsing fails
        sections = [
            Section(
                section_id=str(uuid.uuid4()),
                title="Introduction",
                content=f"This document addresses the topic: {prompt}",
                order=1,
                last_modified_by=user.id,
            ),
            Section(
                section_id=str(uuid.uuid4()),
                title="Main Content",
                content=response.content[:500],
                order=2,
                last_modified_by=user.id,
            ),
            Section(
                section_id=str(uuid.uuid4()),
                title="Conclusion",
                content="Summary and next steps.",
                order=3,
                last_modified_by=user.id,
            ),
        ]

    # Create new version with generated sections
    new_version = DocumentVersion(
        version_id=str(uuid.uuid4()),
        version_number=1,
        sections=sections,
        commit_message="Initial content generation",
    )

    # Update document state
    document.versions = [new_version]  # Replace initial empty version

    # Record the operation
    document.edit_history.append(operation)

    # Update message log
    message_log = add_message_to_log(
        state, "AI Editor", f"Generated initial content with {len(sections)} sections"
    )

    return {
        **state,
        "document": document,
        "current_operation": operation,
        "message_log": message_log,
    }


def format_document(state: EditorState) -> EditorState:
    """Formats the document content for improved readability"""
    document = state["document"]
    user = state["current_user"]

    # Find current version
    current_version = None
    for version in document.versions:
        if version.version_number == document.current_version:
            current_version = version
            break

    if not current_version:
        # Error in state, create message and return unchanged
        message_log = add_message_to_log(
            state, "System", "Error: Could not find current document version"
        )
        return {**state, "message_log": message_log}

    # Start an edit operation
    operation = EditOperation(
        operation_type="format",
        user_id=user.id,
        description="Format document for improved readability",
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Format each section
    formatted_sections = []

    for section in current_version.sections:
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a document formatting expert. Improve the readability and structure of text "
                    "while preserving all information. Add proper formatting, bullet points where appropriate, "
                    "and ensure good paragraph structure.",
                },
                {
                    "role": "user",
                    "content": f"Format this document section titled '{section.title}':\n\n{section.content}",
                },
            ]
        )

        # Create updated section
        formatted_section = Section(
            section_id=section.section_id,
            title=section.title,
            content=response.content,
            order=section.order,
            last_modified=datetime.now().isoformat(),
            last_modified_by=user.id,
        )

        formatted_sections.append(formatted_section)

    # Create new version
    new_version_number = document.current_version + 1
    new_version = DocumentVersion(
        version_id=str(uuid.uuid4()),
        version_number=new_version_number,
        sections=formatted_sections,
        commit_message="Formatted document for improved readability",
    )

    # Update document
    document.versions.append(new_version)
    document.current_version = new_version_number
    document.edit_history.append(operation)

    # Update message log
    message_log = add_message_to_log(
        state,
        "AI Editor",
        f"Formatted {len(formatted_sections)} sections for improved readability",
    )

    return {
        **state,
        "document": document,
        "current_operation": operation,
        "message_log": message_log,
    }


def review_document(state: EditorState) -> EditorState:
    """Reviews the document and provides feedback"""
    document = state["document"]
    user = state["current_user"]

    # Find current version
    current_version = None
    for version in document.versions:
        if version.version_number == document.current_version:
            current_version = version
            break

    if not current_version:
        message_log = add_message_to_log(
            state, "System", "Error: Could not find current document version for review"
        )
        return {**state, "message_log": message_log}

    # Start a review operation
    operation = EditOperation(
        operation_type="review",
        user_id=user.id,
        description="Review document content and structure",
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Prepare full document for review
    full_document = "\n\n".join(
        [
            f"# {section.title}\n{section.content}"
            for section in sorted(current_version.sections, key=lambda s: s.order)
        ]
    )

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a document review specialist. Review this document for clarity, "
                "completeness, coherence, and overall quality. Provide specific feedback "
                "with an overall rating from 1-10.",
            },
            {
                "role": "user",
                "content": f"Review this document titled '{document.title}':\n\n{full_document}",
            },
        ]
    )

    # Change document status based on review
    document.status = "review"

    # Add review to edit history
    document.edit_history.append(operation)

    # Update message log
    message_log = add_message_to_log(
        state,
        "Reviewer",
        f"Completed document review. Feedback: {response.content[:100]}...",
    )

    return {
        **state,
        "document": document,
        "current_operation": operation,
        "message_log": message_log,
    }


def finalize_document(state: EditorState) -> EditorState:
    """Finalizes the document and prepares it for publication"""
    document = state["document"]
    user = state["current_user"]

    # Start an approval operation
    operation = EditOperation(
        operation_type="approve",
        user_id=user.id,
        description="Finalize document for publication",
    )

    # Change document status
    document.status = "approved"

    # Add approval to edit history
    document.edit_history.append(operation)

    # Update message log
    message_log = add_message_to_log(
        state,
        "Publisher",
        f"Document approved and ready for publication. Final version: {document.current_version}",
    )

    return {
        **state,
        "document": document,
        "current_operation": operation,
        "message_log": message_log,
    }


# Create the graph
def create_document_editor_graph() -> StateGraph:
    """Creates the LangGraph for document editing with advanced state management"""
    graph = StateGraph(EditorState)

    # Add nodes
    graph.add_node("initialize", initialize_document)
    graph.add_node("generate_content", generate_document_content)
    graph.add_node("format_document", format_document)
    graph.add_node("review_document", review_document)
    graph.add_node("finalize", finalize_document)

    # Define edges for the linear workflow
    graph.add_edge("initialize", "generate_content")
    graph.add_edge("generate_content", "format_document")
    graph.add_edge("format_document", "review_document")
    graph.add_edge("review_document", "finalize")
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("initialize")

    return graph


def main():
    """Run the document editor LangGraph with advanced state management"""
    if len(sys.argv) < 2:
        print('Usage: python main.py "<document_prompt>"')
        sys.exit(1)

    prompt = sys.argv[1]

    # Initialize the state
    initial_state = {
        "prompt": prompt,
        "document": None,
        "current_user": None,
        "current_operation": None,
        "message_log": [],
    }

    # Create and compile the graph
    graph = create_document_editor_graph().compile()

    print(f"\nCreating document based on: '{prompt}'...\n")

    # Execute the graph
    result = graph.invoke(initial_state)

    # Output results
    document = result["document"]
    snapshot = create_document_snapshot(document)

    print("\n=== DOCUMENT CREATED ===\n")
    print(f"Title: {document.title}")
    print(f"Status: {document.status.upper()}")
    print(f"Versions: {len(document.versions)}")
    print(f"Current Version: {document.current_version}")
    print(f"Total Edits: {len(document.edit_history)}")
    print(f"Word Count: {snapshot['word_count']}")

    # Show document structure
    current_version = None
    for version in document.versions:
        if version.version_number == document.current_version:
            current_version = version
            break

    if current_version:
        print("\n--- Document Structure ---")
        for section in sorted(current_version.sections, key=lambda s: s.order):
            content_preview = (
                section.content[:50] + "..."
                if len(section.content) > 50
                else section.content
            )
            print(f"{section.order}. {section.title}")
            print(f"   {content_preview}")

    # Show edit history summary
    print("\n--- Edit History ---")
    for i, edit in enumerate(document.edit_history, 1):
        print(f"{i}. {edit.operation_type.capitalize()}: {edit.description}")

    # Show message log
    print("\n--- Process Log ---")
    for msg in result["message_log"]:
        print(f"[{msg['sender']}] {msg['message']}")

    print("\nDocument processing complete!")


if __name__ == "__main__":
    main()
