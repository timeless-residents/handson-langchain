"""
LangChain Agent: Conversational Memory Agent
===============================

This module implements a LangChain agent with conversational memory.
The agent can:
- Remember previous interactions within a conversation
- Refer back to previous questions and answers
- Build context over multiple turns of dialogue

This is useful for creating more natural, context-aware conversations.
"""

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0.7)

# Initialize Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Define a simple tool for demonstration
def get_joke(topic):
    """
    Returns a simple joke about the provided topic.

    Args:
        topic (str): Topic to joke about

    Returns:
        str: A joke related to the topic
    """
    jokes = {
        "programming": "Why do programmers prefer dark mode? Because light attracts bugs!",
        "math": "Why was 6 afraid of 7? Because 7 8 9!",
        "physics": "I have a new theory on matter, but I'm afraid it won't work!",
        "food": "I'm on a seafood diet. Every time I see food, I eat it!",
        "animals": "What do you call a bear with no teeth? A gummy bear!",
    }

    # Default joke if topic not found
    return jokes.get(topic.lower(), "What's brown and sticky? A stick!")


# Create tool instances
tools = [
    Tool(
        name="JokeTool",
        func=get_joke,
        description="Useful for getting a joke about a specific topic. Input should be a single topic word.",
    ),
]

# Initialize agent with memory
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
)


def chat_loop():
    print("Conversational Memory Agent")
    print("Type 'exit' to end the conversation")
    print("-" * 50)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)


def main():
    # Run the interactive chat loop
    chat_loop()

    # Alternatively, demonstrate with scripted conversation
    demonstrate_memory()


def demonstrate_memory():
    """Demonstrates memory capabilities with a scripted conversation."""
    print("Demonstrating Memory Capabilities:")
    print("-" * 50)

    # Example conversation to demonstrate memory
    conversation = [
        "Hi there! My name is Alice.",
        "Can you tell me a joke about programming?",
        "That was funny! Now tell me a joke about animals.",
        "What was my name again?",
    ]

    for user_input in conversation:
        print(f"You: {user_input}")

        try:
            response = agent.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)


if __name__ == "__main__":
    # Choose which demo to run
    main()
