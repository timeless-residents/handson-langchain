"""
LangChain Agent: Basic Calculator
===============================

This module implements a LangChain agent that can perform basic calculator operations.
The agent can:
- Perform arithmetic operations (+, -, *, /)
- Handle parentheses and order of operations
- Convert mathematical expressions from natural language to calculations

This is a minimal example to demonstrate a single-purpose agent.
"""

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)


def calculator(expression):
    """
    Evaluates a mathematical expression given as a string.

    Args:
        expression (str): The expression to calculate (e.g., "2 + 2", "3 * 4")

    Returns:
        Union[float, str]: The result of the calculation, or an error message

    Examples:
        >>> calculator("2 + 2")
        4
        >>> calculator("3 * 4")
        12
    """
    try:
        # Clean up the expression
        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")
        expression = expression.strip()

        # Safely evaluate the expression
        result = eval(expression)
        return result
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        return f"Calculation error: {str(e)}"


# Create tool instance
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for when you need to perform mathematical calculations. Input should be a valid mathematical expression as a string.",
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
    "What is 123 + 456?",
    "Calculate 17 * 38",
    "What's the square root of 144?",
    "Divide 1000 by 25",
]


def main():
    print("Testing Basic Calculator Agent:")
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
