"""
LangChain Agent: Multi-Tool Agent
===============================

This module implements a LangChain agent that combines multiple tools.
The agent can:
- Perform calculations
- Fetch current time
- Search the web
- Get current weather (mock implementation)
- Perform simple data analysis

This demonstrates how to create a versatile agent that can handle various types of queries.
"""

import datetime
import random
import statistics

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)

# Initialize DuckDuckGo search
search = DuckDuckGoSearchRun()


# Tool functions
def calculator(expression):
    """
    Evaluates a mathematical expression.

    Args:
        expression (str): The expression to calculate

    Returns:
        Union[float, str]: Result or error message
    """
    try:
        # Clean up the expression
        expression = expression.replace("×", "*")
        expression = expression.replace("÷", "/")
        expression = expression.strip()

        # Safely evaluate the expression
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_current_time(_):
    """
    Returns the current date and time.

    Args:
        _ (Any): Unused parameter

    Returns:
        str: Current date and time
    """
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def get_weather(location):
    """
    Mock function to get weather data.

    Args:
        location (str): Location to get weather for

    Returns:
        str: Weather information
    """
    # This is a mock implementation - in a real app, call a weather API
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy", "Snowy"]
    temperature = random.randint(30, 100)
    condition = random.choice(conditions)

    return (
        f"Weather for {location}:\n"
        f"Temperature: {temperature}°F\n"
        f"Conditions: {condition}\n"
        f"Humidity: {random.randint(20, 95)}%"
    )


def analyze_data(data_str):
    """
    Perform simple statistical analysis on numerical data.

    Args:
        data_str (str): Comma-separated numbers

    Returns:
        str: Statistical analysis results
    """
    try:
        # Parse the input string into a list of numbers
        numbers = [float(x.strip()) for x in data_str.split(",")]

        if not numbers:
            return "Error: No valid numbers provided"

        # Calculate statistics
        mean = statistics.mean(numbers)
        if len(numbers) > 1:
            median = statistics.median(numbers)
            stdev = statistics.stdev(numbers)
            result = (
                f"Analysis results:\n"
                f"Count: {len(numbers)}\n"
                f"Sum: {sum(numbers)}\n"
                f"Mean: {mean:.2f}\n"
                f"Median: {median:.2f}\n"
                f"Standard Deviation: {stdev:.2f}\n"
                f"Min: {min(numbers)}\n"
                f"Max: {max(numbers)}"
            )
        else:
            result = (
                f"Analysis results:\n" f"Count: {len(numbers)}\n" f"Value: {numbers[0]}"
            )

        return result
    except Exception as e:
        return f"Analysis error: {str(e)}"


# Create tool instances
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a mathematical expression as a string.",
    ),
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="Useful for getting the current date and time. No input is needed.",
    ),
    Tool(
        name="WebSearch",
        func=search.run,
        description="Useful for searching the web for specific information. Input should be a search query.",
    ),
    Tool(
        name="Weather",
        func=get_weather,
        description="Useful for getting weather information for a location. Input should be a city or location name.",
    ),
    Tool(
        name="DataAnalysis",
        func=analyze_data,
        description="Useful for analyzing numerical data. Input should be a comma-separated list of numbers.",
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
    "What is 568 * 1234?",
    "What time is it right now?",
    "What's the weather like in San Francisco?",
    "Who is the current CEO of Microsoft?",
    "Analyze these numbers: 12, 45, 67, 89, 23, 45, 78, 90",
    "What's the weather in Tokyo and what time is it there?",
]


def main():
    print("Testing Multi-Tool Agent:")
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
