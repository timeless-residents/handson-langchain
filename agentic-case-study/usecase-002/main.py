"""
LangChain Agent: Weather Information
===============================

This module implements a LangChain agent that provides weather information.
The agent can:
- Fetch current weather conditions for a given location
- Provide weather forecasts
- Answer natural language questions about weather

This example uses a mock weather API for demonstration purposes.
In a production environment, you would replace it with a real weather API.
"""

import random
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_openai import OpenAI

load_dotenv()

# Initialize LLM
llm = OpenAI(temperature=0)


# Mock weather API (replace with real API in production)
def get_weather(location_query):
    """
    Mock function to simulate fetching weather data.

    Args:
        location_query (str): Location to get weather for (city, country, etc.)

    Returns:
        str: Weather information for the specified location
    """
    # In a real implementation, you would call an actual weather API here
    locations = {
        "new york": {
            "temp": random.randint(40, 85),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Snowy"]),
        },
        "london": {
            "temp": random.randint(40, 75),
            "condition": random.choice(["Cloudy", "Rainy", "Foggy", "Partly Cloudy"]),
        },
        "tokyo": {
            "temp": random.randint(50, 90),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
        },
        "sydney": {
            "temp": random.randint(60, 95),
            "condition": random.choice(["Sunny", "Partly Cloudy", "Clear"]),
        },
        "paris": {
            "temp": random.randint(45, 80),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
        },
    }

    # Default weather for unknown locations
    location = location_query.lower()
    weather_data = locations.get(
        location, {"temp": random.randint(50, 80), "condition": "Partly Cloudy"}
    )

    # Get date info
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M")

    return (
        f"Weather for {location_query.title()} on {current_date} at {current_time}:\n"
        f"Temperature: {weather_data['temp']}°F\n"
        f"Condition: {weather_data['condition']}\n"
        f"Humidity: {random.randint(30, 90)}%\n"
        f"Wind Speed: {random.randint(0, 20)} mph"
    )


def get_forecast(location_query):
    """
    Mock function to simulate fetching a weather forecast.

    Args:
        location_query (str): Location to get forecast for

    Returns:
        str: 5-day forecast for the specified location
    """
    # In a real implementation, you would call an actual weather API here
    location = location_query.title()
    current_date = datetime.now()

    forecast = f"5-Day Forecast for {location}:\n\n"

    for i in range(5):
        date = (current_date + timedelta(days=i)).strftime("%Y-%m-%d")
        temp_high = random.randint(60, 95)
        temp_low = random.randint(40, temp_high - 5)
        condition = random.choice(
            ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorms"]
        )
        forecast += f"{date}: High {temp_high}°F, Low {temp_low}°F, {condition}\n"

    return forecast


# Create tool instances
tools = [
    Tool(
        name="CurrentWeather",
        func=get_weather,
        description="Useful for getting current weather conditions for a location. Input should be a city name or location.",
    ),
    Tool(
        name="WeatherForecast",
        func=get_forecast,
        description="Useful for getting a 5-day weather forecast for a location. Input should be a city name or location.",
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
    "What's the current weather in New York?",
    "Give me the 5-day forecast for London",
    "Is it raining in Tokyo right now?",
    "What's the temperature in Sydney?",
]


def main():
    print("Testing Weather Information Agent:")
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
