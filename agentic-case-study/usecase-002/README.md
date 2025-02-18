# LangChain Agent: Weather Information (Usecase-002)

This example demonstrates how to create a LangChain agent that provides weather information and forecasts.

## Overview

The agent uses OpenAI's language model to understand natural language queries about weather and then uses custom weather tools to fetch and return the requested information.

## Features

- Get current weather conditions for a specific location
- Get a 5-day weather forecast
- Answer natural language questions about weather
- Mock API implementation for demonstration purposes

## Requirements

- Python 3.9+
- OpenAI API key (set as environment variable)
- Required packages: see `requirements.txt`

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

Run the script:
```bash
python main.py
```

The script will execute several test queries to demonstrate the agent's capabilities.

## Customization

- You can modify the `queries` list in `main.py` to test different weather-related questions
- In a production environment, replace the mock weather API functions with calls to a real weather API service

## Limitations

- This example uses mock weather data for demonstration purposes
- In a real application, you would need to handle rate limits, API authentication, and error cases from the weather API
- The agent's understanding of weather terminology depends on the underlying language model

## Next Steps

To make this weather agent production-ready, consider:
- Integrating with a real weather API (OpenWeatherMap, Weather.gov, etc.)
- Adding caching to reduce API calls for frequently requested locations
- Implementing proper error handling for API failures
- Adding support for more weather metrics (UV index, air quality, etc.)
- Supporting location detection from user's IP address