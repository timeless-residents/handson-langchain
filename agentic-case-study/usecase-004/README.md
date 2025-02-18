# LangChain Agent: Multi-Tool Agent (Usecase-004)

This example demonstrates how to create a versatile LangChain agent that combines multiple tools to handle a diverse range of queries.

## Overview

The agent uses OpenAI's language model along with several different tools to provide a comprehensive assistant that can perform calculations, search the web, get time information, provide weather updates, and analyze data.

## Features

- Perform mathematical calculations
- Get current date and time
- Search the web for up-to-date information
- Retrieve weather information (mock implementation)
- Analyze numerical data with basic statistics

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

The script will execute several test queries that demonstrate different capabilities of the multi-tool agent.

## Tool Descriptions

1. **Calculator**: Performs mathematical operations (+, -, *, /) on numerical expressions
2. **CurrentTime**: Returns the current date and time
3. **WebSearch**: Searches the web using DuckDuckGo to retrieve information
4. **Weather**: Provides weather information for a given location (mock implementation)
5. **DataAnalysis**: Performs basic statistical analysis on a set of numbers

## Customization

- You can modify the `queries` list in `main.py` to test different types of questions
- Add additional tools to expand the agent's capabilities
- Replace mock implementations (like the weather function) with real API calls

## Limitations

- The agent's capabilities are limited to the tools provided
- Some tools (like weather) use mock data and would need real API integration in production
- Complex queries that require multiple tool uses might sometimes fail
- The agent lacks memory of previous interactions within the same session

## Next Steps

To enhance this multi-tool agent, consider:
- Implementing a persistent memory system
- Adding more sophisticated tools (e.g., image generation, text translation)
- Replacing mock implementations with real API integrations
- Adding authentication and rate limiting for production use
- Implementing a feedback mechanism to improve responses over time