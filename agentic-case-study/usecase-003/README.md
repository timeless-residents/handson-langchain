# LangChain Agent: Web Search (Usecase-003)

This example demonstrates how to create a LangChain agent that can search the web for up-to-date information.

## Overview

The agent uses OpenAI's language model combined with DuckDuckGo search to find and synthesize information from the internet in response to user queries.

## Features

- Perform web searches using DuckDuckGo
- Get up-to-date information beyond the LLM's training cutoff
- Process and summarize search results
- Handle natural language questions that require real-time information

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

- You can modify the `queries` list in `main.py` to test different search queries
- Consider implementing additional search processing in `summarize_search()` to filter or refine results

## Limitations

- The quality of search results depends on DuckDuckGo's index and algorithm
- Web search may be slower than other types of tools due to network latency
- Results may contain irrelevant information that requires additional filtering
- The agent lacks web browsing capabilities (it cannot click through links)

## Next Steps

To enhance this web search agent, consider:
- Adding more search providers (Google, Bing, etc.)
- Implementing web scraping for more detailed information extraction
- Adding memory to store search results for frequently asked questions
- Implementing fact-checking across multiple sources
- Creating specialized search tools for specific domains (news, academic, technical, etc.)