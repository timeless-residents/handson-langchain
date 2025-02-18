# LangChain Agent: Basic Calculator (Usecase-001)

This example demonstrates how to create a simple LangChain agent that can perform basic calculator operations.

## Overview

The agent uses OpenAI's language model to understand natural language queries about calculations and then uses a custom Calculator tool to perform the actual calculations.

## Features

- Perform basic arithmetic operations: addition, subtraction, multiplication, division
- Handle parentheses and follow order of operations
- Convert natural language math queries into calculations
- Provide helpful error messages for invalid expressions

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

You can modify the `queries` list in `main.py` to test different mathematical expressions.

## Limitations

- This simple calculator cannot perform advanced mathematical functions like calculus or symbolic math
- It relies on Python's `eval()` function, which has security implications for user-submitted expressions
- The agent might occasionally struggle with very complex or ambiguously worded requests

## Next Steps

Consider enhancing this basic calculator agent by:
- Adding support for advanced mathematical functions (trigonometry, logarithms, etc.)
- Implementing a safer expression evaluation method
- Supporting step-by-step calculation breakdowns
- Adding memory to track calculation history