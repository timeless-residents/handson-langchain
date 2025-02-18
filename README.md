# LangChain Hands-on Tutorial and Case Studies

A comprehensive learning path from basic LangChain concepts to advanced agent implementations, featuring 100 practical use cases.

## Overview

This repository provides a structured approach to learning LangChain, beginning with fundamental concepts and gradually progressing to complex, specialized agent implementations. The collection serves as both an educational resource and a reference for implementing AI agents in real-world scenarios.

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key
- Basic understanding of Python and API concepts

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/langchain-tutorial.git
   cd langchain-tutorial
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install common dependencies:
   ```bash
   pip install langchain langchain-openai langchain-community python-dotenv
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Repository Structure

The repository is organized in a progressive learning path:

```
langchain-tutorial/
  ├── step1.py           # Basic LLM usage with LangChain
  ├── step2.py           # Multi-tool agent implementation
  ├── steps/             # Additional introductory steps (optional)
  │   ├── step3.py
  │   └── ...
  ├── usecase-001/       # Basic Calculator Agent
  │   ├── main.py
  │   ├── README.md
  │   └── requirements.txt
  ├── usecase-002/       # Weather Information Agent  
  │   ├── main.py
  │   ├── README.md
  │   └── requirements.txt
  ├── ...
  └── usecase-100/
```

## Learning Path

### Part 1: Getting Started with LangChain

#### Step 1: Basic LLM Usage (`step1.py`)
Learn to interact with OpenAI's LLM using LangChain:
- Setting up the environment
- Creating an LLM instance
- Sending basic prompts and receiving responses

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create OpenAI LLM instance
llm = OpenAI()

# Query the LLM
prompt = "What's the weather like today?"
response = llm.invoke(prompt)

print("LLM Response:")
print(response)
```

#### Step 2: Multi-Tool Agent Implementation (`step2.py`)
Build a more advanced LangChain agent with multiple tools:
- Calculator for mathematical operations
- Current time retrieval
- Web search using DuckDuckGo

```python
# Usage examples
queries = [
    "Who won the Nobel Peace Prize in 2023?",
    "What's today's date?",
    "Calculate 1234 × 5678",
    "What is the current population of Tokyo? Please provide specific numbers."
]
```

### Part 2: Comprehensive Use Cases (001-100)

After getting comfortable with the basics in Step 1 and Step 2, explore the 100 use cases organized by increasing complexity and specialization:

#### Foundational Use Cases (001-010)
- **001**: Basic Calculator Agent
- **002**: Weather Information Agent  
- **003**: Web Search Agent
- **004**: Multi-Tool Agent
- **005**: Conversational Memory Agent
- **006**: Document QA Agent
- **007**: Data Analysis Agent
- **008**: Language Translation Agent
- **009**: E-commerce Product Recommendation Agent
- **010**: Code Generation and Explanation Agent

#### Specialized Domains (011-030)
- Healthcare assistants
- Legal document analysis
- Educational tutoring
- Financial planning
- Content creation
- ...

#### Advanced Implementations (031-060)
- Multi-agent systems
- Tool-augmented agents
- Autonomous decision-making
- ...

#### Industry-Specific Solutions (061-100)
- Customer service automation
- Research assistants
- Project management
- ...

## How to Use This Repository

1. **For Beginners**:
   - Start with `step1.py` to understand basic LLM interactions
   - Move to `step2.py` to learn about agents and multiple tools
   - Once comfortable, explore use cases 001-010 in sequence

2. **For Intermediate Users**:
   - Begin with `step2.py` to refresh agent concepts
   - Jump to use cases in your domain of interest (011-030)
   - Experiment with more advanced implementations (031-060)

3. **For Advanced Users**:
   - Use the repository as a reference for specific implementations
   - Contribute improvements or new use cases
   - Adapt the examples for production environments

## Key LangChain Features Demonstrated

Across the examples, you'll learn how to implement:

- **Agent Types**: ZeroShot, ReAct, Conversational, etc.
- **Memory Systems**: Various memory implementations for context retention
- **Tool Integration**: Custom tools, API connections, vector stores
- **Error Handling**: Robust error management strategies
- **Prompt Engineering**: Effective prompt design patterns
- **Chain Composition**: Complex reasoning through chained operations

## Running the Examples

### Basic Steps:
```bash
# Run Step 1
python step1.py

# Run Step 2
python step2.py
```

### Use Cases:
```bash
# Navigate to a specific use case
cd usecase-001

# Install requirements
pip install -r requirements.txt

# Run the use case
python main.py
```

## Error Handling

The examples implement various error handling strategies:
- Input validation errors (ValueError, KeyError)
- Network-related errors (ConnectionError, TimeoutError)
- Runtime errors (RuntimeError)

## Contributing

Contributions are welcome! Please feel free to submit a pull request with:
- New steps or use cases
- Improvements to existing implementations
- Documentation enhancements
- Bug fixes

## Important Notes

- Handle OpenAI API keys securely
- Be aware of potential costs associated with API usage
- Internet connection is required for web search functionality
- Some examples may require additional dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The LangChain community for creating an excellent framework
- OpenAI for providing the underlying language models
- All contributors who have helped improve this collection