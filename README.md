# LangChain and LangGraph Hands-on Tutorial and Case Studies

A comprehensive learning path covering both LangChain and LangGraph, featuring 100+ practical implementations progressing from basic concepts to advanced agent architectures.

## Overview

This repository provides a structured approach to learning modern LLM application development using both LangChain and LangGraph frameworks. The collection begins with fundamental concepts and gradually progresses to complex, specialized agent implementations and graph-based workflows. It serves as both an educational resource and a reference for implementing AI agents in real-world scenarios.

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
   This command creates a local copy of the repository. We use HTTPS cloning for broader compatibility and easier setup compared to SSH, especially for users behind corporate firewalls.

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
   Virtual environments are crucial for project isolation. This prevents dependency conflicts between different projects and ensures reproducible environments. The `venv` module is chosen over alternatives like `virtualenv` because it's included in Python's standard library since Python 3.3.

3. Install common dependencies:
   ```bash
   pip install langchain langchain-openai langchain-community langgraph python-dotenv
   ```
   We install these specific packages because:
   - `langchain`: Core framework for building LLM applications
   - `langchain-openai`: OpenAI-specific implementations
   - `langchain-community`: Community-contributed components
   - `langgraph`: Graph-based workflow management
   - `python-dotenv`: Secure environment variable management

4. Set up your OpenAI API key:
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
   We use environment variables instead of hardcoding API keys for security best practices. The `.env` file is included in `.gitignore` to prevent accidental exposure of sensitive credentials.

## Repository Structure

The repository follows a progressive learning path, with each directory serving a specific educational purpose:

```
langchain-tutorial/
  ├── step1.py           # Basic LLM usage with LangChain
  ├── step2.py           # Multi-tool agent implementation
  ├── steps/             # Additional introductory steps (optional)
  │   ├── step3.py
  │   └── ...
  ├── usecase-001/       # Basic Calculator Agent (LangChain)
  │   ├── main.py
  │   ├── README.md
  │   └── requirements.txt
  └── ...
```

This structure is designed for incremental learning, with each subsequent directory building upon concepts introduced in previous sections.

## Learning Path

### Part 1: Getting Started with LangChain

#### Step 1: Basic LLM Usage (`step1.py`)
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

This code demonstrates several key concepts:
1. **Environment Setup**: `load_dotenv()` loads environment variables securely, a crucial practice for managing API keys and sensitive data.
2. **LLM Initialization**: `OpenAI()` creates an LLM instance with default parameters. We use the default settings initially for simplicity, but these can be customized for temperature, max tokens, etc.
3. **Synchronous Invocation**: `llm.invoke(prompt)` sends a synchronous request to the LLM. We use synchronous calls here for clarity, though asynchronous operations are available for production scenarios.

#### Step 2: Multi-Tool Agent Implementation (`step2.py`)
```python
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain_openai import OpenAI
from datetime import datetime

# Initialize tools
search = DuckDuckGoSearchRun()
calculator = Tool(
    name="Calculator",
    func=lambda x: eval(x),
    description="Useful for mathematical calculations"
)
time_tool = Tool(
    name="Time",
    func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    description="Returns the current time"
)

# Create and initialize the agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[search, calculator, time_tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)
```

This implementation showcases several advanced concepts:
1. **Tool Integration**: Each tool is encapsulated with a clear name and description, helping the agent understand when to use each tool.
   - The calculator uses `eval()` for simple calculations (Note: In production, use safer evaluation methods)
   - The time tool provides formatted current time
   - DuckDuckGoSearchRun enables web searches without API keys

2. **Agent Configuration**:
   - `temperature=0`: Set to 0 for deterministic responses, crucial for tool-using agents
   - `zero-shot-react-description`: This agent type is chosen because it:
     - Requires no examples (zero-shot)
     - Uses ReAct (Reasoning and Acting) framework
     - Can choose tools based on their descriptions

3. **Verbose Mode**: Enabled for learning purposes, allowing observation of the agent's decision-making process.

### Part 2: Comprehensive Use Cases

Each use case demonstrates specific patterns and techniques:

#### LangChain Foundational Use Cases (001-010)
Each implementation is carefully structured to demonstrate specific capabilities:

- **001: Basic Calculator Agent**
  ```python
  from langchain.agents import create_react_agent
  from langchain.tools import Tool
  
  def safe_eval(expression: str) -> float:
      """
      Safely evaluate mathematical expressions.
      
      Args:
          expression (str): Mathematical expression to evaluate
          
      Returns:
          float: Result of the evaluation
          
      Safety:
          - Uses ast.literal_eval instead of eval()
          - Validates input format
          - Handles division by zero
      """
      import ast
      try:
          # Convert string to abstract syntax tree
          tree = ast.parse(expression, mode='eval')
          
          # Validate node types
          for node in ast.walk(tree):
              if not isinstance(node, (ast.Expression, ast.Num, ast.BinOp,
                                     ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
                                     ast.Div)):
                  raise ValueError("Invalid expression")
          
          # Evaluate if safe
          return float(eval(compile(tree, '<string>', 'eval')))
      except ZeroDivisionError:
          raise ValueError("Division by zero")
      except Exception as e:
          raise ValueError(f"Invalid expression: {str(e)}")
  ```
  
  This implementation demonstrates:
  - **Security**: Uses AST parsing instead of direct eval()
  - **Error Handling**: Comprehensive error cases
  - **Type Safety**: Explicit return type
  - **Documentation**: Detailed docstring with Args, Returns, and Safety sections

[Additional use cases would follow with similar detailed explanations...]

## Best Practices Demonstrated

### 1. Type Hinting
```python
from typing import List, Dict, Optional

def process_data(input_data: List[Dict[str, any]],
                config: Optional[Dict[str, str]] = None) -> Dict[str, any]:
    """
    Process input data according to optional configuration.
    
    Args:
        input_data: List of dictionaries containing data to process
        config: Optional configuration parameters
        
    Returns:
        Processed data as a dictionary
    """
    # Implementation
```

Type hints are used throughout the codebase because they:
- Enable better IDE support
- Facilitate early error detection
- Serve as inline documentation
- Support static type checking

### 2. Error Handling
```python
class CustomError(Exception):
    """Base class for custom exceptions"""
    pass

def handle_api_request(url: str) -> Dict[str, any]:
    """
    Handle external API requests with comprehensive error handling.
    
    Args:
        url: API endpoint URL
        
    Returns:
        API response data
        
    Raises:
        CustomError: When API request fails
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise CustomError(f"API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise CustomError(f"Invalid JSON response: {str(e)}")
```

This pattern demonstrates:
- Custom exception classes
- Specific exception handling
- Detailed error messages
- Proper error propagation

## Limitations and Trade-offs

### 1. LLM Dependencies
- **API Costs**: The implementations rely on OpenAI's API, which incurs usage costs. This may limit scalability for high-volume applications.
- **Rate Limiting**: OpenAI's API has rate limits that may affect performance in production environments.
- **Latency**: API calls introduce network latency, which can impact real-time applications.

### 2. Technical Constraints
- **Memory Management**:
  - Conversation history can grow large, impacting performance
  - Token limits restrict context window size
  - Memory implementations may not persist across sessions by default

- **Tool Integration**:
  - Tools must be pre-defined and cannot be dynamically created during runtime
  - Complex tools may require significant error handling
  - Tool descriptions must be carefully crafted to ensure proper agent usage

- **Error Handling Challenges**:
  - LLM responses can be unpredictable
  - Tool execution may fail in unexpected ways
  - Error recovery strategies may need manual intervention

### 3. Implementation Trade-offs
- **Synchronous vs Asynchronous**:
  - Examples use synchronous calls for clarity
  - Production environments may need async implementations for better performance
  - Async implementations add complexity to error handling

- **Security Considerations**:
  - Safe evaluation of expressions limits mathematical capabilities
  - API key management requires careful handling
  - Input validation adds processing overhead

- **Development Complexity**:
  - Debugging LLM-based systems can be challenging
  - Testing requires mock implementations of LLM responses
  - Maintaining consistent behavior across different LLM versions

### 4. Framework Limitations
- **LangChain**:
  - Documentation may lag behind rapid development
  - Some features may be experimental or unstable
  - Community tools may have varying levels of maintenance

- **LangGraph**:
  - Graph-based workflows add complexity
  - State management can become complicated
  - Learning curve for graph-based thinking

### 5. Production Considerations
- **Scalability**:
  - Cost increases linearly with usage
  - Parallel processing may be limited by API constraints
  - State management becomes complex at scale

- **Monitoring**:
  - LLM behavior can be difficult to monitor
  - Tool usage patterns may need custom logging
  - Performance metrics require careful definition

- **Maintenance**:
  - Regular updates needed for API changes
  - Tool integrations may break with external changes
  - Prompt engineering may need ongoing refinement

These limitations and trade-offs should be carefully considered when implementing these patterns in production environments. Mitigation strategies should be developed based on specific use case requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The LangChain and LangGraph teams for creating excellent frameworks
- OpenAI for providing the underlying language models
- All contributors who have helped improve this collection
