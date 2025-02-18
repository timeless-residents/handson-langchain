# Use Case 012: Conditional Branching with LangGraph

This use case demonstrates how to implement conditional branching in a LangGraph application. The agent makes decisions about which path to take based on the complexity of the problem.

## Overview

Building on the simple LangGraph structure from Use Case 011, this example introduces:

1. Conditional routing between nodes
2. Decision-making based on the state
3. Multiple possible execution paths
4. State-dependent routing

## Implementation

This implementation shows a reasoning agent that:
1. Analyzes the complexity of a problem
2. Routes to different solving strategies based on complexity
3. Uses appropriate methods for simple or complex problems
4. Assembles the final answer

## Requirements

```
langgraph>=0.0.10
langchain>=0.1.0
langchain-openai>=0.0.2
```

## Usage

```python
python main.py "What is the sum of the first 10 prime numbers?"
```

## Key Concepts

- **Conditional Edges**: Edges that use functions to determine the next node
- **Routing Functions**: Functions that examine the state and decide where to route execution
- **Multiple Paths**: Creating divergent solution strategies based on problem characteristics