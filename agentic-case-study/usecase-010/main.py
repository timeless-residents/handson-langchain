"""
LangChain Agent: Code Generation and Explanation Agent
===============================

This module implements a LangChain agent for code generation and explanation.
The agent can:
- Generate code in various programming languages
- Explain existing code line by line
- Suggest improvements and optimizations
- Translate code between programming languages
- Debug code and identify potential issues

This is useful for developers seeking assistance with coding tasks.
"""

import re
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize LLMs with different settings
code_generation_llm = ChatOpenAI(
    temperature=0.1
)  # Lower temperature for precise code generation
code_explanation_llm = ChatOpenAI(
    temperature=0.4
)  # Higher temperature for more natural explanations

# Code Generation Prompt
code_generation_template = """
You are an expert software developer. Generate clean, efficient, and well-documented code based on the following requirements.

Requirements:
{requirements}

Programming Language: {language}
Additional Specifications: {specifications}

Your code should follow best practices for {language}, including proper error handling, documentation, and optimizations where appropriate.

Generate only the code without additional explanation:
"""
code_generation_prompt = PromptTemplate(
    input_variables=["requirements", "language", "specifications"],
    template=code_generation_template,
)
code_generation_chain = LLMChain(llm=code_generation_llm, prompt=code_generation_prompt)

# Code Explanation Prompt
code_explanation_template = """
You are an expert programming tutor. Explain the following code in a clear, educational manner.
Break down the explanation by sections or line by line as appropriate.

```{language}
{code}
```

Detailed explanation:
"""
code_explanation_prompt = PromptTemplate(
    input_variables=["code", "language"],
    template=code_explanation_template,
)
code_explanation_chain = LLMChain(
    llm=code_explanation_llm, prompt=code_explanation_prompt
)

# Code Improvement Prompt
code_improvement_template = """
You are a software optimization expert. Review the following code and suggest improvements 
for better performance, readability, maintainability, or security.

```{language}
{code}
```

Improvement suggestions:
"""
code_improvement_prompt = PromptTemplate(
    input_variables=["code", "language"],
    template=code_improvement_template,
)
code_improvement_chain = LLMChain(
    llm=code_explanation_llm, prompt=code_improvement_prompt
)

# Code Translation Prompt
code_translation_template = """
You are an expert polyglot programmer. Translate the following code from {source_language} to {target_language}.
Maintain the same functionality, but use idioms and best practices appropriate for {target_language}.

Original {source_language} code:
```{source_language}
{code}
```

Translated {target_language} code:
"""
code_translation_prompt = PromptTemplate(
    input_variables=["code", "source_language", "target_language"],
    template=code_translation_template,
)
code_translation_chain = LLMChain(
    llm=code_generation_llm, prompt=code_translation_prompt
)

# Code Debugging Prompt
code_debugging_template = """
You are an expert debugging specialist. Analyze the following code and identify potential bugs, 
edge cases, or issues. Also suggest fixes for each issue you find.

```{language}
{code}
```

Issues and fixes:
"""
code_debugging_prompt = PromptTemplate(
    input_variables=["code", "language"],
    template=code_debugging_template,
)
code_debugging_chain = LLMChain(llm=code_generation_llm, prompt=code_debugging_prompt)


# Helper functions for code tools
def generate_code(query: str) -> str:
    """
    Generate code based on requirements.

    Args:
        query: Format should be "requirements|language|specifications"

    Returns:
        str: Generated code
    """
    try:
        parts = query.split("|", 2)
        if len(parts) < 2:
            return "Query must contain at least requirements and language, separated by '|'"

        requirements = parts[0].strip()
        language = parts[1].strip()
        specifications = parts[2].strip() if len(parts) > 2 else ""

        generated_code = code_generation_chain.run(
            requirements=requirements, language=language, specifications=specifications
        )

        # Format the output
        code_with_markdown = f"```{language}\n{generated_code.strip()}\n```"
        return code_with_markdown

    except Exception as e:
        return f"Error generating code: {str(e)}"


def explain_code(query: str) -> str:
    """
    Explain code line by line.

    Args:
        query: Format should be "code|language"

    Returns:
        str: Explanation of the code
    """
    try:
        parts = query.split("|", 1)
        if len(parts) != 2:
            return "Query must contain both code and language, separated by '|'"

        code = parts[0].strip()
        language = parts[1].strip()

        # Extract code if it's in markdown format
        code_pattern = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```")
        match = code_pattern.search(code)
        if match:
            code = match.group(1).strip()

        explanation = code_explanation_chain.run(code=code, language=language)

        return explanation.strip()

    except Exception as e:
        return f"Error explaining code: {str(e)}"


def improve_code(query: str) -> str:
    """
    Suggest improvements for the given code.

    Args:
        query: Format should be "code|language"

    Returns:
        str: Improvement suggestions
    """
    try:
        parts = query.split("|", 1)
        if len(parts) != 2:
            return "Query must contain both code and language, separated by '|'"

        code = parts[0].strip()
        language = parts[1].strip()

        # Extract code if it's in markdown format
        code_pattern = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```")
        match = code_pattern.search(code)
        if match:
            code = match.group(1).strip()

        improvements = code_improvement_chain.run(code=code, language=language)

        return improvements.strip()

    except Exception as e:
        return f"Error improving code: {str(e)}"


def translate_code(query: str) -> str:
    """
    Translate code from one language to another.

    Args:
        query: Format should be "code|source_language|target_language"

    Returns:
        str: Translated code
    """
    try:
        parts = query.split("|", 2)
        if len(parts) != 3:
            return "Query must contain code, source language, and target language, separated by '|'"

        code = parts[0].strip()
        source_language = parts[1].strip()
        target_language = parts[2].strip()

        # Extract code if it's in markdown format
        code_pattern = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```")
        match = code_pattern.search(code)
        if match:
            code = match.group(1).strip()

        translated_code = code_translation_chain.run(
            code=code, source_language=source_language, target_language=target_language
        )

        # Format the output
        code_with_markdown = f"```{target_language}\n{translated_code.strip()}\n```"
        return code_with_markdown

    except Exception as e:
        return f"Error translating code: {str(e)}"


def debug_code(query: str) -> str:
    """
    Debug code and identify potential issues.

    Args:
        query: Format should be "code|language"

    Returns:
        str: Debugging results with issues and fixes
    """
    try:
        parts = query.split("|", 1)
        if len(parts) != 2:
            return "Query must contain both code and language, separated by '|'"

        code = parts[0].strip()
        language = parts[1].strip()

        # Extract code if it's in markdown format
        code_pattern = re.compile(r"```(?:\w+)?\s*([\s\S]*?)\s*```")
        match = code_pattern.search(code)
        if match:
            code = match.group(1).strip()

        debugging_results = code_debugging_chain.run(code=code, language=language)

        return debugging_results.strip()

    except Exception as e:
        return f"Error debugging code: {str(e)}"


# Create tool instances
tools = [
    Tool(
        name="GenerateCode",
        func=generate_code,
        description="Generate code based on requirements. Input format: 'requirements|language|specifications'. The specifications part is optional.",
    ),
    Tool(
        name="ExplainCode",
        func=explain_code,
        description="Explain code line by line. Input format: 'code|language'.",
    ),
    Tool(
        name="ImproveCode",
        func=improve_code,
        description="Suggest improvements for code in terms of performance, readability, maintainability, or security. Input format: 'code|language'.",
    ),
    Tool(
        name="TranslateCode",
        func=translate_code,
        description="Translate code from one programming language to another. Input format: 'code|source_language|target_language'.",
    ),
    Tool(
        name="DebugCode",
        func=debug_code,
        description="Debug code and identify potential issues and fixes. Input format: 'code|language'.",
    ),
]

# Initialize agent
agent = initialize_agent(
    tools,
    code_explanation_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Test queries
queries = [
    "Can you generate a Python function that calculates Fibonacci numbers using dynamic programming?",
    "Explain this code: ```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```",
    "Can you improve this JavaScript code: ```javascript\nfunction findMax(arr) {\n  let max = arr[0];\n  for(let i = 0; i < arr.length; i++) {\n    if(arr[i] > max) max = arr[i];\n  }\n  return max;\n}\n```",
    "Translate this Python code to Java: ```python\ndef process_list(items):\n    return [x * 2 for x in items if x > 0]\n```",
    "Debug this code, it has some issues: ```python\ndef calculate_average(numbers):\n    total = 0\n    for num in numbers:\n        total += num\n    return total / len(numbers)\n```",
]


def main():
    print("Testing Code Generation and Explanation Agent:")
    print("-" * 50)

    for query in queries:
        print(f"\nQuery: {query}")
        try:
            response = agent.invoke(query)
            print(f"Response: {response['output']}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)


if __name__ == "__main__":
    main()
