"""
LangChain Agent: Language Translation Agent
===============================

This module implements a LangChain agent for language translation tasks.
The agent can:
- Detect the language of input text
- Translate text between multiple languages
- Provide cultural context and idiom explanations
- Handle formal vs informal tone adjustments
- Summarize translated content

This demonstrates using LLMs for sophisticated translation tasks.
"""

from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# Initialize LLMs with different temperature settings
translator_llm = ChatOpenAI(
    temperature=0.1
)  # Lower temperature for more accurate translations
creative_llm = ChatOpenAI(
    temperature=0.7
)  # Higher temperature for cultural explanations and examples

# Language detection prompt
language_detection_template = """
You are a language detection expert. Analyze the following text and determine what language it's written in.
Respond with only the language name (e.g., "English", "Spanish", "Japanese", etc.).

Text: {text}

Language:
"""
language_detection_prompt = PromptTemplate(
    input_variables=["text"],
    template=language_detection_template,
)
language_detection_chain = LLMChain(
    llm=translator_llm, prompt=language_detection_prompt
)

# Translation prompt
translation_template = """
You are an expert translator. Translate the following text from {source_language} to {target_language}.
Maintain the original meaning, tone, and nuance as accurately as possible.

{formality_instruction}

Text to translate:
{text}

Translation:
"""
translation_prompt = PromptTemplate(
    input_variables=[
        "text",
        "source_language",
        "target_language",
        "formality_instruction",
    ],
    template=translation_template,
)
translation_chain = LLMChain(llm=translator_llm, prompt=translation_prompt)

# Cultural context prompt
cultural_context_template = """
You are a cultural and linguistic expert. For the following text that has been translated from {source_language} to {target_language},
provide cultural context, explain any idioms, cultural references, or nuances that might be important for someone from a {target_language}-speaking
background to understand.

Original text ({source_language}):
{original_text}

Translated text ({target_language}):
{translated_text}

Cultural context and explanations:
"""
cultural_context_prompt = PromptTemplate(
    input_variables=[
        "original_text",
        "translated_text",
        "source_language",
        "target_language",
    ],
    template=cultural_context_template,
)
cultural_context_chain = LLMChain(llm=creative_llm, prompt=cultural_context_prompt)

# Alternative expressions prompt
alternative_expressions_template = """
You are a language expert. For the following translated text, provide 2-3 alternative ways to express the same meaning in {target_language},
with different levels of formality or in different regional variants.

Translated text:
{translated_text}

Alternative expressions in {target_language}:
"""
alternative_expressions_prompt = PromptTemplate(
    input_variables=["translated_text", "target_language"],
    template=alternative_expressions_template,
)
alternative_expressions_chain = LLMChain(
    llm=creative_llm, prompt=alternative_expressions_prompt
)

# Summarization prompt
summarization_template = """
Summarize the following {target_language} text in {target_language}, capturing the key points while reducing the length by approximately 70%.

Text to summarize:
{text}

Summary:
"""
summarization_prompt = PromptTemplate(
    input_variables=["text", "target_language"],
    template=summarization_template,
)
summarization_chain = LLMChain(llm=translator_llm, prompt=summarization_prompt)


# Helper functions for translation tools
def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text: The text to analyze

    Returns:
        str: The detected language
    """
    return language_detection_chain.run(text=text).strip()


def parse_translation_query(query: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Parse a translation query to extract text, source language, target language, and formality.

    Args:
        query: Format should be "text to translate|source language|target language|[formal/informal]"

    Returns:
        Tuple[str, str, str, Optional[str]]: text, source_lang, target_lang, formality
    """
    parts = query.split("|")
    if len(parts) < 3:
        raise ValueError(
            "Query must contain at least text, source language, and target language, separated by '|'"
        )

    text = parts[0].strip()
    source_lang = parts[1].strip()
    target_lang = parts[2].strip()

    formality = None
    if len(parts) > 3:
        formality = parts[3].strip().lower()
        if formality not in ["formal", "informal"]:
            formality = None

    return text, source_lang, target_lang, formality


def translate_text(query: str) -> str:
    """
    Translate text from one language to another.

    Args:
        query: Format should be "text to translate|source language|target language|[formal/informal]"

    Returns:
        str: Translated text
    """
    try:
        text, source_lang, target_lang, formality = parse_translation_query(query)

        # Auto-detect source language if "auto" is specified
        if source_lang.lower() == "auto":
            source_lang = detect_language(text)

        # Set formality instruction
        formality_instruction = ""
        if formality == "formal":
            formality_instruction = (
                "Use formal language appropriate for professional or official contexts."
            )
        elif formality == "informal":
            formality_instruction = "Use casual, conversational language appropriate for friends or informal situations."

        # Perform translation
        translation = translation_chain.run(
            text=text,
            source_language=source_lang,
            target_language=target_lang,
            formality_instruction=formality_instruction,
        )

        return translation.strip()
    except Exception as e:
        return f"Translation error: {str(e)}\nUse format: 'text|source language|target language|[formal/informal]'"


def provide_cultural_context(query: str) -> str:
    """
    Provide cultural context for a translation.

    Args:
        query: Format should be "original text|translated text|source language|target language"

    Returns:
        str: Cultural context and explanations
    """
    try:
        parts = query.split("|")
        if len(parts) != 4:
            return "Query must contain original text, translated text, source language, and target language, separated by '|'"

        original_text, translated_text, source_lang, target_lang = [
            p.strip() for p in parts
        ]

        context = cultural_context_chain.run(
            original_text=original_text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
        )

        return context.strip()
    except Exception as e:
        return f"Error providing cultural context: {str(e)}"


def suggest_alternative_expressions(query: str) -> str:
    """
    Suggest alternative expressions in the target language.

    Args:
        query: Format should be "translated text|target language"

    Returns:
        str: Alternative expressions
    """
    try:
        parts = query.split("|")
        if len(parts) != 2:
            return "Query must contain translated text and target language, separated by '|'"

        translated_text, target_lang = [p.strip() for p in parts]

        alternatives = alternative_expressions_chain.run(
            translated_text=translated_text, target_language=target_lang
        )

        return alternatives.strip()
    except Exception as e:
        return f"Error suggesting alternatives: {str(e)}"


def summarize_translation(query: str) -> str:
    """
    Summarize translated text.

    Args:
        query: Format should be "text to summarize|target language"

    Returns:
        str: Summarized text
    """
    try:
        parts = query.split("|")
        if len(parts) != 2:
            return "Query must contain text to summarize and target language, separated by '|'"

        text, target_lang = [p.strip() for p in parts]

        summary = summarization_chain.run(text=text, target_language=target_lang)

        return summary.strip()
    except Exception as e:
        return f"Error summarizing: {str(e)}"


# Create tool instances
tools = [
    Tool(
        name="LanguageDetection",
        func=detect_language,
        description="Detects the language of the input text. Input should be the text to analyze.",
    ),
    Tool(
        name="TextTranslation",
        func=translate_text,
        description="Translates text between languages. Input format: 'text to translate|source language|target language|[formal/informal]'. Use 'auto' as source language for automatic detection.",
    ),
    Tool(
        name="CulturalContext",
        func=provide_cultural_context,
        description="Provides cultural context for translations. Input format: 'original text|translated text|source language|target language'.",
    ),
    Tool(
        name="AlternativeExpressions",
        func=suggest_alternative_expressions,
        description="Suggests alternative ways to express the translated text. Input format: 'translated text|target language'.",
    ),
    Tool(
        name="SummarizeTranslation",
        func=summarize_translation,
        description="Summarizes translated text to be more concise. Input format: 'text to summarize|target language'.",
    ),
]

# Initialize agent
agent = initialize_agent(
    tools,
    translator_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Test queries
queries = [
    "What language is this text: 'こんにちは、元気ですか？'",
    "Translate this to Spanish: 'Hello, how are you today? I hope you're doing well.'|English|Spanish|formal",
    "Can you translate 'Break a leg!' from English to Japanese and explain the cultural context?",
    "Translate 'It's raining cats and dogs' to French and suggest some alternative expressions",
    "Translate and then summarize this paragraph: 'Artificial intelligence has made significant strides in recent years. Machine learning models can now generate realistic images, compose music, and even write essays. These advancements raise important questions about creativity, authorship, and the future of human-AI collaboration in various fields.'|English|German",
]


def main():
    print("Testing Language Translation Agent:")
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
