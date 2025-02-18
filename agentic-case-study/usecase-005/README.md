# LangChain Agent: Conversational Memory Agent (Usecase-005)

This example demonstrates how to create a LangChain agent with conversational memory, allowing it to maintain context over multiple turns of dialogue.

## Overview

The agent uses OpenAI's language model combined with a conversation buffer memory to create a more natural and context-aware conversation experience.

## Features

- Remembers previous interactions within a conversation
- Can refer back to information shared earlier in the dialogue
- Maintains context across multiple turns
- Provides more personalized responses based on conversation history
- Includes a simple joke tool to demonstrate functionality

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

The script offers two modes:
1. Interactive chat loop where you can converse with the agent
2. Demonstration with a scripted conversation (uncomment in `main()`)

## Memory Explanation

This agent uses `ConversationBufferMemory`, which is a simple but effective memory implementation that:
- Stores the full conversation history as a list of messages
- Makes this history available to the agent during each interaction
- Allows the agent to reference past exchanges in its responses

## Customization

- Modify the tools available to the agent by adding to the `tools` list
- Change the memory implementation to use different memory types:
  - `ConversationBufferWindowMemory`: Keeps only the last K interactions
  - `ConversationSummaryMemory`: Summarizes old interactions to save tokens
  - `ConversationTokenBufferMemory`: Manages context window by token count

## Limitations

- The basic `ConversationBufferMemory` stores the entire conversation, which can lead to token limits for very long conversations
- The agent lacks persistent storage between sessions
- Memory is limited to what has been explicitly shared in the conversation

## Next Steps

To enhance this conversational memory agent, consider:
- Implementing persistent storage to maintain conversations across sessions
- Adding more sophisticated memory implementations like summary or entity memory
- Combining memory with tools to create assistants that remember preferences
- Implementing memory pruning strategies for very long conversations
- Adding sentiment analysis to track user satisfaction over time