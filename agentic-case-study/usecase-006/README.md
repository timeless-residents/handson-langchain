# LangChain Agent: Document QA Agent (Usecase-006)

This example demonstrates how to create a LangChain agent that can answer questions based on specific documents.

## Overview

The agent loads documents, processes them into chunks, creates vector embeddings for semantic search, and then answers questions based on the content with appropriate citations.

## Features

- Load and process text documents (with potential to expand to PDFs, DOCx, etc.)
- Split documents into manageable chunks
- Create vector embeddings for semantic search
- Retrieve relevant document sections based on questions
- Answer questions with cited sources
- Handle multiple documents simultaneously

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

4. Place documents in the `documents` directory. The script will create a sample document if none exists.

## Usage

Run the script:
```bash
python main.py
```

The script will:
1. Load documents from the `documents` directory
2. Process them into embeddings
3. Run test queries against the document content
4. Display responses with citations

## Customization

- Add more document types by extending the loader logic in `load_and_process_documents()`
- Adjust chunk size and overlap in the text splitter for different document types
- Modify the retriever parameters (like `k`) to control how many chunks are retrieved
- Create additional tools that operate on document content (summarization, translation, etc.)

## Technical Details

The agent uses:
- `RecursiveCharacterTextSplitter` to break documents into smaller chunks
- OpenAI Embeddings to create vector representations of text chunks
- Chroma as the vector store for similarity search
- RetrievalQA chain to combine document retrieval with question answering

## Limitations

- Currently only supports text files (needs extension for PDFs, DOCx, etc.)
- Simple metadata handling (doesn't extract page numbers, sections, etc.)
- No persistent vector store (embeddings are recreated each run)
- Limited to documents provided at startup (can't dynamically add documents)

## Next Steps

To enhance this document QA agent, consider:
- Adding support for more document types (PDF, DOCx, HTML, etc.)
- Implementing persistent vector storage to avoid reprocessing documents
- Adding functionality to dynamically add/remove documents
- Improving citation quality with better metadata extraction
- Implementing a web interface for easier document upload and querying
- Adding summarization capabilities for entire documents or sections