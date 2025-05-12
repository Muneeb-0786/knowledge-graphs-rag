# RAG-Graph: Knowledge Graph-based Retrieval Augmented Generation

A Python-based system for constructing knowledge graphs from text data (specifically SEC Form 10-K documents) using Neo4j, vector embeddings, and LangChain for question answering.

## Overview

This project demonstrates an advanced approach to retrieving and analyzing information from textual data by:

1. Processing and chunking SEC Form 10-K documents
2. Generating vector embeddings for text chunks using Google Gemini API
3. Storing both text chunks and their embeddings in a Neo4j graph database
4. Creating a knowledge graph with relationships between entities
5. Implementing vector similarity search for semantic retrieval
6. Providing question answering capabilities using LangChain and large language models

## Features

- **Document Processing**: Splits Form 10-K documents into manageable chunks with metadata preservation
- **Vector Embeddings**: Generates embeddings using Google's Generative AI API
- **Neo4j Integration**: Stores documents and embeddings in a graph database with vector search capabilities
- **Knowledge Graph Construction**: Creates relationships between entities extracted from text
- **Vector Search**: Performs semantic searches using vector similarity
- **Question Answering**: Uses LangChain to connect the graph database to LLMs for intelligent question answering

## Requirements

- Python 3.8+
- Neo4j Database (4.4+ with vector index capabilities)
- Google Gemini API key
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rag-graph.git
   cd rag-graph
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys and Neo4j credentials:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=neo4j
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### Processing Form 10-K Documents

The system processes Form 10-K JSON files, extracts key sections, and splits them into manageable chunks:

```python
python construct_kg_from_text.py
```

This will:
1. Load and process Form 10-K documents
2. Create chunks with metadata
3. Generate embeddings
4. Store everything in Neo4j
5. Create a knowledge graph structure

### Running Semantic Searches

You can perform semantic searches using vector similarity:

```python
# Import the search function
from construct_kg_from_text import neo4j_vector_search

# Perform a semantic search
results = neo4j_vector_search('Where is Netapp headquartered?')
```

### Asking Questions

The system supports natural language question answering:

```python
from construct_kg_from_text import setup_langchain_qa, prettychain

# Set up QA chain
qa_chain = setup_langchain_qa()

# Ask a question
prettychain("What are the main business risks for Netapp?", qa_chain)
```

## Project Structure

- `construct_kg_from_text.py`: Main script for processing documents and building the knowledge graph
- `graph.py`: Utility functions for working with Neo4j graphs
- `data/form10k/`: Directory containing Form 10-K JSON files
- `.env`: Configuration file for API keys and database credentials

## How It Works

1. **Document Processing**: The system reads Form 10-K JSON files and extracts important sections like Item 1 (Business), Item 1A (Risk Factors), Item 7 (Management's Discussion), etc.

2. **Chunking**: Text is split into manageable chunks (default 2000 characters) with metadata preserved.

3. **Embedding Generation**: Each chunk's text is converted to a vector embedding using Google's Generative AI API.

4. **Neo4j Storage**: Chunks and their embeddings are stored in Neo4j, with appropriate indexes for vector search.

5. **Knowledge Graph**: The system creates relationships between Form entities and their chunks.

6. **Retrieval**: Vector similarity search finds the most relevant text chunks for a query.

7. **Question Answering**: LangChain connects the retrieved context to an LLM to generate answers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

Created by Muneeb Islam (2025)
