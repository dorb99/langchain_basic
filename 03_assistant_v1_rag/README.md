# 03 Assistant v1 RAG

A local-first Retrieval-Augmented Generation (RAG) assistant built with LangChain, Chroma, and Typer.

This project is intentionally structured for teaching **two chain-building approaches**:

1. **Constructor style** (classic LangChain helpers)
2. **Custom style** (your own LCEL runnable composition)

Both are real LangChain chains. The difference is how they are assembled.

## What This Project Covers

- Document ingestion (`.txt` files) into a Chroma vector store.
- Configurable embeddings and retrieval settings.
- Prompt-grounded answering with retrieved context.
- CLI commands for ingestion, search, and Q&A.
- Side-by-side chain construction styles for learning.

## Project Structure

- `main.py` - assistant CLI entry point (`ask` command).
- `assistant/core.py` - answer pipeline and both chain builders:
  - `build_constructor_rag_chain()`
  - `build_custom_rag_chain()`
- `ingestion/ingest.py` - ingest docs and debug search CLI.
- `retrieval/retriever.py` - embeddings, vector store, and retriever.
- `prompts/templates.py` - system + user prompt template.
- `llm/factory.py` - LLM provider/model factory.
- `config/settings.py` - central configuration object.
- `docs/` - sample source documents.

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Add your text files to `docs/` (or any folder you choose).
3. (Optional) Configure models/providers in `config/settings.py`.

## Ingest Documents

Build/update the vector store from text files:

- `python -m ingestion.ingest run --docs-path docs`

Quickly inspect semantic matches in the store:

- `python -m ingestion.ingest search "What is RAG?" --k 3`

## Ask Questions

The `ask` command accepts a positional question and a chain mode.

### Constructor Chain (classic helpers)

- `python main.py ask "What is LangChain?" --mode constructor`

Uses:

- `create_stuff_documents_chain(...)`
- `create_retrieval_chain(...)`

### Custom Chain (LCEL composition)

- `python main.py ask "What is LangChain?" --mode custom`

Builds a runnable chain manually with LCEL (`|`) composition.

## Teaching Note: Constructor vs Custom

- `constructor`: faster to scaffold and very aligned with official examples.
- `custom`: clearer for understanding data flow and extending behavior.
- In practice, teams often start with constructor style, then move to custom composition when they need more control.

## Useful Help Commands

- `python main.py --help`
- `python main.py ask --help`
- `python -m ingestion.ingest --help`

## Current Defaults

From `config/settings.py`:

- LLM provider: `ollama`
- LLM model: `llama3.2:3b`
- Embeddings: `nomic-embed-text`
- Retrieval top-k: `3`

