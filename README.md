# Production RAG System: Ask My Research Papers

A production-grade Retrieval Augmented Generation (RAG) system that answers questions about research papers with cited sources. Built entirely locally — no API keys, no cost.

## Why I Built This

Research papers are dense. Finding specific information across 20-page PDFs is slow and tedious. This system lets you drop in any PDF, ask natural language questions, and get grounded answers with exact page citations — making literature review dramatically faster.

Built as a portfolio project to demonstrate production AI engineering: hybrid retrieval, reranking, citation enforcement, and a CI-gated eval pipeline.

## What It Does

- Ingests PDFs and extracts clean text page by page
- Chunks documents intelligently (650 tokens, 100 overlap) to preserve context
- Embeds chunks locally using nomic-embed-text via Ollama
- Retrieves relevant chunks using hybrid BM25 + vector search
- Reranks candidates using a cross-encoder model for precision
- Generates answers using Llama3 running locally via Ollama
- Enforces citations — every answer includes source file and page number
- Runs an eval pipeline scoring answers against a golden QA dataset

## What It Can Be Used For

- Asking questions across research papers
- Building internal knowledge bases from company documents
- Legal document QA
- Medical literature review
- Any domain where you need cited, grounded answers from PDFs

## Project Structure
```
rag-project/
├── data/                   # Drop your PDFs here
├── src/
│   ├── ingest.py           # Loads PDFs, chunks text, embeds, stores in ChromaDB
│   ├── retrieve.py         # Hybrid BM25 + vector search + cross-encoder reranking
│   ├── answer.py           # Generates cited answers using Llama3
│   └── __init__.py
├── eval/
│   ├── golden_qa.json      # 5 manually verified question-answer pairs
│   ├── run_evals.py        # Eval runner — scores answers, exits 1 if below threshold
│   └── __init__.py
├── prompts/
│   └── qa_prompt.yaml      # Versioned prompt config for the LLM
├── chroma_db/              # Vector database (auto-created, gitignored)
├── .gitignore
└── README.md
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Llama3 via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB |
| RAG framework | LlamaIndex |
| Keyword search | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF parsing | pypdf |
| Eval | Custom keyword-overlap scorer |

## Setup

### 1. Install Ollama and pull models
Download Ollama from https://ollama.com then run:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 2. Clone and create virtual environment
```bash
git clone https://github.com/YOUR_USERNAME/rag-project.git
cd rag-project
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb rank_bm25 sentence-transformers pypdf ragas pytest pyyaml
```

### 4. Add your documents
Drop any PDF into the data/ folder:
```bash
cp ~/Downloads/your-paper.pdf data/
```
> **Note:** For this project I used the original RAG paper — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020). You can download it here: https://arxiv.org/abs/2005.11401

### 5. Ingest documents
```bash
python src/ingest.py
```

### 6. Ask a question
```bash
python -m src.answer
```

### 7. Run the eval pipeline
```bash
python eval/run_evals.py
```

## Example Output
```
Ask a question: What two components does RAG combine?

--- Answer ---
According to the context, RAG combines parametric and non-parametric memory [1].
The non-parametric component guides generation by drawing out specific knowledge
stored in the parametric memory [2].

--- Sources ---
[1] 2005.11401v4.pdf (page 6)
[2] 2005.11401v4.pdf (page 6)
```

## Eval Results
```
[PASS] (0.75) What two components does RAG combine?
[PASS] (0.60) What retriever does RAG use?
[PASS] (1.00) What generator does RAG use?
[PASS] (0.82) What is the difference between RAG-Token and RAG-Sequence?
[PASS] (0.50) What tasks did RAG achieve state of the art on?

Result: 5/5 passed (100%)
EVAL PASSED
```

## Architecture
```
User Query
    │
    ├── Vector Search (semantic similarity via ChromaDB)
    │
    ├── BM25 Search (keyword matching)
    │
    ├── Merge + Deduplicate candidates
    │
    ├── Cross-Encoder Reranker (scores query-chunk pairs)
    │
    ├── Top 5 chunks passed to Llama3
    │
    └── Cited answer returned to user
```
