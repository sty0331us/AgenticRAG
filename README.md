# Agentic RAG Pipeline

Enterprise-oriented multi-agent Retrieval-Augmented Generation (RAG) service built with **LangGraph**, **ChromaDB**, and **Groq** inference (`llama-3.3-70b-versatile`).

The pipeline separates retrieval, reasoning, and verification into specialized agents. Each agent executes an explicit **ReAct** cycle—**Thought → Action → Observation**—and contributes to a shared, auditable workflow state.

---

## Architecture

```
RAG_VS_AgenticRAG/
├── main.py                      # CLI entry point
├── pyproject.toml
├── requirements.txt
├── .env.example
├── README.md
└── agentic_rag/
    ├── __init__.py              # Public API exports
    ├── config.py                # Environment-backed settings (Pydantic)
    ├── exceptions.py            # Domain exception hierarchy
    ├── logging_config.py        # Structured logging
    ├── models.py                # PipelineResult response model
    ├── state.py                 # LangGraph shared state contracts
    ├── prompts.py               # Agent system prompts
    ├── react.py                 # ReAct parsing and audit-trail helpers
    ├── llm.py                   # Inference client factory
    ├── vectorstore.py           # ChromaDB persistence and search
    ├── knowledge.py             # Reference corpus
    ├── ingest.py                # Corpus bootstrap
    ├── agents.py                # Retrieval / Reasoning / Verification / Error
    └── graph.py                 # Workflow compilation + AgenticRAGPipeline
```

### Control flow

```text
Query
  └─► Retrieval Agent      (Thought → Action: vector search → Observation)
        └─► Reasoning Agent (Thought → Action: draft answer → Observation)
              └─► Verification Agent
                    ├─ grounded ──► END (verified answer)
                    └─ not grounded + retries remaining ──► Retrieval (retry)
Any node failure ──► Error-handler Agent ──► END (controlled fallback)
```

---

## Agent responsibilities

### Retrieval agent

| Step | Behavior |
|------|----------|
| Thought | Reformulates the question into an embedding-optimized search query |
| Action | Executes `similarity_search` against ChromaDB |
| Observation | Records retrieved documents and routes to reasoning or error handling |

### Reasoning agent

| Step | Behavior |
|------|----------|
| Thought | Maps retrieved evidence to the question and identifies gaps |
| Action | Produces a context-grounded draft answer with citations |
| Observation | Persists the draft; routes to verification (draft is never final) |

### Verification agent

| Step | Behavior |
|------|----------|
| Thought | Performs claim-level grounding analysis against retrieved context |
| Action | Accepts, corrects, or rejects the draft |
| Observation | Completes the run, or schedules a retrieval retry within configured limits |

### Error-handler agent

Produces a controlled fallback response when an upstream agent fails, preserving auditability for operators and callers.

---

## Classic RAG vs Agentic RAG

| Dimension | Classic RAG | This pipeline |
|-----------|-------------|----------------|
| Execution model | Linear retrieve → generate | Conditional multi-agent graph |
| Reasoning | Implicit inside a single generation call | Explicit `THOUGHT` per agent |
| Acting | One-shot answer generation | Tool actions (search) + draft + accept/reject |
| Observability | Limited intermediate artifacts | Full `react_trace` audit trail |
| Quality control | First generation is final | Verification gate with bounded retries |
| Failure handling | Exceptions or empty answers | Dedicated error-handler node |
| Integration surface | Ad-hoc function return | Typed `PipelineResult` |

Shared with classic RAG: document ingest, embeddings, and vector similarity search.

Differentiating capabilities: multi-agent specialization, LangGraph routing, verification loops, and structured operational telemetry.

---

## Configuration

Copy `.env.example` to `.env` and set required values:

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Inference provider API key (required for query execution) |
| `GROQ_MODEL` | Model identifier (default: `llama-3.3-70b-versatile`) |
| `CHROMA_PERSIST_DIR` | Local ChromaDB persistence path |
| `CHROMA_COLLECTION` | Collection name |
| `RETRIEVAL_TOP_K` | Number of chunks retrieved per search |
| `MAX_VERIFICATION_RETRIES` | Bound on verification-driven retries |
| `LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY in .env
```

---

## Usage

### CLI

```bash
source .venv/bin/activate
python main.py "What is Agentic RAG?"
python main.py "Compare MCP and ACP" --json
python main.py --reingest --log-level DEBUG "Why use multi-agent systems?"
```

### Programmatic API

```python
from agentic_rag import AgenticRAGPipeline
from agentic_rag.ingest import ingest_knowledge

ingest_knowledge()
pipeline = AgenticRAGPipeline()
result = pipeline.invoke("What is Agentic RAG?")

print(result.answer)
print(result.is_grounded)
print(result.react_trace)
```

---

## Design notes

- **Configuration**: validated via Pydantic Settings; secrets are never hard-coded.
- **Observability**: agents emit structured logs and append ReAct steps to shared state.
- **Resilience**: verification retries are bounded; failures route to an error-handler node.
- **API stability**: external callers should depend on `AgenticRAGPipeline` and `PipelineResult`.

## Stack

| Component | Role |
|-----------|------|
| LangGraph | Graph orchestration, shared state, conditional routing |
| ChromaDB | Persistent local vector store |
| Groq | Low-latency chat inference for agent Thought / Action steps |
| Pydantic | Settings and response validation |
