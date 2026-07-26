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
    ├── graph.py                 # Workflow compilation + AgenticRAGPipeline
    └── guardrails/              # Pluggable safety layer (OSS + cloud)
        ├── base.py              # Provider interface + decision model
        ├── local.py             # Default local OSS-style scanners
        ├── llm_guard.py         # ProtectAI llm-guard adapter
        ├── cloud.py             # AWS Bedrock + Azure Content Safety adapters
        ├── factory.py           # Backend selection
        └── nodes.py             # LangGraph input/output guardrail nodes
```

### Control flow

```text
Query
  └─► Input Guardrail     (local | llm-guard | AWS Bedrock | Azure)
        └─► Retrieval Agent
              └─► Reasoning Agent
                    └─► Verification Agent
                          ├─ retry ──► Retrieval
                          └─► Output Guardrail ──► END
Any failure / policy block ──► Error-handler Agent ──► END
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

Produces a controlled fallback response when an upstream node fails, preserving auditability for operators and callers.

---

## Guardrails (open-source + cloud)

Safety checks run as first-class LangGraph nodes on **input** (before retrieval) and **output** (after verification).

| Backend (`GUARDRAIL_BACKEND`) | Type | Notes |
|-------------------------------|------|-------|
| `local` (default) | Built-in OSS-style scanners | Prompt-injection heuristics, toxicity terms, PII patterns — no extra packages |
| `llm-guard` | Open-source (ProtectAI) | Popular Apache-2.0 library for injection / toxicity / sensitive-data scanning |
| `bedrock` | AWS managed | Amazon Bedrock Guardrails — centralized policy, IAM, multi-region scale |
| `azure` | Azure managed | Azure AI Content Safety — hate/violence categories + Prompt Shields |

**Architecture intent:** local / `llm-guard` support rapid iteration and cost control. For production scalability, compliance, and centralized policy management, the same provider interface swaps to **AWS Bedrock Guardrails** or **Azure AI Content Safety** without changing agent nodes.

Related OSS frameworks evaluated for this design (documented in code comments):
- [ProtectAI llm-guard](https://github.com/protectai/llm-guard)
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

Optional installs:

```bash
pip install llm-guard                          # GUARDRAIL_BACKEND=llm-guard
pip install boto3                              # GUARDRAIL_BACKEND=bedrock
pip install azure-ai-contentsafety             # GUARDRAIL_BACKEND=azure
```

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
| `GUARDRAIL_BACKEND` | `local` \| `llm-guard` \| `bedrock` \| `azure` |
| `GUARDRAIL_BEDROCK_ID` | Bedrock Guardrail ID (when backend=`bedrock`) |
| `AZURE_CONTENT_SAFETY_ENDPOINT` / `AZURE_CONTENT_SAFETY_KEY` | Azure Content Safety (when backend=`azure`) |

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
- **Safety**: pluggable guardrails (OSS local / llm-guard, or AWS / Azure managed services).
- **API stability**: external callers should depend on `AgenticRAGPipeline` and `PipelineResult`.

## Stack

| Component | Role |
|-----------|------|
| LangGraph | Graph orchestration, shared state, conditional routing |
| ChromaDB | Persistent local vector store |
| Groq | Low-latency chat inference for agent Thought / Action steps |
| Pydantic | Settings and response validation |
| Guardrails layer | Local OSS scanners + optional ProtectAI llm-guard / AWS Bedrock / Azure |
