# RAG vs Agentic RAG

Multi-agent **Agentic RAG** built with **LangGraph**, **ChromaDB**, and **Groq** free-tier inference (`llama-3.3-70b-versatile`).

This project shows how Agentic RAG differs from a normal (classic) RAG pipeline: specialized agents, shared state, dynamic routing, verification, and retries.

---

## System structure

```
RAG_VS_AgenticRAG/
├── main.py                 # CLI entry — runs the multi-agent graph
├── requirements.txt
├── .env.example
├── README.md
└── agentic_rag/
    ├── __init__.py
    ├── state.py            # Shared TypedDict state (vs one-shot locals in normal RAG)
    ├── llm.py              # Free Groq LLM client
    ├── vectorstore.py      # ChromaDB (same retrieval substrate as normal RAG)
    ├── knowledge.py        # Seed documents about multi-agent / Agentic RAG
    ├── ingest.py           # Load chunks into ChromaDB
    ├── agents.py           # Retrieval / Reasoning / Verification / Error agents
    └── graph.py            # LangGraph orchestration + routing
```

### Runtime flow (Agentic RAG)

```text
                    ┌─────────────────┐
                    │  User query     │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Retrieval agent │  ← ChromaDB similarity search
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Reasoning agent │  ← Draft answer from context
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
              ┌─────│ Verification    │─────┐
              │     │ agent           │     │
              │     └─────────────────┘     │
              │ grounded?                   │ not grounded + retries left
              ▼                             ▼
         ┌─────────┐              ┌─────────────────┐
         │  END    │              │ re-retrieve /   │
         │ answer  │              │ re-reason loop  │
         └─────────┘              └─────────────────┘

Any step failure → Error-handler agent → safe fallback answer → END
```

### Normal RAG (for contrast)

```text
User query → retrieve top-k → single LLM generate → answer (done)
```

No separate verification agent, no dynamic routing, no retry loop, no dedicated error agent.

---

## Normal RAG vs Agentic RAG

| Aspect | Normal RAG | This Agentic RAG |
|--------|------------|------------------|
| Pipeline | Fixed linear: retrieve → generate | Graph of agents with conditional edges |
| Roles | One LLM does everything after retrieval | Specialized agents (retrieve / reason / verify / error) |
| State | Ephemeral locals in one function | Shared `AgenticRAGState` updated by every node |
| Control flow | Always the same path | `next_action` + `route_next_step()` choose the next node |
| Quality check | Usually none (answer returned as-is) | Verification agent checks grounding; can retry |
| Failures | Often bubble up or silent bad answers | Error-handler agent + retries |
| Debugging | Harder (monolithic prompt/call) | Easier (inspect per-agent state fields) |
| Vector DB | ChromaDB / similar | Same idea (ChromaDB) — difference is *orchestration*, not storage |

**Shared with normal RAG:** document ingest, embeddings, similarity search over a vector store.

**Unique to Agentic RAG here:** multi-agent specialization, LangGraph routing, verification, retry, and modular error handling.

---

## Module map (what differs where)

| File | Role | vs normal RAG |
|------|------|----------------|
| `vectorstore.py` / `ingest.py` | Index & search | Closest to normal RAG (same retrieve step) |
| `state.py` | Shared graph state | Normal RAG has no multi-step shared state object |
| `agents.py` | Four specialized nodes | Normal RAG collapses reason+answer into one LLM call |
| `graph.py` | LangGraph + routing | Normal RAG is a script, not a routed graph |
| `llm.py` | Model client | Used by *multiple* agents, not once |
| `main.py` | CLI | Invokes the graph, not a single retrieve→generate |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Set GROQ_API_KEY from https://console.groq.com/keys
```

## Run

```bash
source .venv/bin/activate
python main.py "What is Agentic RAG?"
python main.py "Compare MCP and ACP" --json
python main.py --reingest "Why use multi-agent systems?"
```

Knowledge about multi-agent systems and Agentic RAG is auto-ingested into ChromaDB on first run.

## Stack

- **LangGraph** — graph orchestration, shared state, dynamic routing
- **ChromaDB** — local persistent vector store
- **Groq** — free inference (`llama-3.3-70b-versatile` by default)
