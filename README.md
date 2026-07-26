# RAG vs Agentic RAG

Multi-agent **Agentic RAG** with **LangGraph**, **ChromaDB**, and **Groq** free-tier inference (`llama-3.3-70b-versatile`).

Each agent follows an explicit **ReAct** loop: **Thought → Action → Observation**. That is the main structural difference from normal RAG, which usually does a single retrieve → generate pass with no visible reasoning/acting steps.

---

## Project layout

```
RAG_VS_AgenticRAG/
├── main.py                 # CLI — prints ReAct traces + verified answer
├── requirements.txt
├── .env.example
├── README.md
└── agentic_rag/
    ├── __init__.py
    ├── state.py            # Shared state + ReActStep trace
    ├── react.py            # Thought/Action/Observation helpers
    ├── llm.py              # Groq free LLM client
    ├── vectorstore.py      # ChromaDB (same substrate as normal RAG)
    ├── knowledge.py        # Seed documents
    ├── ingest.py           # Load chunks into ChromaDB
    ├── agents.py           # Retrieval / Reasoning / Verification / Error
    └── graph.py            # LangGraph orchestration + routing
```

---

## High-level graph

```text
                         ┌──────────────┐
                         │  User query  │
                         └──────┬───────┘
                                ▼
              ┌─────────────────────────────────────┐
              │         Retrieval agent             │
              │  Thought → Action → Observation     │
              └──────────────────┬──────────────────┘
                                 ▼
              ┌─────────────────────────────────────┐
              │         Reasoning agent             │
              │  Thought → Action → Observation     │
              └──────────────────┬──────────────────┘
                                 ▼
              ┌─────────────────────────────────────┐
              │       Verification agent            │
              │  Thought → Action → Observation     │
              └───────────┬─────────────┬───────────┘
                 grounded │             │ not grounded
                          ▼             ▼
                     ┌────────┐   ┌─────────────┐
                     │  END   │   │ retry       │──► Retrieval again
                     └────────┘   └─────────────┘

Any failure ──► Error-handler agent (fallback Action) ──► END
```

---

## Agents in detail (reasoning & acting)

### 1. Retrieval agent

| ReAct step | What happens |
|------------|----------------|
| **Thought** | LLM plans what to look up and rewrites a focused `SEARCH_QUERY` |
| **Action** | Tool call: `similarity_search(search_query, k=4)` on ChromaDB |
| **Observation** | How many chunks came back; store docs in shared state |
| **Route** | `reason` if docs found, else `error` |

```text
THOUGHT: "Need definition of Agentic RAG and the three agent roles"
ACTION:  similarity_search("Agentic RAG retrieval reasoning verification", k=4)
OBSERVATION: "Retrieved 4 chunk(s)..."
```

Normal RAG usually searches with the raw user string and never records a search plan.

---

### 2. Reasoning agent

| ReAct step | What happens |
|------------|----------------|
| **Thought** | Analyze evidence: what supports the answer, what is missing |
| **Action** | Produce `DRAFT_ANSWER` grounded only in retrieved context |
| **Observation** | Draft length / ready for verification |
| **Route** | Always `verify` (draft is never final) |

```text
THOUGHT: "Chunks [1] and [12] define Agentic RAG; cite them"
ACTION:  draft_answer_from_context
OBSERVATION: "Draft produced (... chars). Awaiting verification."
```

Normal RAG merges “think” and “answer” into one opaque generate call and returns it immediately.

---

### 3. Verification agent

| ReAct step | What happens |
|------------|----------------|
| **Thought** | Check which draft claims are supported by context |
| **Action** | `accept_answer` / `reject_and_retry_retrieval` / `accept_best_effort` |
| **Observation** | Grounded flag + notes; may loop the graph |
| **Route** | `complete`, or `retrieve` to retry, or `complete` after max retries |

```text
THOUGHT: "Claim about Verification agent is supported by [12]"
ACTION:  accept_answer
OBSERVATION: "Draft accepted as grounded. Completing workflow."
```

Normal RAG has no second agent and no verification-driven retry.

---

### 4. Error-handler agent

| ReAct step | What happens |
|------------|----------------|
| **Thought** | Upstream failed; avoid crashing |
| **Action** | `emit_fallback_answer` |
| **Observation** | Error summary written into state |

---

## ReAct inside one agent (template)

Every specialist agent writes to `react_trace`:

```text
┌──────────────────────────────────────────┐
│ Agent (e.g. Reasoning)                   │
│                                          │
│  1. THOUGHT   — decide strategy / gaps   │
│  2. ACTION    — tool call or draft text  │
│  3. OBSERVE   — write result to state    │
│  4. ROUTE     — set next_action          │
└──────────────────────────────────────────┘
```

Shared fields that make this inspectable:

| State field | Filled by | Purpose |
|-------------|-----------|---------|
| `search_query` | Retrieval Thought | Rewritten vector query |
| `reasoning_thought` | Reasoning Thought | Evidence plan |
| `draft_answer` | Reasoning Action | Intermediate answer |
| `verification_thought` | Verification Thought | Grounding analysis |
| `verified_answer` | Verification Action | Final (or corrected) answer |
| `react_trace` | All agents | Full Thought/Action/Observation log |
| `next_action` | All agents | Graph routing key |

---

## Normal RAG vs this Agentic RAG

| Aspect | Normal RAG | Agentic RAG (this repo) |
|--------|------------|-------------------------|
| Pipeline | retrieve → generate | Graph of ReAct agents |
| Reasoning | Hidden inside one LLM call | Explicit `THOUGHT` per agent |
| Acting | Generate text once | Tool acts (search) + draft + accept/reject |
| Observation | Discarded locals | Stored in `react_trace` + state |
| Control | Fixed order | `next_action` + retries |
| Quality | First answer is final | Verification can reject and re-retrieve |
| Failures | Raise / silent bad answer | Error-handler agent |

**Same as normal RAG:** ingest, embeddings, ChromaDB similarity search.

**Different:** multi-agent ReAct orchestration, shared state, dynamic routing, verification loops.

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
python main.py "Compare MCP and ACP" --json    # full state including react_trace
python main.py --reingest "Why use multi-agent systems?"
```

The CLI prints each agent's Thought → Action → Observation by default, then the verified answer.

## Stack

- **LangGraph** — graph nodes/edges, shared state, dynamic routing
- **ChromaDB** — local persistent vector store (retrieval Action tool)
- **Groq** — free inference for Thought / draft / verify LLM steps
