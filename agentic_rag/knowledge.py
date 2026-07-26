"""Seed knowledge documents for the Agentic RAG demo (multi-agent + Agentic RAG)."""

KNOWLEDGE_CHUNKS = [
    {
        "id": "why_multi_agent",
        "text": (
            "Why multi-agent systems? Single LLM agents face several challenges: "
            "context overload, role confusion, debugging difficulty, and quality dilution. "
            "Multi-agent systems solve these by splitting tasks among agents to reduce burden, "
            "letting agents specialize in distinct cognitive roles, using modular agents to ease "
            "error tracing, and having each agent excel at a focused subtask."
        ),
        "metadata": {"topic": "motivation", "section": "why_multi_agent"},
    },
    {
        "id": "pattern_sequential",
        "text": (
            "Typical multi-agent communication pattern — Sequential (Pipeline): "
            "Agents work one after another, passing results. "
            "Example: Research → Analysis → Writing → Review."
        ),
        "metadata": {"topic": "patterns", "section": "sequential"},
    },
    {
        "id": "pattern_parallel",
        "text": (
            "Typical multi-agent communication pattern — Parallel with aggregation: "
            "Multiple agents run concurrently and results are combined. "
            "Example: SEO analysis, fact-checking, and writing run in parallel."
        ),
        "metadata": {"topic": "patterns", "section": "parallel"},
    },
    {
        "id": "pattern_interactive",
        "text": (
            "Typical multi-agent communication pattern — Interactive dialogue: "
            "Agents exchange messages to clarify or refine. "
            "Example: A requirements agent queries a data agent before finalizing."
        ),
        "metadata": {"topic": "patterns", "section": "interactive"},
    },
    {
        "id": "usecase_market",
        "text": (
            "Real-world multi-agent use case — Automated market report: "
            "Agents & workflow: Research → Data analysis → Writing → Critique → Editing. "
            "Benefit: Faster, accurate, well-rounded reports."
        ),
        "metadata": {"topic": "use_cases", "section": "market_report"},
    },
    {
        "id": "usecase_support",
        "text": (
            "Real-world multi-agent use case — Customer support: "
            "Agents & workflow: Intent detection → Knowledge retrieval → Response → Escalation. "
            "Benefit: Dynamic, personalized, scalable responses."
        ),
        "metadata": {"topic": "use_cases", "section": "customer_support"},
    },
    {
        "id": "usecase_legal",
        "text": (
            "Real-world multi-agent use case — Legal contract review: "
            "Agents & workflow: Clause extraction → Compliance check → Risk analysis → Summary. "
            "Benefit: Thorough, accurate, actionable legal reviews."
        ),
        "metadata": {"topic": "use_cases", "section": "legal_review"},
    },
    {
        "id": "protocol_mcp",
        "text": (
            "Communication protocol — Model Context Protocol (MCP): "
            "JSON-RPC-based interface for LLMs to interact with external tools/services, "
            "enabling modular, real-time collaboration."
        ),
        "metadata": {"topic": "protocols", "section": "mcp"},
    },
    {
        "id": "protocol_acp",
        "text": (
            "Communication protocol — IBM Agent Communication Protocol (ACP): "
            "Standardizes message exchange among autonomous agents for secure, "
            "scalable enterprise workflows."
        ),
        "metadata": {"topic": "protocols", "section": "acp"},
    },
    {
        "id": "framework_langgraph",
        "text": (
            "Framework supporting multi-agent LLM systems — LangGraph: "
            "Graph-based orchestration, shared state, and dynamic routing. "
            "Core concepts: directed graph nodes represent agents/tasks; edges control flow; "
            "shared state (TypedDict) is passed and updated by all agents; "
            "routing logic dynamically determines the next agent based on state."
        ),
        "metadata": {"topic": "frameworks", "section": "langgraph"},
    },
    {
        "id": "framework_others",
        "text": (
            "Other multi-agent frameworks: AutoGen focuses on agent self-organization, "
            "negotiation, and adaptive collaboration. CrewAI emphasizes structured workflows, "
            "strict typed interfaces (Pydantic), and high-fidelity data passing. "
            "BeeAI provides enterprise-grade modular orchestration and uses IBM ACP."
        ),
        "metadata": {"topic": "frameworks", "section": "others"},
    },
    {
        "id": "agentic_rag",
        "text": (
            "Agentic RAG systems combine Retrieval, Reasoning, and Verification using "
            "specialized agents. The Retrieval agent fetches relevant knowledge/data. "
            "The Reasoning agent performs inference and decision-making. "
            "The Verification agent checks results for accuracy and consistency. "
            "Multi-agent design improves reliability and trustworthiness."
        ),
        "metadata": {"topic": "agentic_rag", "section": "overview"},
    },
    {
        "id": "best_practices",
        "text": (
            "Best practices & challenges for multi-agent systems: "
            "Context management — share only relevant info, avoid overload. "
            "Granularity — balance agent count, not too few or too many. "
            "Communication cost — optimize message size and frequency. "
            "Error handling — implement fallback, retries, and error agents."
        ),
        "metadata": {"topic": "best_practices", "section": "challenges"},
    },
]
