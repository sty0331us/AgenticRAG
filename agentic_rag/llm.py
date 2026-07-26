"""
Free-tier LLM setup via Groq (OpenAI-compatible free inference).

Difference from normal RAG
--------------------------
In normal RAG the LLM is called once to generate the final answer.

In Agentic RAG the same client is reused by multiple agents (Reasoning and
Verification) with different temperatures/prompts — specialization by role,
not a single monolithic generate step.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Groq free tier — latest strong open model available without paid OpenAI credits
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """Return a ChatGroq client configured for free-tier inference."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
            "from https://console.groq.com/keys"
        )

    model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )
