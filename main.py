#!/usr/bin/env python3
"""
CLI entrypoint for the Agentic RAG demo.

Difference from normal RAG
--------------------------
A normal RAG CLI would: ingest (optional) → retrieve → one LLM call → print answer.
This CLI invokes a multi-agent LangGraph workflow and can surface Agentic-only
fields: grounded flag, verification notes, draft→verified answers, and errors.
"""

from __future__ import annotations

import argparse
import json
import sys

from agentic_rag.graph import run_agentic_rag
from agentic_rag.ingest import ingest_knowledge


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic RAG with LangGraph + ChromaDB (Groq free inference). "
            "Unlike normal RAG, this runs retrieval → reasoning → verification agents."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is Agentic RAG and how do the Retrieval, Reasoning, and Verification agents work?",
        help="Question to ask the Agentic RAG system",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingest of knowledge chunks into ChromaDB",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full final state as JSON (includes Agentic fields normal RAG lacks)",
    )
    args = parser.parse_args()

    count = ingest_knowledge(force=args.reingest)
    print(f"ChromaDB documents: {count}\n")
    print(f"Query: {args.query}\n")
    print("Running Agentic RAG (retrieval → reasoning → verification)...\n")
    print("(Normal RAG would stop after a single retrieve → generate step.)\n")

    final_state = run_agentic_rag(args.query)

    if args.json:
        print(json.dumps(final_state, indent=2, default=str))
        return 0

    print("=" * 60)
    print("VERIFIED ANSWER  (Agentic: post-verification; Normal RAG: first draft)")
    print("=" * 60)
    print(final_state.get("verified_answer") or "(no answer)")
    print()
    # These metrics are Agentic-specific — classic RAG typically has no grounding check
    print(f"Grounded: {final_state.get('is_grounded')}")
    if final_state.get("verification_notes"):
        print(f"Notes: {final_state['verification_notes']}")
    if final_state.get("errors"):
        print(f"Errors: {final_state['errors']}")
    if final_state.get("retrieved_docs"):
        print(f"\nRetrieved {len(final_state['retrieved_docs'])} chunks from ChromaDB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
