#!/usr/bin/env python3
"""
CLI entrypoint for the Agentic RAG demo.

Difference from normal RAG
--------------------------
Prints verified answers plus optional ReAct traces (Thought / Action / Observation)
from each agent — not just a single generate() string.
"""

from __future__ import annotations

import argparse
import json
import sys

from agentic_rag.graph import run_agentic_rag
from agentic_rag.ingest import ingest_knowledge


def _print_react_trace(trace) -> None:
    if not trace:
        return
    print("=" * 60)
    print("REACT TRACE  (Thought → Action → Observation per agent)")
    print("=" * 60)
    for i, step in enumerate(trace, 1):
        print(f"\n[{i}] Agent: {step.get('agent')}")
        print(f"    Thought:     {step.get('thought')}")
        print(f"    Action:      {step.get('action')}")
        print(f"    Observation: {step.get('observation')}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic RAG with LangGraph + ChromaDB. "
            "Each agent runs Thought → Action → Observation (ReAct)."
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
        help="Print full final state as JSON (includes react_trace)",
    )
    args = parser.parse_args()

    count = ingest_knowledge(force=args.reingest)
    print(f"ChromaDB documents: {count}\n")
    print(f"Query: {args.query}\n")
    print("Running Agentic RAG agents (each: Thought → Action → Observation)...\n")

    final_state = run_agentic_rag(args.query)

    if args.json:
        print(json.dumps(final_state, indent=2, default=str))
        return 0

    _print_react_trace(final_state.get("react_trace") or [])

    print("=" * 60)
    print("VERIFIED ANSWER")
    print("=" * 60)
    print(final_state.get("verified_answer") or "(no answer)")
    print()
    print(f"Grounded: {final_state.get('is_grounded')}")
    if final_state.get("search_query"):
        print(f"Search query (retrieval Thought→Act): {final_state['search_query']}")
    if final_state.get("verification_notes"):
        print(f"Notes: {final_state['verification_notes']}")
    if final_state.get("errors"):
        print(f"Errors: {final_state['errors']}")
    if final_state.get("retrieved_docs"):
        print(f"\nRetrieved {len(final_state['retrieved_docs'])} chunks from ChromaDB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
