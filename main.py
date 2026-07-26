#!/usr/bin/env python3
"""Command-line interface for the Agentic RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from agentic_rag.config import get_settings
from agentic_rag.exceptions import AgenticRAGError, ConfigurationError
from agentic_rag.graph import AgenticRAGPipeline
from agentic_rag.ingest import ingest_knowledge
from agentic_rag.logging_config import configure_logging, get_logger
from agentic_rag.models import PipelineResult

logger = get_logger(__name__)


def _print_react_trace(result: PipelineResult) -> None:
    if not result.react_trace:
        return
    print("=" * 72)
    print("EXECUTION TRACE (Thought → Action → Observation)")
    print("=" * 72)
    for index, step in enumerate(result.react_trace, start=1):
        print(f"\n[{index}] Agent: {step.agent}")
        print(f"    Thought:     {step.thought}")
        print(f"    Action:      {step.action}")
        print(f"    Observation: {step.observation}")
    print()


def _print_result(result: PipelineResult) -> None:
    _print_react_trace(result)
    print("=" * 72)
    print("VERIFIED ANSWER")
    print("=" * 72)
    print(result.answer or "(empty)")
    print()
    print(f"Grounded: {result.is_grounded}")
    if result.search_query:
        print(f"Search query: {result.search_query}")
    if result.verification_notes:
        print(f"Verification notes: {result.verification_notes}")
    if result.errors:
        print(f"Errors: {result.errors}")
    print(f"Retrieved documents: {len(result.retrieved_docs)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-rag",
        description=(
            "Execute the multi-agent Agentic RAG pipeline "
            "(retrieval → reasoning → verification) with ReAct tracing."
        ),
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "What is Agentic RAG and how do the Retrieval, Reasoning, "
            "and Verification agents operate?"
        ),
        help="Natural-language question submitted to the pipeline",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Force re-ingestion of the reference knowledge corpus",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full PipelineResult as JSON",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = get_settings()
        configure_logging(args.log_level or settings.log_level)
        document_count = ingest_knowledge(force=args.reingest, settings=settings)
        logger.info("Indexed document count=%s", document_count)

        pipeline = AgenticRAGPipeline(settings=settings)
        result = pipeline.invoke(args.query)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"Indexed documents: {document_count}\n")
            print(f"Query: {args.query}\n")
            _print_result(result)
        return 0
    except ConfigurationError as exc:
        logger.error("%s", exc)
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except AgenticRAGError as exc:
        logger.error("%s", exc)
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled failure")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
