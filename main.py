#!/usr/bin/env python3
"""Command-line interface for the Agentic RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

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


def _print_sources(result: PipelineResult, *, max_chars: int = 220) -> None:
    if not result.sources:
        return
    print("=" * 72)
    print("RETRIEVED SOURCES")
    print("=" * 72)
    for source in result.sources:
        score = f"{source.score:.3f}" if source.score is not None else "n/a"
        topic = (
            source.metadata.get("topic")
            or source.metadata.get("section")
            or source.metadata.get("source")
            or "unknown"
        )
        snippet = source.text.replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 1] + "…"
        print(f"{source.citation_label} score={score} source={topic}")
        print(f"    {snippet}")
    print()


def _print_metrics(result: PipelineResult) -> None:
    metrics = result.metrics
    if metrics.total_seconds is None and metrics.retrieval_seconds is None:
        return
    print("=" * 72)
    print("LATENCY METRICS (seconds)")
    print("=" * 72)
    for label, value in (
        ("total", metrics.total_seconds),
        ("retrieval", metrics.retrieval_seconds),
        ("reasoning", metrics.reasoning_seconds),
        ("verification", metrics.verification_seconds),
        ("error_handler", metrics.error_handler_seconds),
    ):
        if value is None:
            continue
        print(f"  {label:14s} {value:8.3f}")
    print()


def _print_result(
    result: PipelineResult,
    *,
    show_sources: bool = False,
    show_metrics: bool = False,
) -> None:
    _print_react_trace(result)
    if show_sources:
        _print_sources(result)
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
    if show_metrics:
        print()
        _print_metrics(result)


def _load_batch_queries(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    queries = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not queries:
        raise AgenticRAGError(f"No queries found in batch file: {path}")
    return queries


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
        default=None,
        help="Natural-language question submitted to the pipeline",
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help="Path to a text file with one query per line (# comments allowed)",
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
        "--show-sources",
        action="store_true",
        help="Print retrieved evidence snippets with similarity scores",
    )
    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Print per-agent and total latency metrics",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    default_query = (
        "What is Agentic RAG and how do the Retrieval, Reasoning, "
        "and Verification agents operate?"
    )
    if args.batch_file and args.query:
        print("Provide either a positional query or --batch-file, not both.", file=sys.stderr)
        return 2
    if args.batch_file:
        try:
            queries = _load_batch_queries(args.batch_file)
        except (OSError, AgenticRAGError) as exc:
            print(f"Batch file error: {exc}", file=sys.stderr)
            return 2
    else:
        queries = [args.query or default_query]

    try:
        settings = get_settings()
        configure_logging(args.log_level or settings.log_level)
        document_count = ingest_knowledge(force=args.reingest, settings=settings)
        logger.info("Indexed document count=%s", document_count)

        pipeline = AgenticRAGPipeline(settings=settings)
        results: List[PipelineResult] = []
        for query in queries:
            results.append(pipeline.invoke(query))

        if args.json:
            payload = (
                [result.to_dict() for result in results]
                if len(results) > 1
                else results[0].to_dict()
            )
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Indexed documents: {document_count}\n")
            for index, (query, result) in enumerate(zip(queries, results), start=1):
                if len(results) > 1:
                    print("#" * 72)
                    print(f"BATCH ITEM {index}/{len(results)}")
                    print("#" * 72)
                print(f"Query: {query}\n")
                _print_result(
                    result,
                    show_sources=args.show_sources,
                    show_metrics=args.show_metrics,
                )
                if index < len(results):
                    print()
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
