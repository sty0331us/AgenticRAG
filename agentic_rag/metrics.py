"""Execution timing helpers for pipeline observability."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from agentic_rag.state import AgenticRAGState


def _metrics(state: AgenticRAGState) -> Dict[str, float]:
    raw = state.get("metrics")
    return dict(raw) if isinstance(raw, dict) else {}


def record_metric(state: AgenticRAGState, key: str, value: float) -> None:
    """Persist a numeric metric on shared workflow state."""
    metrics = _metrics(state)
    metrics[key] = float(value)
    state["metrics"] = metrics


def add_metric(state: AgenticRAGState, key: str, delta: float) -> None:
    """Add ``delta`` to an existing metric (useful across verification retries)."""
    metrics = _metrics(state)
    metrics[key] = float(metrics.get(key, 0.0)) + float(delta)
    state["metrics"] = metrics


@contextmanager
def timed_section(state: AgenticRAGState, key: str, *, accumulate: bool = True) -> Iterator[None]:
    """
    Measure wall-clock seconds for a section and store under ``metrics[key]``.

    When ``accumulate`` is True (default), durations from retries are summed.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        if accumulate:
            add_metric(state, key, elapsed)
        else:
            record_metric(state, key, elapsed)


def elapsed_since(start: float) -> float:
    """Return seconds elapsed since ``start`` (from ``time.perf_counter``)."""
    return time.perf_counter() - start
