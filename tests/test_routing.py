"""Unit tests for workflow routing map."""

from __future__ import annotations

from agentic_rag.routing import route_next_step


def test_route_next_step_known_actions() -> None:
    assert route_next_step({"next_action": "retrieve"}) == "retrieval"  # type: ignore[arg-type]
    assert route_next_step({"next_action": "reason"}) == "reasoning"  # type: ignore[arg-type]
    assert route_next_step({"next_action": "verify"}) == "verification"  # type: ignore[arg-type]
    assert route_next_step({"next_action": "error"}) == "error_handler"  # type: ignore[arg-type]
    assert route_next_step({"next_action": "complete"}) == "END"  # type: ignore[arg-type]


def test_route_next_step_unknown_defaults_to_end() -> None:
    assert route_next_step({"next_action": "nope"}) == "END"  # type: ignore[arg-type]
    assert route_next_step({}) == "retrieval"  # type: ignore[arg-type]
