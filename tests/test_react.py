"""Unit tests for ReAct parsing and context formatting."""

from __future__ import annotations

from agentic_rag.react import format_context, parse_labeled_blocks


def test_parse_labeled_blocks_single_line() -> None:
    text = "THOUGHT: plan search\nSEARCH_QUERY: agentic rag retrieval"
    parsed = parse_labeled_blocks(text, ["THOUGHT", "SEARCH_QUERY"])
    assert parsed["THOUGHT"] == "plan search"
    assert parsed["SEARCH_QUERY"] == "agentic rag retrieval"


def test_parse_labeled_blocks_multiline() -> None:
    text = (
        "THOUGHT: line one\n"
        "continued\n"
        "DRAFT_ANSWER: final draft\n"
        "still draft"
    )
    parsed = parse_labeled_blocks(text, ["THOUGHT", "DRAFT_ANSWER"])
    assert parsed["THOUGHT"] == "line one\ncontinued"
    assert parsed["DRAFT_ANSWER"] == "final draft\nstill draft"


def test_parse_labeled_blocks_missing_labels_default_empty() -> None:
    parsed = parse_labeled_blocks("unrelated", ["THOUGHT", "NOTES"])
    assert parsed == {"THOUGHT": "", "NOTES": ""}


def test_format_context_with_scores_and_metadata() -> None:
    formatted = format_context(
        ["chunk a", "chunk b"],
        metadatas=[{"topic": "patterns"}, {"section": "verify"}],
        scores=[0.91, 0.42],
    )
    assert "[1] score=0.910 source=patterns" in formatted
    assert "chunk a" in formatted
    assert "[2] score=0.420 source=verify" in formatted
