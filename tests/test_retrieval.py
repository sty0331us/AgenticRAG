"""Unit tests for multi-query retrieval helpers."""

from __future__ import annotations

from agentic_rag.retrieval_utils import merge_ranked_hits, parse_alt_queries


def test_parse_alt_queries_pipe_and_limit() -> None:
    raw = "alpha | beta | alpha | gamma"
    assert parse_alt_queries(raw, limit=2) == ["alpha", "beta"]


def test_parse_alt_queries_newlines() -> None:
    raw = "first\nsecond\nthird"
    assert parse_alt_queries(raw, limit=5) == ["first", "second", "third"]


def test_parse_alt_queries_empty() -> None:
    assert parse_alt_queries("", limit=3) == []
    assert parse_alt_queries("a | b", limit=0) == []


def test_merge_ranked_hits_keeps_best_score() -> None:
    group_a = (["doc1", "doc2"], [{"id": "a1"}, {"id": "a2"}], [0.5, 0.8])
    group_b = (["doc1", "doc3"], [{"id": "b1"}, {"id": "b3"}], [0.9, 0.7])
    docs, metas, scores = merge_ranked_hits([group_a, group_b], top_k=2)
    assert docs == ["doc1", "doc2"]
    assert scores == [0.9, 0.8]
    assert metas[0]["id"] == "b1"
