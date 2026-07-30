"""ReAct utilities: audit trail updates and structured LLM output parsing."""

from __future__ import annotations

from typing import Dict, List, Optional

from agentic_rag.state import AgenticRAGState, ReActStep


def append_react_step(
    state: AgenticRAGState,
    *,
    agent: str,
    thought: str,
    action: str,
    observation: str,
) -> None:
    """Append one ReAct cycle to the workflow audit trail."""
    step: ReActStep = {
        "agent": agent,
        "thought": thought.strip(),
        "action": action.strip(),
        "observation": observation.strip(),
    }
    trace: List[ReActStep] = list(state.get("react_trace") or [])
    trace.append(step)
    state["react_trace"] = trace


def append_error(state: AgenticRAGState, message: str) -> None:
    """Record an error message on the shared state."""
    errors = list(state.get("errors") or [])
    errors.append(message)
    state["errors"] = errors


def parse_labeled_blocks(text: str, labels: List[str]) -> Dict[str, str]:
    """
    Parse labeled LLM output blocks.

    Expected shape:
        LABEL: value
        OTHER_LABEL: multi-line value...
    """
    result: Dict[str, str] = {label: "" for label in labels}
    current: Optional[str] = None
    buffer: List[str] = []
    upper_map = {label.upper(): label for label in labels}

    def flush() -> None:
        nonlocal buffer, current
        if current is not None:
            result[current] = "\n".join(buffer).strip()
        buffer = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        matched: Optional[str] = None
        for upper, original in upper_map.items():
            if line.upper().startswith(f"{upper}:") or line.upper().startswith(f"{upper} :"):
                matched = original
                after = line.split(":", 1)[1].strip() if ":" in line else ""
                flush()
                current = matched
                buffer = [after] if after else []
                break
        if matched is None and current is not None:
            buffer.append(line)

    flush()
    return result


def format_context(
    docs: List[str],
    metadatas: Optional[List[dict]] = None,
    scores: Optional[List[float]] = None,
) -> str:
    """Format retrieved chunks with stable citation indices and optional provenance."""
    blocks: List[str] = []
    for i, doc in enumerate(docs):
        header_parts = [f"[{i + 1}]"]
        if scores is not None and i < len(scores):
            header_parts.append(f"score={scores[i]:.3f}")
        if metadatas is not None and i < len(metadatas) and metadatas[i]:
            meta = metadatas[i]
            topic = meta.get("topic") or meta.get("section") or meta.get("source")
            if topic:
                header_parts.append(f"source={topic}")
        header = " ".join(header_parts)
        blocks.append(f"{header}\n{doc}")
    return "\n\n".join(blocks)
