"""Helpers shared by ReAct-style agents (Thought → Action → Observation)."""

from __future__ import annotations

from typing import Dict, List

from agentic_rag.state import AgenticRAGState, ReActStep


def append_react_step(
    state: AgenticRAGState,
    *,
    agent: str,
    thought: str,
    action: str,
    observation: str,
) -> None:
    """Append one ReAct cycle to the shared audit trail."""
    step: ReActStep = {
        "agent": agent,
        "thought": thought,
        "action": action,
        "observation": observation,
    }
    trace: List[ReActStep] = list(state.get("react_trace") or [])
    trace.append(step)
    state["react_trace"] = trace


def parse_labeled_blocks(text: str, labels: List[str]) -> Dict[str, str]:
    """
    Parse LLM output of the form:
      LABEL: value
      OTHER_LABEL: multi-line value...
    """
    result: Dict[str, str] = {label: "" for label in labels}
    current: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current
        if current is not None:
            result[current] = "\n".join(buffer).strip()
        buffer = []

    upper_map = {label.upper(): label for label in labels}

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        matched = None
        for upper, original in upper_map.items():
            prefix = f"{upper}:"
            if line.upper().startswith(prefix) or line.upper().startswith(f"{upper} :"):
                matched = original
                # Keep text after the first colon on this line
                after = line.split(":", 1)[1].strip() if ":" in line else ""
                flush()
                current = matched
                buffer = [after] if after else []
                break
        if matched is None and current is not None:
            buffer.append(line)

    flush()
    return result


def format_context(docs: List[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
