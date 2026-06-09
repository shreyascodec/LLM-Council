"""Robust extraction of JSON from messy model output.

Models often wrap JSON in prose, fence it in code blocks, or leave a trailing
comma. We try a sequence of increasingly forgiving strategies and raise if none
succeed, so the caller can take an explicit failure path.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|jsonc)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONRepairError(ValueError):
    pass


def _balanced_span(text: str) -> str | None:
    """Return the first top-level {...} or [...] block by bracket matching,
    ignoring brackets inside strings."""
    start = None
    opener = closer = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start, opener = i, ch
            closer = "}" if ch == "{" else "]"
            break
    if start is None:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # truncated / unbalanced


def _strip_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Best-effort parse. Raises JSONRepairError if nothing usable is found."""
    if text is None:
        raise JSONRepairError("empty model output")

    candidates: list[str] = []

    # 1) fenced blocks first (most reliable when present)
    for m in _FENCE_RE.findall(text):
        candidates.append(m.strip())

    # 2) the whole string
    candidates.append(text.strip())

    # 3) the first balanced bracket span
    span = _balanced_span(text)
    if span:
        candidates.append(span)

    for cand in candidates:
        if not cand:
            continue
        for attempt in (cand, _strip_trailing_commas(cand)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    raise JSONRepairError("no valid JSON object/array found in model output")
