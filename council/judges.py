"""Judges: score candidate answers against the rubric; they never generate.

A judge gets the question, candidate answers, and rubric, and returns per-criterion
scores plus a justification. Two defenses against a judge writing its own answer:
the prompt gives it no room to, and the parser detects and discards any smuggled
answer keys, recording that it did so. Scores are keyed by candidate index, so a
judge can only score the text it was given.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .client import LLMResult, OpenRouterClient
from .config import Config, ModelSpec
from .generators import Candidate
from .jsonparse import JSONRepairError, extract_json

# Keys that indicate a judge tried to (re)write an answer instead of just scoring.
_SMUGGLE_KEYS = {"answer", "my_answer", "response", "corrected_answer",
                 "improved_answer", "rewrite", "better_answer", "solution"}


@dataclass
class JudgeResult:
    model_id: str
    family: str
    ok: bool
    # scores[candidate_index][criterion_name] = float
    scores: dict[int, dict[str, float]] = field(default_factory=dict)
    justification: str = ""
    smuggled_answer_discarded: bool = False
    latency_ms: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw: str = ""


def _build_system(criteria: list[dict], smin: int, smax: int) -> str:
    lines = "\n".join(f'  - "{c["name"]}" ({c["description"]})' for c in criteria)
    names = ", ".join(f'"{c["name"]}"' for c in criteria)
    return (
        "You are a JUDGE on an answer panel. You DO NOT answer the question yourself. "
        "Your only job is to score each candidate answer against the rubric.\n\n"
        f"Rubric criteria, each scored as an integer from {smin} to {smax}:\n{lines}\n\n"
        "Respond with ONLY a JSON object in this exact shape (no prose, no answer of "
        "your own):\n"
        "{\n"
        '  "scores": {\n'
        '     "0": {' + names.replace(", ", ": <int>, ") + ": <int>},\n"
        '     "1": { ... }, ...\n'
        "  },\n"
        '  "justification": "one or two sentences explaining the scores"\n'
        "}\n"
        "Score EVERY candidate by its index. Do not include any answer text of your own."
    )


def _build_user(question: str, candidates: list[Candidate]) -> str:
    parts = [f"QUESTION:\n{question}\n", "CANDIDATE ANSWERS:"]
    for c in candidates:
        parts.append(f"\n[Candidate {c.index}]\n{c.answer}")
    parts.append("\nScore each candidate by index against the rubric now.")
    return "\n".join(parts)


def _coerce_scores(obj: dict, criteria: list[dict], smin: int, smax: int,
                   valid_indices: set[int]) -> dict[int, dict[str, float]]:
    raw_scores = obj.get("scores", {})
    if not isinstance(raw_scores, dict):
        raise JSONRepairError("missing 'scores' object")
    crit_names = [c["name"] for c in criteria]
    out: dict[int, dict[str, float]] = {}
    for k, v in raw_scores.items():
        try:
            idx = int(k)
        except (ValueError, TypeError):
            continue
        if idx not in valid_indices or not isinstance(v, dict):
            continue
        per: dict[str, float] = {}
        for name in crit_names:
            val = v.get(name)
            if isinstance(val, (int, float)):
                per[name] = float(max(smin, min(smax, val)))
        if per:
            out[idx] = per
    if not out:
        raise JSONRepairError("no usable per-candidate scores")
    return out


def _parse_judge(res: LLMResult, criteria, smin, smax, valid_indices) -> JudgeResult:
    base = dict(model_id=res.model_id, family=res.family, latency_ms=res.latency_ms,
                tokens=res.tokens, cost_usd=res.cost_usd, raw=res.text)
    if not res.ok:
        return JudgeResult(ok=False, error=res.error, **base)
    try:
        data = extract_json(res.text)
        if not isinstance(data, dict):
            raise JSONRepairError("expected JSON object")
    except JSONRepairError as e:
        return JudgeResult(ok=False, error=f"judge JSON parse failed: {e}", **base)

    smuggled = any(key in data for key in _SMUGGLE_KEYS)
    # Strip any smuggled answer fields before they can influence anything.
    for key in list(data.keys()):
        if key in _SMUGGLE_KEYS:
            data.pop(key, None)

    try:
        scores = _coerce_scores(data, criteria, smin, smax, valid_indices)
    except JSONRepairError as e:
        return JudgeResult(ok=False, smuggled_answer_discarded=smuggled,
                           error=f"judge scores invalid: {e}", **base)

    return JudgeResult(ok=True, scores=scores,
                       justification=str(data.get("justification", ""))[:500],
                       smuggled_answer_discarded=smuggled, **base)


def run_judges(cfg: Config, client: OpenRouterClient, question: str,
               candidates: list[Candidate]) -> list[JudgeResult]:
    rubric = cfg.section("rubric")
    criteria = rubric["criteria"]
    smin, smax = int(rubric.get("score_min", 0)), int(rubric.get("score_max", 5))
    s = cfg.section("sampling")
    temp = float(s.get("judge_temperature", 0.1))
    max_tokens = int(s.get("judge_max_tokens", s.get("max_tokens", 1024)))
    seed = s.get("seed")

    system = _build_system(criteria, smin, smax)
    user = _build_user(question, candidates)
    valid_indices = {c.index for c in candidates}

    results: list[JudgeResult] = []
    # Sequential and in JSON mode: keeps us under the rate ceiling and parses reliably.
    for spec in cfg.judges:  # type: ModelSpec
        res = client.complete(spec, system, user, temp, max_tokens, seed, json_mode=True)
        results.append(_parse_judge(res, criteria, smin, smax, valid_indices))
    return results
