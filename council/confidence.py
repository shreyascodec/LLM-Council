"""Earned confidence, never self-reported. A model is never asked how confident
it is; confidence is derived from four observable signals, each in [0, 1]:

  inter_judge_agreement : 1 - mean abs difference of judges' candidate totals.
  score_margin          : winner total - runner-up total.
  agent_agreement       : mean pairwise Jaccard overlap of generator answers
                          (weak evidence only, hence low weight).
  verification_pass_rate: fraction of surfaced citations that verified.

    confidence = sum(weight_i * signal_i), minus a one-time high-risk penalty,
    clamped to [0, 1].

All four signals and the score are written into the Decision Object.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def agent_agreement(answers: list[str]) -> float:
    """Mean pairwise Jaccard token overlap across generator answers."""
    toks = [_tokens(a) for a in answers if a]
    if len(toks) < 2:
        return 0.0
    pairs = [(i, j) for i in range(len(toks)) for j in range(i + 1, len(toks))]
    return sum(_jaccard(toks[i], toks[j]) for i, j in pairs) / len(pairs)


def inter_judge_agreement(per_judge_totals: list[dict[int, float]]) -> float:
    """1 - mean absolute difference of normalized candidate totals between judges.
    Returns 0.0 if fewer than two judges produced usable scores."""
    judges = [t for t in per_judge_totals if t]
    if len(judges) < 2:
        return 0.0
    common = set(judges[0])
    for t in judges[1:]:
        common &= set(t)
    if not common:
        return 0.0
    pair_idx = [(i, j) for i in range(len(judges)) for j in range(i + 1, len(judges))]
    diffs: list[float] = []
    for i, j in pair_idx:
        for c in common:
            diffs.append(abs(judges[i][c] - judges[j][c]))
    if not diffs:
        return 0.0
    return max(0.0, 1.0 - (sum(diffs) / len(diffs)))


def verification_pass_rate(citations: list[dict]) -> float:
    """verified / total. No citations -> 1.0 (nothing unverified is presented
    as fact)."""
    if not citations:
        return 1.0
    verified = sum(1 for c in citations if c.get("status") == "verified")
    return verified / len(citations)


def compute_confidence(weights: dict, signals: dict, has_high_risk: bool,
                       high_risk_penalty: float) -> float:
    score = (
        weights.get("inter_judge_agreement", 0) * signals["inter_judge_agreement"]
        + weights.get("score_margin", 0) * signals["score_margin"]
        + weights.get("agent_agreement", 0) * signals["agent_agreement"]
        + weights.get("verification_pass_rate", 0) * signals["verification_pass_rate"]
    )
    if has_high_risk:
        score -= high_risk_penalty
    return max(0.0, min(1.0, score))
