"""Aggregator: combine rubric scores into a winner and earned confidence, and
apply the tie-break / abstention / escalation policy.

  * Fewer than min_usable_candidates real answers -> no_decision.
  * Judges disagree on the winner and the margin is within tie_margin -> a tie
    -> no_decision (no silent coin flip).
  * Otherwise -> decided; below escalate_below_confidence it is still decided but
    flagged for human review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import confidence as conf
from .config import Config
from .generators import Candidate
from .judges import JudgeResult


@dataclass
class Aggregation:
    status: str                       # "decided" | "no_decision"
    winner: Candidate | None
    runner_up: Candidate | None
    confidence: float
    signals: dict
    confidence_method: str
    risks: list[dict] = field(default_factory=list)
    per_candidate_total: dict[int, float] = field(default_factory=dict)
    per_judge_totals: list[dict[int, float]] = field(default_factory=list)
    escalate: bool = False


def _normalized_total(per_criterion: dict[str, float], criteria: list[dict],
                      smin: int, smax: int) -> float:
    """Weighted sum of criteria, each normalized to [0,1]; result in [0,1]."""
    span = max(1e-9, (smax - smin))
    total = 0.0
    for c in criteria:
        raw = per_criterion.get(c["name"])
        if raw is None:
            continue
        total += c["weight"] * ((raw - smin) / span)
    return total


def aggregate(cfg: Config, candidates: list[Candidate],
              judges: list[JudgeResult], citations: list[dict],
              extra_risks: list[dict]) -> Aggregation:
    rubric = cfg.section("rubric")
    criteria = rubric["criteria"]
    smin, smax = int(rubric.get("score_min", 0)), int(rubric.get("score_max", 5))
    policy = cfg.section("policy")
    cset = cfg.section("confidence")

    usable = [c for c in candidates if c.ok and c.answer.strip()]
    ok_judges = [j for j in judges if j.ok and j.scores]
    risks = list(extra_risks)

    # --- Abstention: not enough usable candidates -------------------------
    if len(usable) < int(policy.get("min_usable_candidates", 2)):
        risks.append({"type": "data_gap", "severity": "high",
                      "detail": f"only {len(usable)} usable candidate answer(s); "
                                f"council cannot responsibly decide."})
        return Aggregation(status="no_decision", winner=None, runner_up=None,
                           confidence=0.0,
                           signals=_zero_signals(), confidence_method=_METHOD,
                           risks=risks, escalate=True)

    # --- No usable judges: we have answers but no scoring -> no_decision ---
    if not ok_judges:
        risks.append({"type": "data_gap", "severity": "high",
                      "detail": "no judge produced valid scores; cannot rank answers."})
        return Aggregation(status="no_decision", winner=None, runner_up=None,
                           confidence=0.0, signals=_zero_signals(),
                           confidence_method=_METHOD, risks=risks, escalate=True)

    # --- Per-judge normalized totals, then average across judges ----------
    per_judge_totals: list[dict[int, float]] = []
    for j in ok_judges:
        totals = {idx: _normalized_total(scores, criteria, smin, smax)
                  for idx, scores in j.scores.items()
                  if any(c.index == idx for c in usable)}
        if totals:
            per_judge_totals.append(totals)

    indices = [c.index for c in usable]
    avg_total: dict[int, float] = {}
    for idx in indices:
        vals = [t[idx] for t in per_judge_totals if idx in t]
        if vals:
            avg_total[idx] = sum(vals) / len(vals)

    if not avg_total:
        risks.append({"type": "data_gap", "severity": "high",
                      "detail": "judges did not score the usable candidates."})
        return Aggregation(status="no_decision", winner=None, runner_up=None,
                           confidence=0.0, signals=_zero_signals(),
                           confidence_method=_METHOD, risks=risks, escalate=True)

    ranked = sorted(avg_total.items(), key=lambda kv: kv[1], reverse=True)
    by_index = {c.index: c for c in usable}
    winner = by_index[ranked[0][0]]
    runner_up = by_index[ranked[1][0]] if len(ranked) > 1 else None
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else ranked[0][1]

    # --- Tie detection: judges disagree on winner AND margin is tiny ------
    judge_winners = {max(t.items(), key=lambda kv: kv[1])[0] for t in per_judge_totals}
    judges_disagree = len(judge_winners) > 1
    tie_margin = float(policy.get("tie_margin", 0.07))

    # --- Signals ----------------------------------------------------------
    signals = {
        "inter_judge_agreement": round(conf.inter_judge_agreement(per_judge_totals), 4),
        "score_margin": round(max(0.0, min(1.0, margin)), 4),
        "agent_agreement": round(conf.agent_agreement([c.answer for c in usable]), 4),
        "verification_pass_rate": round(conf.verification_pass_rate(citations), 4),
    }

    if judges_disagree and margin <= tie_margin:
        risks.append({"type": "ambiguity", "severity": "high",
                      "detail": f"judges split on the winner and the score margin "
                                f"({margin:.3f}) is within the tie threshold "
                                f"({tie_margin}); declining rather than coin-flipping."})
        return Aggregation(status="no_decision", winner=None, runner_up=None,
                           confidence=0.0, signals=signals,
                           confidence_method=_METHOD, risks=risks,
                           per_candidate_total=avg_total,
                           per_judge_totals=per_judge_totals, escalate=True)

    has_high_risk = any(r["severity"] == "high" for r in risks)
    confidence = conf.compute_confidence(
        weights=cset.get("weights", {}), signals=signals,
        has_high_risk=has_high_risk,
        high_risk_penalty=float(cset.get("high_risk_penalty", 0.25)),
    )

    escalate = confidence < float(policy.get("escalate_below_confidence", 0.4))
    if escalate:
        risks.append({"type": "data_gap", "severity": "med",
                      "detail": f"confidence {confidence:.2f} is below the escalation "
                                f"threshold; recommend human review before acting."})

    return Aggregation(status="decided", winner=winner, runner_up=runner_up,
                       confidence=confidence, signals=signals,
                       confidence_method=_METHOD, risks=risks,
                       per_candidate_total=avg_total,
                       per_judge_totals=per_judge_totals, escalate=escalate)


_METHOD = (
    "Weighted sum of four observable signals (inter_judge_agreement, score_margin, "
    "agent_agreement, verification_pass_rate) with config weights; a one-time "
    "penalty is subtracted if any high-severity risk is present; result clamped to "
    "[0,1]. No model self-reported confidence is ever used."
)


def _zero_signals() -> dict:
    return {"inter_judge_agreement": 0.0, "score_margin": 0.0,
            "agent_agreement": 0.0, "verification_pass_rate": 0.0}
