"""Post-gate: run after a winner is chosen, before the decision is emitted.

Two jobs: verify the winning answer's citations (verified / unverified / failed,
plus a factual risk if any failed), and a light output-safety pass that downgrades
to refused if an allowed question produced clearly operational harmful content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .citations import verify_all
from .config import Config

_OUTPUT_UNSAFE = re.compile(
    r"\b(step[-\s]?\d|first,|materials needed|ingredients?:)\b.{0,80}"
    r"\b(bomb|explosive|sarin|ricin|nerve\s*agent|methamphetamine|fentanyl)\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class PostGateResult:
    safe: bool
    refusal_reason: str = ""
    citations: list[dict] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)


def run_post_gate(cfg: Config, winning_answer: str, raw_citations: list[dict]) -> PostGateResult:
    cset = cfg.section("citations")
    citations = verify_all(
        raw_citations,
        timeout=float(cset.get("verify_timeout_s", 8)),
        max_urls=int(cset.get("max_urls_per_answer", 6)),
    )

    risks: list[dict] = []
    failed = [c for c in citations if c["status"] == "failed"]
    unverified = [c for c in citations if c["status"] == "unverified"]
    if failed:
        risks.append({"type": "factual", "severity": "med",
                      "detail": f"{len(failed)} citation(s) could not be reached and "
                                f"are marked failed."})
    if unverified:
        risks.append({"type": "factual", "severity": "low",
                      "detail": f"{len(unverified)} citation(s) reachable but not "
                                f"corroborated; treated as unverified, not fact."})

    if _OUTPUT_UNSAFE.search(winning_answer or ""):
        return PostGateResult(safe=False,
                              refusal_reason="winning answer contained operational "
                                             "harmful content; refused at post-gate.",
                              citations=citations, risks=risks)

    return PostGateResult(safe=True, citations=citations, risks=risks)
