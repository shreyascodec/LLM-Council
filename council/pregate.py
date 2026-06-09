"""Pre-gate: decide whether the council should engage at all.

A conservative, rule-based gate rather than an LLM call, so refusals are
deterministic and auditable and cost no requests. It refuses operational
"how to cause harm" requests (serious physical harm, weapons, illegal drug or
malware synthesis), not discussion about sensitive topics. Keyword heuristics
both over- and under-trigger; a calibrated moderation model would go here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern targets *operational* harm intent: a creation/instruction cue near
# a harmful target. We match the cue and target in EITHER order (e.g. "build a
# bomb" and "instructions for a bomb") so phrasing doesn't matter. Word
# boundaries reduce the "Scunthorpe" class of false positives.
_MAKE = r"(?:make|build|construct|assemble|synthesize|manufacture|create|produce|cook|detonate|deploy|instructions?|step[-\s]?by[-\s]?step|how\s+to|guide|recipe|tutorial)"
_WEAPON = r"(?:bomb|explosive|ied|pipe\s*bomb|dirty\s*bomb|nerve\s*agent|bio\s*weapon|bioweapon|chemical\s*weapon|sarin|ricin|vx\s+gas|napalm|grenade)"
_DRUG = r"(?:methamphetamine|crystal\s*meth|\bmeth\b|fentanyl|heroin|cocaine)"

_UNSAFE_PATTERNS: list[tuple[str, str]] = [
    (rf"\b{_MAKE}\b.{{0,40}}\b{_WEAPON}\b", "instructions for building a weapon / WMD"),
    (rf"\b{_WEAPON}\b.{{0,40}}\b{_MAKE}\b", "instructions for building a weapon / WMD"),
    (rf"\b(?:synthesize|manufacture|make|cook|produce)\b.{{0,30}}\b{_DRUG}\b",
     "instructions for manufacturing illegal drugs"),
    (r"\bhow\s+(?:to|do\s+i|can\s+i|could\s+i)\b.{0,40}\b(kill|murder|poison|assassinate)\b.{0,25}\b(someone|a\s+person|people|him|her|them|my\b)",
     "instructions to harm a person"),
    (r"\b(step[-\s]?by[-\s]?step|detailed|best\s+way|how\s+to)\b.{0,30}\b(commit\s+suicide|kill\s+myself|end\s+my\s+life)\b",
     "self-harm instructions"),
    (r"\b(write|create|build|develop|code)\b.{0,30}\b(ransomware|malware|keylogger|computer\s+virus|trojan)\b",
     "creation of malware for attack"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE | re.DOTALL), reason) for p, reason in _UNSAFE_PATTERNS]


@dataclass
class GateResult:
    allow: bool
    reason: str = ""          # human-readable refusal reason (empty if allowed)
    matched_rule: str = ""    # which pattern fired (for the audit log)


def screen(question: str) -> GateResult:
    q = (question or "").strip()
    if not q:
        return GateResult(allow=False, reason="empty question", matched_rule="empty_input")
    for rx, reason in _COMPILED:
        if rx.search(q):
            return GateResult(allow=False, reason=reason, matched_rule=rx.pattern)
    return GateResult(allow=True)
