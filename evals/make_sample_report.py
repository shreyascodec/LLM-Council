"""Generate committed sample artifacts WITHOUT a live API key or network.

This runs the *real* pipeline (pre-gate, generators->parse, judges->parse,
aggregator, confidence, post-gate, schema validation, hash-chained audit) but
swaps the OpenRouter HTTP call and the citation HTTP fetch for deterministic
stubs. It exists so the repo can ship genuine, reproducible artifacts:

  * evals/reports/eval_report.sample.json   (a saved eval report)
  * audit/sample/audit_log.sample.jsonl     (a sample hash chain you can verify)

A real run is identical except the stubs are real network calls:
  python -m council eval
  python -m council audit-verify

Regenerate the samples with:  python -m evals.make_sample_report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from council import postgate  # noqa: E402
from council.client import LLMResult  # noqa: E402
from council.config import load_config  # noqa: E402
from council.council import Council  # noqa: E402

# --- Canned, plausible model outputs keyed by (question keyword, role/family) ---
# Generators answer; judges score by candidate index. Designed to showcase:
#   factual  -> confident decided
#   ambiguous-> judges split -> tie -> no_decision (ambiguity)
#   unknowable-> converging "cannot predict" -> decided but modest confidence
#   citable  -> answer + verified citation

_GEN = {
    "boiling point": {
        "llama": {"answer": "At sea-level pressure (1 atm), pure water boils at 100 degrees Celsius (212 F).", "citations": []},
        "qwen": {"answer": "Water boils at 100 C at standard atmospheric pressure.", "citations": []},
        "glm": {"answer": "100 degrees Celsius at 1 atmosphere of pressure.", "citations": []},
    },
    "rent": {
        "llama": {"answer": "Buying is usually better long-term: you build equity and hedge rising rents, provided you stay 5+ years.", "citations": []},
        "qwen": {"answer": "Renting is often better: more flexibility, no maintenance cost, and you can invest the difference.", "citations": []},
        "glm": {"answer": "It depends entirely on your time horizon, local price-to-rent ratio, and job stability; there is no universal answer.", "citations": []},
    },
    "aapl": {
        "llama": {"answer": "This cannot be known. Future stock prices are not predictable; any specific figure would be fabricated.", "citations": []},
        "qwen": {"answer": "It is impossible to state an exact closing price for a future date; markets are not deterministic.", "citations": []},
        "glm": {"answer": "No one can know a precise future closing price. I will not invent a number.", "citations": []},
    },
    "apollo 11": {
        "llama": {"answer": "Apollo 11 landed humans on the Moon in 1969 (July 20, 1969). Source: https://en.wikipedia.org/wiki/Apollo_11"},
        "qwen": {"answer": "The year was 1969. See https://en.wikipedia.org/wiki/Apollo_11 for details."},
        "glm": {"answer": "1969 - the first crewed Moon landing, per https://en.wikipedia.org/wiki/Apollo_11"},
    },
}

# Judge scores per case: {family: {candidate_index: {criterion: score}}}
_FULL = {"correctness": 5, "relevance": 5, "completeness": 5, "reasoning": 5, "calibration": 5}
_GOOD = {"correctness": 4, "relevance": 4, "completeness": 4, "reasoning": 4, "calibration": 4}
_MID = {"correctness": 3, "relevance": 3, "completeness": 3, "reasoning": 3, "calibration": 3}
_LOW = {"correctness": 2, "relevance": 2, "completeness": 2, "reasoning": 2, "calibration": 2}

_JUDGE = {
    "boiling point": {
        "gemma": {0: _FULL, 1: _GOOD, 2: _GOOD},
        "nemotron": {0: _FULL, 1: _GOOD, 2: _GOOD},
    },
    "rent": {  # judges split -> tie -> no_decision
        "gemma": {0: _FULL, 1: _GOOD, 2: _MID},
        "nemotron": {0: _GOOD, 1: _FULL, 2: _MID},
    },
    "aapl": {  # converge on "cannot know"; honest but unremarkable -> modest conf
        "gemma": {0: _MID, 1: _MID, 2: _MID},
        "nemotron": {0: _GOOD, 1: _MID, 2: _MID},
    },
    "apollo 11": {
        "gemma": {0: _FULL, 1: _GOOD, 2: _GOOD},
        "nemotron": {0: _FULL, 1: _GOOD, 2: _GOOD},
    },
}


def _match_case(text: str) -> str | None:
    t = text.lower()
    for key in ("boiling point", "rent", "aapl", "apollo 11"):
        if key in t:
            return key
    if "closing share price" in t or "apple" in t:
        return "aapl"
    return None


class StubClient:
    """Drop-in replacement for OpenRouterClient.complete with canned outputs."""

    def __init__(self, cfg):
        self.cfg = cfg

    def complete(self, spec, system, user, temperature, max_tokens, seed,
                 json_mode=None) -> LLMResult:
        case = _match_case(user) or _match_case(system) or "boiling point"
        is_judge = system.strip().startswith("You are a JUDGE")
        if is_judge:
            scores = _JUDGE[case].get(spec.family, _JUDGE[case]["gemma"])
            payload = {"scores": {str(k): v for k, v in scores.items()},
                       "justification": f"Scored candidates against the rubric ({spec.family} judge)."}
        else:
            gen = _GEN[case].get(spec.family)
            if gen is None:
                gen = list(_GEN[case].values())[0]
            payload = gen
        return LLMResult(ok=True, text=json.dumps(payload), model_id=spec.id,
                         family=spec.family, latency_ms=420.0, tokens=180, cost_usd=0.0)


def _stub_verify_all(citations, timeout, max_urls):
    """Deterministic, offline citation verification for the sample."""
    out = []
    for c in citations:
        src = (c.get("source") or "")
        status = "verified" if "wikipedia.org/wiki/Apollo_11" in src else "unverified"
        out.append({"claim": c.get("claim", ""), "source": src, "status": status,
                    "detail": "stubbed offline verification for sample artifact"})
    return out


def main() -> int:
    cfg = load_config(require_key=False)

    # Patch the network boundaries only.
    postgate.verify_all = _stub_verify_all  # type: ignore[assignment]

    council = Council(cfg)
    council.client = StubClient(cfg)  # type: ignore[assignment]
    sample_log = ROOT / "audit" / "sample" / "audit_log.sample.jsonl"
    sample_log.parent.mkdir(parents=True, exist_ok=True)
    if sample_log.exists():
        sample_log.unlink()
    from council.audit import AuditLog
    council.audit = AuditLog(sample_log)

    cases = json.loads((ROOT / "evals" / "eval_set.json").read_text(encoding="utf-8"))["cases"]
    results, passed = [], 0
    from evals.run_evals import _check
    for case in cases:
        out = council.decide(case["question"])
        d = out.decision
        ok, notes = _check(case.get("expect", {}), d)
        passed += int(ok and not out.schema_errors)
        results.append({
            "id": case["id"], "kind": case["kind"], "question": case["question"],
            "pass": ok and not out.schema_errors, "notes": notes,
            "schema_errors": out.schema_errors, "status": d["status"],
            "confidence": d["confidence"]["score"], "signals": d["confidence"]["signals"],
            "risks": d["risks"], "citations": d["citations"],
            "decision_id": d["decision_id"], "audit_ref": d["audit_ref"],
        })
        print(f"[{'PASS' if results[-1]['pass'] else 'FAIL'}] {case['id']:<18} "
              f"status={d['status']:<11} conf={d['confidence']['score']:.2f} {' '.join(notes)}")

    report = {
        "_note": "SAMPLE generated offline by evals/make_sample_report.py (model + citation "
                 "HTTP calls stubbed deterministically). Regenerate a live report with "
                 "`python -m council eval` and a real OPENROUTER_API_KEY.",
        "summary": {"total": len(cases), "passed": passed, "failed": len(cases) - passed},
        "config_hash": cfg.config_hash, "results": results,
    }
    report_path = ROOT / "evals" / "reports" / "eval_report.sample.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    chain = council.audit.verify_chain()
    print(f"\n{passed}/{len(cases)} passed. Sample report -> {report_path}")
    print(f"Sample audit chain: ok={chain.ok} length={chain.length} -> {sample_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
