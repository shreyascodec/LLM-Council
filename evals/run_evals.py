"""Eval harness: run every case in eval_set.json through the council, assert the
expected SYSTEM behavior, and write a saved report to evals/reports/.

Run with:  python -m council eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from council.config import REPO_ROOT, load_config
from council.council import Council

EVAL_DIR = REPO_ROOT / "evals"
EVAL_SET = EVAL_DIR / "eval_set.json"
REPORT_DIR = EVAL_DIR / "reports"


def _check(expect: dict, decision: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True

    status_in = expect.get("status_in")
    if status_in and decision["status"] not in status_in:
        ok = False
        notes.append(f"status={decision['status']} not in {status_in}")

    want_risks = expect.get("any_risk_type_in")
    if want_risks:
        have = {r["type"] for r in decision.get("risks", [])}
        if not (have & set(want_risks)):
            ok = False
            notes.append(f"no risk in {want_risks} (have {sorted(have)})")

    if expect.get("exercises_citations"):
        if not decision.get("citations"):
            ok = False
            notes.append("expected at least one citation, found none")

    max_conf = expect.get("max_confidence")
    if max_conf is not None and decision["confidence"]["score"] > max_conf:
        ok = False
        notes.append(f"confidence {decision['confidence']['score']:.2f} exceeds "
                     f"max_confidence {max_conf} (overconfident)")

    return ok, notes


def run_eval_set(config_path: str | None = None) -> int:
    cfg = load_config(config_path)
    council = Council(cfg)
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))["cases"]

    results = []
    passed = 0
    for case in cases:
        out = council.decide(case["question"])
        d = out.decision
        ok, notes = _check(case.get("expect", {}), d)
        if ok and not out.schema_errors:
            passed += 1
        results.append({
            "id": case["id"],
            "kind": case["kind"],
            "question": case["question"],
            "pass": ok and not out.schema_errors,
            "notes": notes,
            "schema_errors": out.schema_errors,
            "status": d["status"],
            "confidence": d["confidence"]["score"],
            "signals": d["confidence"]["signals"],
            "risks": d["risks"],
            "citations": d["citations"],
            "decision_id": d["decision_id"],
            "audit_ref": d["audit_ref"],
        })
        flag = "PASS" if (ok and not out.schema_errors) else "FAIL"
        print(f"[{flag}] {case['id']:<18} status={d['status']:<11} "
              f"conf={d['confidence']['score']:.2f} {' '.join(notes)}")

    report = {
        "summary": {"total": len(cases), "passed": passed, "failed": len(cases) - passed},
        "config_hash": cfg.config_hash,
        "results": results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{passed}/{len(cases)} passed. Report written to {report_path}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(run_eval_set())
