"""Build and validate the Decision Object against schema/decision.schema.json.

A refused / no_decision result is a fully valid Decision Object.
"""
from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from . import SCHEMA_VERSION
from .config import REPO_ROOT

_SCHEMA_PATH = REPO_ROOT / "schema" / "decision.schema.json"
_VALIDATOR: Draft7Validator | None = None


def _validator() -> Draft7Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        _VALIDATOR = Draft7Validator(schema)
    return _VALIDATOR


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_decision_id() -> str:
    return str(uuid.uuid4())


def build_decision(
    *,
    decision_id: str,
    status: str,
    question: str,
    winning_answer: str | None,
    runner_up_answer: str | None,
    confidence_score: float,
    confidence_method: str,
    signals: dict,
    risks: list[dict],
    citations: list[dict],
    provenance: dict,
    audit_ref: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "decision_id": decision_id,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at or now_iso(),
        "status": status,
        "question": question,
        "winning_answer": winning_answer,
        "runner_up_answer": runner_up_answer,
        "confidence": {
            "score": round(float(confidence_score), 4),
            "method": confidence_method,
            "signals": {
                "inter_judge_agreement": float(signals.get("inter_judge_agreement", 0.0)),
                "score_margin": float(signals.get("score_margin", 0.0)),
                "agent_agreement": float(signals.get("agent_agreement", 0.0)),
                "verification_pass_rate": float(signals.get("verification_pass_rate", 0.0)),
            },
        },
        "risks": risks,
        "citations": citations,
        "provenance": provenance,
        "audit_ref": audit_ref,
    }


def validate_decision(decision: dict) -> list[str]:
    """Return a list of human-readable schema errors ([] means valid)."""
    errors = sorted(_validator().iter_errors(decision), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]
