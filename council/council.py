"""Orchestrator: the full pipeline for one question.

  pre-gate -> generators -> judges -> aggregator -> post-gate -> Decision Object
  -> audit log

Every path emits a schema-valid Decision Object and appends one hash-chained
audit record whose hash becomes the decision's audit_ref.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import aggregator, decision as dec, pregate
from .audit import AuditLog
from .client import OpenRouterClient
from .config import Config, load_config
from .generators import Candidate, run_generators
from .judges import run_judges
from .postgate import run_post_gate
from .ratelimit import RateLimiter


@dataclass
class CouncilOutput:
    decision: dict
    schema_errors: list[str]


class Council:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        rl = cfg.section("rate_limit")
        self.limiter = RateLimiter(
            requests_per_minute=int(rl.get("requests_per_minute", 18)),
            requests_per_day=int(rl.get("requests_per_day", 190)),
        )
        self.client = OpenRouterClient(cfg, self.limiter)
        self.audit = AuditLog(cfg.section("audit").get("log_path", "audit/audit_log.jsonl"))

    # -- helpers -----------------------------------------------------------
    def _gen_provenance(self, candidates: list[Candidate]) -> list[dict]:
        return [{
            "model_id": c.model_id or "", "family": c.family,
            "latency_ms": round(c.latency_ms, 1), "tokens": int(c.tokens),
            "ok": bool(c.ok), "error": c.error,
        } for c in candidates]

    def _judge_provenance(self, judges) -> list[dict]:
        out = []
        for j in judges:
            out.append({
                "model_id": j.model_id or "", "family": j.family,
                "rubric_scores": {str(k): v for k, v in j.scores.items()},
                "smuggled_answer_discarded": bool(j.smuggled_answer_discarded),
                "ok": bool(j.ok), "error": j.error,
            })
        return out

    def _finalize(self, decision: dict) -> CouncilOutput:
        # Append first to get the record hash, stamp it as audit_ref, then validate.
        record = self.audit.append(decision)
        decision["audit_ref"] = record["record_hash"]
        errors = dec.validate_decision(decision)
        return CouncilOutput(decision=decision, schema_errors=errors)

    # -- main --------------------------------------------------------------
    def decide(self, question: str) -> CouncilOutput:
        decision_id = dec.new_decision_id()
        created_at = dec.now_iso()
        base_provenance = {
            "generators": [], "judges": [],
            "config_hash": self.cfg.config_hash, "cost_usd": 0.0, "total_latency_ms": 0.0,
        }

        # 1) Pre-gate
        gate = pregate.screen(question)
        if not gate.allow:
            d = dec.build_decision(
                decision_id=decision_id, status="refused", question=question,
                winning_answer=None, runner_up_answer=None,
                confidence_score=0.0,
                confidence_method="pre-gate refusal; no council run",
                signals={}, 
                risks=[{"type": "safety", "severity": "high",
                        "detail": f"pre-gate refused: {gate.reason}"}],
                citations=[], provenance=base_provenance, audit_ref="",
                created_at=created_at,
            )
            return self._finalize(d)

        # 2) Generators
        candidates = run_generators(self.cfg, self.client, question)

        # 3) Judges. Only score usable candidates; if too few, skip judging and
        #    let the aggregator abstain.
        usable = [c for c in candidates if c.ok and c.answer.strip()]
        min_usable = int(self.cfg.section("policy").get("min_usable_candidates", 2))
        judges = (run_judges(self.cfg, self.client, question, usable)
                  if len(usable) >= min_usable else [])

        # provenance + cost/latency tallies
        gen_prov = self._gen_provenance(candidates)
        judge_prov = self._judge_provenance(judges)
        cost = sum(c.cost_usd for c in candidates) + sum(j.cost_usd for j in judges)
        latency = sum(c.latency_ms for c in candidates) + sum(j.latency_ms for j in judges)

        # 4) Post-gate: verify the provisional winner's citations.
        winner_preview = _provisional_winner(self.cfg, candidates, judges)
        raw_citations = winner_preview.citations if winner_preview else []
        post = run_post_gate(self.cfg, winner_preview.answer if winner_preview else "",
                             raw_citations)

        # 5) Aggregate with verified citations feeding verification_pass_rate.
        agg = aggregator.aggregate(self.cfg, candidates, judges,
                                   citations=post.citations, extra_risks=list(post.risks))

        provenance = {
            "generators": gen_prov, "judges": judge_prov,
            "config_hash": self.cfg.config_hash,
            "cost_usd": round(cost, 6), "total_latency_ms": round(latency, 1),
        }

        # 6) A post-gate output-safety refusal overrides a decided winner.
        if agg.status == "decided" and not post.safe:
            d = dec.build_decision(
                decision_id=decision_id, status="refused", question=question,
                winning_answer=None, runner_up_answer=None,
                confidence_score=0.0,
                confidence_method="post-gate output-safety refusal",
                signals=agg.signals,
                risks=agg.risks + [{"type": "safety", "severity": "high",
                                    "detail": post.refusal_reason}],
                citations=post.citations, provenance=provenance, audit_ref="",
                created_at=created_at,
            )
            return self._finalize(d)

        winning = agg.winner.answer if agg.winner else None
        runner = agg.runner_up.answer if agg.runner_up else None
        d = dec.build_decision(
            decision_id=decision_id, status=agg.status, question=question,
            winning_answer=winning, runner_up_answer=runner,
            confidence_score=agg.confidence, confidence_method=agg.confidence_method,
            signals=agg.signals, risks=agg.risks, citations=post.citations,
            provenance=provenance, audit_ref="", created_at=created_at,
        )
        return self._finalize(d)


def _provisional_winner(cfg: Config, candidates, judges) -> Candidate | None:
    """Pick the likely winner using a citation-free aggregation pass, purely to
    decide WHOSE citations to verify. The authoritative ranking happens in the
    real aggregate() call with verified citations."""
    pre = aggregator.aggregate(cfg, candidates, judges, citations=[], extra_risks=[])
    return pre.winner


def run_question(question: str, config_path: str | None = None) -> CouncilOutput:
    cfg = load_config(config_path)
    return Council(cfg).decide(question)
