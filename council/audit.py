"""Append-only, tamper-evident audit log.

Each JSONL line is one record holding index, timestamp, decision_id,
payload_sha256, prev_hash, and:

  record_hash = sha256(index | timestamp | decision_id | payload_sha256 | prev_hash)

Each record commits to the previous record's hash, so editing or deleting any
past entry breaks the chain from that point; verify_chain re-hashes every stored
payload and reports exactly where it broke. The genesis record uses a sentinel
prev_hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

GENESIS_PREV = "0" * 64


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _payload_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(blob)


def _record_hash(index: int, timestamp: str, decision_id: str,
                 payload_sha256: str, prev_hash: str) -> str:
    return _sha256_hex(f"{index}|{timestamp}|{decision_id}|{payload_sha256}|{prev_hash}")


@dataclass
class VerifyReport:
    ok: bool
    length: int
    broken_at: int | None = None
    reason: str = ""


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _last_hash(self) -> tuple[int, str]:
        records = self._read_all()
        if not records:
            return -1, GENESIS_PREV
        last = records[-1]
        return int(last["index"]), str(last["record_hash"])

    def append(self, decision: dict) -> dict:
        """Append a Decision Object; return the chain metadata (incl. record_hash
        to use as the decision's audit_ref)."""
        prev_index, prev_hash = self._last_hash()
        index = prev_index + 1
        timestamp = decision.get("created_at", "")
        decision_id = decision.get("decision_id", "")
        payload_sha = _payload_hash(decision)
        rec_hash = _record_hash(index, timestamp, decision_id, payload_sha, prev_hash)

        record = {
            "index": index,
            "timestamp": timestamp,
            "decision_id": decision_id,
            "payload_sha256": payload_sha,
            "prev_hash": prev_hash,
            "record_hash": rec_hash,
            "payload": decision,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def verify_chain(self) -> VerifyReport:
        records = self._read_all()
        prev_hash = GENESIS_PREV
        for i, rec in enumerate(records):
            if int(rec["index"]) != i:
                return VerifyReport(False, len(records), i, "index out of sequence")
            if rec["prev_hash"] != prev_hash:
                return VerifyReport(False, len(records), i,
                                    "prev_hash does not match previous record")
            if _payload_hash(rec["payload"]) != rec["payload_sha256"]:
                return VerifyReport(False, len(records), i,
                                    "payload was altered (hash mismatch)")
            expected = _record_hash(rec["index"], rec["timestamp"], rec["decision_id"],
                                    rec["payload_sha256"], rec["prev_hash"])
            if expected != rec["record_hash"]:
                return VerifyReport(False, len(records), i, "record_hash mismatch")
            prev_hash = rec["record_hash"]
        return VerifyReport(True, len(records))
