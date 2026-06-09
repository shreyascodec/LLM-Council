"""Command-line entry point.

  python -m council ask "your question here"     # run the council on one question
  python -m council eval                          # run the eval set + write a report
  python -m council audit-verify                  # verify the hash chain integrity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import AuditLog
from .config import REPO_ROOT, load_config
from .council import Council


def _print_decision(decision: dict, schema_errors: list[str]) -> None:
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if schema_errors:
        print("\nSCHEMA VALIDATION ERRORS:", file=sys.stderr)
        for e in schema_errors:
            print(f"  - {e}", file=sys.stderr)
    else:
        d = decision
        print(f"\n[{d['status'].upper()}] confidence={d['confidence']['score']:.2f} "
              f"audit_ref={d['audit_ref'][:12]}…", file=sys.stderr)


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = Council(cfg).decide(args.question)
    _print_decision(out.decision, out.schema_errors)
    return 0 if not out.schema_errors else 2


def cmd_eval(args: argparse.Namespace) -> int:
    # Imported lazily so `ask` doesn't pull eval-only code.
    from evals.run_evals import run_eval_set
    return run_eval_set(config_path=args.config)


def cmd_audit_verify(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, require_key=False)
    path = args.path or cfg.section("audit").get("log_path", "audit/audit_log.jsonl")
    log = AuditLog(path)
    report = log.verify_chain()
    if report.ok:
        print(f"OK: audit chain intact ({report.length} record(s)).")
        return 0
    print(f"BROKEN at index {report.broken_at}: {report.reason} "
          f"(of {report.length} record(s)).", file=sys.stderr)
    return 1


def _force_utf8_stdio() -> None:
    # Model output routinely contains Unicode (e.g. "100 °C"); the default
    # Windows console codepage (cp1252) would crash on it. Reconfigure to UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(prog="council", description="LLM Council")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="run the council on one question")
    p_ask.add_argument("question", help="the question to decide")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="run the eval set and write a report")
    p_eval.set_defaults(func=cmd_eval)

    p_av = sub.add_parser("audit-verify", help="verify the audit hash chain")
    p_av.add_argument("path", nargs="?", default=None,
                      help="path to a JSONL audit log (defaults to config's log_path)")
    p_av.set_defaults(func=cmd_audit_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    # Ensure repo root is importable (for `evals` package) when run as -m council.
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
