# LLM Council — a system that decides, and shows its work

Several models answer a question independently. A separate set of models judges
those answers against a rubric. The system then emits one machine-readable
**Decision Object** with an **earned** confidence score, explicit risks,
**verified** citations, a **safety gate**, and a **tamper-evident audit log**.

The hard part isn't the API calls. It's not lying: not letting a model's own
sense of confidence become the system's confidence, not letting a made-up
citation pass as real, and not letting "the judges disagreed" quietly turn into a
fake consensus. This file explains where I drew those lines, and where I left a
gap on purpose.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 2. Add your key (free models, no credit card)
copy .env.example .env        # cp on macOS/Linux
#   then edit .env and set OPENROUTER_API_KEY=sk-or-...

# 3a. Run the council on one question
python -m council ask "What is the boiling point of water at sea level?"

# 3b. Run the eval set (5 cases) and write evals/reports/eval_report.json
python -m council eval

# 3c. Verify the audit log hash chain
python -m council audit-verify                                      # the live log
python -m council audit-verify audit/sample/audit_log.sample.jsonl  # the shipped sample
```

No key handy? The committed sample artifacts come from the real pipeline with
only the network calls stubbed:

```bash
python -m evals.make_sample_report   # writes evals/reports/eval_report.sample.json
                                     #  + audit/sample/audit_log.sample.jsonl
python -m tests.test_offline         # 33 offline assertions, no key/network needed
```

---

## What runs, in order

```
question
   │
   ▼
PRE-GATE (rule-based safety/scope) ──refuse──▶ Decision Object (status: refused)
   │ allow
   ▼
Generator A | Generator B | Generator C     3 distinct families, run independently
   │            │            │
   └────────────┼────────────┘
                ▼
        3 candidate answers
                ▼
   Judge 1   |   Judge 2        score each candidate against the rubric; never generate
                ▼
        AGGREGATOR              combine rubric scores → winner + earned confidence
                ▼                tie-break / abstention / escalation policy
   POST-GATE (citation verification + output safety)
                ▼
        DECISION OBJECT ──▶ APPEND-ONLY HASH-CHAINED AUDIT LOG
```

Every path — including refusals and `no_decision` — emits a schema-valid
Decision Object and appends exactly one audit record whose hash becomes the
decision's `audit_ref`.

---

## Models

Pinned in `config.yaml`, each with fallbacks, since the free catalog changes
often and I want runs to fail over gracefully:

| Role | Model (`:free`) | Family |
|------|-----------------|--------|
| Generator A | `meta-llama/llama-3.3-70b-instruct:free` | llama |
| Generator B | `qwen/qwen3-next-80b-a3b-instruct:free` | qwen |
| Generator C | `z-ai/glm-4.5-air:free` | glm |
| Judge 1 | `google/gemma-4-31b-it:free` | gemma |
| Judge 2 | `nvidia/nemotron-3-nano-30b-a3b:free` | nemotron |

Five distinct families, so "agreement" isn't one model's mistake repeated three
times, and neither judge shares a family with a generator.

I originally wanted a DeepSeek generator, but there was no DeepSeek `:free` model
available when I pinned these, so I used GLM to keep three genuinely different
families. The actual `model_id` that answered is recorded in every decision's
`provenance`, so if a fallback kicks in it can never hide behind the pinned name.

One bias I can't fully remove: a judge may slightly prefer prose that looks like
its own. I keep the families distinct and lean on `inter_judge_agreement` rather
than trusting any single judge.

---

## The confidence score (earned, never self-reported)

I never ask a model "how confident are you?" — models are systematically
overconfident. The score comes from four observable signals, each scaled to
`[0, 1]`:

| Signal | What it measures |
|--------|------------------|
| `inter_judge_agreement` | how closely the two judges' candidate totals match |
| `score_margin` | winner's total minus runner-up's total (decisiveness) |
| `agent_agreement` | average word overlap (Jaccard) between generator answers |
| `verification_pass_rate` | fraction of surfaced citations that verified |

```text
confidence = 0.35·inter_judge_agreement
           + 0.30·score_margin
           + 0.15·agent_agreement
           + 0.20·verification_pass_rate
if any high-severity risk:  confidence -= 0.25      # one-time penalty
confidence = clamp(confidence, 0, 1)
```

Weights live in `config.yaml`; the code is `council/confidence.py`. The score and
all four input signals are written into every Decision Object, so the number can
be recomputed from the record.

Two choices I made on purpose:
- `agent_agreement` has the lowest weight. Three different families agreeing is
  only mild evidence, because a shared wrong prior can fake it, so I don't let
  agreement dominate.
- `verification_pass_rate = 1.0` when there are no citations — nothing unverified
  is being shown as fact. This can slightly inflate confidence for answers with
  no citations; it's a known trade-off, not an oversight.

---

## Judging rubric

Judges score every candidate `0–5` per criterion (set in `config.yaml`). The
aggregator takes a weighted, normalized sum.

| Criterion | Weight | Question the judge answers |
|-----------|--------|----------------------------|
| correctness | 0.40 | Are the claims accurate and free of fabrication? |
| relevance | 0.20 | Does it directly answer what was asked? |
| completeness | 0.15 | Are the important parts covered without padding? |
| reasoning | 0.15 | Is the logic sound and the conclusion supported? |
| calibration | 0.10 | Does it express appropriate uncertainty vs. overclaim? |

---

## Tie-break, abstention & escalation

- **Abstention (`no_decision`):** fewer than `min_usable_candidates` (=2) real
  answers, or no judge produced valid scores → the council declines instead of
  inventing a winner.
- **Tie-break (`no_decision`):** if the judges disagree on the winner and the
  normalized margin is within `tie_margin` (=0.07) → a real tie → `no_decision`
  with an `ambiguity` risk. No silent coin-flip.
- **Escalation:** a `decided` result whose confidence is below
  `escalate_below_confidence` (=0.40) is still emitted, but tagged with a
  `data_gap` risk recommending human review. I neither fake certainty nor throw
  away a usable answer.

All thresholds live under `policy:` in `config.yaml`.

---

## The audit log (tamper-evident)

`council/audit.py` writes append-only JSONL where each record commits to the
previous record's hash:

```text
record_hash = sha256(index | timestamp | decision_id | payload_sha256 | prev_hash)
```

Editing or deleting any past entry breaks the chain from that point on.
`audit-verify` re-hashes every stored payload and recomputes the chain, then
reports the exact `broken_at` index if anything changed. The offline test suite
proves this by tampering with a middle record and checking it's detected.

---

## Decision Object

Strict schema in `schema/decision.schema.json` (JSON Schema draft-07), validated
on every emit. A `refused` or `no_decision` result is a fully valid Decision
Object — see real examples in `evals/reports/eval_report.sample.json` and
`audit/sample/audit_log.sample.jsonl`.

---

## Rate-limit engineering (free tier ≈ 20 rpm / 200 rpd)

This shaped the design rather than being an afterthought: 3 generators + 2 judges
= 5 calls per question, so ~200/day is only ~40 council runs/day.
`council/ratelimit.py` is a single process-wide sliding-window limiter that every
call passes through, so the concurrent generators can't blow past the per-minute
ceiling, and the run stops before exhausting the daily budget instead of
collecting a wall of 429s. `council/client.py` adds exponential backoff with
jitter (honoring `Retry-After`) on 429 and 5xx.

---

## Failure modes I designed against

| Failure mode | How it's handled |
|------|--------|
| Confidence theater | Confidence is computed only from observable signals; no self-rating is ever read. |
| Correlated errors | 3 distinct families; `agent_agreement` is weighted low and treated as weak evidence. |
| Citation laundering | Real HTTP fetch + keyword corroboration; marked `verified` / `unverified` / `failed`, never silent fact. |
| Judge contamination | Score-only contract; the aggregator detects and discards smuggled-answer keys and records that it did. |
| Rate-limit cliffs | Shared limiter + daily cap + backoff; also detects provider errors wrapped in an HTTP 200 body (e.g. upstream 429) and backs off honoring `retry_after` instead of mislabeling them as parse errors. |
| Malformed model JSON | Fenced/balanced-span extraction, trailing-comma repair, explicit failure path. |
| The tie | Defined tie-break → `no_decision`, no coin-flip. |
| Everyone abstains | `no_decision` when too few usable candidates; never fabricates one. |
| Non-reproducibility | Config hashed into provenance, seeds forwarded; free-model non-determinism remains (see below). |

Known limits I left in on purpose:
- Citation verification is a plausibility check (URL reachable + claim keywords
  present), not real fact-checking. A page can be reachable and on-topic yet not
  actually support the claim.
- The pre-gate is conservative keyword heuristics; it will both over- and
  under-trigger compared to a proper moderation model.
- `inter_judge_agreement` is a simple total-difference measure, not Cohen's
  kappa; fine for two judges, less clean for many.

---

## One thing I intentionally did not automate

I did not automate the tie-break into a winner, and I did not automate the
"ship this decision" threshold. When the two judges disagree on the winner and
the margin is within noise, the cheap move is to just pick one (coin-flip, "trust
the bigger model", or first-by-index). I don't. The system emits `no_decision`
with an explicit `ambiguity` risk and stops.

Why leave it manual: a 1–1 split at a tiny margin is the system honestly saying
"the evidence doesn't separate these answers." Forcing a winner there would be
exactly the dishonesty I'm trying to avoid — turning a disagreement into a fake
consensus. The right next step is a human (or a stronger, more expensive model
called deliberately) who can bring context the council doesn't have. Likewise, a
`decided` result under the confidence floor is surfaced for review rather than
auto-accepted. The discipline is leaving the gap visible instead of papering over
it just because I could.

---

## What I saw in live runs

I ran this against the live free tier; the behavior is worth recording because
it's the whole point of the system.

- **Happy path works.** "Chemical symbol for gold?" → `decided`, winning answer
  "Au", confidence ≈ 0.92, both judges scoring with high agreement.
- **Eval set, live: 4/5.** The factual case decided; the ambiguous and
  unknowable cases returned `no_decision` via the tie-break; the unsafe case was
  refused at the pre-gate. The one miss was the citable case landing last in a
  long run when the judges were rate-limited — it correctly degraded to
  `no_decision`, and since there's no winner, no citations got surfaced.
- **Citation laundering caught.** On one run the winning model cited a
  confident-looking `nist.gov` URL that doesn't exist. The post-gate fetched it,
  got a 404, marked the citation `failed`, and attached a `factual` risk —
  `verification_pass_rate` dropped to 0 and pulled confidence down. The made-up
  source was never shown as fact.
- **Volatile catalog is real.** Several pinned `:free` IDs weren't reachable and
  failed over to `openrouter/free`; the model actually used is always in
  `provenance`, so a fallback can't masquerade as the pinned model.
- **Forcing JSON on generators backfired.** With `response_format` on, a small
  free model emitted a looping JSON blob (`"</</</…"`). So generators answer in
  plain prose (citations pulled from inline URLs) and only judges use JSON mode.
  Degenerate output is also sanitized defensively.
- **Honest failure under load.** Under sustained 429s on the judges, the council
  did not invent scores or a winner — it emitted `no_decision` with a high
  `data_gap` risk ("no judge produced valid scores"). That's the system refusing
  to lie when it can't judge.
- **Errors hidden inside HTTP 200, and a `seed` some providers reject.**
  OpenRouter often returns a provider failure inside a 200 body as
  `{"error": {...}}` with no `choices` — upstream rate limits, and in my runs a
  502 carrying "JAX does not support per-request seed". These used to be
  mislabeled `unparseable success body: 'choices'` and failed over with no
  backoff, hiding the real cause and burning the fallback chain. The client now
  detects the embedded error and reports the true reason in `provenance`, treats
  an embedded 429/5xx like a real throttle (backs off, honoring
  `retry_after_seconds`), and drops an unsupported `seed` and retries the same
  model. After this fix, a generator that had been dying on the seed error
  started returning real answers live. See `council/client.py`; covered by 6
  regression tests in `tests/test_offline.py`.

## With another day

- Swap lexical `agent_agreement` for a cheap embedding-based semantic agreement
  so paraphrases aren't penalized.
- Add a second, model-based moderation pass behind the rule-based pre-gate.
- Use claim-level entailment for citations (does the source actually support the
  claim?) rather than keyword overlap.
- Cache results per question keyed by `config_hash` to stretch the daily budget.

---

## Repo layout

```
.
├── README.md
├── .env.example              # OPENROUTER_API_KEY=...  (never commit the real one)
├── requirements.txt
├── config.yaml               # pinned model IDs, rubric, confidence weights, policy
├── schema/decision.schema.json
├── council/                  # the council (one small module per responsibility)
│   ├── __main__.py           # CLI: ask / eval / audit-verify
│   ├── config.py  ratelimit.py  client.py  jsonparse.py
│   ├── pregate.py  generators.py  judges.py
│   ├── confidence.py  aggregator.py  citations.py  postgate.py
│   ├── audit.py  decision.py  council.py
├── evals/                    # eval set + harness + saved sample report
│   ├── eval_set.json  run_evals.py  make_sample_report.py
│   └── reports/eval_report.sample.json
├── audit/                    # hash-chained log (sample committed; live log gitignored)
│   └── sample/audit_log.sample.jsonl
└── tests/test_offline.py     # 33 assertions, no key/network required
```

## Reproducibility note

The same input and config always takes the same decision path — gates, parsing,
aggregation, policy, and hashing are all deterministic. Free models themselves
are often non-deterministic even at a fixed seed, so the exact answer text can
vary; I pin model IDs and stamp `config_hash` into every record so a decision can
always be traced to the rules that produced it. One wrinkle: some free providers
reject the `seed` parameter, so the client drops it for those calls — the seed is
best-effort, which I'd rather do openly than let a provider error look like a dead
model.
