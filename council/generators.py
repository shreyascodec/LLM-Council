"""Generators: three models from distinct families answer independently.

Each generator call is built only from the question and a fixed system prompt;
no code path feeds one generator's output into another. They run concurrently,
but the shared rate limiter still bounds the actual request rate.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .client import LLMResult, OpenRouterClient
from .config import Config, ModelSpec
from .jsonparse import JSONRepairError, extract_json

# Generators answer in plain prose (forcing JSON on small free models tends to
# loop/garble); citations are extracted from any URLs they include.
_SYSTEM = (
    "You are one independent member of an answer panel. Answer the user's question "
    "directly, concisely, and honestly in plain prose. If you are not sure, say so "
    "plainly rather than inventing specifics. If a factual claim has a real citable "
    "source, include the full source URL inline (e.g. https://...). Never invent a URL."
)

_URL_RE = re.compile(r"https?://[^\s\)\]\}<>\"']+")
# Markdown link: [claim text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]{1,160})\]\((https?://[^\s\)]+)\)")


def _sanitize(text: str) -> str:
    """Trim, collapse runaway whitespace, and cut repeated-character loops."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"[ \t]{4,}", " ", text)
    text = re.sub(r"(\r?\n\s*){4,}", "\n\n", text)
    text = re.sub(r"(.{1,4}?)\1{5,}", r"\1\1\1", text, flags=re.DOTALL)
    return text.strip()


def _extract_citations(text: str) -> list[dict]:
    """Pull URLs from the prose, attaching nearby text as the claim to verify."""
    cites: list[dict] = []
    seen: set[str] = set()

    # Prefer markdown links: the link text is a cleaner claim than the window.
    for m in _MD_LINK_RE.finditer(text):
        url = m.group(2).rstrip(".,;)")
        if url in seen:
            continue
        seen.add(url)
        cites.append({"claim": m.group(1).strip(), "source": url})

    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;)")
        if url in seen:
            continue
        seen.add(url)
        start = max(0, m.start() - 160)
        window = text[start:m.start()].strip().lstrip("([").rstrip("](")
        claim = window.split(". ")[-1][-160:].strip() or "claim associated with cited URL"
        cites.append({"claim": claim, "source": url})
    return cites


@dataclass
class Candidate:
    index: int
    model_id: str
    family: str
    answer: str
    citations: list[dict]          # [{"claim","source"}]
    ok: bool
    latency_ms: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    raw: str = ""


def _parse_candidate(res: LLMResult, index: int) -> Candidate:
    base = dict(index=index, model_id=res.model_id, family=res.family,
                latency_ms=res.latency_ms, tokens=res.tokens, cost_usd=res.cost_usd,
                raw=res.text)
    if not res.ok:
        return Candidate(answer="", citations=[], ok=False, error=res.error, **base)

    text = res.text or ""
    # If a model volunteers a JSON object with an "answer" field, prefer it.
    answer = text
    try:
        data = extract_json(text)
        if isinstance(data, dict) and isinstance(data.get("answer"), str) and data["answer"].strip():
            answer = data["answer"]
    except JSONRepairError:
        pass

    answer = _sanitize(answer)
    if not answer:
        return Candidate(answer="", citations=[], ok=False,
                         error="empty answer after parsing", **base)
    citations = _extract_citations(answer)
    return Candidate(answer=answer, citations=citations, ok=True, **base)


def run_generators(cfg: Config, client: OpenRouterClient, question: str) -> list[Candidate]:
    specs: list[ModelSpec] = cfg.generators
    s = cfg.section("sampling")
    temp = float(s.get("generator_temperature", 0.7))
    max_tokens = int(s.get("max_tokens", 1024))
    seed = s.get("seed")

    def _one(i_spec: tuple[int, ModelSpec]) -> Candidate:
        i, spec = i_spec
        res = client.complete(spec, _SYSTEM, question, temp, max_tokens, seed,
                              json_mode=False)
        return _parse_candidate(res, i)

    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        candidates = list(ex.map(_one, list(enumerate(specs))))
    return sorted(candidates, key=lambda c: c.index)
