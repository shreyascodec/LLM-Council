"""Citation verification. A model-produced citation is treated as a claim about a
source, not a fact. For each one we require a real http(s) URL, fetch it (HEAD
then GET) within a short timeout, and check that salient claim keywords appear at
the source:

  verified   : URL reachable and claim keywords plausibly present.
  unverified : reachable but uncorroborated, non-URL, or skipped.
  failed     : URL unreachable / errored.

This is a plausibility check, not semantic proof.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import requests

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]{4,}")
_STOP = {"this", "that", "with", "from", "have", "https", "http", "www", "com",
         "the", "and", "for", "are", "was", "were", "their", "there", "which"}


def _claim_keywords(claim: str, limit: int = 8) -> list[str]:
    words = [w for w in _WORD_RE.findall((claim or "").lower()) if w not in _STOP]
    # keep order, de-dupe, longest-ish first by simple heuristic
    seen, out = set(), []
    for w in sorted(words, key=len, reverse=True):
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= limit:
            break
    return out


def _fetch_text(url: str, timeout: float) -> tuple[bool, str]:
    headers = {"User-Agent": "LLM-Council-CitationVerifier/1.0"}
    try:
        # Cheap reachability probe first.
        h = requests.head(url, timeout=timeout, allow_redirects=True, headers=headers)
        if h.status_code >= 400 or h.request.method is None:
            # Some servers reject HEAD; fall through to GET.
            pass
    except requests.RequestException:
        pass
    try:
        g = requests.get(url, timeout=timeout, allow_redirects=True, headers=headers)
    except requests.RequestException as e:
        return False, f"unreachable: {e}"
    if g.status_code >= 400:
        return False, f"http {g.status_code}"
    return True, g.text[:200_000]  # cap body we scan


def verify_citation(citation: dict, timeout: float) -> dict:
    claim = (citation.get("claim") or "").strip()
    source = (citation.get("source") or "").strip()
    out = {"claim": claim, "source": source, "status": "unverified", "detail": ""}

    if not source or not _URL_RE.match(source) or not urlparse(source).netloc:
        out["detail"] = "no resolvable http(s) URL provided"
        return out

    reachable, body_or_err = _fetch_text(source, timeout)
    if not reachable:
        out["status"] = "failed"
        out["detail"] = body_or_err
        return out

    keywords = _claim_keywords(claim)
    if not keywords:
        out["detail"] = "URL reachable; no claim text to corroborate"
        return out  # unverified (reachable but uncorroborated)

    body_lc = body_or_err.lower()
    hits = sum(1 for k in keywords if k in body_lc)
    ratio = hits / len(keywords)
    if ratio >= 0.4:
        out["status"] = "verified"
        out["detail"] = f"URL reachable; {hits}/{len(keywords)} claim keywords present"
    else:
        out["detail"] = (f"URL reachable but only {hits}/{len(keywords)} claim "
                         f"keywords found; not corroborated")
    return out


def verify_all(citations: list[dict], timeout: float, max_urls: int) -> list[dict]:
    results: list[dict] = []
    for i, c in enumerate(citations):
        if i >= max_urls:
            results.append({"claim": (c.get("claim") or "").strip(),
                            "source": (c.get("source") or "").strip(),
                            "status": "unverified",
                            "detail": "skipped: exceeded max citations to verify"})
            continue
        results.append(verify_citation(c, timeout))
    return results
