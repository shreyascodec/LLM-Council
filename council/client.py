"""OpenRouter chat-completions client: shared rate limiting, retry/backoff on
429 and 5xx, fallback across model ids, and latency/token reporting."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import requests

from .config import Config, ModelSpec
from .ratelimit import RateLimiter


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    model_id: str = ""
    family: str = ""
    latency_ms: float = 0.0
    tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class OpenRouterClient:
    def __init__(self, cfg: Config, limiter: RateLimiter):
        self.cfg = cfg
        self.limiter = limiter
        self.session = requests.Session()
        rl = cfg.section("rate_limit")
        self.max_retries = int(rl.get("max_retries", 4))
        self.backoff_base = float(rl.get("backoff_base_s", 2.0))
        self.backoff_max = float(rl.get("backoff_max_s", 45.0))
        self.json_mode = bool(cfg.section("provider").get("json_mode", False))

    # -- public ------------------------------------------------------------
    def complete(self, spec: ModelSpec, system: str, user: str,
                 temperature: float, max_tokens: int, seed: int | None,
                 json_mode: bool | None = None) -> LLMResult:
        """Try the model and its fallbacks in order until one succeeds."""
        use_json = self.json_mode if json_mode is None else json_mode
        last_err = "no candidates"
        for model_id in spec.candidates():
            res = self._complete_one(model_id, spec.family, system, user,
                                     temperature, max_tokens, seed, use_json)
            if res.ok:
                return res
            last_err = res.error or "unknown error"
        return LLMResult(ok=False, family=spec.family,
                         error=f"all candidates failed for family '{spec.family}': {last_err}")

    # -- internals ---------------------------------------------------------
    def _complete_one(self, model_id: str, family: str, system: str, user: str,
                      temperature: float, max_tokens: int, seed: int | None,
                      use_json: bool) -> LLMResult:
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.cfg.app_url,
            "X-Title": self.cfg.app_title,
        }
        payload: dict = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        # Drop response_format once and retry if a model rejects it (400).
        json_dropped = False
        if use_json:
            payload["response_format"] = {"type": "json_object"}
        # Some providers reject `seed`; drop it once and retry if so.
        seed_dropped = False

        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            t0 = time.monotonic()
            try:
                resp = self.session.post(url, headers=headers, json=payload,
                                         timeout=self.cfg.request_timeout_s)
            except requests.RequestException as e:
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt)
                    continue
                return LLMResult(ok=False, model_id=model_id, family=family,
                                 error=f"network error: {e}")

            latency_ms = (time.monotonic() - t0) * 1000.0

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as e:
                    return LLMResult(ok=False, model_id=model_id, family=family,
                                     latency_ms=latency_ms,
                                     error=f"unparseable success body: {e}")
                # OpenRouter can return a provider error inside a 200 body as
                # {"error": {...}} with no "choices". Treat throttles/5xx like a
                # real 429 (back off and retry) rather than as a parse failure.
                emb = _embedded_error(data)
                if emb is not None:
                    code, msg, retry_after = emb
                    if ("seed" in payload and not seed_dropped
                            and "seed" in msg.lower()):
                        payload.pop("seed", None)
                        seed_dropped = True
                        continue
                    if code == 429 or code >= 500:
                        if attempt < self.max_retries:
                            self._sleep_backoff(attempt, retry_after=retry_after)
                            continue
                        return LLMResult(ok=False, model_id=model_id, family=family,
                                         latency_ms=latency_ms,
                                         error=f"provider rate/5xx in 200 body "
                                               f"after retries ({code}): {msg}")
                    return LLMResult(ok=False, model_id=model_id, family=family,
                                     latency_ms=latency_ms,
                                     error=f"provider error in 200 body ({code}): {msg}")
                return self._parse_ok(data, model_id, family, latency_ms)

            # Same seed rejection, but as a real (non-200) error.
            if ("seed" in payload and not seed_dropped
                    and "seed" in (resp.text or "").lower()):
                payload.pop("seed", None)
                seed_dropped = True
                continue

            if resp.status_code == 400 and "response_format" in payload and not json_dropped:
                body = (resp.text or "").lower()
                if "response_format" in body or "json" in body or "not supported" in body:
                    payload.pop("response_format", None)
                    json_dropped = True
                    continue

            if resp.status_code in (400, 404):
                return LLMResult(ok=False, model_id=model_id, family=family,
                                 latency_ms=latency_ms,
                                 error=f"model unavailable ({resp.status_code}): "
                                       f"{_short(resp.text)}")

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep_backoff(attempt, resp)
                    continue
                return LLMResult(ok=False, model_id=model_id, family=family,
                                 latency_ms=latency_ms,
                                 error=f"rate/5xx after retries ({resp.status_code})")

            return LLMResult(ok=False, model_id=model_id, family=family,
                             latency_ms=latency_ms,
                             error=f"http {resp.status_code}: {_short(resp.text)}")

        return LLMResult(ok=False, model_id=model_id, family=family,
                         error="exhausted retries")

    def _parse_ok(self, data: dict, model_id: str, family: str,
                  latency_ms: float) -> LLMResult:
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            return LLMResult(ok=False, model_id=model_id, family=family,
                             latency_ms=latency_ms,
                             error=f"unparseable success body: {e}")
        usage = data.get("usage", {}) or {}
        tokens = int(usage.get("total_tokens", 0) or 0)
        cost = float(usage.get("cost", 0.0) or 0.0)
        return LLMResult(ok=True, text=text, model_id=model_id, family=family,
                         latency_ms=latency_ms, tokens=tokens, cost_usd=cost)

    def _sleep_backoff(self, attempt: int, resp: requests.Response | None = None,
                       retry_after: float | None = None) -> None:
        # Honor retry_after / Retry-After if given, else exponential backoff with
        # jitter, capped at backoff_max.
        delay = retry_after
        if delay is None and resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    delay = float(ra)
                except ValueError:
                    delay = None
        if delay is None:
            delay = self.backoff_base * (2 ** attempt)
        delay = min(self.backoff_max, delay)
        delay += random.uniform(0, 0.5 * self.backoff_base)
        time.sleep(delay)


def _embedded_error(data: dict) -> tuple[int, str, float | None] | None:
    """Return (code, message, retry_after) if a provider error is embedded in a
    200 body ({"error": {...}} with no "choices"), else None."""
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if not err:
        return None
    if not isinstance(err, dict):
        return 0, str(err), None

    try:
        code = int(err.get("code"))
    except (TypeError, ValueError):
        code = 0
    msg = str(err.get("message") or "provider error")

    retry_after: float | None = None
    meta = err.get("metadata")
    if isinstance(meta, dict):
        ra = meta.get("retry_after_seconds")
        if ra is None:
            headers = meta.get("headers")
            if isinstance(headers, dict):
                ra = headers.get("Retry-After")
        if ra is not None:
            try:
                retry_after = float(ra)
            except (TypeError, ValueError):
                retry_after = None
        raw = meta.get("raw")
        if raw:
            msg = f"{msg}: {_short(str(raw), 120)}"
    return code, msg, retry_after


def _short(text: str, n: int = 180) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")
