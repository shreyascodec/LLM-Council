"""Load and freeze configuration, hashing it into a config_hash that is stamped
into every Decision Object's provenance for reproducibility."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    fallbacks: tuple[str, ...] = ()

    def candidates(self) -> list[str]:
        """Primary id first, then fallbacks, de-duplicated, order preserved."""
        seen: set[str] = set()
        out: list[str] = []
        for m in (self.id, *self.fallbacks):
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    config_hash: str
    api_key: str
    app_url: str
    app_title: str

    # convenience accessors -------------------------------------------------
    @property
    def base_url(self) -> str:
        return self.raw["provider"]["base_url"]

    @property
    def request_timeout_s(self) -> float:
        return float(self.raw["provider"].get("request_timeout_s", 90))

    @property
    def generators(self) -> list[ModelSpec]:
        return [_to_spec(m) for m in self.raw["generators"]]

    @property
    def judges(self) -> list[ModelSpec]:
        return [_to_spec(m) for m in self.raw["judges"]]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})


def _to_spec(m: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        id=m["id"],
        family=m["family"],
        fallbacks=tuple(m.get("fallbacks", []) or ()),
    )


def _stable_hash(obj: Any) -> str:
    """Deterministic hash of the config content (key order independent)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | os.PathLike[str] | None = None,
                require_key: bool = True) -> Config:
    load_dotenv(REPO_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    # Tolerate config files saved as UTF-8 or a Windows codepage (cp1252).
    data = cfg_path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1252")
    raw = yaml.safe_load(text)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if require_key and not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key. "
            "Get a free key at https://openrouter.ai/keys"
        )

    return Config(
        raw=raw,
        config_hash=_stable_hash(raw),
        api_key=api_key,
        app_url=os.environ.get("OPENROUTER_APP_URL", "https://github.com/llm-council"),
        app_title=os.environ.get("OPENROUTER_APP_TITLE", "LLM Council"),
    )
