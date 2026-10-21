"""Decision providers.

A provider answers a fan-out of typed questions about one utterance. Questions are
`choice` (pick one key of `criteria`) or `noul` (probability that a statement holds).
Answers use one shape for every provider:

    choice -> {"choice": <key of criteria>, "confidence": 0..1}
    noul   -> {"noul": 0..1}

`validate_answers` enforces that shape so a misbehaving provider fails closed
instead of reaching the executor.
"""
from __future__ import annotations

import math
import os
from typing import Any, Protocol

import httpx

from . import config


class ProviderError(RuntimeError):
    pass


class DecisionProvider(Protocol):
    name: str
    model: str
    staged: bool

    def answer(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]: ...


def _prob(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and 0.0 <= f <= 1.0 else None


def validate_answers(questions: dict[str, Any], answers: Any) -> dict[str, Any]:
    if not isinstance(answers, dict):
        raise ProviderError("answers is not an object")
    out: dict[str, Any] = {}
    for key, q in questions.items():
        a = answers.get(key)
        if not isinstance(a, dict):
            raise ProviderError(f"missing answer: {key}")
        if q["type"] == "choice":
            if a.get("choice") not in q["criteria"]:
                raise ProviderError(f"{key}: choice {a.get('choice')!r} is not offered")
            conf = _prob(a.get("confidence"))
            if conf is None:
                raise ProviderError(f"{key}: bad confidence {a.get('confidence')!r}")
            out[key] = {**a, "confidence": conf}
        elif q["type"] == "noul":
            p = _prob(a.get("noul"))
            if p is None:
                raise ProviderError(f"{key}: bad noul {a.get('noul')!r}")
            out[key] = {**a, "noul": p}
        else:
            raise ProviderError(f"{key}: unknown question type {q['type']!r}")
    return out


class JevProvider:
    """TypeSafe System One (hosted). One request per utterance."""

    name = "jev"
    staged = False

    def __init__(self, api_key: str | None = None, model: str | None = None, url: str | None = None,
                 http: httpx.Client | None = None) -> None:
        self.api_key = api_key or config.TYPESAFE_API_KEY
        if not self.api_key:
            raise SystemExit("TYPESAFE_API_KEY is not set (put it in .env)")
        self.model = model or config.JEV_MODEL
        self.url = url or config.TYPESAFE_URL
        self.http = http or httpx.Client(
            timeout=15.0,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            http2=False,
        )
        if http is None:
            # Keep the TCP+TLS connection warm so the first real command is fast.
            try:
                self.http.get("https://api.typesafe.ai/v1/models")
            except Exception:
                pass

    def answer(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        payload = {"state": state, "model": self.model, "questions": questions}
        try:
            r = self.http.post(self.url, json=payload)
            r.raise_for_status()
            return r.json()["answers"]
        except (httpx.HTTPError, ValueError, KeyError) as e:
            raise ProviderError(f"jev: {e}") from e


class LayaProvider:
    """Laya (local, open weights), loaded in-process.

    Laya is accurate only on a small state and a few questions per call, so Brain
    asks it in two stages (`staged`): the action first, then only that action's
    questions.
    """

    name = "laya"
    staged = True

    def __init__(self, repo: str | None = None, subfolder: str | None = None, agent: Any = None) -> None:
        self.repo = repo or config.LAYA_REPO
        self.subfolder = subfolder if subfolder is not None else config.LAYA_SUBFOLDER
        self.model = self.repo + (f"/{self.subfolder}" if self.subfolder else "")
        if agent is None:
            # TensorFlow installs can deadlock Laya's model construction.
            os.environ.setdefault("USE_TF", "0")
            try:
                import laya
            except ImportError:
                raise SystemExit("laya is not installed: uv sync --extra laya") from None
            kw: dict[str, Any] = {"subfolder": self.subfolder} if self.subfolder else {}
            if config.LAYA_DEVICE:
                kw["device"] = config.LAYA_DEVICE
            agent = laya.load(self.repo, **kw)
        self.agent = agent

    def answer(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.agent.predict(state, questions)["answers"]
        except Exception as e:  # noqa: BLE001
            raise ProviderError(f"laya: {e}") from e


PROVIDERS = {"jev": JevProvider, "laya": LayaProvider}


def make_provider(name: str | None = None) -> DecisionProvider:
    name = (name or config.DECISION_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise SystemExit(f"Unknown DECISION_PROVIDER {name!r}; choose one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]()
