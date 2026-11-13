"""Model services used by the curator: a text embedder (shortlisting) and a yes/no pair judge.

Both sit behind small Protocols so the curator can be tested with fakes and the models swapped by config.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable, Protocol

import numpy as np
import requests

from ..errors import CurationError
from ..settings import CurationSettings

log = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised row vectors. Raise CurationError on failure."""


class PairJudge(Protocol):
    name: str

    def is_every(self, a: str, b: str) -> bool:
        """Answer 'is every A a B?' (same thing or a more specific kind). Raise CurationError on failure."""


def _post(session: requests.Session, url: str, payload: dict, *, timeout: float, retries: int,
          sleep: Callable[[float], None]) -> dict:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if 400 <= r.status_code < 500:
                raise CurationError(f"request rejected (HTTP {r.status_code}): {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except CurationError:
            raise
        except (requests.RequestException, ValueError) as e:
            last = e
            log.warning("curation call attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                sleep(2 * attempt)
    raise CurationError(f"curation model call failed after {retries} attempts: {last}")


class LmStudioEmbedder:
    def __init__(self, base_url: str, settings: CurationSettings, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.name, self._url = settings.embed_model, f"{base_url.rstrip('/')}/v1/embeddings"
        self._cfg, self._s, self._sleep = settings, session or requests.Session(), sleep

    def embed(self, texts: list[str]) -> np.ndarray:
        body = _post(self._s, self._url, {"model": self.name, "input": [f"clustering: {t}" for t in texts]},
                     timeout=self._cfg.timeout_s, retries=self._cfg.retries, sleep=self._sleep)
        try:
            v = np.array([d["embedding"] for d in sorted(body["data"], key=lambda d: d.get("index", 0))], float)
        except (KeyError, TypeError, ValueError) as e:
            raise CurationError(f"unexpected embeddings response: {e}") from e
        if v.shape[0] != len(texts):
            raise CurationError(f"expected {len(texts)} embeddings, got {v.shape[0]}")
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


_YES_NO = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}, "required": ["answer"]}


class LmStudioJudge:
    def __init__(self, base_url: str, settings: CurationSettings, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.name, self._url = settings.judge_model, f"{base_url.rstrip('/')}/v1/chat/completions"
        self._cfg, self._s, self._sleep = settings, session or requests.Session(), sleep

    def is_every(self, a: str, b: str) -> bool:
        prompt = (f"{self._cfg.context} Is every '{a}' a '{b}'? Answer yes only if '{a}' is the same thing as, or a "
                  f"specific kind of, '{b}'. Being found together or being related does not count.")
        body = _post(self._s, self._url, {
            "model": self.name, "temperature": 0, "max_tokens": 20, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "yn", "strict": True, "schema": _YES_NO}}},
            timeout=self._cfg.timeout_s, retries=self._cfg.retries, sleep=self._sleep)
        try:
            msg = body["choices"][0]["message"]
            ans = json.loads(msg.get("content") or msg.get("reasoning_content") or "")["answer"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise CurationError(f"unexpected judge response: {e}") from e
        if ans not in ("yes", "no"):
            raise CurationError(f"judge answered {ans!r}")
        return ans == "yes"
