"""Pluggable model backends.

A backend implements one method, :meth:`ModelBackend.generate`, and
returns a :class:`ModelResponse`. The harness ships three backends:

* :class:`StubBackend` — deterministic canned responses, for smoke
  tests and for exercising the harness itself.
* :class:`MockBackend` — regex-routed scripted responses loaded from a
  mapping (great for reproducible demos).
* :class:`OpenAICompatibleBackend` — any HTTP API that speaks the
  OpenAI ``/chat/completions`` shape (OpenAI, vLLM, Ollama, ...).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class ModelResponse:
    """One generation from a model backend."""

    text: str
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class ModelBackend:
    """Interface every model backend must implement."""

    name: str = "base"

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError


class StubBackend(ModelBackend):
    """Deterministic stub useful for testing the harness itself.

    ``mode="echo"`` returns the prompt back verbatim. ``mode="canned"``
    always returns ``response``. ``mode="template"`` formats
    ``response`` with ``{prompt}``.
    """

    def __init__(self, mode: str = "echo", response: str = "") -> None:
        if mode not in ("echo", "canned", "template"):
            raise ValueError(f"unknown stub mode: {mode!r}")
        self.mode = mode
        self.response = response
        self.name = f"stub:{mode}"

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> ModelResponse:
        start = time.perf_counter()
        if self.mode == "echo":
            text = prompt
        elif self.mode == "template":
            text = self.response.format(prompt=prompt)
        else:
            text = self.response
        return ModelResponse(text=text, latency_s=time.perf_counter() - start,
                             meta={"mode": self.mode})


class MockBackend(ModelBackend):
    """Regex-routed scripted responses for reproducible runs.

    Routes are ``(pattern, response)`` pairs; the first pattern that
    matches the prompt wins, otherwise ``default`` is returned.
    Load from JSON with :meth:`from_file`.
    """

    def __init__(self, routes: list[tuple[str, str]] | None = None, default: str = "") -> None:
        self.routes = [(re.compile(p, re.IGNORECASE | re.DOTALL), r) for p, r in (routes or [])]
        self.default = default
        self.name = "mock"

    @classmethod
    def from_file(cls, path: str | Path) -> "MockBackend":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        routes = [(r["pattern"], r["response"]) for r in data.get("routes", [])]
        return cls(routes=routes, default=data.get("default", ""))

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> ModelResponse:
        start = time.perf_counter()
        text = self.default
        matched: str | None = None
        for pattern, response in self.routes:
            if pattern.search(prompt):
                text, matched = response, pattern.pattern
                break
        return ModelResponse(text=text, latency_s=time.perf_counter() - start,
                             meta={"matched_pattern": matched})


class BackendError(RuntimeError):
    """Raised when a remote backend call fails."""


class OpenAICompatibleBackend(ModelBackend):
    """Talk to any OpenAI-compatible ``/chat/completions`` endpoint.

    Uses only the standard library (``urllib``), so no extra
    dependencies are needed.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"openai-compatible:{model}"

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> ModelResponse:
        messages: list[Mapping[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BackendError(f"backend returned HTTP {exc.code}: {exc.read().decode()[:500]}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"backend request failed: {exc.reason}") from exc
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response shape: {str(data)[:500]}") from exc
        return ModelResponse(
            text=text,
            latency_s=time.perf_counter() - start,
            meta={"model": self.model, "finish_reason": data["choices"][0].get("finish_reason")},
        )
