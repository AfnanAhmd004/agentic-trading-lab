"""Language-model backends.

Agents talk to models through one tiny interface, `LLMClient.complete`, so
the same agent code runs against:

* `HeuristicLLM` - a deterministic offline stand-in used for tests and for
  reproducible backtests (no API key, no cost, no nondeterminism);
* `AnthropicLLM` - Claude via the official `anthropic` SDK;
* `CallableLLM` - wrap any function, e.g. a local model served by Ollama.

Backtesting with a live LLM is slow and, if the model has seen the period in
its training data, optimistic. Treat live-LLM backtests as qualitative.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class CallableLLM:
    """Adapter for any `fn(system, prompt) -> str`."""

    def __init__(self, fn: Callable[[str, str], str]):
        self._fn = fn

    def complete(self, system: str, prompt: str) -> str:
        return self._fn(system, prompt)


class AnthropicLLM:
    """Claude backend. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_tokens: int = 400, temperature: float = 0.0):
        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError("pip install anthropic to use AnthropicLLM") from exc
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, system: str, prompt: str) -> str:  # pragma: no cover - network
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


_FIELD = re.compile(r"^\s*([a-z0-9_]+)\s*:\s*(-?[0-9.]+|nan)\s*$", re.MULTILINE)


class HeuristicLLM:
    """Offline stand-in that reads the feature block of a prompt.

    It mimics what we ask a real model to do - weigh trend against
    overextension and volatility - and answers in the same JSON schema, so the
    parsing, validation and fallback paths are exercised exactly as they
    would be with a hosted model.
    """

    def complete(self, system: str, prompt: str) -> str:
        fields = {k: float(v) for k, v in _FIELD.findall(prompt)}
        mom = fields.get("mom_60d", 0.0)
        rsi = fields.get("rsi_14", 50.0)
        vol_ratio = fields.get("vol_ratio", 1.0)
        if any(map(_isnan, (mom, rsi, vol_ratio))):
            return json.dumps({"stance": "flat", "confidence": 0.0, "rationale": "insufficient history"})

        score = max(-1.0, min(1.0, mom / 0.10))  # +/-10% over 60 days saturates
        if rsi > 75 and score > 0:
            score *= 0.5  # trend intact but stretched
        if rsi < 25 and score < 0:
            score *= 0.5
        if vol_ratio > 1.5:
            score *= 0.5  # volatility expanding: lower conviction
        stance = "long" if score > 0.15 else "short" if score < -0.15 else "flat"
        rationale = f"60d momentum {mom:+.1%}, RSI {rsi:.0f}, vol ratio {vol_ratio:.2f}"
        return json.dumps({"stance": stance, "confidence": round(abs(score), 3), "rationale": rationale})


def _isnan(x: float) -> bool:
    return x != x


def parse_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply (models like to add prose)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])
