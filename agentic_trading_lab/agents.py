"""The agent team.

Analysts (quant rules and an LLM) each emit a `Signal`. The portfolio manager
blends them into a target exposure, the risk agent scales or vetoes it, and
the execution agent decides whether the change is worth trading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .llm import LLMClient, parse_json_object
from .messages import Decision, RiskedDecision, Signal


def _nan(x: float) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# --------------------------------------------------------------------------- analysts
class TrendAgent:
    """Classic time-series momentum: follow the 20-day move if price is above/below its 50-day mean."""

    name = "trend"

    def analyse(self, row: pd.Series) -> Signal:
        mom, dist = row["mom_20d"], row["dist_ma50"]
        if _nan(mom) or _nan(dist):
            return Signal(self.name, "flat", 0.0, "warming up")
        if mom > 0 and dist > 0:
            return Signal(self.name, "long", min(1.0, abs(mom) / 0.05), f"20d {mom:+.1%}, above MA50")
        if mom < 0 and dist < 0:
            return Signal(self.name, "short", min(1.0, abs(mom) / 0.05), f"20d {mom:+.1%}, below MA50")
        return Signal(self.name, "flat", 0.0, "trend and MA disagree")


class MeanReversionAgent:
    """Fades RSI extremes. Deliberately low-confidence: it is a counterweight, not a driver."""

    name = "mean_reversion"

    def __init__(self, upper: float = 75.0, lower: float = 25.0, confidence: float = 0.4):
        self.upper, self.lower, self.confidence = upper, lower, confidence

    def analyse(self, row: pd.Series) -> Signal:
        r = row["rsi_14"]
        if _nan(r):
            return Signal(self.name, "flat", 0.0, "warming up")
        if r > self.upper:
            return Signal(self.name, "short", self.confidence, f"RSI {r:.0f} overbought")
        if r < self.lower:
            return Signal(self.name, "long", self.confidence, f"RSI {r:.0f} oversold")
        return Signal(self.name, "flat", 0.0, f"RSI {r:.0f} neutral")


LLM_SYSTEM_PROMPT = (
    "You are a disciplined quantitative research analyst. You are given point-in-time "
    "features for one instrument. Decide a stance for the next trading day. Reply with "
    'ONLY a JSON object: {"stance": "long"|"flat"|"short", "confidence": 0..1, '
    '"rationale": "<one sentence>"}. Prefer "flat" when evidence is mixed.'
)

LLM_FEATURES = ("mom_20d", "mom_60d", "vol_20d", "vol_ratio", "rsi_14", "dist_ma50")


class LLMAnalystAgent:
    """Asks a language model for a structured view, then validates it.

    Bad or unparseable replies never crash the run: they become a flat,
    zero-confidence signal and are counted in `self.failures`.
    """

    name = "llm_analyst"

    def __init__(self, llm: LLMClient, max_confidence: float = 1.0):
        self.llm = llm
        self.max_confidence = max_confidence
        self.failures = 0

    def build_prompt(self, row: pd.Series) -> str:
        lines = [f"{k}: {float(row[k]):.4f}" for k in LLM_FEATURES]
        return "Features (as of today's close):\n" + "\n".join(lines)

    def analyse(self, row: pd.Series) -> Signal:
        try:
            reply = parse_json_object(self.llm.complete(LLM_SYSTEM_PROMPT, self.build_prompt(row)))
            conf = float(reply.get("confidence", 0.0))
            conf = max(0.0, min(self.max_confidence, conf))
            return Signal(self.name, reply["stance"], conf, str(reply.get("rationale", ""))[:200])
        except (ValueError, KeyError, TypeError):
            self.failures += 1
            return Signal(self.name, "flat", 0.0, "invalid model output")


# --------------------------------------------------------------------------- portfolio manager
@dataclass
class PortfolioManager:
    """Confidence-weighted blend of analyst scores."""

    weights: dict[str, float] = field(default_factory=lambda: {"trend": 1.0, "mean_reversion": 0.5, "llm_analyst": 1.0})

    def decide(self, signals: list[Signal]) -> Decision:
        num = sum(self.weights.get(s.source, 0.0) * s.score for s in signals)
        den = sum(self.weights.get(s.source, 0.0) for s in signals) or 1.0
        return Decision(target_exposure=max(-1.0, min(1.0, num / den)), signals=tuple(signals))


# --------------------------------------------------------------------------- risk
@dataclass
class RiskAgent:
    """Volatility targeting, a gross cap and a drawdown circuit breaker."""

    target_vol: float = 0.15  # annualised
    max_leverage: float = 1.0
    max_drawdown: float = 0.15  # cut risk once equity falls this far from its peak
    breaker_scale: float = 0.25

    def review(self, decision: Decision, realised_vol: float, drawdown: float) -> RiskedDecision:
        notes: list[str] = []
        x = decision.target_exposure
        if not _nan(realised_vol) and realised_vol > 0:
            x *= min(self.max_leverage, self.target_vol / realised_vol)
        x = max(-self.max_leverage, min(self.max_leverage, x))
        if drawdown <= -self.max_drawdown:
            x *= self.breaker_scale
            notes.append(f"drawdown breaker ({drawdown:.1%})")
        return RiskedDecision(target_exposure=x, notes=tuple(notes))


# --------------------------------------------------------------------------- execution
@dataclass
class ExecutionAgent:
    """Skips trades smaller than a no-trade band to save costs."""

    min_trade: float = 0.05

    def rebalance(self, current: float, target: float) -> float:
        return target if abs(target - current) >= self.min_trade else current
