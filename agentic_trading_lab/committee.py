"""Deliberation layers: a bull-vs-bear debate before the decision, and a risk committee after it.

Both work offline with transparent rules and accept an LLM for the same role.
Every decision keeps a transcript, so a reviewer can see why the system traded.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import pandas as pd

from .agents import RiskAgent
from .llm import LLMClient, parse_json_object
from .messages import Decision, RiskedDecision, Signal


@dataclass
class Argument:
    side: str  # "bull" | "bear"
    source: str
    weight: float
    text: str
    rebutted: bool = False


@dataclass
class DebatePortfolioManager:
    """Bull and bear researchers argue from the analysts' evidence; a judge decides.

    Offline rules:
    * each side collects the signals that support it (weight = analyst weight × confidence);
    * in each round a side rebuts the opponent's strongest *unrebutted* point if the
      features contradict it (e.g. a trend argument when price is below its 50-day mean),
      which halves that point's weight;
    * the judge goes with the stronger side only if the margin clears ``min_margin``,
      otherwise stays flat. Disagreement therefore costs exposure, which is the point.

    With ``llm`` set, each round's arguments and the verdict come from the model instead
    (JSON, validated; invalid output falls back to the rule-based judge).
    """

    weights: dict[str, float] = field(default_factory=lambda: {"trend": 1.0, "mean_reversion": 0.5, "llm_analyst": 1.0,
                                                                "news": 0.8})
    rounds: int = 2
    min_margin: float = 0.15
    llm: LLMClient | None = None
    uses_context: bool = True
    transcripts: list[dict] = field(default_factory=list)
    llm_failures: int = 0

    def _arguments(self, signals: list[Signal]) -> list[Argument]:
        args = []
        for s in signals:
            if s.stance == "flat" or s.confidence == 0:
                continue
            side = "bull" if s.stance == "long" else "bear"
            args.append(Argument(side, s.source, self.weights.get(s.source, 0.5) * s.confidence, s.rationale))
        return args

    @staticmethod
    def _contradicted(arg: Argument, row: pd.Series) -> bool:
        dist, rsi, volr = row.get("dist_ma50"), row.get("rsi_14"), row.get("vol_ratio")
        if arg.source == "trend":
            return (arg.side == "bull" and dist is not None and dist < 0) or (arg.side == "bear" and dist is not None and dist > 0)
        if arg.source == "mean_reversion":
            return volr is not None and not math.isnan(volr) and volr > 1.3  # fading moves in a vol spike is dangerous
        if arg.source == "news":
            return (arg.side == "bull" and rsi is not None and rsi > 75) or (arg.side == "bear" and rsi is not None and rsi < 25)
        return False

    def decide(self, signals: list[Signal], row: pd.Series | None = None) -> Decision:
        row = row if row is not None else pd.Series(dtype=float)
        args = self._arguments(signals)
        log = []
        for r in range(self.rounds):
            for side, other in (("bull", "bear"), ("bear", "bull")):
                targets = sorted([a for a in args if a.side == other and not a.rebutted], key=lambda a: -a.weight)
                if targets and self._contradicted(targets[0], row):
                    targets[0].rebutted = True
                    targets[0].weight *= 0.5
                    log.append(f"round {r + 1}: {side} rebuts {other}'s {targets[0].source} argument")
        bull = sum(a.weight for a in args if a.side == "bull")
        bear = sum(a.weight for a in args if a.side == "bear")
        total = sum(self.weights.get(s.source, 0.5) for s in signals) or 1.0
        margin = (bull - bear) / total
        verdict = margin if abs(margin) >= self.min_margin else 0.0
        if self.llm is not None:
            verdict = self._llm_verdict(args, row, verdict)
        self.transcripts.append({"date": getattr(row, "name", None), "bull": round(bull, 3), "bear": round(bear, 3),
                                 "verdict": round(verdict, 3), "log": log})
        return Decision(target_exposure=max(-1.0, min(1.0, verdict)), signals=tuple(signals))

    def _llm_verdict(self, args: list[Argument], row: pd.Series, fallback: float) -> float:
        prompt = ("Bull case:\n" + "\n".join(f"- [{a.source}, w={a.weight:.2f}] {a.text}" for a in args if a.side == "bull")
                  + "\nBear case:\n" + "\n".join(f"- [{a.source}, w={a.weight:.2f}] {a.text}" for a in args if a.side == "bear")
                  + "\nFeatures: " + ", ".join(f"{k}={float(v):.3f}" for k, v in row.items() if isinstance(v, (int, float))))
        system = ('You are the research manager judging a bull-vs-bear debate. Weigh evidence quality, not volume. '
                  'Reply ONLY with JSON {"exposure": -1..1, "reason": "<one sentence>"}; prefer 0 when unconvincing.')
        try:
            return max(-1.0, min(1.0, float(parse_json_object(self.llm.complete(system, prompt))["exposure"])))
        except (ValueError, KeyError, TypeError):
            self.llm_failures += 1
            return fallback


@dataclass
class RiskCommittee:
    """Three risk views vote on the size; hard limits apply regardless of the vote.

    * aggressive: vol-target only;
    * neutral: vol-target plus the drawdown breaker;
    * conservative: neutral, halved again when volatility is spiking or news turns sharply negative.

    The committee takes the median view, but the conservative view wins outright while
    the portfolio is in drawdown beyond ``defensive_drawdown``.
    """

    base: RiskAgent = field(default_factory=RiskAgent)
    defensive_drawdown: float = 0.08
    vol_spike: float = 1.5
    uses_context: bool = True
    votes: list[dict] = field(default_factory=list)

    def review(self, decision: Decision, realised_vol: float, drawdown: float, row: pd.Series | None = None) -> RiskedDecision:
        neutral = self.base.review(decision, realised_vol, drawdown)
        aggressive = RiskAgent(self.base.target_vol, self.base.max_leverage, max_drawdown=1.0).review(
            decision, realised_vol, drawdown).target_exposure
        conservative = neutral.target_exposure
        reasons = list(neutral.notes)
        volr = None if row is None else row.get("vol_ratio")
        news = next((s for s in decision.signals if s.source == "news"), None)
        if volr is not None and not math.isnan(volr) and volr > self.vol_spike:
            conservative *= 0.5
            reasons.append(f"vol spike ({volr:.2f}x)")
        if news is not None and news.score <= -0.6:
            conservative *= 0.5
            reasons.append("negative news shock")
        views = {"aggressive": aggressive, "neutral": neutral.target_exposure, "conservative": conservative}
        if drawdown <= -self.defensive_drawdown:
            final, rule = conservative, "conservative (in drawdown)"
        else:
            final, rule = statistics.median(views.values()), "median"
        # hard limit: never exceed the gross cap, whatever the vote
        final = max(-self.base.max_leverage, min(self.base.max_leverage, final))
        self.votes.append({**{k: round(v, 3) for k, v in views.items()}, "final": round(final, 3), "rule": rule})
        return RiskedDecision(target_exposure=final, notes=tuple(reasons + [rule]))
