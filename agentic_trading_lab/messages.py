"""Typed messages passed between agents.

Every agent reads a market snapshot and emits one of these objects, so the
flow of information through the system is explicit and easy to log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Stance = Literal["long", "flat", "short"]

STANCE_TO_SIGN: dict[str, int] = {"long": 1, "flat": 0, "short": -1}


@dataclass(frozen=True)
class Signal:
    """An opinion from an analyst agent."""

    source: str
    stance: Stance
    confidence: float  # 0..1
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.stance not in STANCE_TO_SIGN:
            raise ValueError(f"unknown stance {self.stance!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def score(self) -> float:
        """Signed conviction in [-1, 1]."""
        return STANCE_TO_SIGN[self.stance] * self.confidence


@dataclass(frozen=True)
class Decision:
    """The portfolio manager's combined view, before risk limits."""

    target_exposure: float  # desired position as a fraction of equity, [-1, 1]
    signals: tuple[Signal, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RiskedDecision:
    """A decision after the risk agent has scaled or vetoed it."""

    target_exposure: float
    notes: tuple[str, ...] = field(default_factory=tuple)
