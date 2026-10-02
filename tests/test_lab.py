import json

import numpy as np
import pandas as pd
import pytest

from agentic_trading_lab import (
    CallableLLM,
    HeuristicLLM,
    LLMAnalystAgent,
    MeanReversionAgent,
    TrendAgent,
    run_backtest,
    synthetic_prices,
)
from agentic_trading_lab.agents import PortfolioManager, RiskAgent
from agentic_trading_lab.features import build_features
from agentic_trading_lab.messages import Decision, Signal


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n_days=400, seed=1)


def test_signal_validation():
    with pytest.raises(ValueError):
        Signal("x", "long", 1.5)
    with pytest.raises(ValueError):
        Signal("x", "sideways", 0.5)
    assert Signal("x", "short", 0.5).score == -0.5


def test_features_are_point_in_time(prices):
    """Changing future prices must not change today's features."""
    full = build_features(prices)
    cut = 250
    tampered = prices.copy()
    tampered.iloc[cut + 1 :, 0] *= 3.0
    alt = build_features(tampered)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], alt.iloc[: cut + 1])


def test_backtest_has_no_lookahead(prices):
    """Positions up to t depend only on data up to t."""
    team = [TrendAgent(), MeanReversionAgent(), LLMAnalystAgent(HeuristicLLM())]
    base = run_backtest(prices, team).log
    cut = 300
    tampered = prices.copy()
    tampered.iloc[cut + 1 :, 0] *= np.linspace(1, 2, len(tampered) - cut - 1)
    alt = run_backtest(tampered, team).log
    pd.testing.assert_series_equal(base["position"].iloc[: cut + 1], alt["position"].iloc[: cut + 1])


def test_llm_agent_survives_bad_output(prices):
    row = build_features(prices).iloc[-1]
    agent = LLMAnalystAgent(CallableLLM(lambda s, p: "I think it will go up!"))
    sig = agent.analyse(row)
    assert sig.stance == "flat" and sig.confidence == 0.0 and agent.failures == 1


def test_llm_agent_clamps_confidence(prices):
    row = build_features(prices).iloc[-1]
    reply = json.dumps({"stance": "long", "confidence": 7, "rationale": "very sure"})
    agent = LLMAnalystAgent(CallableLLM(lambda s, p: "Sure. " + reply), max_confidence=0.8)
    assert agent.analyse(row).confidence == 0.8


def test_risk_agent_targets_vol_and_breaks_on_drawdown():
    risk = RiskAgent(target_vol=0.10, max_leverage=1.0, max_drawdown=0.1, breaker_scale=0.5)
    d = Decision(target_exposure=1.0)
    assert risk.review(d, realised_vol=0.20, drawdown=0.0).target_exposure == pytest.approx(0.5)
    tripped = risk.review(d, realised_vol=0.20, drawdown=-0.2)
    assert tripped.target_exposure == pytest.approx(0.25) and tripped.notes


def test_pm_blend_is_bounded():
    pm = PortfolioManager({"a": 1.0, "b": 1.0})
    d = pm.decide([Signal("a", "long", 1.0), Signal("b", "short", 0.5)])
    assert d.target_exposure == pytest.approx(0.25)


def test_costs_reduce_returns(prices):
    team = [TrendAgent()]
    free = run_backtest(prices, team, cost_bps=0).log["ret"].sum()
    costly = run_backtest(prices, team, cost_bps=50).log["ret"].sum()
    assert costly < free
