"""Walk-forward backtest that runs the whole agent team one bar at a time.

Timing convention (the part that most often leaks the future):

    features[t]  -> agents decide position p[t] at the close of bar t
    p[t]         -> earns the simple return from close t to close t+1

Costs are charged on every change in position, in basis points of the traded
notional.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .agents import ExecutionAgent, PortfolioManager, RiskAgent
from .features import build_features
from .metrics import summary


@dataclass
class BacktestResult:
    log: pd.DataFrame
    stats: dict[str, float]
    signal_log: list[dict] = field(default_factory=list)


def run_backtest(
    prices: pd.DataFrame,
    analysts: list,
    pm: PortfolioManager | None = None,
    risk: RiskAgent | None = None,
    execution: ExecutionAgent | None = None,
    cost_bps: float = 2.0,
    keep_signal_log: bool = False,
) -> BacktestResult:
    pm = pm or PortfolioManager()
    risk = risk or RiskAgent()
    execution = execution or ExecutionAgent()

    feats = build_features(prices)
    fwd_ret = prices["close"].pct_change().shift(-1)  # return earned by a position held from t to t+1

    position = 0.0
    equity, peak = 1.0, 1.0
    rows, signal_log = [], []
    dates = feats.index[:-1]  # last bar has no next-day return to earn

    for t in dates:
        row = feats.loc[t]
        signals = [a.analyse(row) for a in analysts]
        decision = pm.decide(signals)
        drawdown = equity / peak - 1
        risked = risk.review(decision, row["vol_20d"], drawdown)
        new_position = execution.rebalance(position, risked.target_exposure)

        trade = abs(new_position - position)
        cost = trade * cost_bps / 1e4
        pnl = new_position * fwd_ret.loc[t] - cost
        equity *= 1 + pnl
        peak = max(peak, equity)
        position = new_position

        rows.append({"date": t, "position": position, "trade": trade, "cost": cost, "ret": pnl, "equity": equity})
        if keep_signal_log:
            signal_log.append({"date": t, **{s.source: s.score for s in signals}, "risk_notes": "; ".join(risked.notes)})

    log = pd.DataFrame(rows).set_index("date")
    return BacktestResult(log=log, stats=summary(log["ret"], log["position"]), signal_log=signal_log)
