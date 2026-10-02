"""Performance statistics for a daily return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(np.sqrt(TRADING_DAYS) * returns.mean() / sd) if sd > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    return float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else 0.0


def summary(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, float]:
    equity = (1 + returns).cumprod()
    out = {
        "cagr": cagr(equity),
        "ann_vol": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(returns),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
    }
    if positions is not None:
        out["avg_turnover"] = float(positions.diff().abs().mean())
        out["avg_gross"] = float(positions.abs().mean())
    return out
