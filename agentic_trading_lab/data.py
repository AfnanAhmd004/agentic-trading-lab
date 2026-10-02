"""Market data: a synthetic regime-switching generator and a CSV loader.

The synthetic generator lets the whole pipeline run offline and reproducibly.
Real data can be dropped in with `load_csv` (any OHLCV export with a date
column and a close column works, e.g. from MetaTrader 5 or a broker).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_prices(
    n_days: int = 1500,
    start_price: float = 100.0,
    seed: int = 7,
    regimes: tuple[tuple[float, float], ...] = ((0.0006, 0.009), (-0.0004, 0.02), (0.0, 0.012)),
    mean_regime_length: int = 120,
) -> pd.DataFrame:
    """Daily prices from a Markov regime-switching random walk.

    Each regime is a (daily drift, daily volatility) pair. Regimes persist for
    roughly `mean_regime_length` days, which gives trend-following and
    volatility-aware agents something real to find.
    """
    rng = np.random.default_rng(seed)
    p_switch = 1.0 / mean_regime_length
    state = 0
    states = np.empty(n_days, dtype=int)
    rets = np.empty(n_days)
    for t in range(n_days):
        if rng.random() < p_switch:
            state = int(rng.integers(len(regimes)))
        mu, sigma = regimes[state]
        states[t] = state
        rets[t] = rng.normal(mu, sigma)
    close = start_price * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2018-01-01", periods=n_days, name="date")
    return pd.DataFrame({"close": close, "regime": states}, index=idx)


def load_csv(path: str, date_col: str = "date", close_col: str = "close") -> pd.DataFrame:
    """Load a price CSV into the frame layout the lab expects."""
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", close_col: "close"}).set_index("date").sort_index()
    if df["close"].isna().any():
        raise ValueError("close column contains NaNs")
    return df[["close"]]
