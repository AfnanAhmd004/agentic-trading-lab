"""Point-in-time features.

Every column at row t uses only prices up to and including t. The backtest
then trades on the *next* bar, so no feature can see the return it is used
to predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    ret = np.log(close).diff()
    f = pd.DataFrame(index=prices.index)
    f["ret_1d"] = ret
    f["mom_20d"] = close / close.shift(20) - 1
    f["mom_60d"] = close / close.shift(60) - 1
    f["vol_20d"] = ret.rolling(20).std() * np.sqrt(252)
    f["vol_ratio"] = ret.rolling(10).std() / ret.rolling(60).std()
    f["rsi_14"] = rsi(close, 14)
    f["dist_ma50"] = close / close.rolling(50).mean() - 1
    return f
