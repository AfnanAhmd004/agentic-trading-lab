"""Does adding agents help? A paired comparison across many synthetic markets.

Each configuration runs on the same 20 synthetic price + news histories (different
seeds), so differences are paired. Reported: mean Sharpe, mean max drawdown, and the
paired Sharpe difference vs the baseline with a 95% interval.

    python examples/committee_study.py      # ~1-2 min
"""
from __future__ import annotations

import numpy as np

from agentic_trading_lab import (DebatePortfolioManager, HeuristicLLM, LLMAnalystAgent, MeanReversionAgent,
                                 NewsAnalystAgent, PortfolioManager, RiskCommittee, TrendAgent, run_backtest,
                                 synthetic_news, synthetic_prices)
from agentic_trading_lab.metrics import summary

SEEDS = range(20)
W = {"trend": 1.0, "mean_reversion": 0.5, "llm_analyst": 1.0, "news": 0.8}


def quant_team():
    return [TrendAgent(), MeanReversionAgent(), LLMAnalystAgent(HeuristicLLM())]


CONFIGS = {
    "baseline: 3 analysts, blend PM": lambda n: dict(analysts=quant_team()),
    "news analyst only": lambda n: dict(analysts=[NewsAnalystAgent(n)], pm=PortfolioManager({"news": 1.0})),
    "+ news analyst": lambda n: dict(analysts=quant_team() + [NewsAnalystAgent(n)], pm=PortfolioManager(W)),
    "+ news, bull/bear debate": lambda n: dict(analysts=quant_team() + [NewsAnalystAgent(n)], pm=DebatePortfolioManager(W)),
    "+ news, debate, risk committee": lambda n: dict(analysts=quant_team() + [NewsAnalystAgent(n)],
                                                      pm=DebatePortfolioManager(W), risk=RiskCommittee()),
}


def main() -> None:
    res = {k: [] for k in CONFIGS}
    bh = []
    for seed in SEEDS:
        prices = synthetic_prices(seed=seed)
        news = synthetic_news(prices, seed=seed)
        r = prices["close"].pct_change().dropna()
        bh.append((summary(r)["sharpe"], summary(r)["max_drawdown"]))
        for name, make in CONFIGS.items():
            s = run_backtest(prices, **make(news), cost_bps=2.0).stats
            res[name].append((s["sharpe"], s["max_drawdown"], s["avg_turnover"]))
    base = np.array(res["baseline: 3 analysts, blend PM"])[:, 0]
    print(f"{'configuration':<34}{'Sharpe':>8}{'max DD':>9}{'turnover':>10}   ΔSharpe vs baseline (95% CI)")
    b = np.array(bh)
    print(f"{'buy & hold':<34}{b[:, 0].mean():>8.2f}{b[:, 1].mean():>9.1%}{'':>10}")
    for name, rows in res.items():
        a = np.array(rows)
        line = f"{name:<34}{a[:, 0].mean():>8.2f}{a[:, 1].mean():>9.1%}{a[:, 2].mean():>10.3f}"
        if not name.startswith("baseline"):
            d = a[:, 0] - base
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            line += f"   {d.mean():+.2f} ({d.mean() - half:+.2f} to {d.mean() + half:+.2f})"
        print(line)


if __name__ == "__main__":
    main()
