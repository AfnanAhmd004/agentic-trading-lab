"""Run the full agent team on synthetic data and compare it with buy-and-hold.

    python examples/run_demo.py                 # offline, deterministic
    python examples/run_demo.py --csv prices.csv  # your own data (date, close)
"""
from __future__ import annotations

import argparse

from agentic_trading_lab import (
    HeuristicLLM,
    LLMAnalystAgent,
    MeanReversionAgent,
    TrendAgent,
    load_csv,
    run_backtest,
    synthetic_prices,
)
from agentic_trading_lab.metrics import summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with date and close columns")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    args = ap.parse_args()

    prices = load_csv(args.csv) if args.csv else synthetic_prices()
    team = [TrendAgent(), MeanReversionAgent(), LLMAnalystAgent(HeuristicLLM())]
    result = run_backtest(prices, team, cost_bps=args.cost_bps)

    bh = prices["close"].pct_change().dropna()
    print(f"{'metric':<14}{'agents':>10}{'buy&hold':>12}")
    for k, v in result.stats.items():
        b = summary(bh).get(k)
        print(f"{k:<14}{v:>10.3f}{'' if b is None else f'{b:>12.3f}'}")


if __name__ == "__main__":
    main()
