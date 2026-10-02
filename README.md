# agentic-trading-lab

A small, readable research framework for **multi-agent trading systems** that mix rule-based quant agents with an **LLM analyst**, wired into a walk-forward backtest with costs and risk controls.

```
features[t] ──► TrendAgent ───────┐
            ├─► MeanReversionAgent ├─► PortfolioManager ─► RiskAgent ─► ExecutionAgent ─► position[t] earns r[t→t+1]
            └─► LLMAnalystAgent ──┘    (confidence blend)   (vol target,   (no-trade band)
                 (JSON, validated)                            DD breaker)
```

## Why this design

- **Explicit messages.** Every agent emits a typed `Signal` / `Decision`, so you can log and audit who said what on each bar.
- **LLM as one voice, not the oracle.** The LLM analyst returns structured JSON; malformed replies degrade to a flat, zero-confidence signal instead of crashing the run.
- **Swappable model backend.** `HeuristicLLM` (offline, deterministic), `AnthropicLLM` (Claude), or `CallableLLM` (wrap anything, e.g. a local model).
- **No lookahead by construction.** Features use data up to `t`; positions decided at `t` earn the `t → t+1` return. Tests check this by tampering with future prices.

## Quick start

```bash
pip install -e ".[dev]"
python examples/run_demo.py               # synthetic regime-switching data, fully offline
python examples/run_demo.py --csv my.csv  # your own data: columns date, close
pytest
```

Example output on the bundled synthetic data (seed 7, 2 bps costs):

| metric | agents | buy & hold |
|---|---:|---:|
| Sharpe | 0.97 | -0.78 |
| Max drawdown | -13.6% | -67.3% |
| Ann. vol | 7.7% | 17.3% |

These numbers come from **synthetic data** and only show that the plumbing works. They are not a claim about real-market performance.

## Using a real LLM

```python
from agentic_trading_lab import AnthropicLLM, LLMAnalystAgent, TrendAgent, run_backtest, load_csv

llm = AnthropicLLM(model="<model-id>")   # needs ANTHROPIC_API_KEY and `pip install -e ".[llm]"`
result = run_backtest(load_csv("prices.csv"), [TrendAgent(), LLMAnalystAgent(llm)])
print(result.stats)
```

Caveats: a live model makes the backtest slow and non-deterministic, and a model may already "know" historical periods from its training data, which makes LLM backtests optimistic. Treat them as qualitative and validate on genuinely out-of-sample data.

## Layout

```
agentic_trading_lab/
  messages.py   Signal / Decision / RiskedDecision
  data.py       synthetic regime-switching prices, CSV loader
  features.py   point-in-time features (momentum, vol, RSI, MA distance)
  llm.py        LLM client interface + backends, JSON extraction
  agents.py     analysts, portfolio manager, risk, execution
  backtest.py   bar-by-bar walk-forward loop with costs
  metrics.py    Sharpe, CAGR, max drawdown, hit rate, turnover
examples/run_demo.py
tests/
```

## Roadmap

- News/sentiment analyst agent over point-in-time headlines
- Multi-asset portfolio with a correlation-aware risk agent
- Debate step between analysts before the PM decides

## License

MIT
