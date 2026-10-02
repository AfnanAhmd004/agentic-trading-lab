# agentic-trading-lab

A small, readable research framework for **multi-agent trading systems** that mix rule-based quant agents with an **LLM analyst** and a **news analyst**. The design follows a trading firm: analysts, a **bull-vs-bear debate** judged by a research manager, a **risk committee**, and execution. Everything is wired into a walk-forward backtest with costs, and every layer is checked for lookahead.

```
features[t] ─► TrendAgent ──────────┐
            ├► MeanReversionAgent ──┤   PortfolioManager (confidence blend)
            ├► LLMAnalystAgent ─────┼─► or DebatePortfolioManager ─► RiskAgent ───────────► ExecutionAgent ─► position[t]
news ≤ t ───► NewsAnalystAgent ─────┘      bull vs bear, rebuttals,    or RiskCommittee       (no-trade band)    earns r[t→t+1]
              (point-in-time store)        judge needs a margin        aggressive/neutral/
                                                                       conservative vote + hard caps
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
python examples/committee_study.py        # paired 20-seed study of news, debate and risk committee
pytest
```

Example output on one synthetic market (seed 7, 2 bps costs):

| metric | agents | buy & hold |
|---|---:|---:|
| Sharpe | 0.97 | -0.78 |
| Max drawdown | -13.6% | -67.3% |
| Ann. vol | 7.7% | 17.3% |

These numbers come from **one synthetic seed**, which turns out to be a lucky one. Across 20 seeds the same team averages a Sharpe of about 0.0 (see below). They show that the plumbing works, not that the strategy has an edge.

## Does adding agents help? (v0.2)

`examples/committee_study.py` runs every configuration on the **same 20 synthetic markets**, so the comparisons are paired. Each market has its own price path and point-in-time headline stream; headline tone follows the hidden regime 60% of the time.

| configuration | Sharpe | max DD | ΔSharpe vs baseline (95% CI) |
|---|---:|---:|---|
| buy & hold | 0.25 | −45.6% | |
| baseline: trend + mean-reversion + LLM analyst, blend PM | 0.02 | −15.8% | |
| news analyst **alone** | **0.24** | −16.0% | +0.23 (+0.05 to +0.40) |
| + news analyst | 0.08 | −13.9% | +0.06 (+0.01 to +0.11) |
| + news, bull/bear debate | 0.09 | −13.9% | +0.07 (+0.02 to +0.13) |
| + news, debate, risk committee | 0.12 | −13.6% | +0.10 (+0.04 to +0.16) |

What this shows:
- **More agents is not more alpha.** The one informative source (news) is worth more on its own than blended with three weak voices that dilute it. A team should weight agents by measured skill, not by headcount.
- **Deliberation and risk layers help a little, and reliably.** The debate and the committee each add a small but statistically positive paired improvement and trim drawdowns. They act mainly by standing aside when evidence conflicts or volatility spikes.
- Every configuration keeps drawdowns around a third of buy-and-hold. Most of that comes from the risk layer's volatility targeting, not from prediction.

## Deliberation layers

- **`NewsAnalystAgent`** reads only headlines published before the decision close, through a `NewsStore` that cannot return the future. It scores sentiment with a lexicon offline, or with an LLM through a validated JSON reply. Unparseable model output becomes a flat signal and is counted.
- **`DebatePortfolioManager`** turns analyst signals into bull and bear arguments. Over several rounds each side rebuts the opponent's strongest point when today's features contradict it (for example a trend case while price is below its 50-day mean); a rebutted point loses half its weight. The judge needs a minimum margin, otherwise it stays flat. With `llm=` the verdict comes from a model, is clamped to [−1, 1], and invalid replies fall back to the rule-based judge. Every bar's transcript is kept.
- **`RiskCommittee`** collects aggressive, neutral and conservative sizing views and takes the median. The conservative view wins outright during drawdowns, and is halved again on volatility spikes or a sharply negative news signal. Hard caps apply whatever the vote. Every vote is logged.

The lookahead test tampers with future prices **and** injects future headlines; positions up to the cut must be identical.

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
  news.py       point-in-time NewsStore, synthetic headlines, NewsAnalystAgent
  committee.py  DebatePortfolioManager (bull vs bear + judge), RiskCommittee
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

- Skill-weighted PM: learn analyst weights from out-of-sample track record
- Multi-asset portfolio with a correlation-aware risk agent
- Real news feed adapter with publication-time auditing

## License

MIT
