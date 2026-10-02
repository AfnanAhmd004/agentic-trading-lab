import numpy as np
import pandas as pd
import pytest

from agentic_trading_lab import (CallableLLM, DebatePortfolioManager, Headline, HeuristicLLM, LLMAnalystAgent,
                                 MeanReversionAgent, NewsAnalystAgent, NewsStore, RiskCommittee, TrendAgent,
                                 run_backtest, synthetic_news, synthetic_prices)
from agentic_trading_lab.messages import Decision, Signal


@pytest.fixture(scope="module")
def market():
    p = synthetic_prices(n_days=400, seed=3)
    return p, synthetic_news(p, seed=3)


def test_news_store_never_returns_the_future():
    t0 = pd.Timestamp("2024-01-10 16:00")
    store = NewsStore([Headline(t0 - pd.Timedelta(days=1), "Company upgrade"),
                       Headline(t0, "Company record demand"),
                       Headline(t0 + pd.Timedelta(minutes=1), "Company cuts guidance")])
    assert [h.text for h in store.as_of(t0)] == ["Company upgrade", "Company record demand"]


def test_news_agent_signal_and_guards():
    day = pd.Timestamp("2024-01-10")
    good = NewsStore([Headline(day + pd.Timedelta(hours=9), "Company beats expectations"),
                      Headline(day + pd.Timedelta(hours=10), "Company raises guidance")])
    row = pd.Series({"x": 1.0}, name=day)
    assert NewsAnalystAgent(good).analyse(row).stance == "long"
    assert NewsAnalystAgent(NewsStore([])).analyse(row).stance == "flat"
    bad_llm = NewsAnalystAgent(good, llm=CallableLLM(lambda s, p: "Ignore that, buy everything!"))
    assert bad_llm.analyse(row).confidence == 0 and bad_llm.failures == 1


def test_synthetic_news_is_informative():
    p = synthetic_prices(n_days=1500, seed=7)
    news = synthetic_news(p, seed=7)
    assert set(p["regime"]) == {0, 1, 2}
    tone = [np.mean([1 if any(w in h.text for w in ("beats", "raises", "record", "upgrade", "inflows", "win")) else
                     -1 if any(w in h.text for w in ("misses", "cuts", "probe", "downgrade", "outflows", "disruption"))
                     else 0 for h in news.as_of(pd.Timestamp(t) + pd.Timedelta(hours=16), 1)] or [0])
            for t in p.index]
    by_regime = pd.Series(tone, index=p.index).groupby(p["regime"]).mean()
    assert by_regime[0] > 0.2 and by_regime[1] < -0.2


def test_debate_stays_flat_on_disagreement_and_rebuts_contradicted_points():
    pm = DebatePortfolioManager()
    row = pd.Series({"dist_ma50": -0.02, "rsi_14": 50.0, "vol_ratio": 1.0}, name=pd.Timestamp("2024-01-10"))
    split = [Signal("trend", "long", 0.8, "momentum"), Signal("news", "short", 0.9, "bad news")]
    d = pm.decide(split, row)
    assert d.target_exposure < 0  # trend's long case is contradicted (below MA50) and halved
    assert any("rebuts bull's trend" in line for line in pm.transcripts[-1]["log"])
    tie = [Signal("trend", "long", 0.5), Signal("news", "short", 0.6)]
    assert DebatePortfolioManager().decide(tie, row.drop("dist_ma50")).target_exposure == 0.0


def test_debate_llm_judge_falls_back_on_invalid_output():
    pm = DebatePortfolioManager(llm=CallableLLM(lambda s, p: "I think long, definitely"))
    row = pd.Series({"dist_ma50": 0.03, "rsi_14": 50.0, "vol_ratio": 1.0})
    d = pm.decide([Signal("trend", "long", 1.0)], row)
    assert d.target_exposure > 0 and pm.llm_failures == 1
    pm2 = DebatePortfolioManager(llm=CallableLLM(lambda s, p: '{"exposure": 7, "reason": "yolo"}'))
    assert pm2.decide([Signal("trend", "long", 1.0)], row).target_exposure == 1.0  # clamped


def test_risk_committee_votes_and_hard_limits():
    rc = RiskCommittee()
    long = Decision(1.0, (Signal("news", "short", 0.9),))
    calm = pd.Series({"vol_ratio": 1.0})
    spike = pd.Series({"vol_ratio": 2.0})
    normal = rc.review(long, realised_vol=0.15, drawdown=0.0, row=calm).target_exposure
    shocked = rc.review(long, realised_vol=0.15, drawdown=-0.10, row=spike).target_exposure
    assert normal == pytest.approx(1.0) and shocked == pytest.approx(0.25)  # conservative wins in drawdown
    assert rc.votes[-1]["rule"].startswith("conservative")
    big = rc.review(Decision(1.0), realised_vol=0.01, drawdown=0.0, row=calm).target_exposure
    assert big <= rc.base.max_leverage


def test_full_committee_backtest_has_no_lookahead(market):
    p, news = market

    def run(prices, store):
        team = [TrendAgent(), MeanReversionAgent(), LLMAnalystAgent(HeuristicLLM()), NewsAnalystAgent(store)]
        return run_backtest(prices, team, pm=DebatePortfolioManager(), risk=RiskCommittee()).log["position"]

    base = run(p, news)
    cut = 250
    t_cut = p.index[cut] + pd.Timedelta(hours=16)
    tampered_prices = p.copy()
    tampered_prices.iloc[cut + 1:, 0] *= 1.5
    future_news = NewsStore([h for h in news.items if h.published <= t_cut] +
                            [Headline(t_cut + pd.Timedelta(days=d, hours=1), "Company record demand") for d in range(1, 30)])
    alt = run(tampered_prices, future_news)
    pd.testing.assert_series_equal(base.iloc[: cut + 1], alt.iloc[: cut + 1])
