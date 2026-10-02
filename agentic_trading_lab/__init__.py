"""agentic-trading-lab: a small multi-agent research framework for systematic trading."""
from .agents import (
    ExecutionAgent,
    LLMAnalystAgent,
    MeanReversionAgent,
    PortfolioManager,
    RiskAgent,
    TrendAgent,
)
from .backtest import BacktestResult, run_backtest
from .data import load_csv, synthetic_prices
from .llm import AnthropicLLM, CallableLLM, HeuristicLLM

__all__ = [
    "AnthropicLLM",
    "BacktestResult",
    "CallableLLM",
    "ExecutionAgent",
    "HeuristicLLM",
    "LLMAnalystAgent",
    "MeanReversionAgent",
    "PortfolioManager",
    "RiskAgent",
    "TrendAgent",
    "load_csv",
    "run_backtest",
    "synthetic_prices",
]

__version__ = "0.1.0"
