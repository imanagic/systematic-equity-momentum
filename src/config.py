from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BacktestConfig:
    tickers: List[str]
    benchmark: str = "SPY"
    start: str = "2016-01-01"
    end: Optional[str] = None  # yfinance uses "today" when None

    lookback_days: int = 252
    skip_days: int = 21

    quantile: float = 0.20
    gross_exposure: float = 1.0

    cost_per_dollar: float = 0.0005


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "JPM", "BAC", "XOM", "CVX", "JNJ", "PG", "KO", "PEP",
    "WMT", "COST", "DIS", "UNH", "HD", "MA"
]
