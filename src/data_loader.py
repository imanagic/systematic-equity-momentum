from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf


def load_adj_close(
    tickers: List[str],
    benchmark: str,
    start: str,
    end: Optional[str],
    cache_path: str = "data/adj_close.csv",
) -> pd.DataFrame:
    """
    Download Adj Close prices for tickers + benchmark and cache to CSV.
    """
    cache_file = Path(cache_path)

    # Use cached data if it exists and contains needed symbols
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        needed = set(tickers + [benchmark])
        if needed.issubset(df.columns):
            return df.sort_index()

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    symbols = tickers + [benchmark]
    data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False)

    if "Adj Close" not in data.columns:
        raise ValueError("yfinance output missing 'Adj Close'. Try upgrading yfinance.")

    adj = data["Adj Close"].dropna(how="all").sort_index()
    adj.to_csv(cache_file)

    return adj
