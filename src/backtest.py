from __future__ import annotations

import numpy as np
import pandas as pd


def _first_trading_day_each_month(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the first trading day for each calendar month in the index.
    """
    s = pd.Series(dates, index=dates)
    firsts = s.groupby([dates.year, dates.month]).min()
    return pd.DatetimeIndex(firsts.values)


def run_long_short_backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    benchmark: str = "SPY",
    quantile: float = 0.20,
    gross_exposure: float = 1.0,
    cost_per_dollar: float = 0.0005,
) -> dict:
    """
    Monthly rebalance on first trading day of each month.

    - Rank by signal
    - Long top quantile, short bottom quantile
    - Equal weight within long and short
    - Market neutral: total long = gross_exposure/2, total short = -gross_exposure/2
    - Transaction costs: turnover * cost_per_dollar (applied on effective rebalance day)

    Returns:
      daily_returns, benchmark_returns, weights, turnover, costs
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in prices columns.")

    universe_cols = [c for c in prices.columns if c != benchmark]

    # Align universe and benchmark
    px = prices[universe_cols].dropna(how="all").sort_index()
    bench_px = prices[benchmark].reindex(px.index).dropna()

    px = px.loc[bench_px.index]
    sig = signal[universe_cols].reindex(px.index)

    asset_rets = px.pct_change().fillna(0.0)
    bench_rets = bench_px.pct_change().fillna(0.0)

    rebalance_dates = _first_trading_day_each_month(px.index)

    weights = pd.DataFrame(0.0, index=px.index, columns=universe_cols)
    prev_w = pd.Series(0.0, index=universe_cols)

    turnover_series = pd.Series(0.0, index=px.index)

    for d in rebalance_dates:
        if d not in px.index:
            continue

        # Apply new weights starting next trading day
        loc = px.index.get_loc(d)
        if loc + 1 >= len(px.index):
            continue
        effective_day = px.index[loc + 1]

        s = sig.loc[d].dropna()
        if len(s) < 5:
            continue

        n = len(s)
        k = max(1, int(np.floor(n * quantile)))

        ranked = s.sort_values(ascending=True)
        short_names = ranked.index[:k]
        long_names = ranked.index[-k:]

        long_w = (gross_exposure / 2.0) / k
        short_w = -(gross_exposure / 2.0) / k

        new_w = pd.Series(0.0, index=universe_cols)
        new_w.loc[long_names] = long_w
        new_w.loc[short_names] = short_w

        turnover = float((new_w - prev_w).abs().sum())
        turnover_series.loc[effective_day] = turnover

        weights.loc[effective_day] = new_w
        prev_w = new_w

    # Forward fill weights
    weights = weights.replace(0.0, np.nan).ffill().fillna(0.0)

    # Use yesterday's weights for today's return
    port_rets = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1)

    costs = turnover_series * cost_per_dollar
    port_rets = port_rets - costs

    return {
        "daily_returns": port_rets,
        "benchmark_returns": bench_rets,
        "weights": weights,
        "turnover": turnover_series,
        "costs": costs,
    }
