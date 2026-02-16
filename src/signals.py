import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    12-1 momentum signal:
    momentum[t] = price[t-skip] / price[t-lookback] - 1

    Using shifted prices avoids using the most recent 'skip' days.
    """
    p_skip = prices.shift(skip)
    p_lb = prices.shift(lookback)
    return (p_skip / p_lb) - 1.0
