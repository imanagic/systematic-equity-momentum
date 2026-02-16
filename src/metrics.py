from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRADING_DAYS = 252


def annualized_return(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    growth = (1.0 + r).prod()
    years = len(r) / TRADING_DAYS
    if years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = daily_returns.dropna() - rf_daily
    if r.empty:
        return float("nan")
    vol = r.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float((r.mean() / vol) * np.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())  # negative value, e.g., -0.25


def alpha_beta_tstat(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_daily: float = 0.0,
) -> dict:
    """
    Regression: (rp - rf) = alpha + beta*(rb - rf) + eps
    Returns: alpha_annualized, beta, alpha_tstat, r2
    """
    rp = portfolio_returns.dropna()
    rb = benchmark_returns.reindex(rp.index).dropna()

    common = rp.index.intersection(rb.index)
    rp = rp.loc[common] - rf_daily
    rb = rb.loc[common] - rf_daily

    if len(common) < 50:
        return {
            "alpha_annualized": float("nan"),
            "beta": float("nan"),
            "alpha_tstat": float("nan"),
            "r2": float("nan"),
        }

    X = sm.add_constant(rb.values)
    y = rp.values
    model = sm.OLS(y, X).fit()

    alpha_daily = float(model.params[0])
    beta = float(model.params[1])
    alpha_t = float(model.tvalues[0])
    r2 = float(model.rsquared)

    return {
        "alpha_annualized": alpha_daily * TRADING_DAYS,
        "beta": beta,
        "alpha_tstat": alpha_t,
        "r2": r2,
    }


def summary_table(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    out = {
        "ann_return": annualized_return(portfolio_returns),
        "ann_vol": annualized_volatility(portfolio_returns),
        "sharpe": sharpe_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
    }
    out.update(alpha_beta_tstat(portfolio_returns, benchmark_returns))
    return pd.DataFrame([out])
