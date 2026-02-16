import numpy as np
import pandas as pd

from src.metrics import sharpe_ratio, max_drawdown, alpha_beta_tstat


def test_sharpe_finite():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    sr = sharpe_ratio(returns)
    assert np.isfinite(sr)


def test_max_drawdown_bounds():
    returns = pd.Series([0.01, -0.02, 0.03, -0.10, 0.02])
    mdd = max_drawdown(returns)
    assert mdd <= 0
    assert mdd >= -1


def test_alpha_beta_outputs_numbers():
    np.random.seed(1)
    rp = pd.Series(np.random.normal(0.001, 0.01, 500))
    rb = pd.Series(np.random.normal(0.0008, 0.009, 500))

    result = alpha_beta_tstat(rp, rb)

    assert np.isfinite(result["beta"])
    assert np.isfinite(result["alpha_tstat"])
