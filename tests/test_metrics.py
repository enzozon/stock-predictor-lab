import numpy as np
import pandas as pd

from backtest.metrics import cagr, compute_metrics, max_drawdown, sharpe, sortino, win_rate


def test_max_drawdown_known_curve():
    equity = pd.Series([1.0, 1.2, 0.6, 0.9, 1.3])
    assert max_drawdown(equity) == -0.5  # 1.2 -> 0.6


def test_cagr_doubling_in_one_year():
    equity = pd.Series(np.linspace(1.0, 2.0, 253))  # 252 retornos
    assert abs(cagr(equity) - 1.0) < 1e-9


def test_sharpe_zero_variance_is_zero():
    assert sharpe(pd.Series([0.01] * 100)) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sharpe(rets) > 0


def test_sortino_ignores_upside_volatility():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.001, 0.01, 1000))
    assert sortino(rets) > sharpe(rets)  # denominador só com downside


def test_win_rate():
    rets = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03])
    assert win_rate(rets) == 0.75  # zeros não contam


def test_compute_metrics_keys():
    rets = pd.Series([0.01, -0.005, 0.002])
    m = compute_metrics(rets)
    assert set(m) == {"sharpe", "sortino", "max_drawdown", "cagr", "win_rate", "total_return"}
