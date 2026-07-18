"""Métricas de performance e risco a partir de retornos diários."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


EPS = 1e-12  # std de série constante retorna ~1e-18 por erro de ponto flutuante


def sharpe(daily_returns: pd.Series) -> float:
    std = daily_returns.std()
    if np.isnan(std) or std < EPS:
        return 0.0
    return float(daily_returns.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    dstd = downside.std()
    if len(downside) < 2 or np.isnan(dstd) or dstd < EPS:
        return 0.0
    return float(daily_returns.mean() / dstd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Pior queda em relação ao pico anterior (número negativo, ex.: -0.35)."""
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (TRADING_DAYS / (len(equity) - 1)) - 1)


def win_rate(daily_returns: pd.Series) -> float:
    active = daily_returns[daily_returns != 0]
    if active.empty:
        return 0.0
    return float((active > 0).mean())


def compute_metrics(daily_returns: pd.Series) -> dict[str, float]:
    equity = (1 + daily_returns).cumprod()
    return {
        "sharpe": sharpe(daily_returns),
        "sortino": sortino(daily_returns),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "win_rate": win_rate(daily_returns),
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
    }
