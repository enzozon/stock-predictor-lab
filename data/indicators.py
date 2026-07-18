"""Indicadores técnicos calculados apenas com janelas passadas (point-in-time correto).

Todas as funções usam rolling/ewm do pandas, que por construção só olham para trás.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = ["ret_1", "ret_5", "ret_21", "sma_ratio", "rsi_14", "vol_21", "volume_z"]


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = gain / loss  # loss==0 -> inf -> RSI 100; 0/0 -> NaN (warmup)
    return 100 - 100 / (1 + rs)


def add_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorna cópia de `prices` (colunas close/volume) com as colunas de FEATURE_COLS."""
    if not {"close", "volume"}.issubset(prices.columns):
        raise ValueError("prices precisa das colunas 'close' e 'volume'")
    df = prices.copy()
    close = df["close"]
    df["ret_1"] = close.pct_change()
    df["ret_5"] = close.pct_change(5)
    df["ret_21"] = close.pct_change(21)
    df["sma_ratio"] = close / sma(close, 20) - 1
    df["rsi_14"] = rsi(close, 14)
    df["vol_21"] = df["ret_1"].rolling(21).std()
    vol_mean = df["volume"].rolling(21).mean()
    vol_std = df["volume"].rolling(21).std()
    df["volume_z"] = (df["volume"] - vol_mean) / vol_std
    return df
