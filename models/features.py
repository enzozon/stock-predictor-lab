"""Construção de features e rótulos sem vazamento de dados futuros.

O rótulo em t usa o retorno de t até t+horizon; portanto ele só é *conhecido*
em t+horizon. `train_cutoff` devolve o corte que garante que todo rótulo do
conjunto de treino já estava realizado antes da data de decisão.
"""

import numpy as np
import pandas as pd

from data.indicators import FEATURE_COLS, add_indicators


def build_features(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Adiciona features + fwd_ret/label. Últimas `horizon` linhas ficam com label NaN."""
    df = add_indicators(prices)
    df["fwd_ret"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = np.where(df["fwd_ret"].isna(), np.nan, (df["fwd_ret"] > 0).astype(float))
    return df


def build_panel(prices_by_ticker: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Painel long (date, ticker, features, fwd_ret, label), sem NaN nas features.

    Linhas com label NaN (rótulo ainda não realizado) são mantidas: são as
    linhas usadas para *scoring* no presente.
    """
    frames = []
    for ticker, prices in prices_by_ticker.items():
        feat = build_features(prices, horizon)
        feat["ticker"] = ticker
        frames.append(feat.reset_index(names="date"))
    panel = pd.concat(frames, ignore_index=True).dropna(subset=FEATURE_COLS)
    return panel.sort_values(["date", "ticker"], ignore_index=True)


def train_cutoff(dates: list, i: int, horizon: int):
    """Data de corte para decisão em dates[i]: treinar apenas com date < corte.

    Uma linha em dates[p] tem rótulo conhecido em dates[p+horizon]; exigir
    p + horizon < i equivale a date < dates[i - horizon].
    """
    return dates[max(i - horizon, 0)]
