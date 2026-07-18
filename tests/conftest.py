"""Fixtures compartilhadas: preços sintéticos e banco em memória."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from db.schema import init_db


def make_prices(n: int = 400, drift: float = 0.0005, vol: float = 0.01,
                seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Random walk geométrico com OHLCV plausível, indexado por dias úteis."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()
