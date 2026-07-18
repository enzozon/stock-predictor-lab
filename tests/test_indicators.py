import numpy as np
import pandas as pd
import pytest

from data.indicators import FEATURE_COLS, add_indicators, rsi, sma
from tests.conftest import make_prices


def test_sma_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(s, 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 2.0
    assert result.iloc[4] == 4.0


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1.0, 31.0))          # só sobe -> RSI 100
    assert rsi(up, 14).iloc[-1] == 100.0
    down = pd.Series(np.arange(31.0, 1.0, -1))    # só cai -> RSI 0
    assert rsi(down, 14).iloc[-1] == 0.0
    noisy = make_prices(100)["close"]
    values = rsi(noisy, 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_add_indicators_creates_all_features():
    df = add_indicators(make_prices(100))
    for col in FEATURE_COLS:
        assert col in df.columns
    # após o warmup (21 dias + pct_change), sem NaN
    assert df[FEATURE_COLS].iloc[30:].notna().all().all()


def test_add_indicators_requires_columns():
    with pytest.raises(ValueError):
        add_indicators(pd.DataFrame({"close": [1.0]}))
