from data.ingest import load_prices, save_prices
from tests.conftest import make_prices


def test_save_and_load_roundtrip(conn):
    df = make_prices(50)
    n = save_prices(conn, "FAKE4.SA", df)
    assert n == 50

    loaded = load_prices(conn, "FAKE4.SA")
    assert len(loaded) == 50
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert abs(loaded["close"].iloc[-1] - df["close"].iloc[-1]) < 1e-6


def test_save_is_idempotent(conn):
    df = make_prices(30)
    save_prices(conn, "FAKE4.SA", df)
    save_prices(conn, "FAKE4.SA", df)  # INSERT OR REPLACE: sem duplicatas
    assert len(load_prices(conn, "FAKE4.SA")) == 30


def test_load_unknown_ticker_returns_empty(conn):
    assert load_prices(conn, "NAOEXISTE").empty
