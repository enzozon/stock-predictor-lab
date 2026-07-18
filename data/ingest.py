"""Ingestão de dados históricos via yfinance com cache em SQLite (tabela prices)."""

import sqlite3

import pandas as pd
import yfinance as yf

from logging_utils import get_logger, log_decision

logger = get_logger("data.ingest")

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def fetch_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Baixa OHLCV diário ajustado. Levanta ValueError se não houver dados."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"nenhum dado retornado para {ticker!r}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[PRICE_COLS]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(subset=["close"])


def save_prices(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> int:
    rows = [
        (ticker, idx.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close, int(r.volume))
        for idx, r in df.iterrows()
    ]
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def load_prices(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM prices "
        "WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
        index_col="date",
        parse_dates=["date"],
    )
    return df


def ingest_all(conn: sqlite3.Connection, tickers: list[str], start: str) -> dict[str, int]:
    """Baixa e persiste todos os tickers; falha em um ticker não derruba os demais."""
    counts: dict[str, int] = {}
    for ticker in tickers:
        try:
            df = fetch_prices(ticker, start)
            counts[ticker] = save_prices(conn, ticker, df)
            log_decision(logger, "ingest_ok", ticker=ticker, rows=counts[ticker])
        except Exception as exc:
            counts[ticker] = 0
            log_decision(logger, "ingest_error", ticker=ticker, error=str(exc))
    return counts
