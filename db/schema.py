"""Schema SQLite: preços (cache), predições, trades, posições e snapshots do portfólio."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,           -- ISO YYYY-MM-DD
    open   REAL, high REAL, low REAL,
    close  REAL NOT NULL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker     TEXT NOT NULL,
    date       TEXT NOT NULL,
    model      TEXT NOT NULL,
    score      REAL NOT NULL,       -- probabilidade de alta no horizonte
    features_json TEXT,             -- features usadas na decisão (auditoria)
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE (ticker, date, model)
);

CREATE TABLE IF NOT EXISTS trades (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    date   TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side   TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    qty    INTEGER NOT NULL CHECK (qty > 0),
    price  REAL NOT NULL CHECK (price > 0),
    reason TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    ticker    TEXT PRIMARY KEY,
    qty       INTEGER NOT NULL,
    avg_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    date      TEXT PRIMARY KEY,
    cash      REAL NOT NULL,
    equity    REAL NOT NULL,        -- cash + valor de mercado das posições
    daily_pnl REAL
);
"""


def get_conn(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(SCHEMA)
