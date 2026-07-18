"""API de leitura do laboratório: ranking, predições, trades, portfólio e performance.

Todas as respostas usam o envelope {"success": bool, "data": ..., "error": ...}.
"""

import sqlite3
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Path, Query

from backtest.metrics import compute_metrics
from config import settings
from db.schema import get_conn, init_db

app = FastAPI(
    title="stock-predictor-lab",
    description="Laboratório educacional de predição de ações. "
    "NÃO é recomendação de investimento (ver DISCLAIMER.md).",
)


def get_db():
    conn = get_conn(settings.db_path)
    init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


DB = Annotated[sqlite3.Connection, Depends(get_db)]
Ticker = Annotated[str, Path(pattern=r"^[A-Z0-9.^\-]+$", max_length=20)]
Limit = Annotated[int, Query(ge=1, le=1000)]


def envelope(data=None, error: str | None = None) -> dict:
    return {"success": error is None, "data": data, "error": error}


@app.get("/health")
def health():
    return envelope({"status": "ok"})


@app.get("/ranking")
def ranking(db: DB, model: str = "gbdt"):
    """Ranking mais recente: score do modelo por ticker, com features de auditoria."""
    last = db.execute(
        "SELECT MAX(date) AS d FROM predictions WHERE model = ?", (model,)
    ).fetchone()
    if last["d"] is None:
        return envelope(error=f"sem predições para o modelo {model!r}; rode o bot antes")
    rows = db.execute(
        "SELECT ticker, date, model, score, features_json FROM predictions "
        "WHERE date = ? AND model = ? ORDER BY score DESC",
        (last["d"], model),
    ).fetchall()
    return envelope([dict(r) for r in rows])


@app.get("/predictions/{ticker}")
def predictions(ticker: Ticker, db: DB, limit: Limit = 100):
    rows = db.execute(
        "SELECT date, model, score FROM predictions WHERE ticker = ? "
        "ORDER BY date DESC LIMIT ?",
        (ticker, limit),
    ).fetchall()
    return envelope([dict(r) for r in rows])


@app.get("/trades")
def trades(db: DB, limit: Limit = 100):
    rows = db.execute(
        "SELECT date, ticker, side, qty, price, reason FROM trades "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return envelope([dict(r) for r in rows])


@app.get("/portfolio")
def portfolio(db: DB):
    positions = [dict(r) for r in db.execute(
        "SELECT ticker, qty, avg_price FROM positions ORDER BY ticker"
    ).fetchall()]
    snapshot = db.execute(
        "SELECT date, cash, equity, daily_pnl FROM portfolio_snapshots "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone()
    return envelope({
        "positions": positions,
        "last_snapshot": dict(snapshot) if snapshot else None,
    })


@app.get("/performance")
def performance(db: DB):
    """Métricas de risco/retorno calculadas sobre os snapshots do bot."""
    rows = db.execute(
        "SELECT date, equity FROM portfolio_snapshots ORDER BY date"
    ).fetchall()
    if len(rows) < 3:
        return envelope(error="snapshots insuficientes para métricas (mínimo 3)")
    equity = pd.Series([r["equity"] for r in rows], index=[r["date"] for r in rows])
    daily_returns = equity.pct_change().dropna()
    return envelope({
        "metrics": compute_metrics(daily_returns),
        "snapshots": [dict(r) for r in rows],
    })
