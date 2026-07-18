"""Bot de paper trading: pontua o universo com o modelo, aplica guardrails e
simula ordens a preço de fechamento, persistindo tudo em SQLite.

Nunca envia ordens reais — não há integração com corretora.
"""

import json
import sqlite3
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from config import Settings
from data.indicators import FEATURE_COLS
from data.ingest import load_prices
from logging_utils import get_logger, log_decision
from models.features import build_panel, train_cutoff
from models.train import fit_predict_scores

logger = get_logger("bot.paper_trader")


class Order(BaseModel):
    """Ordem simulada, validada antes de qualquer efeito no banco."""

    ticker: str = Field(min_length=1, pattern=r"^[A-Z0-9.^\-]+$")
    side: Literal["BUY", "SELL"]
    qty: int = Field(gt=0)
    price: float = Field(gt=0)
    reason: str


class PaperTrader:
    def __init__(self, conn: sqlite3.Connection, settings: Settings, model_name: str = "gbdt"):
        self.conn = conn
        self.settings = settings
        self.model_name = model_name

    # ---------- estado persistido ----------

    def _positions(self) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute("SELECT ticker, qty, avg_price FROM positions").fetchall()
        return {r["ticker"]: r for r in rows}

    def _last_snapshot_before(self, date_iso: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT date, cash, equity FROM portfolio_snapshots WHERE date < ? "
            "ORDER BY date DESC LIMIT 1",
            (date_iso,),
        ).fetchone()

    def _cash(self) -> float:
        row = self.conn.execute(
            "SELECT cash FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return row["cash"] if row else self.settings.initial_cash

    # ---------- execução simulada ----------

    def _execute(self, orders: list[Order], cash: float, date_iso: str) -> float:
        """Aplica ordens validadas: registra trade e atualiza posição/caixa."""
        for order in orders:
            positions = self._positions()
            if order.side == "SELL":
                pos = positions.get(order.ticker)
                if pos is None or pos["qty"] < order.qty:
                    raise ValueError(f"venda de {order.ticker} sem posição suficiente")
            if order.side == "BUY" and order.qty * order.price > cash + 1e-6:
                raise ValueError(f"compra de {order.ticker} excede o caixa disponível")

            with self.conn:
                self.conn.execute(
                    "INSERT INTO trades (date, ticker, side, qty, price, reason) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (date_iso, order.ticker, order.side, order.qty, order.price, order.reason),
                )
                if order.side == "BUY":
                    pos = positions.get(order.ticker)
                    old_qty = pos["qty"] if pos else 0
                    old_avg = pos["avg_price"] if pos else 0.0
                    new_qty = old_qty + order.qty
                    new_avg = (old_qty * old_avg + order.qty * order.price) / new_qty
                    self.conn.execute(
                        "INSERT OR REPLACE INTO positions (ticker, qty, avg_price) VALUES (?, ?, ?)",
                        (order.ticker, new_qty, new_avg),
                    )
                    cash -= order.qty * order.price
                else:
                    pos = positions[order.ticker]
                    remaining = pos["qty"] - order.qty
                    if remaining == 0:
                        self.conn.execute("DELETE FROM positions WHERE ticker = ?", (order.ticker,))
                    else:
                        self.conn.execute(
                            "UPDATE positions SET qty = ? WHERE ticker = ?",
                            (remaining, order.ticker),
                        )
                    cash += order.qty * order.price
        return cash

    def _save_predictions(self, today: pd.DataFrame, date_iso: str) -> None:
        with self.conn:
            for _, row in today.iterrows():
                self.conn.execute(
                    "INSERT OR REPLACE INTO predictions (ticker, date, model, score, features_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        row["ticker"], date_iso, self.model_name, float(row["score"]),
                        json.dumps({c: float(row[c]) for c in FEATURE_COLS}),
                    ),
                )

    # ---------- loop de decisão ----------

    def run_once(self, as_of: str | None = None) -> dict:
        """Um ciclo completo: score -> guardrails -> ordens -> snapshot."""
        s = self.settings
        prices = {t: df for t in s.tickers if not (df := load_prices(self.conn, t)).empty}
        if not prices:
            raise ValueError("sem preços no banco; rode a ingestão antes")

        panel = build_panel(prices, s.horizon_days)
        dates = sorted(panel["date"].unique())
        as_of_ts = pd.Timestamp(as_of) if as_of else dates[-1]
        if as_of_ts not in dates:
            raise ValueError(f"data {as_of_ts.date()} não existe no histórico")
        i = dates.index(as_of_ts)
        if i < s.min_train_days:
            raise ValueError(f"histórico insuficiente até {as_of_ts.date()} "
                             f"({i} datas, mínimo {s.min_train_days})")
        date_iso = as_of_ts.strftime("%Y-%m-%d")

        # 1. score point-in-time (mesma regra anti-lookahead do backtest)
        cutoff = train_cutoff(dates, i, s.horizon_days)
        train = panel[(panel["date"] < cutoff) & panel["label"].notna()]
        today = panel[panel["date"] == as_of_ts]
        scores = fit_predict_scores(
            self.model_name, train[FEATURE_COLS], train["label"], today[FEATURE_COLS]
        )
        today = today.assign(score=scores)
        self._save_predictions(today, date_iso)

        close_by_ticker = dict(zip(today["ticker"], today["close"]))
        positions = self._positions()
        cash = self._cash()

        def mark_to_market(cash_now: float) -> float:
            return cash_now + sum(
                p["qty"] * close_by_ticker.get(t, p["avg_price"])
                for t, p in self._positions().items()
            )

        equity = mark_to_market(cash)

        # 2. guardrail: circuit breaker de perda diária
        last = self._last_snapshot_before(date_iso)
        halted = False
        if last is not None and last["equity"] > 0:
            daily_change = equity / last["equity"] - 1
            if daily_change < -s.daily_loss_limit_pct:
                halted = True
                log_decision(
                    logger, "circuit_breaker", date=date_iso,
                    daily_change=round(daily_change, 4), limit=-s.daily_loss_limit_pct,
                    decision="HALT", reason="stop diário de perda atingido; nenhuma ordem enviada",
                )

        # 3. ordens: sai de quem deixou o top-N, entra em quem chegou
        orders: list[Order] = []
        if not halted:
            ranked = today.nlargest(s.top_n, "score")
            target = set(ranked[ranked["score"] >= s.buy_threshold]["ticker"])

            for ticker, pos in positions.items():
                if ticker not in target and ticker in close_by_ticker:
                    orders.append(Order(
                        ticker=ticker, side="SELL", qty=pos["qty"],
                        price=close_by_ticker[ticker], reason="saiu do top-N",
                    ))

            projected_cash = cash + sum(
                o.qty * o.price for o in orders if o.side == "SELL"
            )
            budget = equity * min(1 / s.top_n, s.max_position_pct)
            for ticker in sorted(target):
                if ticker in positions:
                    continue  # ponytail: mantém posição sem rebalancear tamanho
                price = close_by_ticker[ticker]
                qty = int(min(budget, projected_cash) // price)
                if qty > 0:
                    orders.append(Order(
                        ticker=ticker, side="BUY", qty=qty, price=price,
                        reason=f"entrou no top-{s.top_n}",
                    ))
                    projected_cash -= qty * price

            for _, row in today.iterrows():
                in_target = row["ticker"] in target
                log_decision(
                    logger, "decision", date=date_iso, ticker=row["ticker"],
                    score=round(float(row["score"]), 4),
                    features={c: round(float(row[c]), 6) for c in FEATURE_COLS},
                    decision="BUY/HOLD" if in_target else "AVOID",
                    reason=("score no top-N acima do threshold" if in_target
                            else "score abaixo do top-N ou do threshold"),
                )

        cash = self._execute(orders, cash, date_iso)

        # 4. snapshot do portfólio
        equity_after = mark_to_market(cash)
        baseline = last["equity"] if last is not None else s.initial_cash
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO portfolio_snapshots (date, cash, equity, daily_pnl) "
                "VALUES (?, ?, ?, ?)",
                (date_iso, cash, equity_after, equity_after - baseline),
            )
        return {
            "date": date_iso,
            "halted": halted,
            "orders": [o.model_dump() for o in orders],
            "cash": round(cash, 2),
            "equity": round(equity_after, 2),
        }
