import pytest
from pydantic import ValidationError

from bot.paper_trader import Order, PaperTrader
from config import Settings
from data.ingest import save_prices
from tests.conftest import make_prices


def _settings() -> Settings:
    return Settings(
        tickers=["WINNER", "FLAT", "LOSER"],
        min_train_days=100,
        horizon_days=5,
        top_n=2,
        buy_threshold=0.0,  # garante compras no cenário sintético
        initial_cash=100_000.0,
    )


@pytest.fixture
def seeded(conn):
    save_prices(conn, "WINNER", make_prices(200, drift=0.004, vol=0.005, seed=1))
    save_prices(conn, "FLAT", make_prices(200, drift=0.0, vol=0.01, seed=2))
    save_prices(conn, "LOSER", make_prices(200, drift=-0.003, vol=0.01, seed=3))
    return conn


def test_order_validation():
    with pytest.raises(ValidationError):
        Order(ticker="", side="BUY", qty=10, price=10.0, reason="x")
    with pytest.raises(ValidationError):
        Order(ticker="PETR4.SA", side="BUY", qty=0, price=10.0, reason="x")
    with pytest.raises(ValidationError):
        Order(ticker="PETR4.SA", side="SHORT", qty=10, price=10.0, reason="x")
    with pytest.raises(ValidationError):
        Order(ticker="petr4; DROP TABLE trades", side="BUY", qty=1, price=1.0, reason="x")


def test_run_once_creates_trades_predictions_snapshot(seeded):
    trader = PaperTrader(seeded, _settings(), model_name="logistic")
    result = trader.run_once()

    assert not result["halted"]
    assert any(o["side"] == "BUY" for o in result["orders"])
    assert seeded.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 3
    assert seeded.execute("SELECT COUNT(*) FROM trades").fetchone()[0] >= 1
    assert seeded.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0] == 1
    # conservação de valor: caixa + posições ~= capital inicial
    assert abs(result["equity"] - 100_000.0) < 1.0
    assert result["cash"] >= 0


def test_position_cap_respected(seeded):
    settings = _settings()
    trader = PaperTrader(seeded, settings, model_name="logistic")
    result = trader.run_once()
    for order in result["orders"]:
        if order["side"] == "BUY":
            exposure = order["qty"] * order["price"] / result["equity"]
            assert exposure <= max(1 / settings.top_n, settings.max_position_pct) + 0.01


def test_circuit_breaker_halts_trading(seeded):
    trader = PaperTrader(seeded, _settings(), model_name="logistic")
    # ontem: 100k em caixa + 900k em posições (que não existem mais na tabela)
    # -> equity hoje = só o caixa = queda de ~90% -> circuit breaker
    seeded.execute(
        "INSERT INTO portfolio_snapshots (date, cash, equity, daily_pnl) "
        "VALUES ('2000-01-01', 100000, 1000000, 0)"
    )
    result = trader.run_once()
    assert result["halted"]
    assert result["orders"] == []
    assert seeded.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_execute_sell_updates_position_and_cash(seeded):
    trader = PaperTrader(seeded, _settings())
    cash = trader._execute(
        [Order(ticker="WINNER", side="BUY", qty=10, price=100.0, reason="t")],
        10_000.0, "2020-06-01",
    )
    assert cash == 9_000.0
    cash = trader._execute(
        [Order(ticker="WINNER", side="SELL", qty=4, price=110.0, reason="t")],
        cash, "2020-06-02",
    )
    assert cash == 9_440.0
    pos = seeded.execute("SELECT qty FROM positions WHERE ticker='WINNER'").fetchone()
    assert pos["qty"] == 6


def test_execute_rejects_oversell(seeded):
    trader = PaperTrader(seeded, _settings())
    with pytest.raises(ValueError):
        trader._execute(
            [Order(ticker="WINNER", side="SELL", qty=1, price=10.0, reason="t")],
            1000.0, "2020-06-01",
        )


def test_run_once_insufficient_history(conn):
    save_prices(conn, "WINNER", make_prices(50))
    settings = _settings()
    trader = PaperTrader(conn, settings)
    with pytest.raises(ValueError):
        trader.run_once()
