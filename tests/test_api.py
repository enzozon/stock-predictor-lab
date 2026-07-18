import pytest
from fastapi.testclient import TestClient

from api.main import app, get_db


@pytest.fixture
def client(conn):
    app.dependency_overrides[get_db] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()


def _seed(conn):
    with conn:
        conn.executemany(
            "INSERT INTO predictions (ticker, date, model, score) VALUES (?, ?, 'gbdt', ?)",
            [("AAA", "2024-01-10", 0.7), ("BBB", "2024-01-10", 0.4), ("CCC", "2024-01-10", 0.9)],
        )
        conn.execute(
            "INSERT INTO trades (date, ticker, side, qty, price, reason) "
            "VALUES ('2024-01-10', 'CCC', 'BUY', 100, 10.0, 'test')"
        )
        conn.execute("INSERT INTO positions (ticker, qty, avg_price) VALUES ('CCC', 100, 10.0)")
        conn.executemany(
            "INSERT INTO portfolio_snapshots (date, cash, equity, daily_pnl) VALUES (?, ?, ?, ?)",
            [
                ("2024-01-08", 100000, 100000, 0),
                ("2024-01-09", 99000, 100500, 500),
                ("2024-01-10", 99000, 101000, 500),
            ],
        )


def test_health(client):
    body = client.get("/health").json()
    assert body == {"success": True, "data": {"status": "ok"}, "error": None}


def test_ranking_sorted_by_score(client, conn):
    _seed(conn)
    body = client.get("/ranking").json()
    assert body["success"]
    scores = [row["score"] for row in body["data"]]
    assert scores == sorted(scores, reverse=True)
    assert body["data"][0]["ticker"] == "CCC"


def test_ranking_without_predictions_returns_error(client):
    body = client.get("/ranking").json()
    assert not body["success"]
    assert "sem predições" in body["error"]


def test_predictions_endpoint_and_validation(client, conn):
    _seed(conn)
    body = client.get("/predictions/AAA").json()
    assert body["success"] and len(body["data"]) == 1

    assert client.get("/predictions/AAA?limit=0").status_code == 422
    assert client.get("/predictions/aaa%3Bdrop").status_code == 422  # ticker inválido


def test_portfolio_and_trades(client, conn):
    _seed(conn)
    portfolio = client.get("/portfolio").json()["data"]
    assert portfolio["positions"][0]["ticker"] == "CCC"
    assert portfolio["last_snapshot"]["date"] == "2024-01-10"

    trades = client.get("/trades").json()["data"]
    assert trades[0]["side"] == "BUY"


def test_performance_requires_snapshots(client, conn):
    body = client.get("/performance").json()
    assert not body["success"]

    _seed(conn)
    body = client.get("/performance").json()
    assert body["success"]
    assert set(body["data"]["metrics"]) >= {"sharpe", "sortino", "max_drawdown", "cagr"}
