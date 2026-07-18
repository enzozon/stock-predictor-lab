import pytest

from backtest.engine import buy_and_hold, walk_forward
from models.features import build_panel
from tests.conftest import make_prices

HORIZON = 5


def _panel():
    # WINNER tem drift forte e positivo; os demais andam de lado ou caem
    return build_panel(
        {
            "WINNER": make_prices(300, drift=0.004, vol=0.005, seed=1),
            "FLAT": make_prices(300, drift=0.0, vol=0.01, seed=2),
            "LOSER": make_prices(300, drift=-0.003, vol=0.01, seed=3),
        },
        HORIZON,
    )


def test_walk_forward_runs_and_produces_equity():
    result = walk_forward(_panel(), "logistic", HORIZON, min_train=120, step=21, top_n=1)
    assert len(result.daily_returns) > 50
    assert len(result.equity) == len(result.daily_returns)
    assert not result.picks.empty
    assert set(result.metrics) >= {"sharpe", "max_drawdown", "cagr", "win_rate"}


def test_walk_forward_prefers_trending_ticker():
    result = walk_forward(_panel(), "gbdt", HORIZON, min_train=120, step=21, top_n=1)
    counts = result.picks["ticker"].value_counts()
    assert counts.idxmax() == "WINNER"


def test_walk_forward_rejects_short_history():
    with pytest.raises(ValueError):
        walk_forward(_panel(), "logistic", HORIZON, min_train=1000)


def test_picks_never_use_future_scores():
    """Cortar o painel na data de decisão não pode mudar a escolha daquela data."""
    panel = _panel()
    dates = sorted(panel["date"].unique())
    full = walk_forward(panel, "logistic", HORIZON, min_train=120, step=21, top_n=1)
    first_date = full.picks["date"].min()
    first_pick = full.picks[full.picks["date"] == first_date]["ticker"].iloc[0]

    # mantém 2 datas após a decisão só para o engine ter janela de holding
    truncated = panel[panel["date"] <= dates[122]]
    again = walk_forward(truncated, "logistic", HORIZON, min_train=120, step=21, top_n=1)
    assert again.picks[again.picks["date"] == first_date]["ticker"].iloc[0] == first_pick


def test_buy_and_hold_equity_matches_returns():
    panel = _panel()
    from_date = sorted(panel["date"].unique())[100]
    result = buy_and_hold(panel, from_date)
    assert (result.daily_returns.index > from_date).all()
    assert abs(result.equity.iloc[-1] - (1 + result.daily_returns).prod()) < 1e-9
