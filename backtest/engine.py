"""Backtest walk-forward: re-treina o modelo a cada rebalanceamento usando
apenas rótulos já realizados, escolhe o top-N por score e segura até o próximo
rebalanceamento. Sem vazamento de dados futuros por construção.
"""

from dataclasses import dataclass, field

import pandas as pd

from data.indicators import FEATURE_COLS
from logging_utils import get_logger, log_decision
from models.features import train_cutoff
from models.train import fit_predict_scores

logger = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    model: str
    daily_returns: pd.Series
    equity: pd.Series
    metrics: dict[str, float]
    picks: pd.DataFrame = field(default_factory=pd.DataFrame)


def walk_forward(
    panel: pd.DataFrame,
    model_name: str,
    horizon: int,
    min_train: int = 252,
    step: int = 21,
    top_n: int = 3,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """`panel` vem de models.features.build_panel (long: date, ticker, features, label)."""
    from backtest.metrics import compute_metrics

    dates = sorted(panel["date"].unique())
    if len(dates) <= min_train + 1:
        raise ValueError(f"histórico insuficiente: {len(dates)} datas, mínimo {min_train + 2}")

    close_wide = panel.pivot(index="date", columns="ticker", values="close")
    rets_wide = close_wide.pct_change()

    window_returns: list[pd.Series] = []
    picks_rows: list[dict] = []

    for i in range(min_train, len(dates) - 1, step):
        decision_date = dates[i]
        cutoff = train_cutoff(dates, i, horizon)
        train = panel[(panel["date"] < cutoff) & panel["label"].notna()]
        today = panel[panel["date"] == decision_date]
        if train.empty or today.empty:
            continue

        scores = fit_predict_scores(
            model_name, train[FEATURE_COLS], train["label"], today[FEATURE_COLS]
        )
        ranked = today.assign(score=scores).nlargest(top_n, "score")
        chosen = list(ranked["ticker"])
        for _, row in ranked.iterrows():
            picks_rows.append(
                {"date": decision_date, "ticker": row["ticker"], "score": row["score"]}
            )
        log_decision(
            logger, "rebalance", model=model_name, date=str(decision_date),
            chosen=chosen, train_rows=len(train),
        )

        hold_dates = dates[i + 1 : min(i + step, len(dates) - 1) + 1]
        rets = rets_wide.loc[hold_dates, chosen].mean(axis=1).fillna(0.0)
        # ponytail: custo fixo por rebalanceamento (aprox. turnover total);
        # modelar turnover real por ativo se os custos ficarem relevantes
        rets.iloc[0] -= cost_bps / 10_000
        window_returns.append(rets)

    daily = pd.concat(window_returns)
    equity = (1 + daily).cumprod()
    return BacktestResult(
        model=model_name,
        daily_returns=daily,
        equity=equity,
        metrics=compute_metrics(daily),
        picks=pd.DataFrame(picks_rows),
    )


def buy_and_hold(panel: pd.DataFrame, from_date) -> BacktestResult:
    """Benchmark interno: carteira igualitária com todos os tickers do painel."""
    from backtest.metrics import compute_metrics

    close_wide = panel.pivot(index="date", columns="ticker", values="close")
    rets = close_wide.pct_change().mean(axis=1)
    rets = rets[rets.index > from_date].fillna(0.0)
    equity = (1 + rets).cumprod()
    return BacktestResult(
        model="buy_and_hold", daily_returns=rets, equity=equity,
        metrics=compute_metrics(rets),
    )


def benchmark_result(benchmark_prices: pd.DataFrame, from_date) -> BacktestResult:
    """Benchmark externo (ex.: ^BVSP) a partir de um DataFrame OHLCV."""
    from backtest.metrics import compute_metrics

    rets = benchmark_prices["close"].pct_change()
    rets = rets[rets.index > from_date].fillna(0.0)
    equity = (1 + rets).cumprod()
    return BacktestResult(
        model="benchmark", daily_returns=rets, equity=equity,
        metrics=compute_metrics(rets),
    )
