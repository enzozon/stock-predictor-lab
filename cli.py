"""CLI do laboratório: python cli.py {ingest,backtest,bot,serve}."""

import argparse
import json

import pandas as pd

from config import settings
from db.schema import get_conn, init_db


def _conn():
    conn = get_conn(settings.db_path)
    init_db(conn)
    return conn


def cmd_ingest(_args) -> None:
    from data.ingest import ingest_all

    counts = ingest_all(_conn(), settings.tickers + [settings.benchmark], settings.start_date)
    print(json.dumps(counts, indent=2))


def cmd_backtest(_args) -> None:
    from backtest.engine import benchmark_result, buy_and_hold, walk_forward
    from data.ingest import load_prices
    from models.features import build_panel
    from models.train import MODEL_NAMES

    conn = _conn()
    prices = {t: df for t in settings.tickers if not (df := load_prices(conn, t)).empty}
    if not prices:
        raise SystemExit("sem preços no banco; rode `python cli.py ingest` antes")
    panel = build_panel(prices, settings.horizon_days)
    from_date = sorted(panel["date"].unique())[settings.min_train_days]

    results = [
        walk_forward(
            panel, name, settings.horizon_days,
            min_train=settings.min_train_days, step=settings.rebalance_step_days,
            top_n=settings.top_n, cost_bps=settings.cost_bps,
        )
        for name in MODEL_NAMES
    ]
    results.append(buy_and_hold(panel, from_date))
    bench_prices = load_prices(conn, settings.benchmark)
    if not bench_prices.empty:
        results.append(benchmark_result(bench_prices, from_date))

    table = pd.DataFrame({r.model: r.metrics for r in results}).T
    print(f"\nBacktest walk-forward de {from_date.date()} até {panel['date'].max().date()}")
    print(table.round(3).to_string())


def cmd_bot(args) -> None:
    from bot.paper_trader import PaperTrader

    result = PaperTrader(_conn(), settings, model_name=args.model).run_once(args.as_of)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_serve(_args) -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000)


def main() -> None:
    parser = argparse.ArgumentParser(description="stock-predictor-lab (uso educacional)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="baixa e cacheia dados históricos")
    sub.add_parser("backtest", help="compara modelos vs buy-and-hold vs benchmark")
    bot = sub.add_parser("bot", help="roda um ciclo do paper trading bot")
    bot.add_argument("--model", default="gbdt", choices=["logistic", "gbdt"])
    bot.add_argument("--as-of", default=None, help="data da decisão (YYYY-MM-DD)")
    sub.add_parser("serve", help="sobe a API FastAPI em localhost:8000")

    args = parser.parse_args()
    {"ingest": cmd_ingest, "backtest": cmd_backtest, "bot": cmd_bot, "serve": cmd_serve}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
