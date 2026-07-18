"""Configuração central do projeto via pydantic-settings (prefixo de env: SPL_)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPL_")

    db_path: Path = Path("stock_lab.db")
    tickers: list[str] = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA",
        "WEGE3.SA", "ABEV3.SA", "B3SA3.SA", "BBAS3.SA",
    ]
    benchmark: str = "^BVSP"
    start_date: str = "2018-01-01"

    # Modelo / features
    horizon_days: int = 5           # horizonte do rótulo: retorno futuro de N pregões
    buy_threshold: float = 0.5      # score mínimo (prob. de alta) para entrar

    # Backtest
    min_train_days: int = 252       # histórico mínimo antes da 1ª decisão
    rebalance_step_days: int = 21   # rebalanceia ~1x/mês
    top_n: int = 3
    cost_bps: float = 10.0          # custo por rebalanceamento, em basis points

    # Paper trading / guardrails
    initial_cash: float = 100_000.0
    max_position_pct: float = 0.25          # teto de exposição por ativo
    daily_loss_limit_pct: float = 0.03      # circuit breaker: stop diário de perda


settings = Settings()
