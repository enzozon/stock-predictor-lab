# stock-predictor-lab

Laboratório **educacional** de predição de tendência de ações brasileiras com
backtesting walk-forward rigoroso e bot de **paper trading** (simulação — nunca
executa ordens reais). Leia o [DISCLAIMER](DISCLAIMER.md) antes de qualquer coisa.

## Motivação

Predição de preço de ações é um problema quase impossível — e é exatamente por
isso que é um ótimo exercício de engenharia: obriga a lidar com vazamento de
dados futuros, validação temporal, métricas de risco e guardrails de execução.
O objetivo deste projeto **não** é "ganhar dinheiro", e sim demonstrar o
pipeline completo com a honestidade metodológica que a maioria dos tutoriais
de "IA para trading" ignora.

## Arquitetura

```
cli.py  ──►  data/      ingestão yfinance ──► cache SQLite (prices)
             models/    features point-in-time + 2 modelos (logistic, gbdt)
             backtest/  engine walk-forward + métricas de risco
             bot/       paper trading: score ► guardrails ► ordens ► snapshot
             api/       FastAPI somente leitura sobre o SQLite
             db/        schema único: prices, predictions, trades,
                        positions, portfolio_snapshots
```

Fluxo de decisão (idêntico no backtest e no bot):

1. **Features** em `t`: retornos (1/5/21d), preço vs. SMA20, RSI14,
   volatilidade 21d, z-score de volume — todas com janelas *apenas passadas*.
2. **Rótulo**: direção do retorno nos próximos 5 pregões (`fwd_ret > 0`).
3. **Anti-lookahead**: o rótulo de `t` só é conhecido em `t+5`; o corte de
   treino (`models/features.py:train_cutoff`) garante que toda linha de treino
   tem rótulo **totalmente realizado antes** da data da decisão. Há teste de
   regressão que altera o futuro e verifica que o passado não muda.
4. **Modelos comparados**: Regressão Logística (baseline) vs.
   HistGradientBoosting (sklearn). Sem LSTM: em dados diários com ~2.100
   amostras por ativo, redes recorrentes só adicionam custo e variância.
5. **Re-treino do zero a cada decisão** — sem modelo serializado, o que
   elimina por construção o risco de usar um modelo treinado com dados futuros.

## Schema de dados (SQLite)

| Tabela | Colunas | Papel |
|---|---|---|
| `prices` | ticker, date, OHLCV | cache do histórico (PK ticker+date, idempotente) |
| `predictions` | ticker, date, model, score, features_json | auditoria de cada score |
| `trades` | date, ticker, side, qty, price, reason | ordens simuladas com motivo |
| `positions` | ticker, qty, avg_price | carteira atual |
| `portfolio_snapshots` | date, cash, equity, daily_pnl | curva de equity do bot |

## Resultados do backtest (dados reais, 2019-02-11 → 2026-07-17)

Universo: 8 blue chips da B3 · rebalanceamento mensal · top-3 por score ·
custo de 10 bps por rebalanceamento · re-treino walk-forward.

| Estratégia | Sharpe | Sortino | Max DD | CAGR | Win rate | Retorno total |
|---|---|---|---|---|---|---|
| logistic | 0.55 | 0.70 | −54% | 11.9% | 52.4% | +138% |
| gbdt | 0.57 | 0.74 | −50% | 12.3% | 51.6% | +143% |
| **buy-and-hold (8 ativos)** | **0.70** | **0.89** | −42% | **14.8%** | 52.1% | **+183%** |
| benchmark (^BVSP) | 0.48 | 0.59 | −47% | 8.4% | 51.6% | +84% |

**Leitura honesta:** os dois modelos batem o IBOV, mas **perdem para o
buy-and-hold igualitário do próprio universo**. Ou seja: no período testado, o
"alfa" dos modelos não paga nem o custo da rotatividade — a maior parte do
retorno vem da seleção do universo, não do modelo. Esse é o resultado típico
(e esperado) de features técnicas simples em dados diários, e é publicado aqui
exatamente por isso.

## Bot de paper trading

- Capital fictício inicial: R$100.000. Sem qualquer integração com corretora.
- Ciclo (`python cli.py bot`): score point-in-time → guardrails → ordens a
  preço de fechamento → snapshot em SQLite.
- **Guardrails**:
  - *Circuit breaker*: perda diária simulada > 3% bloqueia todas as ordens do dia;
  - Teto de exposição por ativo (25%) e por caixa disponível;
  - Ordens validadas com Pydantic (side, qty>0, price>0, ticker sanitizado);
  - Oversell e compra sem caixa levantam erro antes de tocar o banco;
  - Toda decisão (inclusive "não comprar") vira log JSON com features, score e motivo.

## API

`python cli.py serve` → `http://127.0.0.1:8000/docs`

| Endpoint | Retorna |
|---|---|
| `GET /ranking` | ranking mais recente por score, com features de auditoria |
| `GET /predictions/{ticker}` | histórico de predições |
| `GET /trades` | ordens simuladas |
| `GET /portfolio` | posições + último snapshot |
| `GET /performance` | Sharpe, Sortino, max drawdown, CAGR sobre a equity do bot |

Envelope padrão: `{"success": bool, "data": ..., "error": ...}`.

## Como rodar

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e . --group dev
python cli.py ingest      # baixa ~8 anos de dados da B3 via yfinance
python cli.py backtest    # compara logistic vs gbdt vs buy-and-hold vs IBOV
python cli.py bot         # um ciclo de paper trading (data mais recente)
python cli.py serve       # API em localhost:8000
pytest                    # 38 testes
```

Configuração via variáveis de ambiente com prefixo `SPL_` (ver `config.py`):
`SPL_TICKERS`, `SPL_TOP_N`, `SPL_DAILY_LOSS_LIMIT_PCT` etc.

## Limitações conhecidas

- **Features técnicas simples** não carregam informação fundamentalista ou de
  fluxo; o resultado do backtest mostra que o alfa é marginal.
- Custo de transação modelado como taxa fixa por rebalanceamento
  (não modela slippage nem book).
- Execução simulada ao **fechamento do próprio dia** da decisão — otimista;
  o conservador seria a abertura do dia seguinte.
- Viés de sobrevivência no universo: os 8 tickers foram escolhidos hoje.
- yfinance não é fonte de dados de nível institucional (splits/proventos
  ajustados de forma retroativa).

## Por que paper trading (e não execução real)?

1. É um projeto educacional; o modelo demonstradamente **não** gera alfa
   confiável (ver tabela acima).
2. Execução real exige homologação de corretora, gestão de risco regulatória e
   responsabilidade fiduciária — fora do escopo por decisão explícita.
3. A simulação já exercita 100% da engenharia relevante: point-in-time,
   persistência, guardrails, observabilidade.
