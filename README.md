# Cross-Sectional Equity Factor Model

A five-signal equity research pipeline with cross-sectional normalization, information-coefficient analysis, portfolio construction, transaction costs, and walk-forward evaluation.

## Signals

- Momentum: 12-1 price return
- Low volatility: negative 252-day realized volatility
- Size: negative log market capitalization
- Value: price-reversal proxy
- Quality: negative idiosyncratic volatility

Signals are winsorized, standardized, and combined into a composite score.

## Verified result

The committed [`results/`](results/README.md) bundle comes from the deterministic offline pipeline:

- CAGR: 11.06%
- Annualized volatility: 3.24%
- Sharpe: 2.021
- Maximum drawdown: -3.15%
- Information ratio: 0.464

Four factor/horizon rows are marked significant after Bonferroni correction: low volatility at 126 days, quality at 63 and 126 days, and the composite at 126 days. Because prices and market capitalizations are synthetic, these metrics are execution evidence, not historical or live-market alpha.

## Run

```bash
python -m pip install -r requirements.txt

# Deterministic and offline
python run_model.py --offline

# Public-market download through yfinance
python run_model.py

python -m pytest -q
```

Generated working files are written to `outputs/`; the compact, reviewed evidence is committed under [`results/`](results/README.md).

## Structure

```text
run_model.py
src/factors/
  universe.py
  signals.py
  scoring.py
  ic_analysis.py
  optimizer.py
  backtest.py
  charts.py
tests/
results/
```

Python 3.10+ is recommended. This software is research code, not investment advice.
