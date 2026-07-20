# Verified results

These files were generated with the repository's deterministic offline pipeline:

```bash
python run_model.py --offline
```

Recorded synthetic-universe portfolio metrics:

- CAGR: 11.06%
- Annualized volatility: 3.24%
- Sharpe ratio: 2.021
- Maximum drawdown: -3.15%
- Annualized alpha versus the synthetic benchmark: 1.24%
- Information ratio: 0.464

Four rows are marked significant after the pipeline's Bonferroni correction: low volatility at 126 days, quality at 63 and 126 days, and the composite at 126 days.

Artifacts:

- `synthetic_backtest_metrics.csv` - portfolio metrics
- `synthetic_ic_summary.csv` - factor information-coefficient evidence
- `synthetic_cumulative_nav.png` - portfolio and benchmark NAV chart

These results use generated prices and market capitalizations. They verify execution and analysis plumbing; they do not establish historical or live-market alpha.
