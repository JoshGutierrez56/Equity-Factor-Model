# Evidence bundles

## Primary: real WRDS research

[`wrds_real_data/`](wrds_real_data/README.md) contains the reviewed aggregate
evidence from CRSP/Compustat. It includes the frozen specification, private-input
hash manifest, coverage, IC inference, cost-aware portfolio summaries, and
portfolio-level monthly returns. It contains no licensed security-level rows.

## Follow-up: locked portfolio engineering

[`portfolio_engineering/`](portfolio_engineering/README.md) preserves the
version-2 baseline and evaluates a separately frozen risk-, constraint-,
turnover-, and volatility-managed implementation. The primary 10-bps Sharpe
moved from 0.466 to 0.472, while turnover and drawdown improved. The paired
block-bootstrap interval includes zero, so the bundle reports an implementation
improvement that is not statistically confirmed rather than an alpha success.

## Legacy: deterministic synthetic validation

The `synthetic_*` files were generated with:

```bash
python run_model.py --offline
```

They record a synthetic-universe Sharpe of 2.021 and several apparently
significant factor/horizon rows. Those figures verify execution and analysis
plumbing only. They are not historical-market evidence and should not be used as
resume performance claims.
