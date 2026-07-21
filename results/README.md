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

## Follow-up: institutional version 3

[`institutional_v3/`](institutional_v3/README.md) preserves the daily-risk,
CVXPY, capacity, market-impact, and borrow-cost experiment. Its primary
2021–2024 Sharpe was 0.372 versus 0.573 for the matched frozen baseline. The
negative result is retained rather than replaced.

## Follow-up: version-4 signal and holding buffer

[`institutional_v4/`](institutional_v4/README.md) contains the post-result,
development-only IC ensemble and 200-entry/500-exit holding-buffer experiment.
On matched 2021–2024 dates, net Sharpe/overlay IR rose to 0.694, 12-month IC to
5.60%, ICIR to 1.087, and monthly turnover fell to 0.561. The paired Sharpe-gain
interval still includes zero and FF5+momentum residual IR remains -0.024, so the
bundle is labeled a retrospective multi-metric improvement—not validated alpha.

## Legacy: deterministic synthetic validation

The `synthetic_*` files were generated with:

```bash
python run_model.py --offline
```

They record a synthetic-universe Sharpe of 2.021 and several apparently
significant factor/horizon rows. Those figures verify execution and analysis
plumbing only. They are not historical-market evidence and should not be used as
resume performance claims.
