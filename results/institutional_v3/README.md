# Institutional Version-3 Evidence

This directory is the separately frozen version-3 implementation experiment.
It leaves the version-2 signal and both earlier portfolio result bundles intact.

## Verdict

The pre-2020 purged development folds selected **assertive**.
In the 2021–2024 retrospective temporal assessment, the locked $100 million,
150-bp borrow scenario recorded a **0.37 Sharpe**, **2.79%
CAGR**, **-11.38% maximum drawdown**, and **0.95
average monthly turnover**.

Against the frozen 10-bps baseline on matched months, the Sharpe difference was
**-0.20** with a paired moving-block 95% interval
of **-1.62 to
0.66**. The formal classification is
**NO_ROBUST_RETROSPECTIVE_IMPROVEMENT** and remains **NO VALIDATED ALPHA**.

![Cumulative wealth](institutional_cumulative_wealth.png)

![Capacity and borrow stress](capacity_borrow_stress.png)

## What changed

- Daily CRSP returns feed a 252-day Ledoit–Wolf shrinkage covariance matrix.
- CVXPY solves a long-short portfolio with dollar, beta, sector, size, position,
  gross-exposure, and turnover controls.
- Realized costs combine lagged CRSP closing bid/ask spreads, ADV/volatility
  nonlinear impact, and explicit short-borrow stress.
- Three parameter candidates were frozen before evaluation and selected only
  from separated pre-2020 development folds. The 2020 calendar year is the
  embargo before the 2021–2024 retrospective temporal assessment.
- Alphalens-style outputs report IC decay, sector IC, score autocorrelation,
  quantile membership turnover, and liquidity buckets.

These ideas are inspired by cvxportfolio, skfolio, Alphalens Reloaded, and
PyPortfolioOpt. Their code and reported performance were not copied.

## Evidence boundary

All dates through 2024 had already been inspected before this experiment. The
2021–2024 segment is therefore a temporal comparison, not a pristine holdout.
Licensed security-level WRDS rows remain in ignored private storage. Public
artifacts contain aggregate metrics only, and this is not investment advice.

Two independent cache-only executions reproduced every public artifact byte for
byte. See `replay_receipt.json` for the recorded SHA-256 hashes.
