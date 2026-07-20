# Real WRDS results

This evidence bundle is generated from CRSP monthly stock data and Compustat
annual fundamentals under specification `7df8f1b088429f7aba5ff930175be9bc29a325e514fb7ba76dbaa950c0089eb7`.

The predeclared 2024+ holdout composite long-short portfolio at 10 bps recorded a 0.45 Sharpe, 1.74% CAGR, and -3.44% maximum drawdown across 11 months. 0 holdout factor/horizon tests survive Bonferroni correction. Development FF5+momentum alpha is 0.02% annualized (HAC t=0.02).

## Evidence boundary

- Universe: point-in-time NYSE/AMEX/Nasdaq common stocks, filtered to price >= $5,
  market capitalization >= $100 million, and the largest 1,000 companies monthly.
- Returns include CRSP delisting returns.
- Fundamentals are linked through CCM and delayed six months after fiscal period end.
- The composite uses fixed equal weights; no holdout return is used to tune it.
- Transaction-cost cases are 0, 10, and 25 bps per dollar traded.
- The development/holdout boundary is 2024-01-01.
- An earlier 2020–2023 engine-validation run exposed a share-class continuity issue.
  That window is part of development; corrected portfolio rules were frozen before
  the previously unexamined 2024 confirmation was opened.
- WRDS currently ends in December 2024, leaving only twelve holdout signal months
  and eleven executable portfolio months. Formal inference requires at least 24
  months, so current holdout statistics are descriptive rather than confirmatory.
- Licensed security-level rows remain under ignored `data/private/` and are not committed.

Coverage: 180 months, 2371 distinct securities,
and an average monthly universe of 1000.

## Artifacts

- `specification.json` — frozen request and model choices
- `data_manifest.json` — aggregate row counts and hashes of ignored private inputs
- `coverage.json` — aggregate coverage diagnostics
- `quality_receipt.json` — duplicate, look-ahead, universe, and return-bound gates
- `headline_summary.json` — machine-readable conclusion and composite metrics
- `ic_summary.csv` — Spearman IC, HAC inference, and Bonferroni correction
- `portfolio_summary.csv` — development, holdout, and full-period portfolio metrics
- `factor_attribution.csv` — FF5 plus momentum alpha and factor loadings
- `monthly_portfolio_returns.csv` — portfolio-level derived returns only

These are historical research results, not live performance or investment advice.
