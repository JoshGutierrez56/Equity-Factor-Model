# Real WRDS results

This evidence bundle is generated from CRSP monthly stock data and Compustat
annual fundamentals under specification `0f193737dd782ea6c31fe09693910789edb99f76efc93d86ec35cae5cdbe1a5a`. All currently
available rows are retrospective. The prospective clock starts on
2025-01-01 and requires 36 never-inspected months.

The locked continuous-score composite long-short portfolio at 10 bps recorded a 0.47 Sharpe, 4.11% CAGR, and -26.62% maximum drawdown across 419 retrospective months. 2 of 7 predeclared hypotheses pass every retrospective direction, Holm, block-bootstrap, hit-rate, era-stability, and monotonicity gate. Retrospective FF5+momentum alpha is 1.06% annualized (HAC t=1.10).

## Evidence boundary

- Universe: point-in-time NYSE/AMEX/Nasdaq common stocks, filtered to price >= $5,
  market capitalization >= $100 million, and the largest 1,000 companies monthly.
- Returns include CRSP delisting returns.
- Fundamentals use Compustat preliminary/final availability dates when present
  and a conservative six-month fallback otherwise.
- The composite uses fixed equal weights; no return-fitted weights are used.
- Non-size signals are neutralized to SIC sector and log market capitalization.
- The primary implementation is a continuous-score, dollar-neutral portfolio;
  the original quintile portfolio remains a comparator.
- Transaction-cost cases are 0, 10, and 25 bps per dollar traded.
- The prospective boundary is 2025-01-01.
- Primary hypotheses use one declared horizon each and Holm correction; other
  horizons form a separately corrected exploratory family.
- Inference includes HAC tests, deterministic 12-month moving-block bootstrap
  intervals, calendar-era stability, and quintile monotonicity.
- The former 2024 holdout was inspected and is no longer called pristine. It is
  part of the retrospective audit. No prospective result exists yet.
- Licensed security-level rows remain under ignored `data/private/` and are not committed.

Coverage: 420 months, 4526 distinct securities,
and an average monthly universe of 1000.

## Artifacts

- `specification.json` — frozen request and model choices
- `data_manifest.json` — aggregate row counts and hashes of ignored private inputs
- `coverage.json` — aggregate coverage diagnostics
- `quality_receipt.json` — duplicate, look-ahead, universe, and return-bound gates
- `headline_summary.json` — machine-readable conclusion and composite metrics
- `research_protocol.json` — locked hypotheses, gates, and prospective policy
- `hypothesis_tests.csv` — primary-hypothesis pass/fail ledger
- `era_stability.csv` — decade-by-decade primary IC evidence
- `quantile_diagnostics.csv` — monotonicity and spread diagnostics
- `ic_summary.csv` — Spearman IC, HAC/bootstrap inference, and Holm correction
- `portfolio_summary.csv` — retrospective, prospective, and full-period metrics
- `factor_attribution.csv` — FF5 plus momentum alpha and factor loadings
- `monthly_portfolio_returns.csv` — portfolio-level derived returns only
- `primary_hypothesis_ic.png` — primary IC estimates and bootstrap intervals
- `composite_cost_sensitivity.png` — 0/10/25-bps cumulative portfolio evidence
- `era_stability_heatmap.png` — calendar-era primary IC stability

These are historical research results, not live performance or investment advice.
