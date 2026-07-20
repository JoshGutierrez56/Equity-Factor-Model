# Point-in-Time U.S. Equity Factor Research

A reproducible cross-sectional research platform for testing economically
motivated equity signals with CRSP, Compustat, and the CRSP/Compustat Merged
link history. The project emphasizes point-in-time controls, explicit hypothesis
families, implementation costs, and honest separation of retrospective evidence
from future validation.

## Research verdict

The 1990–2024 retrospective replication finds a statistically credible
cross-sectional composite, but not a validated standalone trading alpha.

| Primary result | Estimate |
|---|---:|
| Composite 12-month rank IC | **4.22%** |
| HAC t-statistic / raw p-value | **3.18 / 0.0014** |
| Holm-adjusted primary-family p-value | **0.0101** |
| 95% moving-block bootstrap interval | **1.85% to 6.87%** |
| Positive-IC month share | **66.7%** |
| 10-bps continuous long-short Sharpe | **0.47** |
| 10-bps continuous long-short CAGR | **4.11%** |
| Maximum drawdown | **-26.62%** |
| FF5 + momentum alpha | **1.06% annually; HAC t=1.10** |

The composite and gross-profitability quality hypotheses pass every declared
retrospective gate. The portfolio's factor-adjusted alpha does not. The correct
classification is therefore **retrospectively supported signal structure, no
validated prospective alpha**.

![Primary hypothesis IC estimates](results/wrds_real_data/primary_hypothesis_ic.png)

![Composite cost sensitivity](results/wrds_real_data/composite_cost_sensitivity.png)

The prospective ledger starts in 2025 and remains `NOT_STARTED`: WRDS supplied
data only through December 2024. At least 36 genuinely new, never-inspected
months are required before any prospective conclusion is allowed.

## Predeclared hypothesis ledger

Each signal has one primary horizon. Primary tests and exploratory horizons are
separate Holm-corrected families.

| Signal | Primary horizon | Retrospective result |
|---|---:|---|
| Fixed six-signal composite | 12 months | **Retrospectively supported** |
| Gross profitability quality | 12 months | **Retrospectively supported** |
| Low volatility | 12 months | Directional; fails quantile monotonicity |
| Intermediate momentum | 6 months | Directional; fails adjusted significance/bootstrap |
| Conservative investment | 12 months | Directional; fails adjusted significance/monotonicity |
| Book-to-market value | 12 months | Directional; fails adjusted significance/monotonicity |
| Small size | 12 months | **Expected direction rejected** |

The machine-readable gate ledger is
[`results/wrds_real_data/hypothesis_tests.csv`](results/wrds_real_data/hypothesis_tests.csv).

## Why this repository is useful

This is not a collection of optimized backtest settings. It is a compact
institutional-style research system that demonstrates:

- point-in-time CRSP security histories and PERMCO-level share-class consolidation;
- CRSP delisting-adjusted returns;
- CCM-linked Compustat annual fundamentals with preliminary/final availability
  dates when present and a conservative six-month fallback;
- an investable top-1,000-company universe with explicit price and market-cap gates;
- sector and size neutralization without return-fitted factor weights;
- 1/3/6/12-month Spearman IC, HAC inference, Holm correction, and deterministic
  moving-block bootstrap intervals;
- calendar-decade stability and quintile-monotonicity diagnostics;
- continuous-score and quintile portfolios with turnover and 0/10/25-bps costs;
- Fama–French five-factor plus momentum attribution; and
- aggregate-only public evidence, with licensed security rows kept private.

## Research design

- **Sample:** January 1990 through December 2024; 420 months, 4,439 companies,
  and a 1,000-company monthly universe.
- **Universe:** NYSE/AMEX/Nasdaq common stocks, price at least $5, market
  capitalization at least $100 million, largest 1,000 eligible companies monthly.
- **Entity:** CRSP `PERMCO`; share classes use aggregate market equity and
  market-equity-weighted company returns.
- **Signals:** 12–1 momentum, low volatility, size, book-to-market value, gross
  profitability quality, and conservative asset growth.
- **Scoring:** 1st/99th percentile winsorization, SIC-sector neutralization,
  log-market-cap neutralization for non-size signals, cross-sectional z-scores,
  and a fixed equal-weight composite.
- **Primary implementation:** continuous-score dollar-neutral long-short,
  charged 10 bps per dollar traded. Quintile portfolios and alternate costs are
  comparators, not alternate headline searches.
- **Inference:** one primary horizon per hypothesis; other horizons are
  exploratory. Formal gates combine expected direction, Holm-adjusted
  significance, block-bootstrap confidence, hit rate, decade stability, and
  quintile monotonicity.

## Validation chronology

The repository does not relabel inspected data as an untouched holdout.

1. A 2010–2023 engine-validation run exposed a share-class continuity defect.
2. A short 2024 confirmation was inspected and found too short for inference.
3. Version 2 reclassifies all data through 2024 as retrospective, extends the
   replication to 1990, and locks the future-only validation protocol.
4. No parameter changes are permitted before 36 post-2024 observations exist.

This chronology is recorded in
[`specification.json`](results/wrds_real_data/specification.json) and
[`research_protocol.json`](results/wrds_real_data/research_protocol.json).

## Reproduce it

The deterministic synthetic path is credential-free and used by CI:

```bash
python -m pip install -r requirements.txt
python run_model.py --offline
python -m pytest -q
```

The real-data path requires licensed WRDS access. Credentials stay in the
user's normal local WRDS/PostgreSQL configuration and are never accepted as
command-line arguments:

```bash
python -m pip install -r requirements-live.txt
python run_wrds_model.py --scope retrospective --refresh
```

Security-level licensed rows are written only beneath ignored `data/private/`.
Committed artifacts contain aggregate statistics, portfolio-level derived
returns, coverage, quality receipts, and hashes of the private inputs.

## Evidence limitations

- Compustat `funda` is not a vintage database; preliminary/final dates and the
  fallback lag reduce obvious look-ahead but cannot reconstruct every revision.
- SIC groups and log market capitalization are transparent controls, not a
  commercial risk model.
- Linear costs exclude borrow fees, nonlinear impact, and capacity constraints.
- The continuous portfolio is dollar neutral but not explicitly beta neutral;
  realized factor exposures are reported rather than hidden.
- Retrospective support is not live performance. The prospective result remains
  unopened and unavailable.

## Repository map

```text
run_wrds_model.py            # frozen real-data study and evidence writer
src/factors/wrds_data.py     # bounded CRSP/Compustat acquisition
src/factors/real_model.py    # PIT signals, inference, portfolios, diagnostics
results/wrds_real_data/      # primary aggregate evidence and protocol
run_model.py                 # deterministic credential-free CI demonstration
tests/                       # data, inference, security, and artifact contracts
```

Python 3.10+ is recommended. This is research software, not investment advice.
