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

## Locked portfolio-engineering follow-up

After freezing the baseline above, a separate protocol tested whether explicit
risk and implementation controls could convert the same composite score more
efficiently. The primary candidate was fixed before its returns were calculated:
inverse-volatility score scaling; dollar, beta, SIC-sector, and log-size
neutrality; a 1.5% position cap; partial rebalancing with a no-trade band; and
trailing volatility targeting.

| 10-bps result | Frozen baseline | Primary engineered |
|---|---:|---:|
| Sharpe | 0.466 | 0.472 |
| CAGR | 4.11% | 3.28% |
| Annualized volatility | 9.62% | 7.42% |
| Maximum drawdown | -26.62% | -23.21% |
| Average monthly turnover | 0.657 | 0.405 |

The engineered portfolio reduced turnover by 38% and improved drawdown, but its
Sharpe gain was only 0.006. A paired 12-month moving-block bootstrap interval
for that gain was -0.163 to 0.212, so the improvement is not statistically
reliable. Its retrospective FF5-plus-momentum alpha was 2.26% annually
(HAC t=2.54, p=0.011), but this follow-up was motivated after inspecting the
baseline and is not prospective evidence.

The locked risk-plus-turnover comparator reached a descriptive 0.531 Sharpe at
10 bps, but it was not the predeclared primary candidate and is not substituted
after the fact. The honest conclusion remains: better implementation quality,
no validated alpha.

![Engineered cumulative wealth](results/portfolio_engineering/engineered_cumulative_wealth.png)

![Engineered Sharpe and costs](results/portfolio_engineering/engineered_sharpe_costs.png)

The complete protocol and audit trail are in
[`results/portfolio_engineering/`](results/portfolio_engineering/README.md).

## Institutional version-3 implementation

A third, separately frozen experiment tested whether daily risk estimates and a
convex institutional implementation could improve the same unchanged composite.
The architecture uses a 252-day Ledoit-Wolf shrinkage covariance matrix, CVXPY
optimization, residualized alpha forecasts, beta/sector/size/position controls,
lagged CRSP closing-spread estimates, ADV-based nonlinear impact, explicit
borrow costs, and pre-2020 purged candidate selection.

The constraint-only feasibility audit and every pre-result protocol amendment
are preserved in [`institutional_v3_protocol.json`](institutional_v3_protocol.json).
No amendment used a forward return or performance metric.

| 2021-2024 matched result | Frozen 10-bps baseline | Institutional v3 primary |
|---|---:|---:|
| Sharpe | 0.573 | 0.372 |
| CAGR | 5.74% | 2.79% |
| Maximum drawdown | -15.28% | -11.38% |
| Average monthly turnover | 0.627 | 0.948 |

The primary scenario assumes $100 million AUM and 150 bps of annual short-borrow
cost. Its Sharpe difference versus the baseline was -0.201, with a paired
12-month moving-block 95% interval of -1.616 to 0.658. It improved drawdown and
reduced beta exposure, but higher name churn and explicit capacity/borrow costs
overwhelmed those benefits. The formal conclusion is **no robust retrospective
improvement and no validated alpha**.

This negative result is intentionally retained. It demonstrates that a more
sophisticated optimizer does not rescue a modest signal automatically and that
portfolio complexity must earn its implementation costs.

![Institutional v3 cumulative wealth](results/institutional_v3/institutional_cumulative_wealth.png)

![Institutional v3 capacity and borrow stress](results/institutional_v3/capacity_borrow_stress.png)

The complete protocol, purged-selection ledger, cost stress, constraints,
diagnostics, and aggregate monthly evidence are in
[`results/institutional_v3/`](results/institutional_v3/README.md).

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
- a separately frozen beta/sector/size-neutral, turnover-aware portfolio
  engineering experiment with paired block-bootstrap comparison;
- a separately frozen daily-risk/CVXPY implementation with purged candidate
  selection, shrinkage covariance, capacity/borrow stress, and a documented
  negative result;
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
python run_portfolio_engineering.py
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
- Portfolio engineering was designed after the baseline was inspected. It is
  labeled retrospective and cannot serve as an untouched confirmation.
- Retrospective support is not live performance. The prospective result remains
  unopened and unavailable.

## Repository map

```text
run_wrds_model.py            # frozen real-data study and evidence writer
run_portfolio_engineering.py # separately frozen implementation experiment
src/factors/wrds_data.py     # bounded CRSP/Compustat acquisition
src/factors/real_model.py    # PIT signals, inference, portfolios, diagnostics
src/factors/portfolio_engineering.py # risk, constraints, turnover, bootstrap
results/wrds_real_data/      # primary aggregate evidence and protocol
results/portfolio_engineering/ # implementation evidence and integrity receipt
run_model.py                 # deterministic credential-free CI demonstration
tests/                       # data, inference, security, and artifact contracts
```

Python 3.10+ is recommended. This is research software, not investment advice.
