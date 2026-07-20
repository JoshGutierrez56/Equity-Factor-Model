# Point-in-Time U.S. Equity Factor Research

A reproducible cross-sectional equity-research platform built around CRSP,
Compustat, and the CRSP/Compustat Merged link history. The live path tests five
predeclared signals with information coefficients, heteroskedasticity/autocorrelation-
robust inference, walk-forward portfolios, turnover, and explicit costs.

## Bottom line

The real-data result is credible but modest. In the 2010–2023 development sample,
the fixed equal-weight composite produced a 0.97% one-month rank IC and a 2.36%
twelve-month rank IC. Its market-neutral top-minus-bottom portfolio produced a
0.17 Sharpe after 10 bps per dollar traded; no composite IC test survived
Bonferroni correction.

Fama–French five-factor plus momentum attribution reaches the same conclusion:
the development composite long-short alpha is 0.02% annualized after 10 bps
(HAC t-statistic 0.02). No evaluated alpha survives multiple-testing correction.

The previously unexamined 2024 confirmation has only eleven executable months. The composite
long-short portfolio produced a 0.45 descriptive Sharpe after 10 bps, but this is
not enough data for formal inference. The long-only sleeve underperformed the
value-weighted eligible-universe benchmark. These results do not establish a
tradable alpha strategy.

| Sample | Composite 1m IC | Composite 12m IC | 10-bps L/S Sharpe | 10-bps L/S CAGR | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Development, 2010–2023 | 0.97% | 2.36% | 0.17 | 0.93% | -20.13% |
| Descriptive holdout, 2024 | 1.21% | insufficient horizon | 0.45 | 1.74% | -3.44% |

The reviewed evidence and full audit disclosure are in
[`results/wrds_real_data/`](results/wrds_real_data/README.md).

## Research design

- **Universe:** point-in-time NYSE/AMEX/Nasdaq common stocks; price at least $5,
  market capitalization at least $100 million, and the largest 1,000 companies
  each month.
- **Entity:** CRSP `PERMCO`. Multiple share classes are consolidated using
  aggregate market equity and market-equity-weighted returns.
- **Returns:** CRSP monthly returns compounded with CRSP delisting returns.
- **Fundamentals:** Compustat annual data linked through CCM and delayed six
  months after fiscal period end.
- **Signals:** 12–1 momentum, low volatility, size, book-to-market value, and
  gross profitability quality.
- **Scoring:** 1st/99th percentile winsorization, SIC-sector neutralization,
  cross-sectional z-scores, and fixed equal composite weights.
- **Evaluation:** monthly Spearman IC at 1/3/6/12 months, HAC mean tests,
  Bonferroni correction, long-only and long-short quintiles, and 0/10/25-bps
  cost cases. Portfolio returns are also regressed on Fama–French five factors
  plus momentum with HAC inference.
- **Confirmation:** portfolio rules were frozen before 2024 was opened. A
  minimum-inference guard was added after seeing that WRDS supplied only twelve
  signal months; returns and construction were unchanged. Formal
  significance requires 24 months, so the one-year holdout is descriptive.

## Run it

The deterministic synthetic path remains available for CI and code review:

```bash
python -m pip install -r requirements.txt
python run_model.py --offline
python -m pytest -q
```

The live path requires licensed WRDS access. Credentials stay in the user's
normal local WRDS/PostgreSQL configuration and are never accepted as CLI flags:

```bash
python -m pip install -r requirements-live.txt

# Inspect development only; writes private rows under ignored data/private/
python run_wrds_model.py --scope development --refresh

# Open the frozen holdout and write aggregate public evidence
python run_wrds_model.py --scope full
```

Security-level licensed rows are ignored by Git. The committed result bundle
contains only aggregate IC statistics, portfolio-level monthly returns, coverage,
and hashes of the private inputs.

## Evidence boundaries and limitations

- Compustat `funda` can contain later restatements. The six-month lag prevents
  obvious announcement look-ahead but is not a true vintage database.
- SIC groups are a transparent sector proxy, not a commercial risk model.
- Transaction costs are linear and do not model borrow fees, impact, or capacity.
- Equal-weight quintiles can overstate capacity relative to an institutional
  implementation.
- WRDS coverage available in this run ended in December 2024, leaving a short
  confirmation period. The holdout should be extended as new data become available.
- An earlier engine-validation split exposed a share-class continuity defect.
  It was corrected without changing factor orientation, weights, quantiles, or
  cost assumptions; the full history is recorded in `specification.json`.

## Repository map

```text
run_model.py                 # legacy synthetic/yfinance demonstration
run_wrds_model.py            # frozen real-data research pipeline
src/factors/wrds_data.py     # bounded CRSP/Compustat acquisition
src/factors/real_model.py    # PIT signals, IC, portfolios, costs, evidence
src/factors/                 # reusable legacy scoring/optimizer modules
tests/                       # offline contracts and regression tests
results/wrds_real_data/      # primary aggregate evidence
results/synthetic_*          # legacy execution evidence
```

Python 3.10+ is recommended. This is research software, not investment advice.
