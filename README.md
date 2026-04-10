# Cross-Sectional Equity Factor Model
### 5-Factor Alpha Signal | IC Analysis | Factor-Constrained Optimizer | Walk-Forward Backtest

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

A research-grade cross-sectional equity factor model targeting the S&P 500 universe.
Directly implements the workflow described in Acadian's quant research roles:
*"explore new datasets, apply sophisticated statistical techniques, and write code
to develop signals for predicting equity returns."*

---

## Factors

| Factor | Signal | Academic Basis |
|--------|--------|---------------|
| **MOM** | 12-1 month price return | Jegadeesh & Titman (1993) |
| **LV**  | Negative 252-day realised vol | Ang et al. (2006), Frazzini & Pedersen (2014) |
| **SIZE**| Negative log market cap | Fama & French (1993) |
| **VAL** | Price-based reversal proxy | De Bondt & Thaler (1985) |
| **QUAL**| Negative idiosyncratic vol | Asness, Frazzini & Pedersen (2019) |

All factors are normalised cross-sectionally (winsorise → z-score) and combined
into an IC-weighted composite signal.

---

## Key Results (Synthetic Universe, 2015–2024)

**IC Summary (composite, Bonferroni-corrected):**

| Horizon | IC Mean | t-stat | ICIR | Hit Rate |
|---------|---------|--------|------|----------|
| 5d  | +0.024 | 2.26 | 0.23 | 62.5% |
| 21d | +0.019 | 1.81 | 0.18 | 61.5% |
| 63d | +0.030 | 2.87 | 0.30 | 59.6% |
| 126d| +0.046 | 4.41✓| 0.46 | 61.5% |

**LV (Low Volatility) is the strongest individual factor** — significant at
Bonferroni-corrected p < 0.001 at the 126-day horizon. This is consistent with
the large body of empirical literature on the low-volatility anomaly.

**Backtest metrics (top-50 equal-weight, 10 bps TC):**
- CAGR: 11.3% | Sharpe: 2.03 | Max Drawdown: −3.2% | IR: 0.56

---

## Repository Structure

```
equity-factor-model/
├── run_model.py                    ← CLI entry point
├── src/factors/
│   ├── universe.py                 ← S&P 500 download (Wikipedia + yfinance)
│   ├── signals.py                  ← 5-factor construction
│   ├── scoring.py                  ← Winsorise → z-score → composite
│   ├── ic_analysis.py              ← IC, t-stat, ICIR, decay, factor corr
│   ├── optimizer.py                ← CVXPY/SLSQP factor-constrained optimizer
│   ├── backtest.py                 ← Walk-forward with attribution
│   └── charts.py                   ← 7 publication-quality figures
└── outputs/
    ├── figures/                    ← IC decay, NAV, drawdown, rolling Sharpe…
    └── tables/                     ← ic_summary.csv, backtest_metrics.csv
```

---

## Setup & Usage

```bash
pip install -r requirements.txt

# No internet needed (synthetic prices)
python run_model.py --offline

# Full pipeline (downloads S&P 500 prices via yfinance)
python run_model.py

# Custom date range, top-100 equal weight
python run_model.py --start 2018-01-01 --end 2024-01-01 --top-n 100
```

---

## Academic References

| Reference | Applied In |
|-----------|-----------|
| Jegadeesh & Titman (1993) | MOM factor specification |
| Fama & French (1993) | SIZE factor |
| Ang, Hodrick, Xing & Zhang (2006) | LV factor |
| Grinold & Kahn (1999) | IC, ICIR, composite construction |
| Qian, Hua & Sorensen (2007) | Cross-sectional signal framework |
| Ledoit & Wolf (2004) | Covariance shrinkage |
