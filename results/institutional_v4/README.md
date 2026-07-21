# Institutional Version-4 Follow-up

This is an explicitly **post-result retrospective follow-up**. It preserves the
version-2 and version-3 evidence and does not create an untouched holdout.

## Verdict

At the locked $100 million / 150-bp borrow scenario, the 2021–2024 assessment
recorded a **0.694 net Sharpe**, **5.31% CAGR**,
**-12.39% maximum drawdown**, and **0.561
average monthly turnover**. Its self-financing overlay information ratio was
**0.694**; this equals the net Sharpe because
the sleeve return itself is the active return. The stricter FF5+momentum
factor-residual information ratio was **-0.024**.

The version-4 12-month rank IC was **0.0560** with ICIR
**1.087**, versus **0.0277** and
**0.401** for the frozen equal-weight composite over the same
dates. Formal classification: **RETROSPECTIVE_MULTI_METRIC_IMPROVEMENT**; alpha status:
**NO VALIDATED ALPHA**.

![Matched cumulative wealth](v4_cumulative_wealth.png)

![Metric comparison](v4_metric_comparison.png)

## What changed

One development-only score was fixed before evaluation. Monthly ICs from
2010–2019 were averaged uniformly across 1, 3, 6, and 12-month horizons,
clipped at zero, normalized, and shrunk 50% toward equal weights. `SIZE_SCORE`
is excluded because size is an explicit risk constraint.

- `MOM_SCORE`: 23.3%
- `LV_SCORE`: 32.9%
- `VALUE_SCORE`: 10.0%
- `QUALITY_SCORE`: 23.8%
- `INVESTMENT_SCORE`: 10.0%

The portfolio adds an entry/exit buffer: new positions require an absolute
residual-score rank of 200 or better, while existing holdings remain eligible
through rank 500. Monthly turnover is capped at 0.75 and each position is
limited by both a 2% box constraint and lagged ADV capacity.

## Metric definitions

- Sharpe uses monthly net returns annualized by square-root of 12.
- IC is monthly Spearman rank correlation between the score and forward returns.
- ICIR is mean monthly IC divided by its monthly standard deviation.
- Overlay IR is annualized mean net sleeve return divided by its annualized
  tracking-error contribution. For this dollar-neutral self-financing sleeve,
  it is mathematically the same number as net Sharpe—not a second independent win.
- Factor-residual IR is annualized FF5+momentum regression alpha divided by
  annualized regression-residual volatility.
- Benchmark-relative IR remains in the portfolio summary for transparency, but
  it is not the main IR for a dollar-neutral sleeve.

## Evidence boundary

All dates through 2024 had already been inspected before version 4. Licensed
security-level WRDS rows and weight paths remain under ignored `data/private/`.
Committed files contain only aggregate evidence. This is not investment advice.
