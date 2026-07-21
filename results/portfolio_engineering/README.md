# Portfolio Engineering Evidence

This is a separately frozen, retrospective implementation experiment. It does
not alter the version-2 composite signal or overwrite its baseline artifacts.

## Verdict

The primary engineered portfolio recorded a **0.47 Sharpe at
10 bps**, versus **0.47** for the frozen baseline. Its CAGR
was **3.28%**, maximum drawdown **-23.21%**,
and average monthly turnover **0.41**.

The paired 12-month moving-block estimate of the Sharpe improvement was
**0.01**, with a 95% interval of
**-0.16 to
0.21**. FF5 plus momentum
alpha was **2.26% annually** (HAC
t=2.54, p=0.011).

The classification is **RETROSPECTIVE_IMPROVEMENT_NOT_BOOTSTRAP_CONFIRMED**. All evidence remains
retrospective and the project still makes **no validated prospective-alpha
claim**.

![Cumulative wealth](engineered_cumulative_wealth.png)

![Sharpe and costs](engineered_sharpe_costs.png)

## Frozen design

- Frozen signal: the unchanged six-signal composite from specification
  `0f193737dd782ea6c31fe09693910789edb99f76efc93d86ec35cae5cdbe1a5a`.
- Primary portfolio: inverse-volatility score scaling; dollar, beta, SIC-sector,
  and log-size projection; 1.5% absolute position cap; 35% partial rebalance;
  five-basis-point weight-change no-trade band; and trailing 10% volatility
  targeting.
- Costs: 0, 10, and 25 bps per dollar traded.
- Comparators: the original continuous-score baseline, risk-neutral construction,
  and risk-plus-turnover construction.
- No signal weight, risk parameter, or candidate was selected from the resulting
  forward-return evidence.

## Public artifacts

- `protocol.json` - frozen experiment specification and acceptance criteria
- `comparison_receipt.json` - baseline-versus-primary conclusion
- `portfolio_summary.csv` - cost and performance comparison
- `factor_attribution.csv` - FF5 plus momentum attribution
- `portfolio_era_stability.csv` - calendar-decade results
- `engineering_diagnostics.csv` - aggregate exposure and concentration checks
- `monthly_portfolio_returns.csv` - aggregate portfolio returns only
- `baseline_integrity.json` - hashes and exact baseline-replay checks
- `quality_receipt.json` - protocol, privacy, exposure, and replay gates

Licensed security-level WRDS rows remain under ignored private storage. This
directory contains no security identifiers and no investment recommendation.
