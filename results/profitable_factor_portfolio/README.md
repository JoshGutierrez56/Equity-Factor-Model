# Profitable-factor portfolio follow-up

This is an explicitly **retrospective, performance-informed allocation study**.
It preserves institutional v4 and combines its net stock-selection return with
established Fama-French and momentum test-portfolio returns. It is designed to
harvest known premiums; it does **not** claim new factor-neutral alpha.

## Primary result

For the 47 matched formation months from 2021 through November 2024, with the
v4 sleeve already net of its locked $100 million cost model and a conservative
**100-bp annual haircut** applied to the factor sleeve:

- combined unlevered Sharpe: **1.026** versus **0.694** for v4;
- CAGR: **4.69%**;
- annualized volatility: **4.58%**;
- maximum drawdown: **-6.43%**;
- factor-tilt IR versus the otherwise identical equal-factor combination:
  **0.639**;
- active IR versus v4: **-0.127**.

The development-only volatility target hit its 2.0x leverage cap. That secondary
risk-scaled presentation recorded **1.026 Sharpe**,
**9.36% CAGR**, and **-12.57%**
maximum drawdown. Leverage does not improve Sharpe and is not the primary claim.

The gross academic factor sleeve recorded **1.010 Sharpe**;
after the 100-bp haircut it recorded **0.820**. The paired
12-month block interval for the unlevered Sharpe improvement over v4 is
**[-0.180,
1.121]**, so the improvement is not statistically confirmed.

![Cumulative wealth](factor_combination_cumulative_wealth.png)

![Cost stress](factor_cost_stress.png)

## Fixed weights

Factor weights use 1990-2019 data only: 50% equal weight plus 50% normalized
positive development Sharpe, with a 30% cap.

- `MKTRF`: 22.0%
- `SMB`: 12.0%
- `HML`: 12.9%
- `RMW`: 19.2%
- `CMA`: 16.7%
- `UMD`: 17.2%

The 2010-2019 inverse-volatility mix is **31.2% v4** and
**68.8% factor sleeve**. The risk-scaled diagnostic uses
**2.00x** leverage, fixed from development data and capped at 2.0x.

## IC and IR interpretation

The combined portfolio has no cross-sectional IC of its own. The applicable
stock-selection diagnostic remains v4's 12-month rank IC of 5.60% and ICIR of
1.087. The factor-tilt IR measures the benefit of the development-only tilt
relative to equal factor weights. The active IR versus v4 is also reported and
is negative because diversification improved risk-adjusted performance while
slightly lowering average return.

## Evidence boundary

Fama-French factors are academic test portfolios, not directly tradable
instruments. The flat 0/50/100/150-bp factor haircuts are stress scenarios, not
a security-level spread, impact, financing, and borrow reconstruction. All dates
had already been inspected, the Sharpe interval includes zero, and no prospective
alpha claim is made. Licensed security-level rows and weights remain uncommitted.
This is not investment advice.
