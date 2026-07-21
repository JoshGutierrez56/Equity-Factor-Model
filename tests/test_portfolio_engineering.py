import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from factors.portfolio_engineering import (
    ENGINEERING_STRATEGIES,
    PortfolioEngineeringSpec,
    add_trailing_risk_features,
    aggregate_diagnostics,
    compute_engineered_portfolios,
    compute_engineering_summary,
    engineering_specification_hash,
    paired_sharpe_improvement_ci,
)
from factors.real_model import RealModelSpec, build_point_in_time_panel
from factors.wrds_data import WRDSResearchRequest
from tests.test_real_model import _mock_inputs


def _panel() -> pd.DataFrame:
    crsp, fundamentals = _mock_inputs(n_securities=120)
    request = WRDSResearchRequest.create(
        "2020-01-01", "2021-12-31", "2021-01-01", max_universe=100
    )
    return build_point_in_time_panel(crsp, fundamentals, request, RealModelSpec())


def _spec() -> PortfolioEngineeringSpec:
    return PortfolioEngineeringSpec(
        trailing_vol_months=12,
        minimum_vol_months=6,
        trailing_beta_months=12,
        minimum_beta_months=6,
        portfolio_volatility_minimum_months=6,
        portfolio_volatility_lookback=12,
        maximum_absolute_weight=0.03,
        bootstrap_repetitions=50,
        bootstrap_block_months=4,
    )


def test_engineering_specification_is_deterministic_and_signal_locked():
    spec = PortfolioEngineeringSpec()
    assert engineering_specification_hash("a" * 64, spec) == engineering_specification_hash(
        "a" * 64, spec
    )
    public = spec.public_dict()
    assert public["signal"] == "COMPOSITE"
    assert public["signal_policy"].startswith("frozen composite")


def test_trailing_risk_features_do_not_use_later_returns():
    panel = _panel()
    spec = _spec()
    original = add_trailing_risk_features(panel, spec)
    changed = panel.copy()
    final_date = changed["date"].max()
    changed.loc[changed["date"] == final_date, "total_ret"] = 5.0
    revised = add_trailing_risk_features(changed, spec)
    prior = original["date"] < final_date
    columns = ["trailing_volatility", "trailing_beta"]
    assert np.allclose(
        original.loc[prior, columns].to_numpy(dtype=float),
        revised.loc[prior, columns].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_engineered_portfolios_apply_locked_constraints_and_costs():
    monthly = compute_engineered_portfolios(_panel(), _spec())
    assert set(ENGINEERING_STRATEGIES).issubset(set(monthly["strategy"]))
    assert {0.0, 10.0, 25.0} == set(monthly["cost_bps"])
    unique = monthly.drop_duplicates(["date", "strategy"])
    assert unique["net_exposure"].abs().max() < 1e-5
    assert unique["beta_exposure"].abs().max() < 1e-4
    assert unique["size_exposure"].abs().max() < 1e-4
    assert unique["maximum_sector_net_exposure"].max() < 1e-4
    assert unique["maximum_absolute_weight"].max() <= _spec().maximum_absolute_weight + 1e-6
    same = monthly[monthly["strategy"] == "risk_turnover_vol_scaled"].pivot(
        index="date", columns="cost_bps", values="net_return"
    )
    assert (same[0.0] >= same[10.0]).all()
    assert (same[10.0] >= same[25.0]).all()


def test_engineering_summaries_and_bootstrap_are_aggregate_only():
    monthly = compute_engineered_portfolios(_panel(), _spec())
    summary = compute_engineering_summary(monthly)
    diagnostics = aggregate_diagnostics(monthly)
    assert not summary.empty
    assert not diagnostics.empty
    assert "sharpe" in summary
    assert "mean_beta_exposure" in diagnostics
    assert "mean_abs_beta_exposure" in diagnostics
    assert not {"permno", "permco", "ticker"}.intersection(summary.columns)

    baseline = monthly[monthly["strategy"] == "risk_neutral"].copy()
    baseline["strategy"] = "baseline"
    combined = pd.concat([baseline, monthly], ignore_index=True)
    observed, lower, upper = paired_sharpe_improvement_ci(
        combined,
        "baseline",
        "risk_turnover_vol_scaled",
        10.0,
        _spec(),
    )
    assert np.isfinite([observed, lower, upper]).all()
    assert lower <= upper
