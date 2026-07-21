from __future__ import annotations

import numpy as np
import pandas as pd

from factors.profitable_factor_portfolio import (
    ProfitableFactorPortfolioSpec,
    build_combined_returns,
    fit_factor_weights,
    fit_sleeve_mix,
    merge_factor_and_v4_returns,
    profitable_factor_portfolio_hash,
)


def _factors() -> pd.DataFrame:
    dates = pd.date_range("1990-01-31", "2024-12-31", freq="ME")
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"dateff": dates})
    for index, factor in enumerate(ProfitableFactorPortfolioSpec().factors):
        frame[factor] = 0.001 + index * 0.0001 + rng.normal(0.0, 0.02, len(frame))
    return frame


def _v4() -> pd.DataFrame:
    dates = pd.date_range("2010-01-31", "2024-11-30", freq="ME")
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        "date": dates,
        "aum_usd": 100_000_000.0,
        "borrow_bps": 150.0,
        "gross_return": rng.normal(0.006, 0.03, len(dates)),
        "net_return": rng.normal(0.004, 0.03, len(dates)),
    })


def test_factor_weights_use_development_only() -> None:
    factors = _factors()
    weights, _ = fit_factor_weights(factors)
    changed = factors.copy()
    mask = changed["dateff"] >= pd.Timestamp("2021-01-01")
    changed.loc[mask, list(ProfitableFactorPortfolioSpec().factors)] *= -1_000.0
    changed_weights, _ = fit_factor_weights(changed)
    pd.testing.assert_series_equal(weights, changed_weights)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= ProfitableFactorPortfolioSpec().maximum_factor_weight + 1e-12


def test_alignment_uses_next_calendar_month() -> None:
    weights, _ = fit_factor_weights(_factors())
    merged = merge_factor_and_v4_returns(_factors(), _v4(), weights)
    assert len(merged) == len(_v4())
    assert merged.iloc[0]["factor_month"] == pd.Period("2010-02", freq="M")
    assert merged.iloc[-1]["factor_month"] == pd.Period("2024-12", freq="M")


def test_sleeve_mix_and_cost_stresses_are_fixed_pre_assessment() -> None:
    factors = _factors()
    weights, _ = fit_factor_weights(factors)
    merged = merge_factor_and_v4_returns(factors, _v4(), weights)
    sleeve_weights, leverage = fit_sleeve_mix(merged)
    assert np.isclose(sleeve_weights.sum(), 1.0)
    assert (sleeve_weights > 0.0).all()
    assert 0.0 < leverage <= ProfitableFactorPortfolioSpec().maximum_combined_leverage
    monthly = build_combined_returns(merged, sleeve_weights, leverage)
    assert set(monthly["factor_cost_bps"].unique()) == {0.0, 50.0, 100.0, 150.0}
    primary = monthly[np.isclose(monthly["factor_cost_bps"], 100.0)]
    assert np.allclose(
        primary["factor_sleeve_gross_return"]
        - primary["factor_sleeve_net_return"],
        0.01 / 12.0,
    )


def test_portfolio_hash_is_deterministic_and_chained() -> None:
    assert profitable_factor_portfolio_hash("abc") == profitable_factor_portfolio_hash("abc")
    assert profitable_factor_portfolio_hash("abc") != profitable_factor_portfolio_hash("def")
