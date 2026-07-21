import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from factors.daily_wrds import DailyCRSPRequest, build_daily_query, normalize_crsp_daily
from factors.institutional_v3 import (
    InstitutionalV3Spec,
    OptimizerCandidate,
    _daily_snapshot,
    institutional_specification_hash,
    run_candidate_backtest,
    select_candidate,
)


def _synthetic_inputs(n=120, months=3, seed=7):
    rng = np.random.default_rng(seed)
    month_dates = pd.date_range("2020-03-31", periods=months, freq="ME")
    panel_rows = []
    for date in month_dates:
        scores = np.linspace(-2.5, 2.5, n)
        rng.shuffle(scores)
        for i in range(n):
            panel_rows.append({
                "date": date,
                "permno": 10000 + i,
                "permco": 20000 + i,
                "COMPOSITE": scores[i],
                "fwd_1m": 0.004 * scores[i] + rng.normal(0, 0.03),
                "market_equity_millions": 250 + 10 * i,
                "sector": i % 6,
                "period": "retrospective",
            })
    daily_rows = []
    for date in pd.bdate_range("2019-01-01", "2020-06-30"):
        market = rng.normal(0.0002, 0.008)
        for i in range(n):
            price = 10 + i / 5
            daily_rows.append({
                "date": date,
                "permno": 10000 + i,
                "ret": market * (0.8 + (i % 8) / 10) + rng.normal(0, 0.01),
                "price": price,
                "vol": 200000 + 1000 * i,
                "bid": price - 0.01,
                "ask": price + 0.01,
                "bidlo": price - 0.20,
                "askhi": price + 0.20,
                "shrout": 10000,
            })
    return pd.DataFrame(panel_rows), normalize_crsp_daily(pd.DataFrame(daily_rows))


def _spec():
    return InstitutionalV3Spec(
        evaluation_start="2020-01-01",
        development_end="2020-04-30",
        temporal_assessment_start="2020-05-01",
        candidate_count=120,
        daily_lookback_days=80,
        minimum_daily_observations=50,
        liquidity_lookback_days=21,
        maximum_absolute_weight=0.025,
        maximum_monthly_turnover=2.0,
        development_folds=(("2020-03-01", "2020-05-31"),),
        aum_stress_usd=(10_000_000.0,),
        borrow_stress_bps=(150.0,),
        primary_aum_usd=10_000_000.0,
        primary_borrow_bps=150.0,
        candidates=(OptimizerCandidate("test", 0.005, 4.0, 0.001),),
    )


def test_daily_query_and_normalization_are_bounded_and_conservative():
    query = build_daily_query(2020, [10003, 10001])
    assert "2020-01-01" in query and "10001,10003" in query
    _, daily = _synthetic_inputs(n=4, months=1)
    assert daily["spread_fraction"].min() >= 0.0002
    assert (daily["dollar_volume"] > 0).all()
    assert DailyCRSPRequest().maximum_permnos == 6000


def test_snapshot_uses_only_history_available_at_formation():
    panel, daily = _synthetic_inputs()
    spec = _spec()
    formation = panel["date"].min()
    cross = panel[panel["date"] == formation]
    original = _daily_snapshot(formation, cross, daily, spec)
    changed = daily.copy()
    changed.loc[changed["date"] > formation, "ret"] = 9.0
    revised = _daily_snapshot(formation, cross, changed, spec)
    assert original is not None and revised is not None
    assert np.allclose(original["covariance"], revised["covariance"])


def test_convex_backtest_enforces_exposures_and_costs():
    panel, daily = _synthetic_inputs()
    spec = _spec()
    snapshots = []
    for date, cross in panel.groupby("date"):
        value = _daily_snapshot(date, cross, daily, spec)
        assert value is not None
        snapshots.append(value)
    monthly = run_candidate_backtest(snapshots, spec.candidates[0], spec)
    assert len(monthly) == len(snapshots)
    assert monthly["gross_exposure"].between(1.99, 2.01).all()
    assert monthly["net_exposure"].abs().max() < 1e-5
    assert monthly["beta_exposure"].abs().max() <= spec.beta_tolerance + 1e-4
    assert monthly["size_exposure"].abs().max() <= spec.size_tolerance + 1e-4
    assert monthly["maximum_sector_net_exposure"].max() <= spec.sector_tolerance + 1e-4
    assert (monthly["net_return"] <= monthly["gross_return"]).all()


def test_candidate_selection_and_hash_are_deterministic():
    dates = pd.date_range("2020-03-31", periods=3, freq="ME")
    result = pd.DataFrame({
        "date": dates,
        "period": ["development"] * 3,
        "candidate": ["test"] * 3,
        "aum_usd": [10_000_000.0] * 3,
        "borrow_bps": [150.0] * 3,
        "net_return": [0.01, -0.005, 0.015],
        "benchmark_return": [0.005] * 3,
        "turnover": [0.2] * 3,
    })
    spec = _spec()
    winner, table = select_candidate({"test": result}, spec)
    assert winner == "test" and not table.empty
    first = institutional_specification_hash("a" * 64, "b" * 64, spec)
    assert first == institutional_specification_hash("a" * 64, "b" * 64, spec)
