from __future__ import annotations

import numpy as np
import pandas as pd

from factors.institutional_v4 import (
    InstitutionalV4Spec,
    apply_ensemble_score,
    capacity_position_caps,
    eligible_names,
    fit_development_ensemble,
    institutional_v4_hash,
)


def _panel() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for date in (pd.Timestamp("2018-01-31"), pd.Timestamp("2022-01-31")):
        for i in range(80):
            row = {"date": date}
            for feature in InstitutionalV4Spec().signal_features:
                row[feature] = rng.normal()
            for horizon in InstitutionalV4Spec().score_horizons:
                row[f"fwd_{horizon}m"] = row["QUALITY_SCORE"] + rng.normal(scale=0.2)
            rows.append(row)
    return pd.DataFrame(rows)


def test_ensemble_fit_ignores_temporal_assessment_returns() -> None:
    panel = _panel()
    weights, _ = fit_development_ensemble(panel)
    changed = panel.copy()
    mask = changed["date"] >= pd.Timestamp("2021-01-01")
    for horizon in InstitutionalV4Spec().score_horizons:
        changed.loc[mask, f"fwd_{horizon}m"] *= -1000
    changed_weights, _ = fit_development_ensemble(changed)
    assert weights == changed_weights
    assert np.isclose(sum(weights.values()), 1.0)
    assert all(value > 0.0 for value in weights.values())


def test_apply_ensemble_score_requires_eighty_percent_weight_coverage() -> None:
    panel = _panel().head(2)
    weights = {feature: 0.2 for feature in InstitutionalV4Spec().signal_features}
    panel.loc[panel.index[0], list(weights)[:1]] = np.nan
    panel.loc[panel.index[1], list(weights)[:2]] = np.nan
    scored = apply_ensemble_score(panel, weights)
    assert np.isfinite(scored.loc[scored.index[0], "V4_SCORE"])
    assert pd.isna(scored.loc[scored.index[1], "V4_SCORE"])


def test_hysteresis_keeps_existing_names_inside_exit_buffer() -> None:
    cross = pd.DataFrame(
        {"absolute_score_rank": [1, 2, 3, 4, 5]}, index=[10, 11, 12, 13, 14]
    )
    previous = pd.Series([0.1, -0.1], index=[13, 99])
    spec = InstitutionalV4Spec(entry_rank_count=2, exit_rank_count=4)
    names = eligible_names(cross, previous, spec)
    assert set(names) == {10, 11, 13}
    assert 99 not in names


def test_capacity_caps_respect_adv_and_hard_position_limit() -> None:
    cross = pd.DataFrame({"adv_usd": [1_000_000.0, 100_000_000.0]}, index=[1, 2])
    caps = capacity_position_caps(cross)
    assert np.isclose(caps.loc[1], 0.001)
    assert np.isclose(caps.loc[2], 0.02)


def test_v4_hash_is_deterministic_and_chained_to_v3() -> None:
    first = institutional_v4_hash("abc")
    assert first == institutional_v4_hash("abc")
    assert first != institutional_v4_hash("def")
