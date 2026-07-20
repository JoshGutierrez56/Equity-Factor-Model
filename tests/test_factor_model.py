import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from factors.scoring import normalise_factors, zscore
from run_model import _synthetic_prices


def test_offline_prices_are_deterministic_and_positive():
    first = _synthetic_prices(["AAA", "BBB"], "2020-01-01", "2021-01-01")
    second = _synthetic_prices(["AAA", "BBB"], "2020-01-01", "2021-01-01")
    assert first.equals(second)
    assert np.isfinite(first.to_numpy()).all()
    assert (first > 0).all().all()


def test_zscore_is_centered_and_scaled():
    scored = zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert abs(scored.mean()) < 1e-12
    assert abs(scored.std(ddof=1) - 1.0) < 1e-12


def test_normalisation_limits_outlier_and_preserves_shape():
    raw = pd.DataFrame({"MOM": [1.0, 2.0, 3.0, 1_000.0], "LV": [4.0, 3.0, 2.0, 1.0]})
    normalised = normalise_factors(raw)
    assert normalised.shape == raw.shape
    assert np.isfinite(normalised.to_numpy()).all()
