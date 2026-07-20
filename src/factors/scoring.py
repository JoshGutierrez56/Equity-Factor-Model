"""
scoring.py
==========
Cross-sectional factor normalisation and composite score construction.

Pipeline
--------
1. Winsorise each factor at [2.5%, 97.5%] to remove outliers
2. Z-score normalise cross-sectionally (mean=0, std=1 across universe)
3. Combine into a composite signal using IC-weighted or equal weights
4. Apply universe filter (drop NaN-heavy tickers)

References
----------
Grinold & Kahn (1999) "Active Portfolio Management" — Ch. 1, alpha signals
Barra / MSCI Risk Model Handbook — factor normalisation conventions
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factors.signals import FACTOR_NAMES

logger = logging.getLogger(__name__)


def winsorise(
    series: pd.Series,
    lower: float = 0.025,
    upper: float = 0.975,
) -> pd.Series:
    """Clip extreme values at the given quantiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score normalisation."""
    mu  = series.mean()
    std = series.std(ddof=1)
    if std < 1e-10:
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - mu) / std


def normalise_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply winsorise → z-score to every factor column in a cross-section.

    Parameters
    ----------
    df : pd.DataFrame  — rows=tickers, cols=factor names

    Returns
    -------
    pd.DataFrame  — same shape, normalised
    """
    result = pd.DataFrame(index=df.index)
    for col in df.columns:
        raw    = df[col].dropna()
        clipped = winsorise(raw)
        result[col] = zscore(clipped)
    return result


def composite_score(
    df_norm: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Combine normalised factor columns into a single composite alpha signal.

    Parameters
    ----------
    df_norm : pd.DataFrame  — normalised factors (rows=tickers, cols=factors)
    weights : dict, optional  — {factor_name: weight}. Equal weight if None.

    Returns
    -------
    pd.Series  — composite score per ticker, z-scored
    """
    cols = [c for c in FACTOR_NAMES if c in df_norm.columns]
    if not cols:
        return pd.Series(dtype=float)

    if weights is None:
        w = {c: 1.0 / len(cols) for c in cols}
    else:
        total = sum(weights.get(c, 0) for c in cols)
        w     = {c: weights.get(c, 0) / total for c in cols}

    composite = sum(df_norm[c] * w[c] for c in cols if c in df_norm.columns)
    return zscore(composite.dropna()).rename("COMPOSITE")


class FactorScorer:
    """
    Process a factor panel into normalised scores and composite alphas.

    Parameters
    ----------
    panel   : dict of {date: DataFrame (tickers × factors)}
    weights : optional composite weights
    min_coverage : minimum fraction of factors required per ticker
    """

    def __init__(
        self,
        panel:        Dict[pd.Timestamp, pd.DataFrame],
        weights:      Optional[Dict[str, float]] = None,
        min_coverage: float = 0.6,
    ) -> None:
        self.panel        = panel
        self.weights      = weights
        self.min_coverage = min_coverage

    def process(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Return normalised factor exposures and composite scores for each date.

        Returns
        -------
        dict  — {date: DataFrame with cols = FACTOR_NAMES + ["COMPOSITE"]}
        """
        scored = {}
        for dt, raw_df in self.panel.items():
            # Filter tickers with enough factor coverage
            coverage = raw_df.notna().mean(axis=1)
            df = raw_df.loc[coverage >= self.min_coverage].copy()

            if df.empty:
                continue

            # Normalise
            norm_df = normalise_factors(df)

            # Composite
            norm_df["COMPOSITE"] = composite_score(norm_df, self.weights)
            scored[dt] = norm_df

        logger.info("Scored %d dates; avg universe size = %.0f",
                    len(scored),
                    np.mean([len(v) for v in scored.values()]))
        return scored

    def factor_exposure_history(
        self,
        scored: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
        factor: str = "MOM",
    ) -> pd.DataFrame:
        """
        Return a wide DataFrame of one factor across all dates.
        Rows = dates, columns = tickers.
        """
        if scored is None:
            scored = self.process()
        frames = {dt: df[factor] for dt, df in scored.items() if factor in df.columns}
        return pd.DataFrame(frames).T.sort_index()
