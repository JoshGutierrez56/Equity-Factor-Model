"""
ic_analysis.py
==============
Information Coefficient analysis: the core statistical validation
that every quantitative equity researcher must produce.

Metrics
-------
IC (Information Coefficient)
    Spearman rank correlation between factor score at time t and
    forward return at t+h. Measures how well the factor ranks stocks.

IC Mean & t-statistic
    IC_mean / (IC_std / sqrt(T)) — tests whether IC is significantly
    different from zero. |t| > 2 is a typical publication threshold.

IC Information Ratio (ICIR)
    IC_mean / IC_std  — analogous to Sharpe ratio for a signal.
    ICIR > 0.5 is considered strong for a single factor.

IC Decay
    How does IC fall as the forecast horizon h increases from 1 week
    to 12 months? A factor with slow decay is more investable.

Factor Correlation
    Pairwise Pearson correlation of factor scores cross-sectionally.
    Correlated factors add little diversification value in a composite.

Multiple Hypothesis Correction
    With 5 factors tested simultaneously, the family-wise error rate
    inflates. We apply Bonferroni correction to IC t-statistics.

References
----------
Grinold & Kahn (1999) "Active Portfolio Management"
Qian, Hua & Sorensen (2007) "Quantitative Equity Portfolio Management"
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from factors.signals import FACTOR_NAMES

logger = logging.getLogger(__name__)


class ICAnalyzer:
    """
    Compute Information Coefficients for each factor and the composite.

    Parameters
    ----------
    scored  : dict of {date: DataFrame (tickers × factors + COMPOSITE)}
    returns : pd.DataFrame  — wide daily returns (rows=dates, cols=tickers)
    """

    def __init__(
        self,
        scored:  Dict[pd.Timestamp, pd.DataFrame],
        returns: pd.DataFrame,
    ) -> None:
        self.scored  = scored
        self.returns = returns

    # ------------------------------------------------------------------
    # Core IC computation
    # ------------------------------------------------------------------

    def compute_ic(
        self,
        factor:  str = "COMPOSITE",
        horizon: int = 21,      # trading days (≈ 1 month)
    ) -> pd.Series:
        """
        Quarterly IC time series for a given factor and forward horizon.

        Parameters
        ----------
        factor  : column name in scored DataFrames
        horizon : forward return window in trading days

        Returns
        -------
        pd.Series  — IC per date, indexed by rebalancing date
        """
        ic_series = {}
        dates = sorted(self.scored.keys())

        for dt in dates:
            scores = self.scored[dt].get(factor) if factor in self.scored[dt].columns \
                     else self.scored[dt].get("COMPOSITE")
            if scores is None:
                continue
            scores = scores.dropna()

            # Forward return: compounded over horizon trading days
            try:
                dt_loc = self.returns.index.get_indexer([dt], method="ffill")[0]
                end_loc = dt_loc + horizon
                if end_loc >= len(self.returns):
                    continue
                fwd = (1 + self.returns.iloc[dt_loc + 1: end_loc + 1]).prod() - 1
            except Exception:
                continue

            common = scores.index.intersection(fwd.index)
            if len(common) < 20:
                continue

            ic, _ = stats.spearmanr(scores[common], fwd[common])
            ic_series[dt] = float(ic)

        return pd.Series(ic_series, name=f"IC_{factor}_{horizon}d")

    # ------------------------------------------------------------------
    # IC summary statistics
    # ------------------------------------------------------------------

    def ic_summary(
        self,
        horizons: List[int] = [5, 21, 63, 126, 252],
    ) -> pd.DataFrame:
        """
        IC mean, std, t-stat, ICIR, and hit rate for each factor × horizon.

        Returns
        -------
        pd.DataFrame  — MultiIndex (factor, horizon)
        """
        all_factors = [f for f in FACTOR_NAMES + ["COMPOSITE"]
                       if any(f in df.columns for df in self.scored.values())]

        rows = []
        n_tests = len(all_factors) * len(horizons)   # for Bonferroni

        for factor in all_factors:
            for h in horizons:
                ic_ts = self.compute_ic(factor=factor, horizon=h)
                if ic_ts.empty:
                    continue

                n   = len(ic_ts)
                mu  = ic_ts.mean()
                std = ic_ts.std(ddof=1)
                t   = mu / (std / np.sqrt(n)) if std > 0 else np.nan

                # Bonferroni-corrected p-value
                p_raw = 2 * (1 - stats.t.cdf(abs(t), df=n - 1)) if not np.isnan(t) else np.nan
                p_bonf = min(p_raw * n_tests, 1.0) if not np.isnan(p_raw) else np.nan

                rows.append({
                    "factor":        factor,
                    "horizon_days":  h,
                    "IC_mean":       round(mu, 4),
                    "IC_std":        round(std, 4),
                    "t_stat":        round(t, 3) if not np.isnan(t) else np.nan,
                    "p_value":       round(p_raw, 4) if not np.isnan(p_raw) else np.nan,
                    "p_bonferroni":  round(p_bonf, 4) if not np.isnan(p_bonf) else np.nan,
                    "ICIR":          round(mu / std, 3) if std > 0 else np.nan,
                    "hit_rate":      round((ic_ts > 0).mean(), 3),
                    "n_periods":     n,
                    "significant":   p_bonf < 0.05 if not np.isnan(p_bonf) else False,
                })

        df = pd.DataFrame(rows).set_index(["factor", "horizon_days"])
        return df

    # ------------------------------------------------------------------
    # IC decay curve
    # ------------------------------------------------------------------

    def ic_decay(
        self,
        factor:   str = "COMPOSITE",
        horizons: List[int] = [1, 5, 10, 21, 42, 63, 126, 252],
    ) -> pd.Series:
        """
        IC as a function of forecast horizon — the "decay curve."

        A factor with IC still positive at 63-day horizon is a long-horizon
        signal (better for low-turnover strategies).
        """
        ics = {}
        for h in horizons:
            ic_ts = self.compute_ic(factor=factor, horizon=h)
            ics[h] = ic_ts.mean() if not ic_ts.empty else np.nan
        return pd.Series(ics, name=f"IC_decay_{factor}")

    # ------------------------------------------------------------------
    # Cross-sectional factor correlation
    # ------------------------------------------------------------------

    def factor_correlation(self) -> pd.DataFrame:
        """
        Average cross-sectional Pearson correlation matrix across all dates.
        High correlation (|ρ| > 0.5) suggests factor redundancy.
        """
        corr_matrices = []
        factors = [f for f in FACTOR_NAMES if any(
            f in df.columns for df in self.scored.values()
        )]

        for dt, df in self.scored.items():
            sub = df[factors].dropna()
            if len(sub) < 20:
                continue
            corr_matrices.append(sub.corr(method="pearson"))

        if not corr_matrices:
            return pd.DataFrame()

        avg = pd.concat(corr_matrices).groupby(level=0).mean()
        return avg.reindex(factors, axis=0).reindex(factors, axis=1)
