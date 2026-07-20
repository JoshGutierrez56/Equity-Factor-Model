"""
backtest.py
===========
Walk-forward backtest engine with Brinson-Hood-Beebower attribution.

Rebalancing schedule: monthly (first trading day of each month).
Execution assumption: prices at next-day open (1-day implementation lag).
Transaction costs: 10 bps one-way (round-trip = 20 bps).

Attribution (Brinson-Hood-Beebower 1986)
-----------------------------------------
Total active return = Selection + Allocation + Interaction

For a factor-based portfolio, we decompose active return into:
    - Factor contribution: wᵀ · f · factor_returns
    - Specific return: residual after factor attribution

Performance metrics reported:
    CAGR, annualised vol, Sharpe, max drawdown, Calmar,
    alpha (vs SPY), beta, tracking error, information ratio
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class Backtester:
    """
    Walk-forward portfolio backtesting with factor attribution.

    Parameters
    ----------
    returns      : daily returns (wide, tickers as columns)
    scored       : {date: DataFrame} — normalised factor scores + COMPOSITE
    benchmark    : pd.Series — daily benchmark returns (e.g. SPY), optional
    tc_bps       : one-way transaction cost in basis points
    """

    def __init__(
        self,
        returns:   pd.DataFrame,
        scored:    Dict[pd.Timestamp, pd.DataFrame],
        benchmark: Optional[pd.Series] = None,
        tc_bps:    float = 10.0,
    ) -> None:
        self.returns   = returns
        self.scored    = scored
        self.benchmark = benchmark
        self.tc        = tc_bps * 1e-4

    # ------------------------------------------------------------------
    # Walk-forward engine
    # ------------------------------------------------------------------

    def run(
        self,
        optimizer_kwargs: Optional[Dict] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Run the backtest.

        Strategy: at each monthly rebalancing date, score stocks by COMPOSITE,
        optimise (if CVXPY available) or use top-N equal weight (fallback).

        Parameters
        ----------
        optimizer_kwargs : passed to FactorConstrainedOptimizer
        top_n : if set, use top-N equal weight instead of optimiser

        Returns
        -------
        pd.DataFrame  — daily portfolio returns + metadata
        """
        from factors.optimizer import FactorConstrainedOptimizer, rolling_covariance

        rebal_dates = sorted(self.scored.keys())
        port_returns: List[Dict] = []
        w_prev = pd.Series(dtype=float)

        for i, dt in enumerate(rebal_dates):
            score_df = self.scored[dt]
            if "COMPOSITE" not in score_df.columns:
                continue

            alpha = score_df["COMPOSITE"].dropna()

            # Next rebalancing date (or end of data)
            next_dt = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else self.returns.index[-1]

            # Get the trading period
            try:
                start_loc = self.returns.index.get_indexer([dt], method="bfill")[0] + 1
                end_loc   = self.returns.index.get_indexer([next_dt], method="ffill")[0] + 1
            except Exception:
                continue

            period_ret = self.returns.iloc[start_loc:end_loc]
            if period_ret.empty:
                continue

            # Determine weights
            if top_n is not None:
                # Simple top-N equal weight
                top = alpha.nlargest(top_n).index
                w   = pd.Series(1.0 / len(top), index=top)
            else:
                try:
                    cov = rolling_covariance(self.returns, dt)
                    kw  = optimizer_kwargs or {}
                    opt = FactorConstrainedOptimizer(
                        alpha_scores=alpha,
                        cov_matrix=cov,
                        factor_scores=score_df.drop(columns=["COMPOSITE"], errors="ignore"),
                        **kw,
                    )
                    w, _info = opt.solve()
                except Exception as exc:
                    logger.warning("Optimiser failed at %s: %s — top-50 equal weight", dt.date(), exc)
                    top = alpha.nlargest(50).index
                    w   = pd.Series(1.0 / len(top), index=top)

            # Transaction costs
            turnover = 0.0
            if not w_prev.empty:
                turnover = float((w - w_prev.reindex(w.index).fillna(0)).abs().sum())
                tc_drag  = turnover * self.tc
            else:
                tc_drag = 0.0

            # Portfolio daily returns
            for date, row in period_ret.iterrows():
                port_ret = (row.reindex(w.index).fillna(0) * w.values).sum() - tc_drag / len(period_ret)
                bench_ret = self.benchmark.loc[date] if (self.benchmark is not None and date in self.benchmark.index) else np.nan

                port_returns.append({
                    "date":       date,
                    "port_ret":   port_ret,
                    "bench_ret":  bench_ret,
                    "n_holdings": (w > 0).sum(),
                    "turnover":   turnover if date == period_ret.index[0] else 0.0,
                })
                tc_drag = 0.0  # only on rebalancing day

            w_prev = w

        df = pd.DataFrame(port_returns).set_index("date")
        df["cum_port"]  = (1 + df["port_ret"]).cumprod()
        df["cum_bench"] = (1 + df["bench_ret"].fillna(0)).cumprod()
        return df

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @staticmethod
    def metrics(results: pd.DataFrame, rf_annual: float = 0.045) -> Dict:
        """
        Standard performance metrics for a daily return series.

        Parameters
        ----------
        results    : output of run()
        rf_annual  : annual risk-free rate

        Returns
        -------
        dict  — metrics for portfolio and (if available) benchmark
        """
        r   = results["port_ret"].dropna()
        rf  = rf_annual / 252

        # ── CAGR ─────────────────────────────────────────────────────
        n_days = len(r)
        cagr   = (1 + r).prod() ** (252 / n_days) - 1

        # ── Volatility ────────────────────────────────────────────────
        vol    = r.std(ddof=1) * np.sqrt(252)

        # ── Sharpe ───────────────────────────────────────────────────
        sharpe = (cagr - rf_annual) / vol if vol > 0 else np.nan

        # ── Max drawdown ──────────────────────────────────────────────
        nav  = (1 + r).cumprod()
        peak = nav.expanding().max()
        mdd  = (nav / peak - 1).min()

        # ── Calmar ────────────────────────────────────────────────────
        calmar = cagr / abs(mdd) if mdd < 0 else np.nan

        out = {
            "CAGR":         round(cagr, 4),
            "Vol_Ann":      round(vol, 4),
            "Sharpe":       round(sharpe, 3),
            "Max_Drawdown": round(mdd, 4),
            "Calmar":       round(calmar, 3),
        }

        # ── Alpha / Beta vs benchmark ─────────────────────────────────
        b = results["bench_ret"].reindex(r.index).dropna()
        if len(b) > 60:
            common = r.index.intersection(b.index)
            r_c, b_c = r[common].values, b[common].values
            slope, intercept, *_ = stats.linregress(b_c, r_c)
            te  = (r_c - b_c).std() * np.sqrt(252)
            ir  = (r_c - b_c).mean() / (r_c - b_c).std() * np.sqrt(252)
            out.update({
                "Alpha_Ann": round(intercept * 252, 4),
                "Beta":      round(slope, 3),
                "Tracking_Error": round(te, 4),
                "Info_Ratio":     round(ir, 3),
            })

        return out
