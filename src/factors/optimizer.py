"""
optimizer.py
============
Factor-constrained mean-variance portfolio optimizer.

Construction
------------
Given a vector of alpha scores α (the composite factor signal) and a
covariance matrix Σ, solve:

    max  αᵀw - λ/2 · wᵀΣw
    s.t. Σ w_i = 1
         w_i ≥ 0       (long-only constraint)
         |w_i| ≤ w_max  (position concentration limit)
         |f_k exposure| ≤ f_limit  (factor neutrality constraints)
         TE(w, w_bench) ≤ TE_max   (tracking error budget)

The factor neutrality constraints prevent the portfolio from being
accidentally dominated by a single factor (e.g. all momentum, no value).

Solver: CVXPY with CLARABEL backend. Falls back to scipy SLSQP.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False
    logger.warning("cvxpy not installed — using scipy SLSQP fallback")


class FactorConstrainedOptimizer:
    """
    Long-only factor-constrained portfolio optimizer.

    Parameters
    ----------
    alpha_scores : pd.Series — composite signal per ticker (z-scored)
    cov_matrix   : pd.DataFrame — covariance matrix of returns
    factor_scores: pd.DataFrame — raw factor exposures (tickers × factors)
    benchmark_w  : pd.Series, optional — benchmark weights for TE constraint
    """

    def __init__(
        self,
        alpha_scores:  pd.Series,
        cov_matrix:    pd.DataFrame,
        factor_scores: Optional[pd.DataFrame] = None,
        benchmark_w:   Optional[pd.Series]    = None,
        risk_aversion: float = 1.0,
        w_max:         float = 0.05,        # max position size
        te_max:        float = 0.04,        # max annualised tracking error
        factor_limit:  float = 0.3,         # max |z-score| on any factor
    ) -> None:
        # Align on common tickers
        common = alpha_scores.dropna().index
        if cov_matrix is not None:
            common = common.intersection(cov_matrix.index)
        if factor_scores is not None:
            common = common.intersection(factor_scores.index)

        self.alpha   = alpha_scores.reindex(common).fillna(0).values
        self.tickers = common.tolist()
        n = len(common)

        # Covariance
        if cov_matrix is not None:
            self.Sigma = cov_matrix.reindex(common, axis=0).reindex(common, axis=1).values
        else:
            self.Sigma = np.eye(n) * (0.20 / np.sqrt(252)) ** 2

        # Factor exposures for constraints
        self.F = (factor_scores.reindex(common).fillna(0).values
                  if factor_scores is not None else None)

        # Benchmark
        self.w_bench = (benchmark_w.reindex(common).fillna(1 / n).values
                        if benchmark_w is not None else np.ones(n) / n)

        self.lam          = risk_aversion
        self.w_max        = w_max
        self.te_max       = te_max
        self.factor_limit = factor_limit

    # ------------------------------------------------------------------
    # CVXPY solver
    # ------------------------------------------------------------------

    def _solve_cvxpy(self) -> Tuple[np.ndarray, Dict]:
        import cvxpy as cp
        n   = len(self.alpha)
        w   = cp.Variable(n, name="weights")
        obj = cp.Maximize(self.alpha @ w - self.lam / 2 * cp.quad_form(w, self.Sigma))

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.w_max,
            # Tracking error constraint: (w-wb)ᵀΣ(w-wb) ≤ TE²/252 (quarterly)
            cp.quad_form(w - self.w_bench, self.Sigma) <= (self.te_max ** 2 / 4),
        ]

        # Factor neutrality
        if self.F is not None:
            for k in range(self.F.shape[1]):
                constraints += [
                    self.F[:, k] @ w <= self.factor_limit,
                    self.F[:, k] @ w >= -self.factor_limit,
                ]

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            logger.warning("CVXPY infeasible — returning equal weight")
            ew = np.ones(n) / n
            return ew, {"status": "infeasible"}

        w_opt = np.clip(w.value, 0, 1)
        w_opt /= w_opt.sum()

        info = {
            "status":        prob.status,
            "objective":     float(prob.value),
            "ex_ante_te":    float(np.sqrt(max(0, (w_opt - self.w_bench) @ self.Sigma @ (w_opt - self.w_bench)) * 4)),
            "port_vol_ann":  float(np.sqrt(max(0, w_opt @ self.Sigma @ w_opt) * 252)),
        }
        return w_opt, info

    # ------------------------------------------------------------------
    # scipy SLSQP fallback
    # ------------------------------------------------------------------

    def _solve_scipy(self) -> Tuple[np.ndarray, Dict]:
        from scipy.optimize import minimize
        n = len(self.alpha)

        def neg_obj(w):
            return -(self.alpha @ w - self.lam / 2 * w @ self.Sigma @ w)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        if self.F is not None:
            for k in range(self.F.shape[1]):
                fk = self.F[:, k]
                constraints += [
                    {"type": "ineq", "fun": lambda w, f=fk: self.factor_limit - f @ w},
                    {"type": "ineq", "fun": lambda w, f=fk: f @ w + self.factor_limit},
                ]

        bounds = [(0, self.w_max)] * n
        res = minimize(neg_obj, np.ones(n) / n, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})

        w_opt = np.clip(res.x, 0, 1)
        w_opt /= w_opt.sum()
        return w_opt, {"status": "slsqp_ok" if res.success else "slsqp_failed"}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[pd.Series, Dict]:
        """
        Run the optimiser.

        Returns
        -------
        weights : pd.Series  — optimal portfolio weights (indexed by ticker)
        info    : dict       — solver diagnostics
        """
        if _HAS_CVXPY:
            w_arr, info = self._solve_cvxpy()
        else:
            w_arr, info = self._solve_scipy()

        weights = pd.Series(w_arr, index=self.tickers, name="weight")
        return weights, info


def rolling_covariance(
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    window:   int = 252,
    min_obs:  int = 63,
) -> pd.DataFrame:
    """
    Compute a sample covariance matrix over a trailing window.
    Uses Ledoit-Wolf shrinkage toward the scaled identity matrix.
    """
    idx_end   = returns.index.get_indexer([end_date], method="ffill")[0]
    idx_start = max(idx_end - window, 0)
    window_ret = returns.iloc[idx_start:idx_end].dropna(how="all", axis=1)

    # Drop tickers with too few observations
    ok = window_ret.notna().sum() >= min_obs
    window_ret = window_ret.loc[:, ok]

    # Sample covariance
    S = window_ret.cov().values
    n, T = S.shape[0], len(window_ret)

    # Ledoit-Wolf analytical shrinkage (Oracle approximating)
    mu_hat = np.trace(S) / n
    rho    = min(((n + 1 - 2) / ((T - 2) * (n + 1 - 2) + n)), 1.0)
    S_shrunk = (1 - rho) * S + rho * mu_hat * np.eye(n)

    return pd.DataFrame(S_shrunk, index=window_ret.columns, columns=window_ret.columns)
