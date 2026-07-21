"""Retrospective version-4 signal and low-churn implementation research.

Version 4 is deliberately separate from the frozen version-2 and version-3
evidence.  It addresses two observed weaknesses with one predeclared design:

* a development-only, shrinkage IC ensemble replaces the equal-weight score;
* entry/exit hysteresis keeps existing holdings after they leave the entry set.

All dates are still retrospective.  Nothing in this module creates a
prospective alpha claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

from factors.institutional_v3 import _realized_costs, _sector_matrix, _zscore
from factors.real_model import _hac_mean_test, _performance_metrics, _turnover


@dataclass(frozen=True)
class InstitutionalV4Spec:
    """Fixed post-result follow-up choices; never a pristine holdout design."""

    signal_features: tuple[str, ...] = (
        "MOM_SCORE",
        "LV_SCORE",
        "VALUE_SCORE",
        "QUALITY_SCORE",
        "INVESTMENT_SCORE",
    )
    score_horizons: tuple[int, ...] = (1, 3, 6, 12)
    development_start: str = "2010-01-01"
    development_end: str = "2019-12-31"
    temporal_assessment_start: str = "2021-01-01"
    ensemble_shrinkage_to_equal: float = 0.50
    risk_pool_count: int = 500
    entry_rank_count: int = 200
    exit_rank_count: int = 500
    daily_lookback_days: int = 252
    minimum_daily_observations: int = 126
    minimum_daily_coverage: float = 0.70
    gross_exposure: float = 2.0
    maximum_absolute_weight: float = 0.02
    maximum_monthly_turnover: float = 0.75
    beta_tolerance: float = 0.05
    size_tolerance: float = 0.05
    sector_tolerance: float = 0.04
    expected_return_scale: float = 0.006
    risk_aversion: float = 3.0
    turnover_penalty: float = 0.001
    primary_aum_usd: float = 100_000_000.0
    aum_stress_usd: tuple[float, ...] = (10_000_000.0, 100_000_000.0, 500_000_000.0)
    primary_borrow_bps: float = 150.0
    borrow_stress_bps: tuple[float, ...] = (50.0, 150.0, 300.0)
    nonlinear_impact_coefficient: float = 0.10
    adv_participation_cap: float = 0.10
    solver_order: tuple[str, ...] = ("CLARABEL", "OSQP", "SCS")
    bootstrap_repetitions: int = 1000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260723

    def public_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key in (
            "signal_features", "score_horizons", "aum_stress_usd",
            "borrow_stress_bps", "solver_order",
        ):
            output[key] = list(output[key])
        output["signal_policy"] = (
            "nonnegative multi-horizon development IC weights, normalized then "
            "shrunk 50% toward equal weights; SIZE excluded because the portfolio "
            "is explicitly size neutral"
        )
        output["membership_policy"] = (
            "new positions require absolute residual-score rank <= 200; existing "
            "positions are explicitly carried into the next risk pool and may "
            "remain through rank 500"
        )
        output["capacity_policy"] = (
            "each absolute position is capped at the lesser of 2% and 10% of "
            "lagged median ADV divided by $100m AUM"
        )
        output["claim_policy"] = (
            "post-result retrospective follow-up; no outcome may be called "
            "validated or prospective alpha"
        )
        return output


def institutional_v4_hash(
    institutional_v3_sha256: str,
    spec: InstitutionalV4Spec | None = None,
) -> str:
    spec = spec or InstitutionalV4Spec()
    payload = {
        "institutional_v3_sha256": institutional_v3_sha256,
        "institutional_v4": spec.public_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def fit_development_ensemble(
    panel: pd.DataFrame,
    spec: InstitutionalV4Spec | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit one transparent IC ensemble using development dates only."""
    spec = spec or InstitutionalV4Spec()
    development = panel[
        (panel["date"] >= pd.Timestamp(spec.development_start))
        & (panel["date"] <= pd.Timestamp(spec.development_end))
    ].copy()
    rows: list[dict[str, Any]] = []
    for feature in spec.signal_features:
        for horizon in spec.score_horizons:
            target = f"fwd_{horizon}m"
            for date, cross in development.groupby("date", sort=True):
                pair = cross[[feature, target]].dropna()
                if len(pair) < 50:
                    continue
                rows.append({
                    "date": pd.Timestamp(date),
                    "feature": feature,
                    "horizon_months": int(horizon),
                    "rank_ic": float(pair[feature].corr(pair[target], method="spearman")),
                    "n": int(len(pair)),
                })
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        equal = 1.0 / len(spec.signal_features)
        return {feature: equal for feature in spec.signal_features}, diagnostics
    raw = diagnostics.groupby("feature")["rank_ic"].mean().reindex(spec.signal_features)
    positive = raw.clip(lower=0.0).fillna(0.0)
    if float(positive.sum()) <= 1e-12:
        ic_weights = pd.Series(1.0 / len(positive), index=positive.index)
    else:
        ic_weights = positive / positive.sum()
    equal = pd.Series(1.0 / len(positive), index=positive.index)
    shrink = float(spec.ensemble_shrinkage_to_equal)
    final = shrink * equal + (1.0 - shrink) * ic_weights
    return {str(key): float(value) for key, value in final.items()}, diagnostics


def apply_ensemble_score(
    panel: pd.DataFrame,
    weights: dict[str, float],
    output_column: str = "V4_SCORE",
) -> pd.DataFrame:
    frame = panel.copy()
    score = pd.Series(0.0, index=frame.index, dtype=float)
    available = pd.Series(0.0, index=frame.index, dtype=float)
    for feature, weight in weights.items():
        values = pd.to_numeric(frame[feature], errors="coerce")
        score = score.add(values.fillna(0.0) * float(weight), fill_value=0.0)
        available = available.add(values.notna().astype(float) * float(weight), fill_value=0.0)
    frame[output_column] = (score / available.where(available > 0.0)).where(available >= 0.80)
    return frame


def _build_snapshot(
    formation_date: pd.Timestamp,
    cross: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV4Spec,
    retained_permnos: pd.Index | None = None,
) -> dict[str, Any] | None:
    cross = cross.drop_duplicates("permco").dropna(
        subset=["permno", "V4_SCORE", "fwd_1m"]
    ).copy()
    if len(cross) < spec.risk_pool_count:
        return None
    cross["_raw_abs_score"] = cross["V4_SCORE"].abs()
    ranked = cross.nlargest(spec.risk_pool_count, "_raw_abs_score")
    retained_permnos = pd.Index([]) if retained_permnos is None else pd.Index(retained_permnos)
    retained = cross[cross["permno"].astype(int).isin(retained_permnos.astype(int))]
    cross = pd.concat([ranked, retained], ignore_index=True).drop_duplicates("permco")
    cross["permno"] = cross["permno"].astype(int)
    end = pd.Timestamp(formation_date)
    start = end - pd.Timedelta(days=int(spec.daily_lookback_days * 1.75))
    history = daily[
        (daily["date"] <= end)
        & (daily["date"] >= start)
        & (daily["permno"].isin(cross["permno"]))
    ].copy()
    if history.empty:
        return None
    return_matrix = history.pivot(index="date", columns="permno", values="ret").tail(
        spec.daily_lookback_days
    )
    coverage = return_matrix.notna().mean()
    eligible = coverage[
        (coverage >= spec.minimum_daily_coverage)
        & (return_matrix.notna().sum() >= spec.minimum_daily_observations)
    ].index
    cross = cross[cross["permno"].isin(eligible)].copy()
    if len(cross) < spec.exit_rank_count:
        return None
    permnos = cross["permno"].astype(int).tolist()
    matrix = return_matrix.reindex(columns=permnos).astype(float).clip(-0.30, 0.30)
    matrix = matrix.fillna(matrix.mean()).fillna(0.0)
    covariance = LedoitWolf(assume_centered=False).fit(matrix.to_numpy()).covariance_
    covariance = (covariance + covariance.T) / 2.0 * 21.0
    covariance += np.eye(len(covariance)) * 1e-8
    market = matrix.mean(axis=1)
    market_variance = float(market.var(ddof=1))
    betas = [
        float(matrix[column].cov(market) / market_variance)
        if market_variance > 1e-12 else 1.0
        for column in matrix
    ]
    recent = history[history["date"] > end - pd.Timedelta(days=100)].copy()
    liquidity = recent.groupby("permno").agg(
        adv_usd=("dollar_volume", "median"),
        spread_fraction=("spread_fraction", "median"),
        daily_volatility=("ret", "std"),
    )
    cross = cross.set_index("permno").loc[permnos]
    cross["beta"] = np.asarray(betas, dtype=float)
    cross["size_z"] = _zscore(np.log(cross["market_equity_millions"].clip(lower=1.0)))
    cross = cross.join(liquidity, how="left")
    cross["adv_usd"] = cross["adv_usd"].fillna(cross["adv_usd"].median()).clip(lower=100_000.0)
    cross["spread_fraction"] = cross["spread_fraction"].fillna(
        cross["spread_fraction"].median()
    ).clip(0.0002, 0.02)
    cross["daily_volatility"] = cross["daily_volatility"].fillna(
        cross["daily_volatility"].median()
    ).clip(0.002, 0.20)
    cross["raw_score_z"] = _zscore(cross["V4_SCORE"]).clip(-3.0, 3.0)
    sector_controls = pd.get_dummies(
        cross["sector"].fillna(-1).astype(str), drop_first=True, dtype=float
    ).to_numpy(dtype=float)
    controls = np.column_stack([
        np.ones(len(cross)), cross["beta"].to_numpy(dtype=float),
        cross["size_z"].to_numpy(dtype=float), sector_controls,
    ])
    raw_score = cross["raw_score_z"].to_numpy(dtype=float)
    ridge = np.eye(controls.shape[1]) * 1e-8
    coefficients = np.linalg.pinv(controls.T @ controls + ridge) @ (controls.T @ raw_score)
    residual = raw_score - controls @ coefficients
    cross["score_z"] = _zscore(pd.Series(residual, index=cross.index)).clip(-3.0, 3.0)
    cross["absolute_score_rank"] = cross["score_z"].abs().rank(
        ascending=False, method="first"
    ).astype(int)
    cross = cross[cross["absolute_score_rank"] <= spec.exit_rank_count].copy()
    covariance_frame = pd.DataFrame(covariance, index=permnos, columns=permnos)
    covariance = covariance_frame.loc[cross.index, cross.index].to_numpy(dtype=float)
    return {
        "date": end,
        "cross": cross,
        "covariance": covariance,
        "risk_observations": int(len(matrix)),
        "risk_universe": int(len(cross)),
    }


def build_v4_snapshots(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV4Spec | None = None,
) -> list[dict[str, Any]]:
    spec = spec or InstitutionalV4Spec()
    output = []
    for date, cross in panel.groupby("date", sort=True):
        if date < pd.Timestamp(spec.development_start):
            continue
        snapshot = _build_snapshot(date, cross, daily, spec)
        if snapshot is not None:
            output.append(snapshot)
    return output


def build_snapshots_and_weight_path(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV4Spec | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the risk path while explicitly carrying current holdings forward.

    This function never reads a forward return.  It exists so the holding
    buffer is applied before the risk-pool truncation, rather than after a name
    has already disappeared from the optimizer universe.
    """
    spec = spec or InstitutionalV4Spec()
    snapshots: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []
    previous = pd.Series(dtype=float)
    for date, cross in panel.groupby("date", sort=True):
        if date < pd.Timestamp(spec.development_start):
            continue
        retained = previous[previous.abs() > 1e-8].index
        snapshot = _build_snapshot(date, cross, daily, spec, retained_permnos=retained)
        if snapshot is None:
            continue
        weights, solve = _solve_weights(snapshot, previous, spec)
        snapshots.append(snapshot)
        path.append({
            "date": snapshot["date"], "weights": weights,
            "turnover": _turnover(weights, previous), **solve,
        })
        previous = weights
    return snapshots, path


def eligible_names(
    cross: pd.DataFrame,
    previous: pd.Series,
    spec: InstitutionalV4Spec | None = None,
) -> pd.Index:
    """Entry names plus buffered existing holdings, with no return information."""
    spec = spec or InstitutionalV4Spec()
    entry = cross.index[cross["absolute_score_rank"] <= spec.entry_rank_count]
    held = previous[previous.abs() > 1e-8].index.intersection(cross.index)
    buffered = cross.index[
        (cross.index.isin(held))
        & (cross["absolute_score_rank"] <= spec.exit_rank_count)
    ]
    return entry.union(buffered, sort=False)


def capacity_position_caps(
    cross: pd.DataFrame,
    spec: InstitutionalV4Spec | None = None,
) -> pd.Series:
    spec = spec or InstitutionalV4Spec()
    capacity = (
        spec.adv_participation_cap
        * cross["adv_usd"].astype(float)
        / spec.primary_aum_usd
    )
    return capacity.clip(lower=1e-5, upper=spec.maximum_absolute_weight)


def _solve_weights(
    snapshot: dict[str, Any],
    previous: pd.Series,
    spec: InstitutionalV4Spec,
) -> tuple[pd.Series, dict[str, Any]]:
    full_cross = snapshot["cross"]
    names = eligible_names(full_cross, previous, spec)
    cross = full_cross.loc[names].copy()
    covariance_frame = pd.DataFrame(
        snapshot["covariance"], index=full_cross.index, columns=full_cross.index
    )
    covariance = covariance_frame.loc[names, names].to_numpy(dtype=float)
    score = cross["score_z"].to_numpy(dtype=float)
    positive = score > 0
    negative = score < 0
    minimum_side = int(np.ceil(1.0 / spec.maximum_absolute_weight))
    if positive.sum() < minimum_side or negative.sum() < minimum_side:
        raise RuntimeError(f"v4 insufficient names per side at {snapshot['date'].date()}")
    prior = previous.reindex(names, fill_value=0.0).to_numpy(dtype=float)
    exit_turnover = float(previous.drop(names, errors="ignore").abs().sum())
    spread = cross["spread_fraction"].to_numpy(dtype=float)
    cap = capacity_position_caps(cross, spec).to_numpy(dtype=float)
    capacity_weight = np.clip(
        cross["adv_usd"].to_numpy(dtype=float) / spec.primary_aum_usd, 1e-4, 1.0
    )
    w = cp.Variable(len(cross))
    trade = w - prior
    objective = cp.Maximize(
        spec.expected_return_scale * score @ w
        - spec.risk_aversion * cp.quad_form(w, cp.psd_wrap(covariance))
        - spec.turnover_penalty * cp.norm1(trade)
        - cp.sum(cp.multiply(spread, cp.abs(trade)))
        - 0.01 * cp.sum(cp.multiply(1.0 / capacity_weight, cp.square(trade)))
        - spec.primary_borrow_bps * 1e-4 / 12.0 * cp.sum(cp.pos(-w))
    )
    sectors = _sector_matrix(cross)
    constraints = [
        cp.sum(w[positive]) == spec.gross_exposure / 2.0,
        cp.sum(w[negative]) == -spec.gross_exposure / 2.0,
        w[positive] >= 0.0,
        w[negative] <= 0.0,
        cp.abs(w) <= cap,
        cp.abs(cross["beta"].to_numpy(dtype=float) @ w) <= spec.beta_tolerance,
        cp.abs(cross["size_z"].to_numpy(dtype=float) @ w) <= spec.size_tolerance,
        cp.abs(sectors @ w) <= spec.sector_tolerance,
    ]
    if not previous.empty:
        constraints.append(cp.norm1(trade) + exit_turnover <= spec.maximum_monthly_turnover)
    problem = cp.Problem(objective, constraints)
    status, solver_used = "not_solved", "none"
    for solver in spec.solver_order:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        status, solver_used = str(problem.status), solver
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and w.value is not None:
            break
    if w.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"v4 optimizer failed at {snapshot['date'].date()}: {status}; "
            f"eligible={len(cross)}; exit_turnover={exit_turnover:.6f}"
        )
    weights = pd.Series(np.asarray(w.value).reshape(-1), index=names, dtype=float)
    weights[weights.abs() < 1e-8] = 0.0
    return weights, {
        "solver": solver_used,
        "solver_status": status,
        "objective_value": float(problem.value),
        "exit_turnover": exit_turnover,
        "eligible_names": int(len(names)),
        "buffered_holdings": int(len(names.difference(
            full_cross.index[full_cross["absolute_score_rank"] <= spec.entry_rank_count]
        ))),
        "minimum_position_cap": float(cap.min()),
    }


def build_weight_path(
    snapshots: Iterable[dict[str, Any]],
    spec: InstitutionalV4Spec | None = None,
) -> list[dict[str, Any]]:
    """Sequentially solve weights without computing any forward return."""
    spec = spec or InstitutionalV4Spec()
    previous = pd.Series(dtype=float)
    path: list[dict[str, Any]] = []
    for snapshot in snapshots:
        weights, solve = _solve_weights(snapshot, previous, spec)
        path.append({
            "date": snapshot["date"],
            "weights": weights,
            "turnover": _turnover(weights, previous),
            **solve,
        })
        previous = weights
    return path


def feasibility_scan(
    snapshots: Iterable[dict[str, Any]],
    spec: InstitutionalV4Spec | None = None,
    weight_path: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return public constraint-only diagnostics for a solved weight path."""
    spec = spec or InstitutionalV4Spec()
    path = weight_path if weight_path is not None else build_weight_path(snapshots, spec)
    return pd.DataFrame([
        {key: value for key, value in row.items() if key != "weights"}
        for row in path
    ])


def run_v4_backtest(
    snapshots: Iterable[dict[str, Any]],
    spec: InstitutionalV4Spec | None = None,
    weight_path: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    spec = spec or InstitutionalV4Spec()
    snapshots = list(snapshots)
    path = weight_path if weight_path is not None else build_weight_path(snapshots, spec)
    if len(path) != len(snapshots):
        raise ValueError("weight path and snapshots have different lengths")
    previous = pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for snapshot, solved in zip(snapshots, path):
        cross = snapshot["cross"]
        weights = solved["weights"]
        solve = {key: value for key, value in solved.items() if key not in {"date", "weights", "turnover"}}
        gross_return = float((weights * cross["fwd_1m"].reindex(weights.index)).sum())
        turnover = _turnover(weights, previous)
        benchmark_weights = cross["market_equity_millions"].clip(lower=0.0)
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
        benchmark_return = float((benchmark_weights * cross["fwd_1m"]).sum())
        sectors = _sector_matrix(cross.loc[weights.index])
        for aum in spec.aum_stress_usd:
            for borrow_bps in spec.borrow_stress_bps:
                costs = _realized_costs(weights, previous, cross, aum, borrow_bps, spec)
                rows.append({
                    "date": snapshot["date"],
                    "period": (
                        "temporal_assessment"
                        if snapshot["date"] >= pd.Timestamp(spec.temporal_assessment_start)
                        else "development"
                    ),
                    "strategy": "v4_hysteresis",
                    "aum_usd": float(aum),
                    "borrow_bps": float(borrow_bps),
                    "gross_return": gross_return,
                    "net_return": gross_return - costs["total_cost"],
                    "turnover": turnover,
                    "benchmark_return": benchmark_return,
                    "universe_size": int(len(cross)),
                    "long_count": int((weights > 0).sum()),
                    "short_count": int((weights < 0).sum()),
                    "gross_exposure": float(weights.abs().sum()),
                    "net_exposure": float(weights.sum()),
                    "beta_exposure": float((cross["beta"].reindex(weights.index) * weights).sum()),
                    "size_exposure": float((cross["size_z"].reindex(weights.index) * weights).sum()),
                    "maximum_sector_net_exposure": float(np.max(np.abs(sectors @ weights.to_numpy()))),
                    "maximum_absolute_weight": float(weights.abs().max()),
                    "effective_breadth": float(1.0 / np.square(weights.abs() / weights.abs().sum()).sum()),
                    "median_spread_bps": float(cross["spread_fraction"].median() * 1e4),
                    "median_adv_usd": float(cross["adv_usd"].median()),
                    "risk_observations": int(snapshot["risk_observations"]),
                    "risk_universe": int(snapshot["risk_universe"]),
                    **costs,
                    **solve,
                })
        previous = weights
    return pd.DataFrame(rows).sort_values(["date", "aum_usd", "borrow_bps"]).reset_index(drop=True)


def score_ic_summary(
    panel: pd.DataFrame,
    score_columns: tuple[str, ...] = ("COMPOSITE", "V4_SCORE"),
    spec: InstitutionalV4Spec | None = None,
) -> pd.DataFrame:
    spec = spec or InstitutionalV4Spec()
    rows = []
    periods = {
        "development": (
            pd.Timestamp(spec.development_start), pd.Timestamp(spec.development_end)
        ),
        "temporal_assessment": (
            pd.Timestamp(spec.temporal_assessment_start), panel["date"].max()
        ),
    }
    for period, (start, end) in periods.items():
        sample = panel[(panel["date"] >= start) & (panel["date"] <= end)]
        for score_column in score_columns:
            for horizon in spec.score_horizons:
                values = []
                for _, cross in sample.groupby("date", sort=True):
                    pair = cross[[score_column, f"fwd_{horizon}m"]].dropna()
                    if len(pair) >= 50:
                        values.append(float(pair[score_column].corr(
                            pair[f"fwd_{horizon}m"], method="spearman"
                        )))
                series = pd.Series(values, dtype=float)
                if len(series) < 3:
                    continue
                t_stat, p_value = _hac_mean_test(series, maxlags=max(0, horizon - 1))
                rows.append({
                    "period": period,
                    "score": score_column,
                    "horizon_months": int(horizon),
                    "months": int(len(series)),
                    "mean_rank_ic": float(series.mean()),
                    "rank_ic_std": float(series.std(ddof=1)),
                    "icir": float(series.mean() / series.std(ddof=1)),
                    "hac_t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "positive_month_fraction": float((series > 0).mean()),
                })
    return pd.DataFrame(rows).sort_values(["period", "score", "horizon_months"])


def factor_residual_information_ratio(
    monthly: pd.DataFrame,
    fama_french: pd.DataFrame,
) -> dict[str, float]:
    """Return FF5+momentum alpha divided by annualized residual volatility."""
    portfolio = monthly.copy()
    portfolio["factor_month"] = portfolio["date"].dt.to_period("M") + 1
    factors = fama_french.copy()
    factors["factor_month"] = factors["dateff"].dt.to_period("M")
    columns = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    merged = portfolio.merge(
        factors[["factor_month", *columns]], on="factor_month", how="inner"
    ).dropna(subset=["net_return", *columns])
    if len(merged) < 12:
        return {
            "n_months": int(len(merged)), "alpha_annualized": np.nan,
            "residual_volatility_annualized": np.nan,
            "factor_residual_information_ratio": np.nan,
        }
    X = sm.add_constant(merged[columns].astype(float), has_constant="add")
    fit = sm.OLS(merged["net_return"].astype(float), X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3}
    )
    alpha = float(fit.params["const"] * 12.0)
    residual_vol = float(fit.resid.std(ddof=1) * np.sqrt(12.0))
    return {
        "n_months": int(fit.nobs),
        "alpha_annualized": alpha,
        "alpha_hac_t_stat": float(fit.tvalues["const"]),
        "alpha_p_value": float(fit.pvalues["const"]),
        "residual_volatility_annualized": residual_vol,
        "factor_residual_information_ratio": (
            float(alpha / residual_vol) if residual_vol > 1e-12 else np.nan
        ),
        "r_squared": float(fit.rsquared),
    }


def summarize_v4(
    monthly: pd.DataFrame,
    spec: InstitutionalV4Spec | None = None,
) -> pd.DataFrame:
    spec = spec or InstitutionalV4Spec()
    rows = []
    for period in ("full_retrospective", "development", "temporal_assessment"):
        sample = monthly if period == "full_retrospective" else monthly[monthly["period"] == period]
        for (aum, borrow), group in sample.groupby(["aum_usd", "borrow_bps"]):
            metrics = _performance_metrics(group.sort_values("date"))
            if metrics:
                rows.append({
                    "period": period, "strategy": "v4_hysteresis",
                    "aum_usd": aum, "borrow_bps": borrow, **metrics,
                    "mean_spread_cost": float(group["spread_cost"].mean()),
                    "mean_impact_cost": float(group["impact_cost"].mean()),
                    "mean_borrow_cost": float(group["borrow_cost"].mean()),
                    "maximum_adv_participation": float(group["maximum_adv_participation"].max()),
                    "capacity_breaches": int(group["capacity_breaches"].sum()),
                })
    return pd.DataFrame(rows).sort_values(["period", "aum_usd", "borrow_bps"])
