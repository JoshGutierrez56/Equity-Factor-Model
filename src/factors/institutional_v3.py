"""Institutional version-3 portfolio research for the frozen equity signal.

The implementation borrows proven ideas rather than performance claims from
cvxportfolio (explicit objectives/constraints/costs), skfolio (purged temporal
model selection), Alphalens (factor diagnostics), and PyPortfolioOpt
(shrinkage-risk comparators). The existing signal and earlier result bundles
remain immutable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf

from factors.real_model import _performance_metrics, _turnover


@dataclass(frozen=True)
class OptimizerCandidate:
    name: str
    expected_return_scale: float
    risk_aversion: float
    turnover_penalty: float


@dataclass(frozen=True)
class InstitutionalV3Spec:
    """Frozen version-3 research choices, never tuned on the reporting result."""

    signal: str = "COMPOSITE"
    evaluation_start: str = "2010-01-01"
    development_end: str = "2019-12-31"
    temporal_assessment_start: str = "2021-01-01"
    candidate_count: int = 200
    daily_lookback_days: int = 252
    minimum_daily_observations: int = 126
    minimum_daily_coverage: float = 0.70
    liquidity_lookback_days: int = 63
    gross_exposure: float = 2.0
    maximum_absolute_weight: float = 0.02
    maximum_monthly_turnover: float = 1.75
    beta_tolerance: float = 0.05
    size_tolerance: float = 0.05
    sector_tolerance: float = 0.04
    primary_aum_usd: float = 100_000_000.0
    aum_stress_usd: tuple[float, ...] = (10_000_000.0, 100_000_000.0, 500_000_000.0)
    primary_borrow_bps: float = 150.0
    borrow_stress_bps: tuple[float, ...] = (50.0, 150.0, 300.0)
    nonlinear_impact_coefficient: float = 0.10
    adv_participation_cap: float = 0.10
    purge_months: int = 12
    selection_turnover_penalty: float = 0.25
    selection_drawdown_penalty: float = 0.25
    candidates: tuple[OptimizerCandidate, ...] = (
        OptimizerCandidate("conservative", 0.0040, 8.0, 0.0025),
        OptimizerCandidate("balanced", 0.0050, 5.0, 0.0015),
        OptimizerCandidate("assertive", 0.0060, 3.0, 0.0010),
    )
    development_folds: tuple[tuple[str, str], ...] = (
        ("2010-01-01", "2012-12-31"),
        ("2014-01-01", "2016-12-31"),
        ("2018-01-01", "2019-12-31"),
    )
    solver_order: tuple[str, ...] = ("CLARABEL", "OSQP", "SCS")
    bootstrap_repetitions: int = 1000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260722

    def public_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["candidates"] = [asdict(candidate) for candidate in self.candidates]
        output["aum_stress_usd"] = list(self.aum_stress_usd)
        output["borrow_stress_bps"] = list(self.borrow_stress_bps)
        output["development_folds"] = [list(fold) for fold in self.development_folds]
        output["solver_order"] = list(self.solver_order)
        output["signal_policy"] = "unchanged frozen version-2 composite"
        output["alpha_forecast_policy"] = (
            "cross-sectional composite forecast residualized against lagged beta, "
            "log-size, and sector exposures before constrained optimization"
        )
        output["risk_model"] = "252-day Ledoit-Wolf covariance; no forward returns"
        output["cost_model"] = (
            "lagged CRSP closing bid/ask spread with conservative fallback; "
            "ADV/volatility nonlinear impact; explicit short-borrow stress"
        )
        output["validation_policy"] = (
            "candidate selected only from purged, pre-2020 development folds; "
            "2021-2024 is a retrospective temporal assessment, not a pristine holdout"
        )
        return output


def institutional_specification_hash(
    baseline_sha256: str,
    engineering_sha256: str,
    spec: InstitutionalV3Spec | None = None,
) -> str:
    spec = spec or InstitutionalV3Spec()
    payload = {
        "baseline_specification_sha256": baseline_sha256,
        "portfolio_engineering_sha256": engineering_sha256,
        "institutional_v3": spec.public_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _daily_snapshot(
    formation_date: pd.Timestamp,
    cross: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV3Spec,
) -> dict[str, Any] | None:
    """Create one point-in-time shrinkage-risk and liquidity snapshot."""
    cross = cross.drop_duplicates("permco").copy()
    cross = cross.dropna(subset=["permno", spec.signal, "fwd_1m"])
    if len(cross) < 100:
        return None
    cross["_abs_score"] = pd.to_numeric(cross[spec.signal], errors="coerce").abs()
    cross = cross.nlargest(min(spec.candidate_count, len(cross)), "_abs_score")
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
    eligible_permnos = coverage[
        (coverage >= spec.minimum_daily_coverage)
        & (return_matrix.notna().sum() >= spec.minimum_daily_observations)
    ].index
    cross = cross[cross["permno"].isin(eligible_permnos)].copy()
    if len(cross) < 100:
        return None
    # Guarantee enough names on both sides to support the frozen box constraint.
    positive = cross[pd.to_numeric(cross[spec.signal], errors="coerce") > 0]
    negative = cross[pd.to_numeric(cross[spec.signal], errors="coerce") < 0]
    minimum_side = int(np.ceil(1.0 / spec.maximum_absolute_weight))
    if len(positive) < minimum_side or len(negative) < minimum_side:
        return None
    permnos = cross["permno"].astype(int).tolist()
    matrix = return_matrix.reindex(columns=permnos).astype(float)
    matrix = matrix.clip(-0.30, 0.30)
    matrix = matrix.fillna(matrix.mean()).fillna(0.0)
    covariance = LedoitWolf(assume_centered=False).fit(matrix.to_numpy()).covariance_
    covariance = (covariance + covariance.T) / 2.0 * 21.0
    covariance += np.eye(len(covariance)) * 1e-8

    market = matrix.mean(axis=1)
    market_variance = float(market.var(ddof=1))
    betas = []
    for column in matrix:
        value = (
            float(matrix[column].cov(market) / market_variance)
            if market_variance > 1e-12
            else 1.0
        )
        betas.append(value)

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
    cross["raw_score_z"] = _zscore(cross[spec.signal]).clip(-3.0, 3.0)
    sector_controls = pd.get_dummies(
        cross["sector"].fillna(-1).astype(str), drop_first=True, dtype=float
    ).to_numpy(dtype=float)
    controls = np.column_stack([
        np.ones(len(cross)),
        cross["beta"].to_numpy(dtype=float),
        cross["size_z"].to_numpy(dtype=float),
        sector_controls,
    ])
    raw_score = cross["raw_score_z"].to_numpy(dtype=float)
    ridge = np.eye(controls.shape[1]) * 1e-8
    coefficients = np.linalg.pinv(controls.T @ controls + ridge) @ (controls.T @ raw_score)
    residual_score = raw_score - controls @ coefficients
    cross["score_z"] = _zscore(pd.Series(residual_score, index=cross.index)).clip(-3.0, 3.0)
    return {
        "date": end,
        "cross": cross,
        "covariance": covariance,
        "risk_observations": int(len(matrix)),
        "risk_universe": int(len(cross)),
    }


def build_daily_snapshots(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV3Spec | None = None,
) -> list[dict[str, Any]]:
    """Build point-in-time risk snapshots without using future information."""
    spec = spec or InstitutionalV3Spec()
    frame = panel[panel["date"] >= pd.Timestamp(spec.evaluation_start)]
    output = []
    for formation_date, cross in frame.groupby("date", sort=True):
        snapshot = _daily_snapshot(formation_date, cross, daily, spec)
        if snapshot is not None:
            output.append(snapshot)
    return output


def _sector_matrix(cross: pd.DataFrame) -> np.ndarray:
    dummies = pd.get_dummies(cross["sector"].fillna(-1).astype(str), dtype=float)
    return dummies.to_numpy(dtype=float).T


def _solve_weights(
    snapshot: dict[str, Any],
    previous: pd.Series,
    candidate: OptimizerCandidate,
    spec: InstitutionalV3Spec,
) -> tuple[pd.Series, dict[str, Any]]:
    cross = snapshot["cross"]
    index = cross.index.astype(int)
    n = len(cross)
    score = cross["score_z"].to_numpy(dtype=float)
    positive = score > 0
    negative = score < 0
    prior = previous.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    exit_turnover = float(previous.drop(index, errors="ignore").abs().sum())
    covariance = np.asarray(snapshot["covariance"], dtype=float)
    spread = cross["spread_fraction"].to_numpy(dtype=float)
    capacity_weight = np.clip(
        cross["adv_usd"].to_numpy(dtype=float) / spec.primary_aum_usd,
        1e-4,
        1.0,
    )

    w = cp.Variable(n)
    trade = w - prior
    expected = candidate.expected_return_scale * score @ w
    risk = candidate.risk_aversion * cp.quad_form(w, cp.psd_wrap(covariance))
    linear_cost = cp.sum(cp.multiply(spread, cp.abs(trade)))
    impact_proxy = cp.sum(cp.multiply(1.0 / capacity_weight, cp.square(trade)))
    borrow = spec.primary_borrow_bps * 1e-4 / 12.0 * cp.sum(cp.pos(-w))
    objective = cp.Maximize(
        expected
        - risk
        - candidate.turnover_penalty * cp.norm1(trade)
        - linear_cost
        - 0.01 * impact_proxy
        - borrow
    )
    sectors = _sector_matrix(cross)
    constraints = [
        cp.sum(w[positive]) == spec.gross_exposure / 2.0,
        cp.sum(w[negative]) == -spec.gross_exposure / 2.0,
        w[positive] >= 0.0,
        w[negative] <= 0.0,
        cp.abs(w) <= spec.maximum_absolute_weight,
        cp.abs(cross["beta"].to_numpy(dtype=float) @ w) <= spec.beta_tolerance,
        cp.abs(cross["size_z"].to_numpy(dtype=float) @ w) <= spec.size_tolerance,
        cp.abs(sectors @ w) <= spec.sector_tolerance,
    ]
    if not previous.empty:
        constraints.append(cp.norm1(trade) + exit_turnover <= spec.maximum_monthly_turnover)
    problem = cp.Problem(objective, constraints)
    status = "not_solved"
    solver_used = "none"
    for solver in spec.solver_order:
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, warm_start=True, verbose=False)
        except Exception:
            continue
        status = str(problem.status)
        solver_used = solver
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and w.value is not None:
            break
    if w.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"v3 optimizer failed at {snapshot['date'].date()} for "
            f"{candidate.name}: {status}; exit_turnover={exit_turnover:.6f}"
        )
    weights = pd.Series(np.asarray(w.value).reshape(-1), index=index, dtype=float)
    weights[weights.abs() < 1e-8] = 0.0
    return weights, {
        "solver": solver_used,
        "solver_status": status,
        "objective_value": float(problem.value),
        "exit_turnover": exit_turnover,
    }


def _realized_costs(
    weights: pd.Series,
    previous: pd.Series,
    cross: pd.DataFrame,
    aum_usd: float,
    borrow_bps: float,
    spec: InstitutionalV3Spec,
) -> dict[str, float]:
    names = weights.index.union(previous.index)
    trade = weights.reindex(names, fill_value=0.0) - previous.reindex(names, fill_value=0.0)
    aligned = cross.reindex(names)
    fallback_spread = float(cross["spread_fraction"].median())
    spread = aligned["spread_fraction"].fillna(fallback_spread).clip(0.0002, 0.02)
    spread_cost = float((trade.abs() * spread).sum())
    fallback_adv = float(cross["adv_usd"].median())
    adv = aligned["adv_usd"].fillna(fallback_adv).clip(lower=100_000.0)
    fallback_vol = float(cross["daily_volatility"].median())
    volatility = aligned["daily_volatility"].fillna(fallback_vol).clip(0.002, 0.20)
    dollars = trade.abs() * float(aum_usd)
    participation = (dollars / adv).clip(lower=0.0)
    impact_rate = spec.nonlinear_impact_coefficient * volatility * np.sqrt(participation)
    impact_cost = float((trade.abs() * impact_rate).sum())
    short_gross = float((-weights.clip(upper=0.0)).sum())
    borrow_cost = short_gross * float(borrow_bps) * 1e-4 / 12.0
    capacity_breaches = int((participation > spec.adv_participation_cap).sum())
    return {
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "borrow_cost": borrow_cost,
        "total_cost": spread_cost + impact_cost + borrow_cost,
        "maximum_adv_participation": float(participation.max()) if len(participation) else 0.0,
        "capacity_breaches": capacity_breaches,
    }


def run_candidate_backtest(
    snapshots: Iterable[dict[str, Any]],
    candidate: OptimizerCandidate,
    spec: InstitutionalV3Spec | None = None,
) -> pd.DataFrame:
    """Run one sequential candidate with explicit realized cost scenarios."""
    spec = spec or InstitutionalV3Spec()
    previous = pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        cross = snapshot["cross"]
        weights, solve = _solve_weights(snapshot, previous, candidate, spec)
        forward = cross["fwd_1m"].reindex(weights.index).astype(float)
        gross_return = float((weights * forward).sum())
        turnover = _turnover(weights, previous)
        benchmark_weights = cross["market_equity_millions"].clip(lower=0.0)
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
        benchmark_return = float((benchmark_weights * cross["fwd_1m"]).sum())
        sectors = _sector_matrix(cross)
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
                    "candidate": candidate.name,
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
                    "beta_exposure": float((cross["beta"] * weights).sum()),
                    "size_exposure": float((cross["size_z"] * weights).sum()),
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


def _primary_scenario(monthly: pd.DataFrame, spec: InstitutionalV3Spec) -> pd.DataFrame:
    return monthly[
        np.isclose(monthly["aum_usd"], spec.primary_aum_usd)
        & np.isclose(monthly["borrow_bps"], spec.primary_borrow_bps)
    ].copy()


def select_candidate(
    candidate_results: dict[str, pd.DataFrame],
    spec: InstitutionalV3Spec | None = None,
) -> tuple[str, pd.DataFrame]:
    """Select once using only frozen pre-2020 purged development folds."""
    spec = spec or InstitutionalV3Spec()
    rows = []
    for name, monthly in candidate_results.items():
        primary = _primary_scenario(monthly, spec)
        fold_scores = []
        for fold_number, (start, end) in enumerate(spec.development_folds, start=1):
            fold = primary[
                (primary["date"] >= pd.Timestamp(start))
                & (primary["date"] <= pd.Timestamp(end))
                & (primary["date"] <= pd.Timestamp(spec.development_end))
            ].copy()
            metrics = _performance_metrics(fold.sort_values("date"))
            if not metrics:
                continue
            selection_score = (
                metrics["sharpe"]
                - spec.selection_turnover_penalty * metrics["average_monthly_turnover"]
                - spec.selection_drawdown_penalty * abs(metrics["max_drawdown"])
            )
            fold_scores.append(float(selection_score))
            rows.append({
                "candidate": name,
                "fold": fold_number,
                "start": start,
                "end": end,
                "n_months": metrics["n_months"],
                "sharpe": metrics["sharpe"],
                "average_monthly_turnover": metrics["average_monthly_turnover"],
                "max_drawdown": metrics["max_drawdown"],
                "selection_score": selection_score,
            })
        rows.append({
            "candidate": name,
            "fold": "median",
            "start": spec.evaluation_start,
            "end": spec.development_end,
            "n_months": int(sum(row["candidate"] == name and row["fold"] != "median" for row in rows)),
            "sharpe": np.nan,
            "average_monthly_turnover": np.nan,
            "max_drawdown": np.nan,
            "selection_score": float(np.median(fold_scores)) if fold_scores else -np.inf,
        })
    table = pd.DataFrame(rows)
    median = table[table["fold"] == "median"].copy()
    candidate_order = {candidate.name: i for i, candidate in enumerate(spec.candidates)}
    median["_order"] = median["candidate"].map(candidate_order)
    winner = median.sort_values(["selection_score", "_order"], ascending=[False, True]).iloc[0]
    return str(winner["candidate"]), table.drop(columns=["_order"], errors="ignore")


def summarize_v3(monthly: pd.DataFrame, spec: InstitutionalV3Spec | None = None) -> pd.DataFrame:
    spec = spec or InstitutionalV3Spec()
    rows = []
    for period in ("full_retrospective", "development", "temporal_assessment"):
        sample = monthly if period == "full_retrospective" else monthly[monthly["period"] == period]
        for (candidate, aum, borrow), group in sample.groupby(["candidate", "aum_usd", "borrow_bps"]):
            metrics = _performance_metrics(group.sort_values("date"))
            if metrics:
                rows.append({
                    "period": period,
                    "candidate": candidate,
                    "aum_usd": aum,
                    "borrow_bps": borrow,
                    **metrics,
                    "mean_spread_cost": float(group["spread_cost"].mean()),
                    "mean_impact_cost": float(group["impact_cost"].mean()),
                    "mean_borrow_cost": float(group["borrow_cost"].mean()),
                    "maximum_adv_participation": float(group["maximum_adv_participation"].max()),
                    "capacity_breaches": int(group["capacity_breaches"].sum()),
                })
    return pd.DataFrame(rows).sort_values(["period", "aum_usd", "borrow_bps"]).reset_index(drop=True)


def factor_diagnostics(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    spec: InstitutionalV3Spec | None = None,
) -> dict[str, pd.DataFrame]:
    """Alphalens-style aggregate diagnostics for the unchanged signal."""
    spec = spec or InstitutionalV3Spec()
    frame = panel[panel["date"] >= pd.Timestamp(spec.evaluation_start)].copy()
    ic_rows = []
    sector_rows = []
    autocorrelation_rows = []
    quantile_rows = []
    previous_scores = pd.Series(dtype=float)
    previous_top = set()
    previous_bottom = set()
    for date, cross in frame.groupby("date", sort=True):
        usable = cross.dropna(subset=[spec.signal]).drop_duplicates("permco").set_index("permco")
        if len(usable) < 50:
            continue
        score = usable[spec.signal].astype(float)
        if not previous_scores.empty:
            common = score.index.intersection(previous_scores.index)
            autocorrelation = score.loc[common].corr(previous_scores.loc[common], method="spearman")
        else:
            autocorrelation = np.nan
        n_side = max(1, int(len(score) * 0.20))
        top = set(score.nlargest(n_side).index)
        bottom = set(score.nsmallest(n_side).index)
        quantile_rows.append({
            "date": date,
            "rank_autocorrelation": autocorrelation,
            "top_quantile_turnover": (
                1.0 - len(top & previous_top) / len(top) if previous_top else np.nan
            ),
            "bottom_quantile_turnover": (
                1.0 - len(bottom & previous_bottom) / len(bottom) if previous_bottom else np.nan
            ),
        })
        for horizon in (1, 3, 6, 12):
            target = f"fwd_{horizon}m"
            pair = usable[[spec.signal, target]].dropna()
            if len(pair) >= 30:
                ic_rows.append({
                    "date": date,
                    "horizon_months": horizon,
                    "rank_ic": float(pair[spec.signal].corr(pair[target], method="spearman")),
                    "n": int(len(pair)),
                })
            for sector, group in usable.groupby("sector"):
                pair = group[[spec.signal, target]].dropna()
                if len(pair) >= 20:
                    sector_rows.append({
                        "date": date,
                        "sector": str(sector),
                        "horizon_months": horizon,
                        "rank_ic": float(pair[spec.signal].corr(pair[target], method="spearman")),
                        "n": int(len(pair)),
                    })
        previous_scores = score
        previous_top = top
        previous_bottom = bottom
    ic = pd.DataFrame(ic_rows)
    if not ic.empty:
        ic_summary = ic.groupby("horizon_months").agg(
            months=("date", "nunique"),
            mean_rank_ic=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            positive_month_fraction=("rank_ic", lambda x: float((x > 0).mean())),
        ).reset_index()
        ic_summary["naive_t_stat"] = (
            ic_summary["mean_rank_ic"]
            / ic_summary["rank_ic_std"]
            * np.sqrt(ic_summary["months"])
        )
    else:
        ic_summary = pd.DataFrame()
    sector = pd.DataFrame(sector_rows)
    sector_summary = (
        sector.groupby(["sector", "horizon_months"]).agg(
            months=("date", "nunique"), mean_rank_ic=("rank_ic", "mean")
        ).reset_index()
        if not sector.empty else pd.DataFrame()
    )
    turnover = pd.DataFrame(quantile_rows)

    liquidity = daily.groupby("permno").agg(
        median_adv_usd=("dollar_volume", "median"),
        median_spread_bps=("spread_fraction", lambda x: float(x.median() * 1e4)),
    )
    liquidity["adv_bucket"] = pd.qcut(
        liquidity["median_adv_usd"].rank(method="first"), 5,
        labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"],
    )
    liquidity_summary = liquidity.groupby("adv_bucket", observed=False).agg(
        securities=("median_adv_usd", "size"),
        median_adv_usd=("median_adv_usd", "median"),
        median_spread_bps=("median_spread_bps", "median"),
    ).reset_index()
    return {
        "ic_decay": ic_summary,
        "sector_ic": sector_summary,
        "turnover_autocorrelation": turnover,
        "liquidity": liquidity_summary,
    }


def paired_sharpe_difference_ci(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    repetitions: int = 1000,
    block_months: int = 12,
    seed: int = 20260722,
) -> tuple[float, float, float]:
    """Paired moving-block interval for candidate-minus-baseline Sharpe."""
    left = baseline.set_index("date")["net_return"].astype(float)
    right = candidate.set_index("date")["net_return"].astype(float)
    paired = pd.concat([left.rename("base"), right.rename("candidate")], axis=1).dropna()
    if len(paired) < max(24, block_months * 2):
        return np.nan, np.nan, np.nan

    def sharpe(values: np.ndarray) -> float:
        std = float(np.std(values, ddof=1))
        return float(np.mean(values) / std * np.sqrt(12.0)) if std > 1e-12 else np.nan

    observed = sharpe(paired["candidate"].to_numpy()) - sharpe(paired["base"].to_numpy())
    rng = np.random.default_rng(seed)
    n = len(paired)
    starts = np.arange(0, n - block_months + 1)
    draws = []
    for _ in range(repetitions):
        indices = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, start + block_months))
        sample = paired.iloc[np.asarray(indices[:n])]
        draws.append(
            sharpe(sample["candidate"].to_numpy())
            - sharpe(sample["base"].to_numpy())
        )
    lower, upper = np.nanquantile(draws, [0.025, 0.975])
    return float(observed), float(lower), float(upper)
