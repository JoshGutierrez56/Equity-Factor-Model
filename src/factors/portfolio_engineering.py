"""Locked, point-in-time portfolio engineering for the real-data factor study.

The signal specification is intentionally unchanged.  This module tests whether
transparent risk and implementation controls convert the frozen composite score
into a more efficient portfolio.  All risk estimates use information available
at the formation month and every public output remains aggregate-only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from factors.real_model import _performance_metrics, _turnover


ENGINEERING_STRATEGIES = (
    "risk_neutral",
    "risk_turnover_managed",
    "risk_turnover_vol_scaled",
)


@dataclass(frozen=True)
class PortfolioEngineeringSpec:
    """Predeclared portfolio choices; none is fitted to realized forward returns."""

    signal: str = "COMPOSITE"
    transaction_cost_bps: tuple[float, ...] = (0.0, 10.0, 25.0)
    trailing_vol_months: int = 24
    minimum_vol_months: int = 18
    trailing_beta_months: int = 36
    minimum_beta_months: int = 24
    inverse_volatility_power: float = 1.0
    gross_exposure: float = 2.0
    maximum_absolute_weight: float = 0.015
    rebalance_fraction: float = 0.35
    no_trade_weight_change: float = 0.0005
    target_annualized_volatility: float = 0.10
    portfolio_volatility_lookback: int = 24
    portfolio_volatility_minimum_months: int = 12
    minimum_leverage_multiplier: float = 0.50
    maximum_leverage_multiplier: float = 1.50
    projection_ridge: float = 1e-8
    projection_iterations: int = 30
    bootstrap_repetitions: int = 1000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260721

    def public_dict(self) -> dict:
        output = asdict(self)
        output["transaction_cost_bps"] = list(self.transaction_cost_bps)
        output["risk_model"] = (
            "trailing diagonal volatility with rolling market beta and explicit "
            "sector/log-size exposure projection"
        )
        output["signal_policy"] = "frozen composite score; no return-fitted signal weights"
        output["implementation_lag"] = "formation at month t; return begins in month t+1"
        return output


def engineering_specification_hash(
    baseline_specification_sha256: str,
    spec: PortfolioEngineeringSpec,
) -> str:
    payload = {
        "baseline_specification_sha256": baseline_specification_sha256,
        "portfolio_engineering": spec.public_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def add_trailing_risk_features(
    panel: pd.DataFrame,
    spec: PortfolioEngineeringSpec,
) -> pd.DataFrame:
    """Add formation-date volatility, beta, and size controls without look-ahead."""
    frame = panel.sort_values(["permco", "date"]).copy()
    frame["lag_market_equity"] = frame.groupby("permco")[
        "market_equity_millions"
    ].shift(1)

    def market_return(cross: pd.DataFrame) -> float:
        usable = cross[["total_ret", "lag_market_equity"]].dropna()
        usable = usable[usable["lag_market_equity"] > 0]
        if usable.empty:
            return float(pd.to_numeric(cross["total_ret"], errors="coerce").mean())
        weights = usable["lag_market_equity"] / usable["lag_market_equity"].sum()
        return float((weights * usable["total_ret"]).sum())

    market = frame.groupby("date", sort=True).apply(
        market_return, include_groups=False
    )
    frame["market_proxy_return"] = frame["date"].map(market)

    def trailing(group: pd.DataFrame) -> pd.DataFrame:
        returns = pd.to_numeric(group["total_ret"], errors="coerce").astype(float)
        market_returns = pd.to_numeric(
            group["market_proxy_return"], errors="coerce"
        ).astype(float)
        volatility = returns.rolling(
            spec.trailing_vol_months,
            min_periods=spec.minimum_vol_months,
        ).std(ddof=1) * np.sqrt(12.0)
        covariance = returns.rolling(
            spec.trailing_beta_months,
            min_periods=spec.minimum_beta_months,
        ).cov(market_returns)
        market_variance = market_returns.rolling(
            spec.trailing_beta_months,
            min_periods=spec.minimum_beta_months,
        ).var(ddof=1)
        return pd.DataFrame(
            {
                "trailing_volatility": volatility,
                "trailing_beta": covariance / market_variance.where(
                    market_variance > 1e-12
                ),
            },
            index=group.index,
        )

    risk = frame.groupby("permco", group_keys=False).apply(
        trailing, include_groups=False
    )
    frame[["trailing_volatility", "trailing_beta"]] = risk[
        ["trailing_volatility", "trailing_beta"]
    ]
    frame["log_market_cap"] = np.log(
        pd.to_numeric(frame["market_equity_millions"], errors="coerce").where(
            frame["market_equity_millions"] > 0
        )
    )
    return frame.drop(columns=["lag_market_equity"])


def _exposure_matrix(cross: pd.DataFrame) -> pd.DataFrame:
    """Create full-rank dollar, beta, size, and sector constraint exposures."""
    index = cross.index
    beta = pd.to_numeric(cross["trailing_beta"], errors="coerce").astype(float)
    beta_fill = float(beta.dropna().median()) if beta.notna().any() else 1.0
    beta = beta.fillna(beta_fill)
    size = pd.to_numeric(cross["log_market_cap"], errors="coerce").astype(float)
    size_fill = float(size.dropna().median()) if size.notna().any() else 0.0
    size = size.fillna(size_fill)
    size_std = size.std(ddof=1)
    size = (size - size.mean()) / size_std if size_std > 1e-12 else size * 0.0
    sectors = pd.get_dummies(
        cross["sector"].fillna(-1).astype(str), prefix="sector", dtype=float
    )
    if sectors.shape[1] > 1:
        sectors = sectors.iloc[:, 1:]
    else:
        sectors = sectors.iloc[:, 0:0]
    return pd.concat(
        [
            pd.Series(1.0, index=index, name="dollar"),
            beta.rename("beta"),
            size.rename("size"),
            sectors,
        ],
        axis=1,
    ).astype(float)


def _project_neutral(
    values: pd.Series,
    exposures: pd.DataFrame,
    ridge: float,
) -> pd.Series:
    aligned = values.reindex(exposures.index, fill_value=0.0).astype(float)
    x = exposures.to_numpy(dtype=float)
    y = aligned.to_numpy(dtype=float)
    gram = x.T @ x + np.eye(x.shape[1]) * ridge
    coefficients = np.linalg.pinv(gram) @ (x.T @ y)
    return pd.Series(y - x @ coefficients, index=exposures.index, dtype=float)


def _normalize_and_cap(
    values: pd.Series,
    exposures: pd.DataFrame,
    spec: PortfolioEngineeringSpec,
) -> pd.Series:
    weights = values.astype(float).copy()
    x = exposures.to_numpy(dtype=float)
    gram = x.T @ x + np.eye(x.shape[1]) * spec.projection_ridge
    projection_coefficients = np.linalg.pinv(gram) @ x.T
    for _ in range(spec.projection_iterations):
        y = weights.reindex(exposures.index, fill_value=0.0).to_numpy(dtype=float)
        weights = pd.Series(
            y - x @ (projection_coefficients @ y),
            index=exposures.index,
            dtype=float,
        )
        gross = float(weights.abs().sum())
        if gross <= 1e-12:
            return pd.Series(dtype=float)
        weights *= spec.gross_exposure / gross
        weights = weights.clip(
            -spec.maximum_absolute_weight,
            spec.maximum_absolute_weight,
        )
    # The last operation remains the box projection so the declared cap is exact.
    # Repeated affine/box projections converge to a nearly neutral feasible point;
    # diagnostics and quality gates enforce the remaining exposure tolerances.
    return weights


def _ideal_weights(
    cross: pd.DataFrame,
    spec: PortfolioEngineeringSpec,
) -> tuple[pd.Series, pd.DataFrame]:
    usable = cross.dropna(subset=[spec.signal, "fwd_1m"]).drop_duplicates("permco")
    usable = usable.set_index("permco")
    volatility = pd.to_numeric(
        usable["trailing_volatility"], errors="coerce"
    ).astype(float)
    lower, upper = volatility.quantile([0.10, 0.90]) if volatility.notna().any() else (np.nan, np.nan)
    fallback = float(volatility.dropna().median()) if volatility.notna().any() else np.nan
    volatility = volatility.fillna(fallback)
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower > 0:
        volatility = volatility.clip(lower=lower, upper=upper)
    else:
        volatility = pd.Series(1.0, index=usable.index)
    score = pd.to_numeric(usable[spec.signal], errors="coerce").clip(-3.0, 3.0)
    raw = score / volatility.pow(spec.inverse_volatility_power)
    exposures = _exposure_matrix(usable)
    return _normalize_and_cap(raw, exposures, spec), exposures


def _managed_weights(
    ideal: pd.Series,
    previous: pd.Series,
    exposures: pd.DataFrame,
    spec: PortfolioEngineeringSpec,
) -> pd.Series:
    prior = previous.reindex(ideal.index, fill_value=0.0)
    desired_change = ideal - prior
    desired_change = desired_change.where(
        desired_change.abs() >= spec.no_trade_weight_change,
        0.0,
    )
    candidate = prior + spec.rebalance_fraction * desired_change
    return _normalize_and_cap(candidate, exposures, spec)


def _diagnostics(
    weights: pd.Series,
    cross: pd.DataFrame,
    exposures: pd.DataFrame,
    leverage_multiplier: float,
) -> dict:
    aligned = weights.reindex(exposures.index, fill_value=0.0)
    sector_columns = [column for column in exposures if column.startswith("sector_")]
    sector_exposure = exposures[sector_columns].T @ aligned if sector_columns else pd.Series(dtype=float)
    normalized = aligned.abs() / max(float(aligned.abs().sum()), 1e-12)
    effective_breadth = float(1.0 / np.square(normalized).sum())
    return {
        "gross_exposure": float(aligned.abs().sum()),
        "net_exposure": float(aligned.sum()),
        "beta_exposure": float((exposures["beta"] * aligned).sum()),
        "size_exposure": float((exposures["size"] * aligned).sum()),
        "maximum_sector_net_exposure": (
            float(sector_exposure.abs().max()) if not sector_exposure.empty else 0.0
        ),
        "maximum_absolute_weight": float(aligned.abs().max()),
        "effective_breadth": effective_breadth,
        "leverage_multiplier": float(leverage_multiplier),
        "risk_feature_coverage": float(cross["trailing_volatility"].notna().mean()),
    }


def compute_engineered_portfolios(
    panel: pd.DataFrame,
    spec: PortfolioEngineeringSpec | None = None,
) -> pd.DataFrame:
    """Construct locked portfolio variants sequentially with formation-date data."""
    spec = spec or PortfolioEngineeringSpec()
    risk_panel = add_trailing_risk_features(panel, spec)
    previous = {strategy: pd.Series(dtype=float) for strategy in ENGINEERING_STRATEGIES}
    managed_gross_history: list[float] = []
    rows: list[dict] = []

    for date, cross in risk_panel.groupby("date", sort=True):
        cross = cross.dropna(subset=[spec.signal, "fwd_1m"])
        if len(cross) < 50:
            continue
        ideal, exposures = _ideal_weights(cross, spec)
        if ideal.empty:
            continue
        managed = _managed_weights(
            ideal,
            previous["risk_turnover_managed"],
            exposures,
            spec,
        )
        if managed.empty:
            continue

        if len(managed_gross_history) >= spec.portfolio_volatility_minimum_months:
            trailing = pd.Series(managed_gross_history[-spec.portfolio_volatility_lookback:])
            realized_volatility = float(trailing.std(ddof=1) * np.sqrt(12.0))
            multiplier = (
                spec.target_annualized_volatility / realized_volatility
                if realized_volatility > 1e-12
                else 1.0
            )
            multiplier = float(np.clip(
                multiplier,
                spec.minimum_leverage_multiplier,
                spec.maximum_leverage_multiplier,
            ))
        else:
            multiplier = 1.0
        max_weight = float(managed.abs().max())
        if max_weight > 0:
            multiplier = min(
                multiplier,
                spec.maximum_absolute_weight / max_weight,
            )
        vol_scaled = managed * multiplier

        strategies = {
            "risk_neutral": ideal,
            "risk_turnover_managed": managed,
            "risk_turnover_vol_scaled": vol_scaled,
        }
        forward = cross.drop_duplicates("permco").set_index("permco")["fwd_1m"]
        benchmark_weights = cross.drop_duplicates("permco").set_index("permco")[
            "market_equity_millions"
        ]
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
        benchmark_return = float(
            (benchmark_weights * forward.reindex(benchmark_weights.index)).sum()
        )

        for strategy, weights in strategies.items():
            turnover = _turnover(weights, previous[strategy])
            gross_return = float((weights * forward.reindex(weights.index)).sum())
            diagnostics = _diagnostics(
                weights,
                cross.drop_duplicates("permco").set_index("permco"),
                exposures,
                multiplier if strategy == "risk_turnover_vol_scaled" else 1.0,
            )
            for cost_bps in spec.transaction_cost_bps:
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "period": str(cross["period"].iloc[0]),
                        "signal": spec.signal,
                        "strategy": strategy,
                        "cost_bps": float(cost_bps),
                        "gross_return": gross_return,
                        "net_return": gross_return - turnover * float(cost_bps) * 1e-4,
                        "turnover": turnover,
                        "benchmark_return": benchmark_return,
                        "universe_size": int(len(cross)),
                        "long_count": int((weights > 0).sum()),
                        "short_count": int((weights < 0).sum()),
                        **diagnostics,
                    }
                )
            previous[strategy] = weights
        managed_gross_history.append(
            float((managed * forward.reindex(managed.index)).sum())
        )

    return pd.DataFrame(rows).sort_values(
        ["date", "strategy", "cost_bps"]
    ).reset_index(drop=True)


def compute_engineering_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["strategy", "cost_bps"]
    for period in ("retrospective", "prospective", "full"):
        sample = monthly if period == "full" else monthly[monthly["period"] == period]
        for values, group in sample.groupby(keys):
            metrics = _performance_metrics(group.sort_values("date"))
            if metrics:
                rows.append({"period": period, **dict(zip(keys, values)), **metrics})
    return pd.DataFrame(rows).sort_values(["period", *keys]).reset_index(drop=True)


def portfolio_era_stability(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    retrospective = monthly[monthly["period"] == "retrospective"].copy()
    retrospective["era"] = (retrospective["date"].dt.year // 10 * 10).astype(str) + "s"
    for keys, group in retrospective.groupby(["strategy", "cost_bps", "era"]):
        metrics = _performance_metrics(group.sort_values("date"))
        if metrics:
            rows.append({
                "strategy": keys[0],
                "cost_bps": float(keys[1]),
                "era": keys[2],
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(["strategy", "cost_bps", "era"])


def paired_sharpe_improvement_ci(
    monthly: pd.DataFrame,
    baseline_strategy: str,
    candidate_strategy: str,
    cost_bps: float,
    spec: PortfolioEngineeringSpec,
) -> tuple[float, float, float]:
    """Moving-block CI for the paired candidate-minus-baseline Sharpe change."""
    sample = monthly[
        (monthly["period"] == "retrospective")
        & (monthly["cost_bps"] == float(cost_bps))
        & (monthly["strategy"].isin([baseline_strategy, candidate_strategy]))
    ]
    wide = sample.pivot(index="date", columns="strategy", values="net_return").dropna()
    if len(wide) < max(8, spec.bootstrap_block_months):
        return np.nan, np.nan, np.nan

    def sharpe(values: np.ndarray) -> float:
        std = float(np.std(values, ddof=1))
        return float(np.mean(values) / std * np.sqrt(12.0)) if std > 0 else np.nan

    baseline = wide[baseline_strategy].to_numpy(dtype=float)
    candidate = wide[candidate_strategy].to_numpy(dtype=float)
    observed = sharpe(candidate) - sharpe(baseline)
    rng = np.random.default_rng(spec.bootstrap_seed)
    n = len(wide)
    blocks_needed = int(np.ceil(n / spec.bootstrap_block_months))
    offsets = np.arange(spec.bootstrap_block_months)
    differences = np.empty(spec.bootstrap_repetitions, dtype=float)
    for repetition in range(spec.bootstrap_repetitions):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % n
        draw = indices.ravel()[:n]
        differences[repetition] = sharpe(candidate[draw]) - sharpe(baseline[draw])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(lower), float(upper)


def aggregate_diagnostics(monthly: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "gross_exposure",
        "net_exposure",
        "beta_exposure",
        "size_exposure",
        "maximum_sector_net_exposure",
        "maximum_absolute_weight",
        "effective_breadth",
        "leverage_multiplier",
        "risk_feature_coverage",
    ]
    unique = monthly.drop_duplicates(["date", "strategy"])
    rows = []
    for strategy, group in unique.groupby("strategy"):
        row = {"strategy": strategy, "n_months": int(len(group))}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"mean_{column}"] = float(values.mean())
            row[f"mean_abs_{column}"] = float(values.abs().mean())
            row[f"max_abs_{column}"] = float(values.abs().max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)
