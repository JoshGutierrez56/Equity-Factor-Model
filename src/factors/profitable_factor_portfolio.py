"""Retrospective portfolio that combines v4 with established factor premiums.

The stock-selection v4 sleeve remains frozen.  This module adds a distinct
allocation layer whose purpose is to harvest known factor premiums rather than
to claim a new factor-neutral alpha.  Every fitted quantity uses data ending in
2019; 2020 is an embargo year and 2021-2024 is a retrospective assessment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfitableFactorPortfolioSpec:
    """Fixed, transparent rules for the post-v4 factor-allocation follow-up."""

    factors: tuple[str, ...] = ("mktrf", "smb", "hml", "rmw", "cma", "umd")
    factor_development_start: str = "1990-01-01"
    factor_development_end: str = "2019-12-31"
    sleeve_development_start: str = "2010-01-01"
    sleeve_development_end: str = "2019-12-31"
    embargo_start: str = "2020-01-01"
    assessment_start: str = "2021-01-01"
    equal_weight_blend: float = 0.50
    positive_sharpe_blend: float = 0.50
    maximum_factor_weight: float = 0.30
    primary_factor_cost_bps: float = 100.0
    factor_cost_stress_bps: tuple[float, ...] = (0.0, 50.0, 100.0, 150.0)
    primary_v4_aum_usd: float = 100_000_000.0
    primary_v4_borrow_bps: float = 150.0
    target_annualized_volatility: float = 0.08
    maximum_combined_leverage: float = 2.0
    bootstrap_repetitions: int = 5_000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260724

    def public_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["factors"] = list(self.factors)
        output["factor_cost_stress_bps"] = list(self.factor_cost_stress_bps)
        output["factor_weight_policy"] = (
            "50% equal weight plus 50% normalized positive development Sharpe; "
            "weights capped at 30% with excess redistributed pro rata"
        )
        output["sleeve_mix_policy"] = (
            "inverse-volatility weights fitted on 2010-2019 monthly v4 net and "
            "factor-sleeve returns"
        )
        output["risk_scaling_policy"] = (
            "development-only 8% volatility target capped at 2.0x leverage"
        )
        output["claim_policy"] = (
            "retrospective, performance-informed factor allocation; intended to "
            "harvest known premiums, not demonstrate factor-neutral alpha"
        )
        return output


def profitable_factor_portfolio_hash(
    institutional_v4_sha256: str,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> str:
    spec = spec or ProfitableFactorPortfolioSpec()
    payload = {
        "institutional_v4_sha256": institutional_v4_sha256,
        "profitable_factor_portfolio": spec.public_dict(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _redistribute_cap(weights: pd.Series, maximum: float) -> pd.Series:
    result = weights.astype(float).copy()
    for _ in range(100):
        over = result > maximum + 1e-12
        if not bool(over.any()):
            break
        excess = float((result.loc[over] - maximum).sum())
        result.loc[over] = maximum
        available = ~over
        if excess <= 1e-12 or float(result.loc[available].sum()) <= 1e-12:
            break
        result.loc[available] += excess * result.loc[available] / result.loc[available].sum()
    return result / result.sum()


def fit_factor_weights(
    factors: pd.DataFrame,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit a shrinkage factor tilt using only the fixed development sample."""
    spec = spec or ProfitableFactorPortfolioSpec()
    frame = factors.copy()
    frame["dateff"] = pd.to_datetime(frame["dateff"])
    development = frame[
        (frame["dateff"] >= pd.Timestamp(spec.factor_development_start))
        & (frame["dateff"] <= pd.Timestamp(spec.factor_development_end))
    ].copy()
    if len(development) < 120:
        raise ValueError("factor development sample must contain at least 120 months")
    values = development.loc[:, spec.factors].astype(float)
    annual_mean = values.mean() * 12.0
    annual_volatility = values.std(ddof=1) * np.sqrt(12.0)
    sharpe = annual_mean / annual_volatility.replace(0.0, np.nan)
    positive = sharpe.clip(lower=0.0).fillna(0.0)
    sharpe_weights = (
        positive / positive.sum()
        if float(positive.sum()) > 1e-12
        else pd.Series(1.0 / len(spec.factors), index=spec.factors)
    )
    equal = pd.Series(1.0 / len(spec.factors), index=spec.factors, dtype=float)
    weights = (
        spec.equal_weight_blend * equal
        + spec.positive_sharpe_blend * sharpe_weights
    )
    weights = _redistribute_cap(weights, spec.maximum_factor_weight)
    diagnostics = pd.DataFrame({
        "factor": list(spec.factors),
        "development_annualized_mean": annual_mean.reindex(spec.factors).to_numpy(),
        "development_annualized_volatility": annual_volatility.reindex(spec.factors).to_numpy(),
        "development_sharpe": sharpe.reindex(spec.factors).to_numpy(),
        "positive_rolling_60m_mean_fraction": [
            float((values[factor].rolling(60).mean().dropna() > 0.0).mean())
            for factor in spec.factors
        ],
        "final_weight": weights.reindex(spec.factors).to_numpy(),
    })
    return weights.reindex(spec.factors), diagnostics


def merge_factor_and_v4_returns(
    factors: pd.DataFrame,
    v4_monthly: pd.DataFrame,
    factor_weights: pd.Series,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> pd.DataFrame:
    """Align formation-month v4 returns with next-calendar-month factors."""
    spec = spec or ProfitableFactorPortfolioSpec()
    factor_frame = factors.copy()
    factor_frame["dateff"] = pd.to_datetime(factor_frame["dateff"])
    factor_frame["factor_month"] = factor_frame["dateff"].dt.to_period("M")
    factor_frame["factor_sleeve_gross_return"] = (
        factor_frame.loc[:, spec.factors].astype(float)
        * factor_weights.reindex(spec.factors)
    ).sum(axis=1)
    factor_frame["equal_factor_gross_return"] = factor_frame.loc[:, spec.factors].astype(
        float
    ).mean(axis=1)

    v4 = v4_monthly[
        np.isclose(v4_monthly["aum_usd"].astype(float), spec.primary_v4_aum_usd)
        & np.isclose(v4_monthly["borrow_bps"].astype(float), spec.primary_v4_borrow_bps)
    ].copy()
    v4["date"] = pd.to_datetime(v4["date"])
    v4["factor_month"] = v4["date"].dt.to_period("M") + 1
    merged = v4.merge(
        factor_frame[[
            "factor_month", "factor_sleeve_gross_return", "equal_factor_gross_return",
        ]],
        on="factor_month",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(v4):
        raise ValueError(
            f"factor alignment lost {len(v4) - len(merged)} of {len(v4)} v4 months"
        )
    merged["research_period"] = np.select(
        [
            merged["date"] <= pd.Timestamp(spec.sleeve_development_end),
            (merged["date"] >= pd.Timestamp(spec.embargo_start))
            & (merged["date"] < pd.Timestamp(spec.assessment_start)),
        ],
        ["development", "embargo"],
        default="temporal_assessment",
    )
    return merged.sort_values("date").reset_index(drop=True)


def fit_sleeve_mix(
    merged: pd.DataFrame,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> tuple[pd.Series, float]:
    """Fit inverse-volatility sleeve weights and capped leverage pre-2020."""
    spec = spec or ProfitableFactorPortfolioSpec()
    development = merged[
        (merged["date"] >= pd.Timestamp(spec.sleeve_development_start))
        & (merged["date"] <= pd.Timestamp(spec.sleeve_development_end))
    ]
    vol = pd.Series({
        "v4": float(development["net_return"].std(ddof=1)),
        "factor": float(development["factor_sleeve_gross_return"].std(ddof=1)),
    })
    if (vol <= 0.0).any() or not np.isfinite(vol).all():
        raise ValueError("development sleeve volatility must be positive and finite")
    inverse = 1.0 / vol
    weights = inverse / inverse.sum()
    combined = (
        weights["v4"] * development["net_return"]
        + weights["factor"] * development["factor_sleeve_gross_return"]
    )
    development_vol = float(combined.std(ddof=1) * np.sqrt(12.0))
    leverage = min(
        spec.maximum_combined_leverage,
        spec.target_annualized_volatility / development_vol,
    )
    return weights, float(leverage)


def build_combined_returns(
    merged: pd.DataFrame,
    sleeve_weights: pd.Series,
    leverage: float,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> pd.DataFrame:
    spec = spec or ProfitableFactorPortfolioSpec()
    rows: list[pd.DataFrame] = []
    for factor_cost_bps in spec.factor_cost_stress_bps:
        frame = merged.copy()
        factor_haircut = float(factor_cost_bps) * 1e-4 / 12.0
        frame["factor_cost_bps"] = float(factor_cost_bps)
        frame["factor_sleeve_net_return"] = (
            frame["factor_sleeve_gross_return"] - factor_haircut
        )
        frame["equal_factor_net_return"] = frame["equal_factor_gross_return"] - factor_haircut
        frame["combined_unlevered_return"] = (
            sleeve_weights["v4"] * frame["net_return"]
            + sleeve_weights["factor"] * frame["factor_sleeve_net_return"]
        )
        frame["equal_factor_combination_return"] = (
            sleeve_weights["v4"] * frame["net_return"]
            + sleeve_weights["factor"] * frame["equal_factor_net_return"]
        )
        frame["combined_risk_scaled_return"] = leverage * frame["combined_unlevered_return"]
        frame["v4_sleeve_weight"] = float(sleeve_weights["v4"])
        frame["factor_sleeve_weight"] = float(sleeve_weights["factor"])
        frame["combined_leverage"] = float(leverage)
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    public_columns = [
        "date", "factor_month", "research_period", "factor_cost_bps",
        "gross_return", "net_return", "factor_sleeve_gross_return",
        "factor_sleeve_net_return", "equal_factor_gross_return",
        "equal_factor_net_return", "combined_unlevered_return",
        "combined_risk_scaled_return", "equal_factor_combination_return",
        "v4_sleeve_weight", "factor_sleeve_weight", "combined_leverage",
    ]
    output["factor_month"] = output["factor_month"].astype(str)
    return output.loc[:, public_columns].sort_values(
        ["date", "factor_cost_bps"]
    ).reset_index(drop=True)


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    values = pd.Series(returns, dtype=float).dropna()
    if len(values) < 3:
        return {}
    annualized_volatility = float(values.std(ddof=1) * np.sqrt(12.0))
    annualized_mean = float(values.mean() * 12.0)
    wealth = (1.0 + values).cumprod()
    return {
        "n_months": int(len(values)),
        "annualized_mean": annualized_mean,
        "cagr": float(wealth.iloc[-1] ** (12.0 / len(values)) - 1.0),
        "annualized_volatility": annualized_volatility,
        "sharpe": float(annualized_mean / annualized_volatility),
        "maximum_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "hit_rate": float((values > 0.0).mean()),
    }


def summarize_combined_returns(
    monthly: pd.DataFrame,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> pd.DataFrame:
    spec = spec or ProfitableFactorPortfolioSpec()
    strategies = {
        "v4_net": "net_return",
        "factor_sleeve": "factor_sleeve_net_return",
        "combined_unlevered": "combined_unlevered_return",
        "combined_risk_scaled": "combined_risk_scaled_return",
        "equal_factor_combination": "equal_factor_combination_return",
    }
    rows: list[dict[str, Any]] = []
    for (period, factor_cost_bps), group in monthly.groupby(
        ["research_period", "factor_cost_bps"], sort=True
    ):
        for strategy, column in strategies.items():
            metrics = performance_metrics(group.sort_values("date")[column])
            rows.append({
                "research_period": period,
                "factor_cost_bps": float(factor_cost_bps),
                "strategy": strategy,
                **metrics,
            })
    return pd.DataFrame(rows).sort_values(
        ["research_period", "factor_cost_bps", "strategy"]
    ).reset_index(drop=True)


def active_information_ratio(active_returns: pd.Series) -> float:
    values = pd.Series(active_returns, dtype=float).dropna()
    tracking_error = float(values.std(ddof=1) * np.sqrt(12.0))
    return float(values.mean() * 12.0 / tracking_error) if tracking_error > 1e-12 else np.nan


def paired_block_sharpe_interval(
    candidate: pd.Series,
    baseline: pd.Series,
    spec: ProfitableFactorPortfolioSpec | None = None,
) -> tuple[float, float, float]:
    spec = spec or ProfitableFactorPortfolioSpec()
    candidate_values = pd.Series(candidate, dtype=float).to_numpy()
    baseline_values = pd.Series(baseline, dtype=float).to_numpy()
    if len(candidate_values) != len(baseline_values):
        raise ValueError("candidate and baseline return series must have equal length")
    n = len(candidate_values)
    rng = np.random.default_rng(spec.bootstrap_seed)

    def _sharpe(values: np.ndarray) -> float:
        return float(values.mean() / values.std(ddof=1) * np.sqrt(12.0))

    differences = []
    for _ in range(spec.bootstrap_repetitions):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend(
                ((np.arange(start, start + spec.bootstrap_block_months) % n).tolist())
            )
        sample = np.asarray(indices[:n], dtype=int)
        differences.append(
            _sharpe(candidate_values[sample]) - _sharpe(baseline_values[sample])
        )
    observed = _sharpe(candidate_values) - _sharpe(baseline_values)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(lower), float(upper)
