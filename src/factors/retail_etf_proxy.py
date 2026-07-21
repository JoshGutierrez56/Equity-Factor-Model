"""Executable long-only ETF proxy for the frozen profitable-factor sleeve."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factors.profitable_factor_portfolio import performance_metrics


@dataclass(frozen=True)
class RetailETFProxySpec:
    tickers: tuple[str, ...] = ("SPY", "IWM", "VLUE", "QUAL", "MTUM")
    weights: tuple[float, ...] = (
        0.21982583947529455,
        0.12006665204590708,
        0.12912581605042373,
        0.3590985669774964,
        0.1718831254508782,
    )
    assessment_start: str = "2021-01-01"
    assessment_end: str = "2025-12-31"
    post_freeze_start: str = "2026-01-01"
    beta_development_end: str = "2019-12-31"
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0)
    primary_cost_bps: float = 10.0
    bootstrap_repetitions: int = 5_000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260725

    def weight_series(self) -> pd.Series:
        weights = pd.Series(self.weights, index=self.tickers, dtype=float)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("ETF target weights must sum to one")
        return weights


def month_end_prices(daily_prices: pd.DataFrame, spec: RetailETFProxySpec | None = None) -> pd.DataFrame:
    """Return complete month-end adjusted prices for the locked ETF set."""
    spec = spec or RetailETFProxySpec()
    prices = daily_prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index().loc[:, list(spec.tickers)].astype(float)
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError("No complete ETF price observations")
    complete = prices.dropna(how="any")
    month_key = complete.index.to_period("M")
    month_ends = complete.groupby(month_key, sort=True).tail(1)
    return month_ends.dropna(how="any")


def _static_weight_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, values in asset_returns.iterrows():
        gross = float((values * weights).sum())
        drifted = weights * (1.0 + values) / (1.0 + gross)
        one_way_turnover = float(0.5 * (weights - drifted).abs().sum())
        cost = one_way_turnover * float(cost_bps) * 1e-4
        rows.append({
            "date": pd.Timestamp(date),
            "gross_return": gross,
            "one_way_turnover": one_way_turnover,
            "transaction_cost": cost,
            "net_return": gross - cost,
        })
    return pd.DataFrame(rows)


def build_monthly_returns(
    daily_prices: pd.DataFrame,
    spec: RetailETFProxySpec | None = None,
) -> pd.DataFrame:
    """Build locked ETF, SPY, and equal-weight monthly return series."""
    spec = spec or RetailETFProxySpec()
    month_prices = month_end_prices(daily_prices, spec)
    returns = month_prices.pct_change().dropna(how="any")
    target = spec.weight_series()
    equal = pd.Series(1.0 / len(spec.tickers), index=spec.tickers)
    frames: list[pd.DataFrame] = []
    for cost in spec.cost_bps:
        proxy = _static_weight_returns(returns, target, cost)
        benchmark = _static_weight_returns(returns, equal, cost)
        proxy["cost_bps"] = float(cost)
        proxy["spy_return"] = returns["SPY"].to_numpy()
        proxy["equal_weight_return"] = benchmark["net_return"].to_numpy()
        proxy["research_period"] = np.select(
            [
                proxy["date"] <= pd.Timestamp(spec.beta_development_end),
                proxy["date"].between(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
                proxy["date"].between(pd.Timestamp(spec.assessment_start), pd.Timestamp(spec.assessment_end)),
                proxy["date"] >= pd.Timestamp(spec.post_freeze_start),
            ],
            ["development", "embargo", "primary_assessment", "post_freeze_extension"],
            default="pre_development",
        )
        frames.append(proxy)
    return pd.concat(frames, ignore_index=True).sort_values(["cost_bps", "date"]).reset_index(drop=True)


def fit_spy_beta(monthly: pd.DataFrame, spec: RetailETFProxySpec | None = None) -> float:
    """Estimate one fixed hedge beta using development data only."""
    spec = spec or RetailETFProxySpec()
    frame = monthly[
        np.isclose(monthly["cost_bps"], spec.primary_cost_bps)
        & (monthly["research_period"] == "development")
    ]
    if len(frame) < 36:
        raise ValueError("At least 36 development months are required for beta")
    variance = float(frame["spy_return"].var(ddof=1))
    if variance <= 0:
        raise ValueError("SPY development variance must be positive")
    return float(frame[["net_return", "spy_return"]].cov().iloc[0, 1] / variance)


def active_information_ratio(candidate: pd.Series, benchmark: pd.Series) -> float:
    active = pd.Series(candidate, dtype=float).reset_index(drop=True) - pd.Series(
        benchmark, dtype=float
    ).reset_index(drop=True)
    tracking_error = float(active.std(ddof=1) * np.sqrt(12.0))
    return float(active.mean() * 12.0 / tracking_error) if tracking_error > 1e-12 else np.nan


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = {
        "etf_proxy": "net_return",
        "spy": "spy_return",
        "equal_weight_etfs": "equal_weight_return",
    }
    for (period, cost), group in monthly.groupby(["research_period", "cost_bps"]):
        for strategy, column in columns.items():
            rows.append({
                "research_period": period,
                "cost_bps": float(cost),
                "strategy": strategy,
                **performance_metrics(group.sort_values("date")[column]),
            })
    return pd.DataFrame(rows).sort_values(["research_period", "cost_bps", "strategy"])


def paired_sharpe_interval(
    candidate: pd.Series,
    benchmark: pd.Series,
    spec: RetailETFProxySpec | None = None,
) -> tuple[float, float, float]:
    spec = spec or RetailETFProxySpec()
    left = pd.Series(candidate, dtype=float).to_numpy()
    right = pd.Series(benchmark, dtype=float).to_numpy()
    if len(left) != len(right) or len(left) < 12:
        raise ValueError("Matched candidate and benchmark samples are required")
    rng = np.random.default_rng(spec.bootstrap_seed)

    def sharpe(values: np.ndarray) -> float:
        return float(values.mean() / values.std(ddof=1) * np.sqrt(12.0))

    differences: list[float] = []
    for _ in range(spec.bootstrap_repetitions):
        indices: list[int] = []
        while len(indices) < len(left):
            start = int(rng.integers(0, len(left)))
            indices.extend(
                ((np.arange(start, start + spec.bootstrap_block_months) % len(left)).tolist())
            )
        sample = np.asarray(indices[: len(left)])
        differences.append(sharpe(left[sample]) - sharpe(right[sample]))
    observed = sharpe(left) - sharpe(right)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return float(observed), float(lower), float(upper)
