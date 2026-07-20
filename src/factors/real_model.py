"""Point-in-time factor construction and walk-forward evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from factors.wrds_data import WRDSResearchRequest, normalize_compustat, normalize_crsp_monthly


FACTOR_COLUMNS = ("MOM", "LV", "SIZE", "VALUE", "QUALITY", "INVESTMENT")
PERIODS = ("retrospective", "prospective", "full")
HYPOTHESIS_REGISTRY = {
    "MOM": {
        "primary_horizon_months": 6,
        "expected_sign": 1,
        "rationale": "Intermediate-horizon return continuation.",
    },
    "LV": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "The low-risk anomaly predicts higher risk-adjusted returns.",
    },
    "SIZE": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "Smaller eligible companies may earn a size premium.",
    },
    "VALUE": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "High book-to-market firms may earn a value premium.",
    },
    "QUALITY": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "Gross profitability predicts persistent operating quality.",
    },
    "INVESTMENT": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "Conservative asset growth may outperform aggressive investment.",
    },
    "COMPOSITE": {
        "primary_horizon_months": 12,
        "expected_sign": 1,
        "rationale": "A fixed diversified blend should improve signal breadth.",
    },
}


@dataclass(frozen=True)
class RealModelSpec:
    """Predeclared model/evaluation choices; no return-fitted factor weights."""

    horizons_months: tuple[int, ...] = (1, 3, 6, 12)
    transaction_cost_bps: tuple[float, ...] = (0.0, 10.0, 25.0)
    quantile_fraction: float = 0.20
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    min_factor_coverage: int = 4
    min_inference_months: int = 24
    min_prospective_months: int = 36
    sector_neutral: bool = True
    size_neutral: bool = True
    bootstrap_repetitions: int = 1000
    bootstrap_block_months: int = 12
    bootstrap_seed: int = 20260720

    def public_dict(self) -> dict:
        output = asdict(self)
        output["horizons_months"] = list(self.horizons_months)
        output["transaction_cost_bps"] = list(self.transaction_cost_bps)
        output["factor_columns"] = list(FACTOR_COLUMNS)
        output["composite_weights"] = "equal; fixed ex ante"
        output["hypothesis_registry"] = HYPOTHESIS_REGISTRY
        output["research_entity"] = "CRSP PERMCO company"
        output["share_class_treatment"] = (
            "aggregate market equity and market-equity-weight monthly returns; "
            "largest class carries identifiers"
        )
        output["implementation_lag"] = "signal at month t; return begins in month t+1"
        output["risk_attribution"] = "Fama-French five factors plus momentum; HAC(3)"
        output["multiple_testing"] = (
            "Holm family-wise correction, separated into predeclared primary and "
            "exploratory horizon families"
        )
        output["validation_policy"] = (
            "all data before holdout_start are retrospective; only later, never-"
            "inspected months may be called prospective"
        )
        return output


def specification_hash(request: WRDSResearchRequest, spec: RealModelSpec) -> str:
    payload = {"request": request.public_dict(), "model": spec.public_dict()}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _primary_company_security(crsp: pd.DataFrame) -> pd.DataFrame:
    """Consolidate share classes and represent each company with its largest class."""
    frame = normalize_crsp_monthly(crsp)
    valid_me = frame["market_equity_millions"].where(
        frame["market_equity_millions"] > 0
    )
    frame = frame.assign(_security_me=valid_me)
    totals = (
        frame.groupby(["date", "permco"], as_index=False)["_security_me"]
        .sum(min_count=1)
        .rename(columns={"_security_me": "company_market_equity_millions"})
    )
    weighted = frame.assign(
        _weighted_return=frame["total_ret"] * frame["_security_me"]
    ).groupby(["date", "permco"], as_index=False).agg(
        _weighted_return=("_weighted_return", "sum"),
        _return_weight=("_security_me", "sum"),
    )
    weighted["company_total_ret"] = (
        weighted["_weighted_return"] / weighted["_return_weight"]
    )
    ranked = frame.sort_values(
        ["date", "permco", "_security_me", "permno"],
        ascending=[True, True, False, True],
    )
    primary = ranked.drop_duplicates(["date", "permco"], keep="first")
    primary = primary.merge(totals, on=["date", "permco"], how="left")
    primary = primary.merge(
        weighted[["date", "permco", "company_total_ret"]],
        on=["date", "permco"],
        how="left",
    )
    primary["market_equity_millions"] = primary["company_market_equity_millions"]
    primary["total_ret"] = primary["company_total_ret"]
    return primary.drop(columns=["_security_me", "company_market_equity_millions"])


def _fundamental_signals(fundamentals: pd.DataFrame, lag_months: int) -> pd.DataFrame:
    frame = normalize_compustat(fundamentals, lag_months).copy()
    frame = frame.sort_values(["permno", "datadate"])
    preferred = frame[["pstkrv", "pstkl", "pstk"]].bfill(axis=1).iloc[:, 0].fillna(0.0)
    equity = frame["seq"].copy()
    equity = equity.fillna(frame["ceq"] + frame["pstk"].fillna(0.0))
    equity = equity.fillna(frame["at"] - frame["lt"])
    frame["book_equity_millions"] = equity + frame["txditc"].fillna(0.0) - preferred
    revenue = frame["revt"].fillna(frame["sale"])
    frame["gross_profitability"] = (revenue - frame["cogs"]) / frame["at"]
    prior_assets = frame.groupby("permno")["at"].shift(1)
    frame["asset_growth"] = frame["at"] / prior_assets - 1.0
    return frame[
        [
            "permno", "gvkey", "datadate", "availability_date",
            "availability_source", "book_equity_millions", "gross_profitability",
            "asset_growth", "at", "ib",
        ]
    ].sort_values(["permno", "availability_date"])


def _merge_fundamentals(monthly: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly.iloc[0:0].copy()
    return pd.merge_asof(
        monthly.sort_values(["date", "permno"]),
        fundamentals.sort_values(["availability_date", "permno"]),
        left_on="date",
        right_on="availability_date",
        by="permno",
        direction="backward",
        allow_exact_matches=True,
    )


def _rolling_compound(values: pd.Series, window: int, shift: int) -> pd.Series:
    base = (1.0 + values).shift(shift)
    return base.rolling(window, min_periods=max(6, window - 3)).apply(np.prod, raw=True) - 1.0


def _forward_compound(values: pd.Series, horizon: int) -> pd.Series:
    array = values.to_numpy(dtype=float)
    output = np.full(len(array), np.nan)
    for i in range(len(array) - horizon):
        future = array[i + 1 : i + horizon + 1]
        if np.isfinite(future).sum() == horizon:
            output[i] = np.prod(1.0 + future) - 1.0
    return pd.Series(output, index=values.index)


def _cross_section_score(
    frame: pd.DataFrame,
    column: str,
    spec: RealModelSpec,
) -> pd.Series:
    values = frame[column].replace([np.inf, -np.inf], np.nan)
    valid = values.dropna()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(valid) < 20:
        return output
    clipped = valid.clip(valid.quantile(spec.winsor_lower), valid.quantile(spec.winsor_upper))
    if spec.sector_neutral:
        sectors = frame.loc[clipped.index, "sector"].fillna(-1)
        counts = sectors.map(sectors.value_counts())
        group_mean = clipped.groupby(sectors).transform("mean")
        clipped = clipped - group_mean.where(counts >= 5, clipped.mean())
    if spec.size_neutral and column != "SIZE":
        log_size = np.log(
            pd.to_numeric(
                frame.loc[clipped.index, "market_equity_millions"], errors="coerce"
            ).where(lambda values: values > 0)
        )
        valid_size = clipped.notna() & log_size.notna()
        if valid_size.sum() >= 20 and log_size.loc[valid_size].std(ddof=1) > 1e-12:
            x = np.column_stack(
                [np.ones(int(valid_size.sum())), log_size.loc[valid_size].to_numpy()]
            )
            y = clipped.loc[valid_size].to_numpy()
            fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
            clipped.loc[valid_size] = y - fitted
    std = clipped.std(ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        output.loc[clipped.index] = 0.0
    else:
        output.loc[clipped.index] = (clipped - clipped.mean()) / std
    return output


def build_point_in_time_panel(
    crsp: pd.DataFrame,
    fundamentals: pd.DataFrame,
    request: WRDSResearchRequest,
    spec: RealModelSpec | None = None,
) -> pd.DataFrame:
    """Construct monthly signals using only information available at each date."""
    spec = spec or RealModelSpec()
    monthly = _primary_company_security(crsp)
    fundamental_signals = _fundamental_signals(fundamentals, request.accounting_lag_months)
    monthly = _merge_fundamentals(monthly, fundamental_signals)
    monthly = monthly.sort_values(["permco", "date"]).reset_index(drop=True)
    monthly["_month_id"] = monthly["date"].dt.year * 12 + monthly["date"].dt.month
    gap = monthly.groupby("permco")["_month_id"].diff().fillna(1).ne(1)
    monthly["_continuity_segment"] = gap.groupby(monthly["permco"]).cumsum()
    grouped = monthly.groupby(["permco", "_continuity_segment"], group_keys=False)
    monthly["MOM"] = grouped["total_ret"].transform(
        lambda values: _rolling_compound(values, window=11, shift=2)
    )
    monthly["LV"] = -grouped["total_ret"].transform(
        lambda values: values.shift(1).rolling(12, min_periods=9).std(ddof=1) * np.sqrt(12.0)
    )
    market_equity = pd.to_numeric(
        monthly["market_equity_millions"], errors="coerce"
    ).astype(float)
    monthly["SIZE"] = -np.log(market_equity.where(market_equity > 0))
    monthly["VALUE"] = (
        monthly["book_equity_millions"] / monthly["market_equity_millions"]
    ).where(monthly["book_equity_millions"] > 0)
    monthly["QUALITY"] = monthly["gross_profitability"]
    monthly["INVESTMENT"] = -monthly["asset_growth"]
    monthly["sector"] = np.floor(pd.to_numeric(monthly["siccd"], errors="coerce") / 100.0)
    for horizon in spec.horizons_months:
        monthly[f"fwd_{horizon}m"] = grouped["total_ret"].transform(
            lambda values, h=horizon: _forward_compound(values, h)
        )
    eligible = monthly[
        (monthly["date"] >= pd.Timestamp(request.start))
        & (monthly["date"] <= pd.Timestamp(request.end))
        & (monthly["price"] >= request.min_price)
        & (monthly["market_equity_millions"] >= request.min_market_cap_millions)
    ].copy()
    eligible["market_cap_rank"] = eligible.groupby("date")[
        "market_equity_millions"
    ].rank(method="first", ascending=False)
    eligible = eligible[eligible["market_cap_rank"] <= request.max_universe].copy()
    score_columns = []
    for factor in FACTOR_COLUMNS:
        score_column = f"{factor}_SCORE"
        eligible[score_column] = np.nan
        for _, index in eligible.groupby("date").groups.items():
            cross = eligible.loc[index]
            eligible.loc[index, score_column] = _cross_section_score(
                cross, factor, spec
            ).to_numpy()
        score_columns.append(score_column)
    coverage = eligible[score_columns].notna().sum(axis=1)
    eligible["COMPOSITE"] = eligible[score_columns].mean(axis=1).where(
        coverage >= spec.min_factor_coverage
    )
    eligible["period"] = np.where(
        eligible["date"] < pd.Timestamp(request.holdout_start),
        "retrospective",
        "prospective",
    )
    if (eligible["availability_date"].dropna() > eligible.loc[
        eligible["availability_date"].notna(), "date"
    ]).any():
        raise AssertionError("fundamental observation crossed the point-in-time boundary")
    return eligible.drop(columns=["_month_id", "_continuity_segment"]).sort_values(
        ["date", "permco"]
    ).reset_index(drop=True)


def _hac_mean_test(values: pd.Series, maxlags: int) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    if len(clean) < 3:
        return np.nan, np.nan
    try:
        import statsmodels.api as sm

        fit = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(1, int(maxlags))}
        )
        return float(fit.tvalues[0]), float(fit.pvalues[0])
    except Exception:
        sem = clean.std(ddof=1) / np.sqrt(len(clean))
        t_stat = clean.mean() / sem if sem > 0 else np.nan
        p_value = 2.0 * stats.t.sf(abs(t_stat), df=len(clean) - 1) if np.isfinite(t_stat) else np.nan
        return float(t_stat), float(p_value)


def _moving_block_bootstrap_mean_ci(
    values: pd.Series,
    repetitions: int,
    block_months: int,
    seed: int,
) -> tuple[float, float]:
    """Deterministic circular moving-block confidence interval for a mean."""
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < max(8, block_months) or repetitions <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(len(clean) / block_months))
    offsets = np.arange(block_months)
    means = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, len(clean), size=blocks_needed)
        indices = (starts[:, None] + offsets[None, :]) % len(clean)
        means[repetition] = clean[indices.ravel()[: len(clean)]].mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Return Holm family-wise adjusted p-values with monotonicity enforced."""
    clean = pd.to_numeric(p_values, errors="coerce")
    output = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = clean.dropna().sort_values()
    if valid.empty:
        return output
    adjusted_sorted = []
    running = 0.0
    total = len(valid)
    for rank, value in enumerate(valid.to_numpy(dtype=float)):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted_sorted.append(running)
    output.loc[valid.index] = adjusted_sorted
    return output


def _monthly_ic_values(
    sample: pd.DataFrame,
    score_column: str,
    forward_column: str,
) -> pd.Series:
    observations = {}
    for date, cross in sample.groupby("date"):
        pair = cross[[score_column, forward_column]].dropna()
        if len(pair) >= 30:
            value = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
            if np.isfinite(value):
                observations[pd.Timestamp(date)] = float(value)
    return pd.Series(observations, dtype=float).sort_index()


def compute_ic_summary(panel: pd.DataFrame, spec: RealModelSpec) -> pd.DataFrame:
    score_map = {factor: f"{factor}_SCORE" for factor in FACTOR_COLUMNS}
    score_map["COMPOSITE"] = "COMPOSITE"
    rows = []
    for period in PERIODS:
        sample = panel if period == "full" else panel[panel["period"] == period]
        for factor, score_column in score_map.items():
            for horizon in spec.horizons_months:
                forward_column = f"fwd_{horizon}m"
                series = _monthly_ic_values(sample, score_column, forward_column)
                if series.empty:
                    continue
                t_stat, p_value = _hac_mean_test(series, maxlags=horizon - 1)
                ci_lower, ci_upper = _moving_block_bootstrap_mean_ci(
                    series,
                    repetitions=spec.bootstrap_repetitions,
                    block_months=spec.bootstrap_block_months,
                    seed=(
                        spec.bootstrap_seed
                        + list(score_map).index(factor) * 100
                        + horizon
                        + PERIODS.index(period) * 1000
                    ),
                )
                primary = (
                    HYPOTHESIS_REGISTRY[factor]["primary_horizon_months"] == horizon
                )
                rows.append({
                    "period": period,
                    "factor": factor,
                    "horizon_months": horizon,
                    "mean_ic": float(series.mean()),
                    "ic_std": float(series.std(ddof=1)),
                    "icir": float(series.mean() / series.std(ddof=1)) if series.std(ddof=1) > 0 else np.nan,
                    "hac_t_stat": t_stat,
                    "p_value": p_value,
                    "bootstrap_ci_lower": ci_lower,
                    "bootstrap_ci_upper": ci_upper,
                    "hit_rate": float((series > 0).mean()),
                    "n_months": int(len(series)),
                    "test_family": "primary" if primary else "exploratory",
                    "primary_hypothesis": bool(primary),
                })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["p_holm"] = np.nan
    for (_, _), index in output.groupby(["period", "test_family"]).groups.items():
        output.loc[index, "p_holm"] = _holm_adjust(output.loc[index, "p_value"])
    output["minimum_required_months"] = np.where(
        output["period"] == "prospective",
        spec.min_prospective_months,
        spec.min_inference_months,
    )
    output["inference_eligible"] = (
        output["n_months"] >= output["minimum_required_months"]
    )
    output["significant_5pct"] = (
        output["inference_eligible"] & (output["p_holm"] < 0.05)
    )
    return output.sort_values(["period", "factor", "horizon_months"]).reset_index(drop=True)


def _weights_for_cross_section(
    cross: pd.DataFrame,
    score_column: str,
    quantile_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    usable = cross[["permco", score_column, "market_equity_millions"]].dropna()
    usable = usable.drop_duplicates("permco").set_index("permco")
    n_side = max(1, int(np.floor(len(usable) * quantile_fraction)))
    ranked = usable[score_column].sort_values()
    bottom = ranked.head(n_side).index
    top = ranked.tail(n_side).index
    long_only = pd.Series(1.0 / len(top), index=top, dtype=float)
    long_short = pd.concat([
        pd.Series(1.0 / len(top), index=top, dtype=float),
        pd.Series(-1.0 / len(bottom), index=bottom, dtype=float),
    ]).groupby(level=0).sum()
    centered = usable[score_column].clip(-3.0, 3.0)
    centered = centered - centered.mean()
    positive = centered.clip(lower=0.0)
    negative = -centered.clip(upper=0.0)
    if positive.sum() > 0 and negative.sum() > 0:
        continuous = positive / positive.sum() - negative / negative.sum()
    else:
        continuous = pd.Series(dtype=float)
    return long_only, long_short, continuous


def _turnover(current: pd.Series, previous: pd.Series) -> float:
    names = current.index.union(previous.index)
    return float((current.reindex(names, fill_value=0.0) - previous.reindex(names, fill_value=0.0)).abs().sum())


def compute_monthly_portfolios(panel: pd.DataFrame, spec: RealModelSpec) -> pd.DataFrame:
    score_map = {factor: f"{factor}_SCORE" for factor in FACTOR_COLUMNS}
    score_map["COMPOSITE"] = "COMPOSITE"
    rows = []
    for signal, score_column in score_map.items():
        previous = {
            "long_only": pd.Series(dtype=float),
            "long_short": pd.Series(dtype=float),
            "continuous_long_short": pd.Series(dtype=float),
        }
        for date, cross in panel.groupby("date", sort=True):
            cross = cross.dropna(subset=[score_column, "fwd_1m"])
            if len(cross) < 50:
                continue
            long_only, long_short, continuous = _weights_for_cross_section(
                cross, score_column, spec.quantile_fraction
            )
            forward = cross.drop_duplicates("permco").set_index("permco")["fwd_1m"]
            benchmark_weights = cross.drop_duplicates("permco").set_index("permco")[
                "market_equity_millions"
            ]
            benchmark_weights = benchmark_weights / benchmark_weights.sum()
            benchmark_return = float((benchmark_weights * forward.reindex(benchmark_weights.index)).sum())
            for strategy, weights in (
                ("long_only", long_only),
                ("long_short", long_short),
                ("continuous_long_short", continuous),
            ):
                if weights.empty:
                    continue
                gross_return = float((weights * forward.reindex(weights.index)).sum())
                turnover = _turnover(weights, previous[strategy])
                for cost_bps in spec.transaction_cost_bps:
                    rows.append({
                        "date": pd.Timestamp(date),
                        "period": str(cross["period"].iloc[0]),
                        "signal": signal,
                        "strategy": strategy,
                        "cost_bps": float(cost_bps),
                        "gross_return": gross_return,
                        "net_return": gross_return - turnover * float(cost_bps) * 1e-4,
                        "turnover": turnover,
                        "benchmark_return": benchmark_return,
                        "universe_size": int(len(cross)),
                        "long_count": int((weights > 0).sum()),
                        "short_count": int((weights < 0).sum()),
                    })
                previous[strategy] = weights
    return pd.DataFrame(rows).sort_values(
        ["date", "signal", "strategy", "cost_bps"]
    ).reset_index(drop=True)


def _performance_metrics(group: pd.DataFrame) -> dict:
    returns = group["net_return"].dropna().astype(float)
    benchmark = group.loc[returns.index, "benchmark_return"].astype(float)
    n = len(returns)
    if n < 3:
        return {}
    wealth = (1.0 + returns).cumprod()
    benchmark_wealth = (1.0 + benchmark).cumprod()
    cagr = float(wealth.iloc[-1] ** (12.0 / n) - 1.0) if (wealth > 0).all() else np.nan
    benchmark_cagr = float(benchmark_wealth.iloc[-1] ** (12.0 / n) - 1.0)
    vol = float(returns.std(ddof=1) * np.sqrt(12.0))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0)) if returns.std(ddof=1) > 0 else np.nan
    drawdown = wealth / wealth.cummax() - 1.0
    active = returns - benchmark
    information_ratio = float(active.mean() / active.std(ddof=1) * np.sqrt(12.0)) if active.std(ddof=1) > 0 else np.nan
    beta, alpha_ann = np.nan, np.nan
    if benchmark.std(ddof=1) > 0:
        slope, intercept, *_ = stats.linregress(benchmark, returns)
        beta, alpha_ann = float(slope), float(intercept * 12.0)
    return {
        "n_months": n,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
        "average_monthly_turnover": float(group.loc[returns.index, "turnover"].mean()),
        "benchmark_cagr": benchmark_cagr,
        "information_ratio_vs_benchmark": information_ratio,
        "alpha_annualized": alpha_ann,
        "beta": beta,
    }


def compute_portfolio_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["signal", "strategy", "cost_bps"]
    for period in PERIODS:
        sample = monthly if period == "full" else monthly[monthly["period"] == period]
        for values, group in sample.groupby(keys):
            metrics = _performance_metrics(group.sort_values("date"))
            if metrics:
                rows.append({"period": period, **dict(zip(keys, values)), **metrics})
    return pd.DataFrame(rows).sort_values(["period", *keys]).reset_index(drop=True)


def compute_factor_attribution(
    monthly: pd.DataFrame,
    fama_french: pd.DataFrame,
    spec: RealModelSpec,
) -> pd.DataFrame:
    """Regress portfolio returns on FF5 plus momentum with HAC inference."""
    factors = fama_french.copy()
    factors["factor_month"] = factors["dateff"].dt.to_period("M")
    portfolios = monthly.copy()
    portfolios["factor_month"] = portfolios["date"].dt.to_period("M") + 1
    merged = portfolios.merge(
        factors.drop(columns=["dateff"]), on="factor_month", how="inner"
    )
    factor_columns = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]
    rows = []
    for period in PERIODS:
        sample = merged if period == "full" else merged[merged["period"] == period]
        for keys, group in sample.groupby(["signal", "strategy", "cost_bps"]):
            clean = group.dropna(subset=["net_return", "rf", *factor_columns]).copy()
            if len(clean) < 8:
                continue
            import statsmodels.api as sm

            y = clean["net_return"].astype(float)
            if keys[1] == "long_only":
                y = y - clean["rf"].astype(float)
            X = sm.add_constant(clean[factor_columns].astype(float), has_constant="add")
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            row = {
                "period": period,
                "signal": keys[0],
                "strategy": keys[1],
                "cost_bps": float(keys[2]),
                "n_months": int(fit.nobs),
                "alpha_annualized": float(fit.params["const"] * 12.0),
                "alpha_hac_t_stat": float(fit.tvalues["const"]),
                "alpha_p_value": float(fit.pvalues["const"]),
                "r_squared": float(fit.rsquared),
                "minimum_required_months": int(
                    spec.min_prospective_months
                    if period == "prospective"
                    else spec.min_inference_months
                ),
                "inference_eligible": bool(
                    fit.nobs
                    >= (
                        spec.min_prospective_months
                        if period == "prospective"
                        else spec.min_inference_months
                    )
                ),
                "test_family": (
                    "primary"
                    if keys == ("COMPOSITE", "continuous_long_short", 10.0)
                    else "exploratory"
                ),
            }
            for factor in factor_columns:
                row[f"beta_{factor}"] = float(fit.params[factor])
            rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["alpha_p_holm"] = np.nan
    for (_, _), index in output.groupby(["period", "test_family"]).groups.items():
        output.loc[index, "alpha_p_holm"] = _holm_adjust(
            output.loc[index, "alpha_p_value"]
        )
    output["alpha_significant_5pct"] = (
        output["inference_eligible"] & (output["alpha_p_holm"] < 0.05)
    )
    return output.sort_values(
        ["period", "signal", "strategy", "cost_bps"]
    ).reset_index(drop=True)


def compute_era_stability(panel: pd.DataFrame, spec: RealModelSpec) -> pd.DataFrame:
    """Evaluate each predeclared primary hypothesis by calendar decade."""
    score_map = {factor: f"{factor}_SCORE" for factor in FACTOR_COLUMNS}
    score_map["COMPOSITE"] = "COMPOSITE"
    retrospective = panel[panel["period"] == "retrospective"]
    rows = []
    for factor, registry in HYPOTHESIS_REGISTRY.items():
        horizon = int(registry["primary_horizon_months"])
        series = _monthly_ic_values(
            retrospective, score_map[factor], f"fwd_{horizon}m"
        )
        for decade, values in series.groupby((series.index.year // 10) * 10):
            t_stat, p_value = _hac_mean_test(values, maxlags=horizon - 1)
            mean_ic = float(values.mean())
            rows.append({
                "factor": factor,
                "primary_horizon_months": horizon,
                "era": f"{int(decade)}s",
                "n_months": int(len(values)),
                "mean_ic": mean_ic,
                "hac_t_stat": t_stat,
                "p_value": p_value,
                "hit_rate": float((values > 0).mean()),
                "expected_sign": int(registry["expected_sign"]),
                "expected_direction_observed": bool(
                    mean_ic * int(registry["expected_sign"]) > 0
                ),
            })
    return pd.DataFrame(rows).sort_values(["factor", "era"]).reset_index(drop=True)


def compute_quantile_diagnostics(
    panel: pd.DataFrame,
    spec: RealModelSpec,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Test cross-sectional monotonicity without publishing security-level rows."""
    score_map = {factor: f"{factor}_SCORE" for factor in FACTOR_COLUMNS}
    score_map["COMPOSITE"] = "COMPOSITE"
    rows = []
    for period in PERIODS:
        sample = panel if period == "full" else panel[panel["period"] == period]
        for factor, registry in HYPOTHESIS_REGISTRY.items():
            horizon = int(registry["primary_horizon_months"])
            score_column = score_map[factor]
            forward_column = f"fwd_{horizon}m"
            monthly = []
            for date, cross in sample.groupby("date"):
                usable = cross[[score_column, forward_column]].dropna().copy()
                if len(usable) < max(50, quantiles * 10):
                    continue
                usable["quantile"] = pd.qcut(
                    usable[score_column].rank(method="first"),
                    quantiles,
                    labels=False,
                ) + 1
                returns = usable.groupby("quantile")[forward_column].mean()
                if len(returns) == quantiles:
                    monthly.append({
                        "date": pd.Timestamp(date),
                        **{f"q{int(q)}": float(value) for q, value in returns.items()},
                    })
            monthly_frame = pd.DataFrame(monthly)
            if monthly_frame.empty:
                continue
            quantile_columns = [f"q{number}" for number in range(1, quantiles + 1)]
            means = monthly_frame[quantile_columns].mean()
            spread = monthly_frame[f"q{quantiles}"] - monthly_frame["q1"]
            spread_t, spread_p = _hac_mean_test(spread, maxlags=horizon - 1)
            monotonicity = stats.spearmanr(
                np.arange(1, quantiles + 1), means.to_numpy(dtype=float)
            ).statistic
            rows.append({
                "period": period,
                "factor": factor,
                "primary_horizon_months": horizon,
                "n_months": int(len(monthly_frame)),
                **{f"mean_q{number}": float(means[f"q{number}"]) for number in range(1, quantiles + 1)},
                "top_minus_bottom_mean": float(spread.mean()),
                "top_minus_bottom_hac_t_stat": spread_t,
                "top_minus_bottom_p_value": spread_p,
                "quantile_monotonicity_spearman": float(monotonicity),
            })
    return pd.DataFrame(rows).sort_values(["period", "factor"]).reset_index(drop=True)


def compute_hypothesis_summary(
    ic_summary: pd.DataFrame,
    era_stability: pd.DataFrame,
    quantile_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Apply transparent pass/fail gates to the primary retrospective tests."""
    rows = []
    for factor, registry in HYPOTHESIS_REGISTRY.items():
        horizon = int(registry["primary_horizon_months"])
        match = ic_summary[
            (ic_summary["period"] == "retrospective")
            & (ic_summary["factor"] == factor)
            & (ic_summary["horizon_months"] == horizon)
        ]
        quantile = quantile_diagnostics[
            (quantile_diagnostics["period"] == "retrospective")
            & (quantile_diagnostics["factor"] == factor)
        ]
        eras = era_stability[era_stability["factor"] == factor]
        if match.empty:
            continue
        result = match.iloc[0]
        expected_sign = int(registry["expected_sign"])
        positive_era_fraction = (
            float(eras["expected_direction_observed"].mean()) if not eras.empty else np.nan
        )
        monotonicity = (
            float(quantile.iloc[0]["quantile_monotonicity_spearman"])
            if not quantile.empty else np.nan
        )
        gates = {
            "direction_gate": bool(result["mean_ic"] * expected_sign > 0),
            "holm_significance_gate": bool(result["significant_5pct"]),
            "bootstrap_gate": bool(result["bootstrap_ci_lower"] * expected_sign > 0),
            "hit_rate_gate": bool(result["hit_rate"] >= 0.55),
            "era_stability_gate": bool(positive_era_fraction >= 0.75),
            "monotonicity_gate": bool(monotonicity >= 0.50),
        }
        if all(gates.values()):
            classification = "RETROSPECTIVELY_SUPPORTED"
        elif gates["direction_gate"]:
            classification = "DIRECTIONAL_BUT_NOT_VALIDATED"
        else:
            classification = "DIRECTION_REJECTED"
        rows.append({
            "factor": factor,
            "primary_horizon_months": horizon,
            "economic_rationale": registry["rationale"],
            "expected_sign": expected_sign,
            "mean_ic": float(result["mean_ic"]),
            "p_holm": float(result["p_holm"]),
            "bootstrap_ci_lower": float(result["bootstrap_ci_lower"]),
            "bootstrap_ci_upper": float(result["bootstrap_ci_upper"]),
            "hit_rate": float(result["hit_rate"]),
            "positive_era_fraction": positive_era_fraction,
            "quantile_monotonicity_spearman": monotonicity,
            **gates,
            "classification": classification,
            "prospective_validation_status": "NOT_STARTED",
        })
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)


def coverage_summary(panel: pd.DataFrame) -> dict:
    return {
        "start_date": panel["date"].min().date().isoformat(),
        "end_date": panel["date"].max().date().isoformat(),
        "months": int(panel["date"].nunique()),
        "unique_securities": int(panel["permno"].nunique()),
        "unique_companies": int(panel["permco"].nunique()),
        "average_monthly_universe": float(panel.groupby("date").size().mean()),
        "median_monthly_universe": float(panel.groupby("date").size().median()),
        "fundamental_coverage": float(panel["book_equity_millions"].notna().mean()),
        "composite_coverage": float(panel["COMPOSITE"].notna().mean()),
        "retrospective_months": int(panel.loc[panel["period"] == "retrospective", "date"].nunique()),
        "prospective_months": int(panel.loc[panel["period"] == "prospective", "date"].nunique()),
        "reported_date_availability_share": float(
            panel["availability_source"].isin(
                ["compustat_pdate", "compustat_fdate"]
            ).mean()
        ),
        "preliminary_date_availability_share": float(
            panel["availability_source"].eq("compustat_pdate").mean()
        ),
        "six_month_fallback_share": float(
            panel["availability_source"].eq("six_month_fallback").mean()
        ),
    }


def quality_receipt(panel: pd.DataFrame, request: WRDSResearchRequest) -> dict:
    dated = panel.dropna(subset=["availability_date"])
    checks = {
        "duplicate_company_month_rows": int(panel.duplicated(["permco", "date"]).sum()),
        "future_fundamental_rows": int((dated["availability_date"] > dated["date"]).sum()),
        "monthly_universe_above_cap": int(
            (panel.groupby("date").size() > request.max_universe).sum()
        ),
        "returns_below_minus_one": int((panel["total_ret"] < -1.0).sum()),
        "nonfinite_composite_scores": int(
            np.isinf(pd.to_numeric(panel["COMPOSITE"], errors="coerce")).sum()
        ),
    }
    return {
        "status": "PASS" if all(value == 0 for value in checks.values()) else "FAIL",
        "checks": checks,
        "max_abs_monthly_company_return": float(panel["total_ret"].abs().max()),
        "min_monthly_company_return": float(panel["total_ret"].min()),
        "max_monthly_company_return": float(panel["total_ret"].max()),
    }


def public_monthly_results(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return only portfolio-level derived data safe for a public evidence bundle."""
    allowed = [
        "date", "period", "signal", "strategy", "cost_bps", "gross_return",
        "net_return", "turnover", "benchmark_return", "universe_size",
        "long_count", "short_count",
    ]
    return monthly.loc[:, allowed].copy()
