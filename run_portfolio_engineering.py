"""Run the frozen retrospective portfolio-engineering experiment.

This script never changes the factor signal or the version-2 baseline bundle.
It validates a separately frozen protocol, reads licensed rows only from ignored
private storage, and writes aggregate public evidence to a separate directory.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row(frame: pd.DataFrame, strategy: str, cost_bps: float) -> pd.Series:
    match = frame[
        (frame["period"] == "retrospective")
        & (frame["strategy"] == strategy)
        & (frame["cost_bps"] == float(cost_bps))
    ]
    if len(match) != 1:
        raise RuntimeError(f"expected one summary row for {strategy} at {cost_bps} bps")
    return match.iloc[0]


def _write_charts(results_dir: Path, monthly: pd.DataFrame, summary: pd.DataFrame) -> None:
    colors = {
        "baseline_continuous_long_short": "#64748b",
        "risk_neutral": "#0f766e",
        "risk_turnover_managed": "#2563eb",
        "risk_turnover_vol_scaled": "#7c3aed",
    }
    labels = {
        "baseline_continuous_long_short": "Frozen baseline",
        "risk_neutral": "Risk neutral",
        "risk_turnover_managed": "Risk + turnover",
        "risk_turnover_vol_scaled": "Primary engineered",
    }
    sample = monthly[
        (monthly["period"] == "retrospective") & (monthly["cost_bps"] == 10.0)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for strategy, group in sample.groupby("strategy"):
        wealth = (1.0 + group.sort_values("date")["net_return"]).cumprod()
        ax.plot(
            group.sort_values("date")["date"],
            wealth,
            label=labels.get(strategy, strategy),
            color=colors.get(strategy),
            linewidth=2.0 if strategy == "risk_turnover_vol_scaled" else 1.5,
        )
    ax.set_yscale("log")
    ax.set_title("Retrospective cumulative wealth at 10 bps")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "engineered_cumulative_wealth.png", dpi=170)
    plt.close(fig)

    retrospective = summary[summary["period"] == "retrospective"].copy()
    strategies = list(labels)
    costs = [0.0, 10.0, 25.0]
    x = np.arange(len(strategies))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for offset, cost in enumerate(costs):
        values = []
        for strategy in strategies:
            match = retrospective[
                (retrospective["strategy"] == strategy)
                & (retrospective["cost_bps"] == cost)
            ]
            values.append(float(match.iloc[0]["sharpe"]) if not match.empty else np.nan)
        ax.bar(x + (offset - 1) * width, values, width, label=f"{cost:g} bps")
    ax.axhline(0.0, color="#0f172a", linewidth=0.8)
    ax.set_xticks(x, [labels[value] for value in strategies], rotation=12, ha="right")
    ax.set_ylabel("Annualized Sharpe")
    ax.set_title("Locked implementation comparison across transaction costs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "engineered_sharpe_costs.png", dpi=170)
    plt.close(fig)


def _write_readme(
    results_dir: Path,
    protocol: dict,
    summary: pd.DataFrame,
    attribution: pd.DataFrame,
    comparison: dict,
) -> None:
    baseline = _row(summary, "baseline_continuous_long_short", 10.0)
    primary = _row(summary, "risk_turnover_vol_scaled", 10.0)
    alpha = attribution[
        (attribution["period"] == "retrospective")
        & (attribution["strategy"] == "risk_turnover_vol_scaled")
        & (attribution["cost_bps"] == 10.0)
    ].iloc[0]
    text = f"""# Portfolio Engineering Evidence

This is a separately frozen, retrospective implementation experiment. It does
not alter the version-2 composite signal or overwrite its baseline artifacts.

## Verdict

The primary engineered portfolio recorded a **{primary['sharpe']:.2f} Sharpe at
10 bps**, versus **{baseline['sharpe']:.2f}** for the frozen baseline. Its CAGR
was **{primary['cagr']:.2%}**, maximum drawdown **{primary['max_drawdown']:.2%}**,
and average monthly turnover **{primary['average_monthly_turnover']:.2f}**.

The paired 12-month moving-block estimate of the Sharpe improvement was
**{comparison['paired_sharpe_improvement']:.2f}**, with a 95% interval of
**{comparison['paired_sharpe_improvement_ci_lower']:.2f} to
{comparison['paired_sharpe_improvement_ci_upper']:.2f}**. FF5 plus momentum
alpha was **{alpha['alpha_annualized']:.2%} annually** (HAC
t={alpha['alpha_hac_t_stat']:.2f}, p={alpha['alpha_p_value']:.3f}).

The classification is **{comparison['classification']}**. All evidence remains
retrospective and the project still makes **no validated prospective-alpha
claim**.

![Cumulative wealth](engineered_cumulative_wealth.png)

![Sharpe and costs](engineered_sharpe_costs.png)

## Frozen design

- Frozen signal: the unchanged six-signal composite from specification
  `{protocol['baseline_specification_sha256']}`.
- Primary portfolio: inverse-volatility score scaling; dollar, beta, SIC-sector,
  and log-size projection; 1.5% absolute position cap; 35% partial rebalance;
  five-basis-point weight-change no-trade band; and trailing 10% volatility
  targeting.
- Costs: 0, 10, and 25 bps per dollar traded.
- Comparators: the original continuous-score baseline, risk-neutral construction,
  and risk-plus-turnover construction.
- No signal weight, risk parameter, or candidate was selected from the resulting
  forward-return evidence.

## Public artifacts

- `protocol.json` - frozen experiment specification and acceptance criteria
- `comparison_receipt.json` - baseline-versus-primary conclusion
- `portfolio_summary.csv` - cost and performance comparison
- `factor_attribution.csv` - FF5 plus momentum attribution
- `portfolio_era_stability.csv` - calendar-decade results
- `engineering_diagnostics.csv` - aggregate exposure and concentration checks
- `monthly_portfolio_returns.csv` - aggregate portfolio returns only
- `baseline_integrity.json` - hashes and exact baseline-replay checks
- `quality_receipt.json` - protocol, privacy, exposure, and replay gates

Licensed security-level WRDS rows remain under ignored private storage. This
directory contains no security identifiers and no investment recommendation.
"""
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--max-universe", type=int, default=1000)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-market-cap-millions", type=float, default=100.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/private"))
    parser.add_argument(
        "--baseline-results-dir", type=Path, default=Path("results/wrds_real_data")
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/portfolio_engineering")
    )
    parser.add_argument(
        "--protocol", type=Path, default=Path("portfolio_engineering_protocol.json")
    )
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from factors.portfolio_engineering import (
        PortfolioEngineeringSpec,
        aggregate_diagnostics,
        compute_engineered_portfolios,
        compute_engineering_summary,
        engineering_specification_hash,
        paired_sharpe_improvement_ci,
        portfolio_era_stability,
    )
    from factors.real_model import (
        RealModelSpec,
        build_point_in_time_panel,
        compute_factor_attribution,
        compute_monthly_portfolios,
    )
    from factors.wrds_data import WRDSResearchRequest, load_or_fetch_private_inputs

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    baseline_specification = json.loads(
        (args.baseline_results_dir / "specification.json").read_text(encoding="utf-8")
    )
    baseline_hash = baseline_specification["specification_sha256"]
    engineering_spec = PortfolioEngineeringSpec()
    engineering_hash = engineering_specification_hash(baseline_hash, engineering_spec)
    if protocol.get("baseline_specification_sha256") != baseline_hash:
        raise RuntimeError("protocol does not reference the current frozen baseline")
    if protocol.get("portfolio_engineering_sha256") != engineering_hash:
        raise RuntimeError("portfolio-engineering parameters changed after protocol freeze")
    if protocol.get("status") != "FROZEN_BEFORE_RETURN_EVALUATION":
        raise RuntimeError("portfolio-engineering protocol is not frozen")
    frozen_hashes = protocol.get("baseline_artifact_hashes_at_freeze", {})
    current_hashes = {
        "portfolio_summary_sha256": _file_hash(
            args.baseline_results_dir / "portfolio_summary.csv"
        ),
        "monthly_portfolio_returns_sha256": _file_hash(
            args.baseline_results_dir / "monthly_portfolio_returns.csv"
        ),
        "research_protocol_sha256": _file_hash(
            args.baseline_results_dir / "research_protocol.json"
        ),
    }
    if frozen_hashes != current_hashes:
        raise RuntimeError("frozen baseline artifacts changed after protocol freeze")

    request = WRDSResearchRequest.create(
        args.start,
        args.end,
        args.holdout_start,
        max_universe=args.max_universe,
        min_price=args.min_price,
        min_market_cap_millions=args.min_market_cap_millions,
    )
    crsp, fundamentals, fama_french, cache_info = load_or_fetch_private_inputs(
        request, args.cache_dir, refresh=False
    )
    panel = build_point_in_time_panel(crsp, fundamentals, request, RealModelSpec())

    baseline_monthly = compute_monthly_portfolios(panel, RealModelSpec())
    baseline_monthly = baseline_monthly[
        (baseline_monthly["signal"] == "COMPOSITE")
        & (baseline_monthly["strategy"] == "continuous_long_short")
    ].copy()
    baseline_monthly["strategy"] = "baseline_continuous_long_short"
    engineered_monthly = compute_engineered_portfolios(panel, engineering_spec)
    monthly = pd.concat([baseline_monthly, engineered_monthly], ignore_index=True)
    monthly = monthly[monthly["period"] == "retrospective"].copy()
    summary = compute_engineering_summary(monthly)
    summary = summary[summary["period"] == "retrospective"].copy()
    attribution = compute_factor_attribution(
        monthly,
        fama_french,
        RealModelSpec(),
        primary_key=("COMPOSITE", "risk_turnover_vol_scaled", 10.0),
    )
    attribution = attribution[attribution["period"] == "retrospective"].copy()
    eras = portfolio_era_stability(monthly)
    diagnostics = aggregate_diagnostics(engineered_monthly)

    committed_baseline = pd.read_csv(args.baseline_results_dir / "portfolio_summary.csv")
    committed_baseline = committed_baseline[
        (committed_baseline["period"] == "retrospective")
        & (committed_baseline["signal"] == "COMPOSITE")
        & (committed_baseline["strategy"] == "continuous_long_short")
    ].sort_values("cost_bps")
    replayed = summary[
        summary["strategy"] == "baseline_continuous_long_short"
    ].sort_values("cost_bps")
    metric_columns = [
        "cagr", "annualized_volatility", "sharpe", "max_drawdown",
        "average_monthly_turnover",
    ]
    baseline_differences = {
        column: float(np.max(np.abs(
            committed_baseline[column].to_numpy(dtype=float)
            - replayed[column].to_numpy(dtype=float)
        )))
        for column in metric_columns
    }
    baseline_integrity = {
        "schema": "equity-factor-baseline-integrity.v1",
        "baseline_specification_sha256": baseline_hash,
        "baseline_artifact_hashes": {
            name: _file_hash(args.baseline_results_dir / name)
            for name in (
                "specification.json", "research_protocol.json",
                "portfolio_summary.csv", "monthly_portfolio_returns.csv",
            )
        },
        "maximum_absolute_metric_differences": baseline_differences,
        "exact_replay": all(value <= 1e-12 for value in baseline_differences.values()),
    }

    baseline = _row(summary, "baseline_continuous_long_short", 10.0)
    primary = _row(summary, "risk_turnover_vol_scaled", 10.0)
    observed, ci_lower, ci_upper = paired_sharpe_improvement_ci(
        monthly,
        "baseline_continuous_long_short",
        "risk_turnover_vol_scaled",
        10.0,
        engineering_spec,
    )
    primary_diagnostics = diagnostics[
        diagnostics["strategy"] == "risk_turnover_vol_scaled"
    ].iloc[0]
    gates = {
        "sharpe_improved": bool(primary["sharpe"] > baseline["sharpe"]),
        "turnover_reduced": bool(
            primary["average_monthly_turnover"]
            < baseline["average_monthly_turnover"]
        ),
        "maximum_drawdown_not_worse": bool(
            primary["max_drawdown"] >= baseline["max_drawdown"]
        ),
        "paired_bootstrap_ci_excludes_zero": bool(ci_lower > 0),
        "mean_absolute_beta_exposure_below_0_05": bool(
            primary_diagnostics["mean_abs_beta_exposure"] < 0.05
        ),
    }
    if all(gates.values()):
        classification = "ROBUST_RETROSPECTIVE_ENGINEERING_IMPROVEMENT"
    elif gates["sharpe_improved"] and gates["turnover_reduced"]:
        classification = "RETROSPECTIVE_IMPROVEMENT_NOT_BOOTSTRAP_CONFIRMED"
    else:
        classification = "NO_ROBUST_ENGINEERING_IMPROVEMENT"
    comparison = {
        "schema": "equity-factor-portfolio-engineering-comparison.v1",
        "portfolio_engineering_sha256": engineering_hash,
        "classification": classification,
        "alpha_classification": "NO_VALIDATED_ALPHA",
        "all_evidence_retrospective": True,
        "baseline_10bps": baseline.to_dict(),
        "primary_engineered_10bps": primary.to_dict(),
        "paired_sharpe_improvement": observed,
        "paired_sharpe_improvement_ci_lower": ci_lower,
        "paired_sharpe_improvement_ci_upper": ci_upper,
        "gates": gates,
    }

    constraint_checks = {
        "baseline_exact_replay_failures": int(not baseline_integrity["exact_replay"]),
        "protocol_hash_mismatch": 0,
        "prospective_rows_published": int((monthly["period"] != "retrospective").sum()),
        "mean_absolute_beta_above_0_05": int(
            primary_diagnostics["mean_abs_beta_exposure"] >= 0.05
        ),
        "mean_absolute_net_above_0_001": int(
            primary_diagnostics["mean_abs_net_exposure"] >= 0.001
        ),
        "max_position_above_declared_cap": int(
            primary_diagnostics["max_abs_maximum_absolute_weight"]
            > engineering_spec.maximum_absolute_weight + 1e-6
        ),
    }
    quality = {
        "schema": "equity-factor-portfolio-engineering-quality.v1",
        "status": "PASS" if all(value == 0 for value in constraint_checks.values()) else "FAIL",
        "checks": constraint_checks,
        "licensed_rows_committed": False,
    }
    if quality["status"] != "PASS":
        raise RuntimeError(f"portfolio-engineering quality gate failed: {constraint_checks}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results_dir / "portfolio_summary.csv", index=False)
    attribution.to_csv(args.results_dir / "factor_attribution.csv", index=False)
    eras.to_csv(args.results_dir / "portfolio_era_stability.csv", index=False)
    diagnostics.to_csv(args.results_dir / "engineering_diagnostics.csv", index=False)
    public_columns = [
        "date", "period", "signal", "strategy", "cost_bps", "gross_return",
        "net_return", "turnover", "benchmark_return", "universe_size",
        "long_count", "short_count", "gross_exposure", "net_exposure",
        "beta_exposure", "size_exposure", "maximum_sector_net_exposure",
        "maximum_absolute_weight", "effective_breadth", "leverage_multiplier",
        "risk_feature_coverage",
    ]
    available_columns = [column for column in public_columns if column in monthly]
    monthly[available_columns].to_csv(
        args.results_dir / "monthly_portfolio_returns.csv", index=False
    )
    (args.results_dir / "comparison_receipt.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    (args.results_dir / "baseline_integrity.json").write_text(
        json.dumps(baseline_integrity, indent=2), encoding="utf-8"
    )
    (args.results_dir / "quality_receipt.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    private_manifest = {
        "schema": "equity-factor-portfolio-engineering-private-data-manifest.v1",
        "portfolio_engineering_sha256": engineering_hash,
        "licensed_rows_committed": False,
        "source": cache_info["source"],
        "private_input_hashes": {
            "crsp_monthly_sha256": _file_hash(Path(cache_info["crsp_cache"])),
            "compustat_annual_sha256": _file_hash(Path(cache_info["compustat_cache"])),
            "fama_french_monthly_sha256": _file_hash(Path(cache_info["fama_french_cache"])),
        },
    }
    (args.results_dir / "data_manifest.json").write_text(
        json.dumps(private_manifest, indent=2), encoding="utf-8"
    )
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    _write_charts(args.results_dir, monthly, summary)
    _write_readme(args.results_dir, protocol, summary, attribution, comparison)

    print(json.dumps({
        "status": "PASS",
        "portfolio_engineering_sha256": engineering_hash,
        "classification": classification,
        "alpha_classification": "NO_VALIDATED_ALPHA",
        "baseline_sharpe_10bps": float(baseline["sharpe"]),
        "primary_sharpe_10bps": float(primary["sharpe"]),
        "primary_turnover": float(primary["average_monthly_turnover"]),
        "primary_max_drawdown": float(primary["max_drawdown"]),
        "licensed_rows_committed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
