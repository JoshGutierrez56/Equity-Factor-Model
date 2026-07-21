"""Run the frozen retrospective version-4 equity-factor follow-up."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _one(frame: pd.DataFrame, **filters) -> pd.Series:
    output = frame.copy()
    for column, value in filters.items():
        if isinstance(value, float):
            output = output[np.isclose(pd.to_numeric(output[column]), value)]
        else:
            output = output[output[column] == value]
    if len(output) != 1:
        raise RuntimeError(f"expected one row for {filters}; found {len(output)}")
    return output.iloc[0]


def _plot(
    results_dir: Path,
    baseline: pd.DataFrame,
    v3: pd.DataFrame,
    v4: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for label, frame, color in (
        ("Frozen baseline (10 bps)", baseline, "#64748b"),
        ("Institutional v3", v3, "#b91c1c"),
        ("V4 signal + holding buffer", v4, "#0369a1"),
    ):
        wealth = (1.0 + frame.sort_values("date").set_index("date")["net_return"]).cumprod()
        ax.plot(wealth.index, wealth, label=label, color=color, linewidth=2)
    ax.set_title("Matched 2021–2024 retrospective cumulative wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "v4_cumulative_wealth.png", dpi=170)
    plt.close(fig)

    metrics = comparison.set_index("implementation")[[
        "sharpe", "factor_residual_information_ratio", "average_monthly_turnover"
    ]]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    colors = ["#64748b", "#b91c1c", "#0369a1"]
    for ax, column, title in zip(
        axes,
        metrics.columns,
        ("Net Sharpe", "Factor-residual IR", "Monthly turnover"),
    ):
        values = metrics[column].astype(float)
        ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(values)), ["Baseline", "V3", "V4"], rotation=20)
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(results_dir / "v4_metric_comparison.png", dpi=170)
    plt.close(fig)


def _write_readme(
    results_dir: Path,
    receipt: dict,
    summary: pd.DataFrame,
    ic: pd.DataFrame,
    comparison: pd.DataFrame,
    weights: dict[str, float],
) -> None:
    primary = _one(
        summary, period="temporal_assessment", aum_usd=100_000_000.0,
        borrow_bps=150.0,
    )
    v4_ir = _one(comparison, implementation="v4")
    ic12 = _one(ic, period="temporal_assessment", score="V4_SCORE", horizon_months=12)
    base_ic12 = _one(ic, period="temporal_assessment", score="COMPOSITE", horizon_months=12)
    weight_lines = "\n".join(
        f"- `{feature}`: {weight:.1%}" for feature, weight in weights.items()
    )
    text = f"""# Institutional Version-4 Follow-up

This is an explicitly **post-result retrospective follow-up**. It preserves the
version-2 and version-3 evidence and does not create an untouched holdout.

## Verdict

At the locked $100 million / 150-bp borrow scenario, the 2021–2024 assessment
recorded a **{primary['sharpe']:.3f} net Sharpe**, **{primary['cagr']:.2%} CAGR**,
**{primary['max_drawdown']:.2%} maximum drawdown**, and **{primary['average_monthly_turnover']:.3f}
average monthly turnover**. Its self-financing overlay information ratio was
**{v4_ir['overlay_information_ratio']:.3f}**; this equals the net Sharpe because
the sleeve return itself is the active return. The stricter FF5+momentum
factor-residual information ratio was **{v4_ir['factor_residual_information_ratio']:.3f}**.

The version-4 12-month rank IC was **{ic12['mean_rank_ic']:.4f}** with ICIR
**{ic12['icir']:.3f}**, versus **{base_ic12['mean_rank_ic']:.4f}** and
**{base_ic12['icir']:.3f}** for the frozen equal-weight composite over the same
dates. Formal classification: **{receipt['classification']}**; alpha status:
**NO VALIDATED ALPHA**.

![Matched cumulative wealth](v4_cumulative_wealth.png)

![Metric comparison](v4_metric_comparison.png)

## What changed

One development-only score was fixed before evaluation. Monthly ICs from
2010–2019 were averaged uniformly across 1, 3, 6, and 12-month horizons,
clipped at zero, normalized, and shrunk 50% toward equal weights. `SIZE_SCORE`
is excluded because size is an explicit risk constraint.

{weight_lines}

The portfolio adds an entry/exit buffer: new positions require an absolute
residual-score rank of 200 or better, while existing holdings remain eligible
through rank 500. Monthly turnover is capped at 0.75 and each position is
limited by both a 2% box constraint and lagged ADV capacity.

## Metric definitions

- Sharpe uses monthly net returns annualized by square-root of 12.
- IC is monthly Spearman rank correlation between the score and forward returns.
- ICIR is mean monthly IC divided by its monthly standard deviation.
- Overlay IR is annualized mean net sleeve return divided by its annualized
  tracking-error contribution. For this dollar-neutral self-financing sleeve,
  it is mathematically the same number as net Sharpe—not a second independent win.
- Factor-residual IR is annualized FF5+momentum regression alpha divided by
  annualized regression-residual volatility.
- Benchmark-relative IR remains in the portfolio summary for transparency, but
  it is not the main IR for a dollar-neutral sleeve.

## Evidence boundary

All dates through 2024 had already been inspected before version 4. Licensed
security-level WRDS rows and weight paths remain under ignored `data/private/`.
Committed files contain only aggregate evidence. This is not investment advice.
"""
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/institutional_v4"))
    parser.add_argument("--protocol", type=Path, default=Path("institutional_v4_protocol.json"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from factors.daily_wrds import DailyCRSPRequest, load_or_fetch_daily_crsp
    from factors.institutional_v3 import paired_sharpe_difference_ci
    from factors.institutional_v4 import (
        InstitutionalV4Spec,
        apply_ensemble_score,
        build_snapshots_and_weight_path,
        factor_residual_information_ratio,
        feasibility_scan,
        fit_development_ensemble,
        institutional_v4_hash,
        run_v4_backtest,
        score_ic_summary,
        summarize_v4,
    )
    from factors.real_model import RealModelSpec, _performance_metrics, build_point_in_time_panel
    from factors.wrds_data import WRDSResearchRequest, load_or_fetch_private_inputs

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_RETURN_EVALUATION":
        raise RuntimeError("institutional-v4 protocol is not frozen")
    spec = InstitutionalV4Spec()
    computed_hash = institutional_v4_hash(protocol["institutional_v3_sha256"], spec)
    if computed_hash != protocol["institutional_v4_sha256"]:
        raise RuntimeError("institutional-v4 specification changed after freeze")
    frozen = protocol["frozen_artifact_hashes"]
    current = {
        "v3_portfolio_summary_sha256": _hash(Path("results/institutional_v3/portfolio_summary.csv")),
        "v3_comparison_receipt_sha256": _hash(Path("results/institutional_v3/comparison_receipt.json")),
        "v3_artifact_manifest_sha256": _hash(Path("results/institutional_v3/artifact_manifest.json")),
        "baseline_ic_summary_sha256": _hash(Path("results/wrds_real_data/ic_summary.csv")),
    }
    if current != frozen:
        raise RuntimeError("earlier committed evidence changed after v4 freeze")

    baseline_spec = json.loads(Path("results/wrds_real_data/specification.json").read_text())
    request = WRDSResearchRequest.create(
        "1990-01-01", "2025-12-31", "2025-01-01",
        max_universe=1000, min_price=5.0, min_market_cap_millions=100.0,
    )
    print("[v4] loading frozen monthly inputs", flush=True)
    crsp, fundamentals, fama_french, cache_info = load_or_fetch_private_inputs(
        request, args.cache_dir, refresh=False
    )
    panel_path = args.cache_dir / f"institutional_v3_panel_{baseline_spec['specification_sha256'][:12]}.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_point_in_time_panel(crsp, fundamentals, request, RealModelSpec())
        panel = panel[(panel["date"] >= pd.Timestamp("2010-01-01")) & (panel["date"] < pd.Timestamp("2025-01-01"))]
        panel.to_parquet(panel_path, index=False)
    weights, training_ic = fit_development_ensemble(panel, spec)
    panel = apply_ensemble_score(panel, weights)
    print(f"[v4] development-only ensemble fixed: {json.dumps(weights, sort_keys=True)}", flush=True)

    permnos = panel["permno"].dropna().astype(int).unique().tolist()
    daily, daily_manifest = load_or_fetch_daily_crsp(
        permnos,
        DailyCRSPRequest(start="2009-01-01", end="2024-12-31"),
        args.cache_dir / "crsp_daily",
        refresh=False,
    )
    path_bundle_file = args.cache_dir / f"institutional_v4_path_bundle_{computed_hash[:12]}.pkl"
    if path_bundle_file.exists() and not args.refresh:
        with path_bundle_file.open("rb") as handle:
            bundle = pickle.load(handle)
        snapshots = bundle["snapshots"]
        weight_path = bundle["weight_path"]
    else:
        print("[v4] building risk snapshots and constraint-only holding path", flush=True)
        snapshots, weight_path = build_snapshots_and_weight_path(panel, daily, spec)
        with path_bundle_file.open("wb") as handle:
            pickle.dump(
                {"snapshots": snapshots, "weight_path": weight_path},
                handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
    print(f"[v4] constraint-only path ready: {len(snapshots)} snapshots", flush=True)
    feasibility = feasibility_scan(snapshots, spec, weight_path)
    if (feasibility.iloc[1:]["turnover"] > spec.maximum_monthly_turnover + 1e-4).any():
        raise RuntimeError("v4 feasibility turnover constraint failed before return evaluation")
    print("[v4] feasibility path passed; evaluating frozen returns once", flush=True)
    monthly = run_v4_backtest(snapshots, spec, weight_path)
    summary = summarize_v4(monthly, spec)
    ic = score_ic_summary(panel, spec=spec)

    primary = monthly[
        np.isclose(monthly["aum_usd"], spec.primary_aum_usd)
        & np.isclose(monthly["borrow_bps"], spec.primary_borrow_bps)
        & (monthly["period"] == "temporal_assessment")
    ].copy()
    baseline_public = pd.read_csv(
        "results/wrds_real_data/monthly_portfolio_returns.csv", parse_dates=["date"]
    )
    baseline = baseline_public[
        (baseline_public["signal"] == "COMPOSITE")
        & (baseline_public["strategy"] == "continuous_long_short")
        & np.isclose(baseline_public["cost_bps"], 10.0)
        & (baseline_public["date"] >= pd.Timestamp(spec.temporal_assessment_start))
    ].copy()
    v3_public = pd.read_csv(
        "results/institutional_v3/monthly_portfolio_returns.csv", parse_dates=["date"]
    )
    v3 = v3_public[
        np.isclose(v3_public["aum_usd"], spec.primary_aum_usd)
        & np.isclose(v3_public["borrow_bps"], spec.primary_borrow_bps)
        & (v3_public["period"] == "temporal_assessment")
    ].copy()
    observed, lower, upper = paired_sharpe_difference_ci(
        baseline, primary, repetitions=spec.bootstrap_repetitions,
        block_months=spec.bootstrap_block_months, seed=spec.bootstrap_seed,
    )
    comparison_rows = []
    for name, frame in (("baseline", baseline), ("v3", v3), ("v4", primary)):
        performance = _performance_metrics(frame.sort_values("date"))
        residual = factor_residual_information_ratio(frame, fama_french)
        comparison_rows.append({
            "implementation": name,
            **performance,
            "overlay_information_ratio": performance["sharpe"],
            **residual,
        })
    comparison = pd.DataFrame(comparison_rows)
    base_metrics = _one(comparison, implementation="baseline")
    v3_metrics = _one(comparison, implementation="v3")
    v4_metrics = _one(comparison, implementation="v4")
    base_ic12 = _one(ic, period="temporal_assessment", score="COMPOSITE", horizon_months=12)
    v4_ic12 = _one(ic, period="temporal_assessment", score="V4_SCORE", horizon_months=12)
    gates = {
        "sharpe_exceeds_baseline": bool(v4_metrics["sharpe"] > base_metrics["sharpe"]),
        "sharpe_exceeds_v3": bool(v4_metrics["sharpe"] > v3_metrics["sharpe"]),
        "turnover_not_above_baseline": bool(
            v4_metrics["average_monthly_turnover"] <= base_metrics["average_monthly_turnover"]
        ),
        "twelve_month_ic_exceeds_composite": bool(
            v4_ic12["mean_rank_ic"] > base_ic12["mean_rank_ic"]
        ),
        "factor_residual_ir_exceeds_baseline": bool(
            v4_metrics["factor_residual_information_ratio"]
            > base_metrics["factor_residual_information_ratio"]
        ),
        "paired_bootstrap_ci_excludes_zero": bool(lower > 0.0),
    }
    constraint_checks = {
        "gross_exposure_violations": int((primary["gross_exposure"].sub(spec.gross_exposure).abs() > 1e-4).sum()),
        "position_cap_violations": int((primary["maximum_absolute_weight"] > spec.maximum_absolute_weight + 1e-5).sum()),
        "turnover_cap_violations": int((primary.iloc[1:]["turnover"] > spec.maximum_monthly_turnover + 1e-4).sum()),
        "beta_tolerance_violations": int((primary["beta_exposure"].abs() > spec.beta_tolerance + 1e-4).sum()),
        "size_tolerance_violations": int((primary["size_exposure"].abs() > spec.size_tolerance + 1e-4).sum()),
        "sector_tolerance_violations": int((primary["maximum_sector_net_exposure"] > spec.sector_tolerance + 1e-4).sum()),
        "solver_failures": int((~primary["solver_status"].isin(["optimal", "optimal_inaccurate"])).sum()),
    }
    quality_pass = all(value == 0 for value in constraint_checks.values())
    if all(gates[key] for key in (
        "sharpe_exceeds_baseline", "sharpe_exceeds_v3",
        "turnover_not_above_baseline", "twelve_month_ic_exceeds_composite",
        "factor_residual_ir_exceeds_baseline",
    )) and quality_pass:
        classification = "RETROSPECTIVE_MULTI_METRIC_IMPROVEMENT"
    elif quality_pass:
        classification = "PARTIAL_RETROSPECTIVE_IMPROVEMENT"
    else:
        classification = "QUALITY_GATE_FAILURE"
    receipt = {
        "schema": "equity-factor-institutional-v4-comparison.v1",
        "institutional_v4_sha256": computed_hash,
        "classification": classification,
        "alpha_classification": "NO_VALIDATED_ALPHA",
        "all_evidence_retrospective": True,
        "reporting_clarification": {
            "overlay_information_ratio": (
                "For a self-financing dollar-neutral overlay, active return is "
                "the sleeve net return and overlay IR therefore equals net Sharpe."
            ),
            "factor_residual_information_ratio": (
                "Annualized FF5+momentum intercept divided by annualized residual "
                "volatility; retained as the stricter distinctiveness diagnostic."
            ),
            "clarification_added_after_evaluation": True,
            "strategy_or_return_series_changed": False,
        },
        "matched_sharpe_difference_vs_baseline": float(observed),
        "matched_sharpe_difference_ci_lower": float(lower),
        "matched_sharpe_difference_ci_upper": float(upper),
        "gates": gates,
    }
    quality = {
        "schema": "equity-factor-institutional-v4-quality.v1",
        "status": "PASS" if quality_pass else "FAIL",
        "checks": constraint_checks,
        "licensed_rows_committed": False,
        "security_level_weights_committed": False,
    }
    if not quality_pass:
        raise RuntimeError(f"v4 quality failure: {constraint_checks}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results_dir / "portfolio_summary.csv", index=False)
    comparison.to_csv(args.results_dir / "implementation_comparison.csv", index=False)
    ic.to_csv(args.results_dir / "ic_comparison.csv", index=False)
    training_ic.groupby(["feature", "horizon_months"]).agg(
        months=("date", "nunique"), mean_development_rank_ic=("rank_ic", "mean")
    ).reset_index().to_csv(args.results_dir / "development_ic.csv", index=False)
    pd.DataFrame([
        {"feature": feature, "ensemble_weight": weight}
        for feature, weight in weights.items()
    ]).to_csv(args.results_dir / "ensemble_weights.csv", index=False)
    feasibility.to_csv(args.results_dir / "feasibility_receipt.csv", index=False)
    public_columns = [
        "date", "period", "strategy", "aum_usd", "borrow_bps", "gross_return",
        "net_return", "turnover", "benchmark_return", "universe_size", "long_count",
        "short_count", "gross_exposure", "net_exposure", "beta_exposure",
        "size_exposure", "maximum_sector_net_exposure", "maximum_absolute_weight",
        "effective_breadth", "median_spread_bps", "median_adv_usd", "spread_cost",
        "impact_cost", "borrow_cost", "total_cost", "maximum_adv_participation",
        "capacity_breaches", "risk_observations", "risk_universe", "solver",
        "solver_status", "eligible_names", "buffered_holdings", "minimum_position_cap",
    ]
    monthly[public_columns].to_csv(args.results_dir / "monthly_portfolio_returns.csv", index=False)
    (args.results_dir / "comparison_receipt.json").write_text(json.dumps(receipt, indent=2))
    (args.results_dir / "quality_receipt.json").write_text(json.dumps(quality, indent=2))
    (args.results_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    manifest = {
        "schema": "equity-factor-institutional-v4-private-data-manifest.v1",
        "institutional_v4_sha256": computed_hash,
        "monthly_source": cache_info["source"],
        "daily_source": daily_manifest["source"],
        "daily_number_of_rows": daily_manifest["number_of_rows"],
        "daily_number_of_permnos": daily_manifest["number_of_permnos"],
        "snapshot_count": len(snapshots),
        "licensed_rows_committed": False,
        "security_level_weights_committed": False,
    }
    (args.results_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    _plot(args.results_dir, baseline, v3, primary, comparison)
    _write_readme(args.results_dir, receipt, summary, ic, comparison, weights)
    artifacts = {
        path.name: _hash(path) for path in sorted(args.results_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    (args.results_dir / "artifact_manifest.json").write_text(json.dumps({
        "schema": "equity-factor-institutional-v4-artifact-manifest.v1",
        "institutional_v4_sha256": computed_hash,
        "artifacts": artifacts,
    }, indent=2))
    print(json.dumps({
        "status": "PASS", "classification": classification,
        "institutional_v4_sha256": computed_hash,
        "temporal_sharpe": float(v4_metrics["sharpe"]),
        "temporal_turnover": float(v4_metrics["average_monthly_turnover"]),
        "temporal_12m_ic": float(v4_ic12["mean_rank_ic"]),
        "temporal_12m_icir": float(v4_ic12["icir"]),
        "overlay_information_ratio": float(v4_metrics["overlay_information_ratio"]),
        "factor_residual_information_ratio": float(v4_metrics["factor_residual_information_ratio"]),
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
