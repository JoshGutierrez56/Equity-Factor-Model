"""Run the frozen institutional version-3 equity-factor experiment."""
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


def _row(frame: pd.DataFrame, **filters) -> pd.Series:
    output = frame.copy()
    for column, value in filters.items():
        output = output[np.isclose(output[column], value) if isinstance(value, float) else output[column] == value]
    if len(output) != 1:
        raise RuntimeError(f"expected one row for {filters}; found {len(output)}")
    return output.iloc[0]


def _plot(results_dir: Path, monthly: pd.DataFrame, summary: pd.DataFrame) -> None:
    primary = monthly[
        np.isclose(monthly["aum_usd"], 100_000_000.0)
        & np.isclose(monthly["borrow_bps"], 150.0)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    wealth = (1.0 + primary.set_index("date")["net_return"]).cumprod()
    ax.plot(wealth.index, wealth, label="Institutional v3 — $100m / 150-bp borrow", linewidth=2)
    ax.axvline(pd.Timestamp("2021-01-01"), color="#b91c1c", linestyle="--", linewidth=1,
               label="Retrospective temporal assessment begins")
    ax.set_title("Frozen institutional implementation — cumulative wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "institutional_cumulative_wealth.png", dpi=170)
    plt.close(fig)

    stress = summary[summary["period"] == "temporal_assessment"].copy()
    pivot = stress.pivot(index="aum_usd", columns="borrow_bps", values="sharpe")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{value:g} bps" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"${value/1e6:g}m" for value in pivot.index])
    ax.set_xlabel("Annual short-borrow stress")
    ax.set_ylabel("Assumed AUM")
    ax.set_title("2021–2024 retrospective Sharpe — capacity/borrow stress")
    for i, aum in enumerate(pivot.index):
        for j, borrow in enumerate(pivot.columns):
            ax.text(j, i, f"{pivot.loc[aum, borrow]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(results_dir / "capacity_borrow_stress.png", dpi=170)
    plt.close(fig)


def _write_readme(results_dir: Path, receipt: dict, summary: pd.DataFrame) -> None:
    primary = _row(
        summary,
        period="temporal_assessment",
        aum_usd=100_000_000.0,
        borrow_bps=150.0,
    )
    comparison = receipt["matched_comparison"]
    text = f"""# Institutional Version-3 Evidence

This directory is the separately frozen version-3 implementation experiment.
It leaves the version-2 signal and both earlier portfolio result bundles intact.

## Verdict

The pre-2020 purged development folds selected **{receipt['selected_candidate']}**.
In the 2021–2024 retrospective temporal assessment, the locked $100 million,
150-bp borrow scenario recorded a **{primary['sharpe']:.2f} Sharpe**, **{primary['cagr']:.2%}
CAGR**, **{primary['max_drawdown']:.2%} maximum drawdown**, and **{primary['average_monthly_turnover']:.2f}
average monthly turnover**.

Against the frozen 10-bps baseline on matched months, the Sharpe difference was
**{comparison['sharpe_difference']:.2f}** with a paired moving-block 95% interval
of **{comparison['sharpe_difference_ci_lower']:.2f} to
{comparison['sharpe_difference_ci_upper']:.2f}**. The formal classification is
**{receipt['classification']}** and remains **NO VALIDATED ALPHA**.

![Cumulative wealth](institutional_cumulative_wealth.png)

![Capacity and borrow stress](capacity_borrow_stress.png)

## What changed

- Daily CRSP returns feed a 252-day Ledoit–Wolf shrinkage covariance matrix.
- CVXPY solves a long-short portfolio with dollar, beta, sector, size, position,
  gross-exposure, and turnover controls.
- Realized costs combine lagged CRSP closing bid/ask spreads, ADV/volatility
  nonlinear impact, and explicit short-borrow stress.
- Three parameter candidates were frozen before evaluation and selected only
  from separated pre-2020 development folds. The 2020 calendar year is the
  embargo before the 2021–2024 retrospective temporal assessment.
- Alphalens-style outputs report IC decay, sector IC, score autocorrelation,
  quantile membership turnover, and liquidity buckets.

These ideas are inspired by cvxportfolio, skfolio, Alphalens Reloaded, and
PyPortfolioOpt. Their code and reported performance were not copied.

## Evidence boundary

All dates through 2024 had already been inspected before this experiment. The
2021–2024 segment is therefore a temporal comparison, not a pristine holdout.
Licensed security-level WRDS rows remain in ignored private storage. Public
artifacts contain aggregate metrics only, and this is not investment advice.

Two independent cache-only executions reproduced every public artifact byte for
byte. See `replay_receipt.json` for the recorded SHA-256 hashes.
"""
    (results_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/institutional_v3"))
    parser.add_argument("--protocol", type=Path, default=Path("institutional_v3_protocol.json"))
    parser.add_argument("--refresh-daily", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from factors.daily_wrds import DailyCRSPRequest, load_or_fetch_daily_crsp
    from factors.institutional_v3 import (
        InstitutionalV3Spec,
        build_daily_snapshots,
        factor_diagnostics,
        institutional_specification_hash,
        paired_sharpe_difference_ci,
        run_candidate_backtest,
        select_candidate,
        summarize_v3,
    )
    from factors.real_model import RealModelSpec, _performance_metrics, build_point_in_time_panel
    from factors.wrds_data import WRDSResearchRequest, load_or_fetch_private_inputs

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_REAL_RETURN_EVALUATION":
        raise RuntimeError("institutional-v3 protocol is not frozen")
    baseline_spec = json.loads(Path("results/wrds_real_data/specification.json").read_text())
    engineering_protocol = json.loads(Path("portfolio_engineering_protocol.json").read_text())
    spec = InstitutionalV3Spec()
    computed_hash = institutional_specification_hash(
        baseline_spec["specification_sha256"],
        engineering_protocol["portfolio_engineering_sha256"],
        spec,
    )
    if computed_hash != protocol.get("institutional_v3_sha256"):
        raise RuntimeError("institutional-v3 specification changed after freeze")
    frozen_hashes = protocol["frozen_artifact_hashes"]
    current_hashes = {
        "baseline_portfolio_summary_sha256": _hash(Path("results/wrds_real_data/portfolio_summary.csv")),
        "engineering_portfolio_summary_sha256": _hash(Path("results/portfolio_engineering/portfolio_summary.csv")),
        "engineering_comparison_receipt_sha256": _hash(Path("results/portfolio_engineering/comparison_receipt.json")),
    }
    if current_hashes != frozen_hashes:
        raise RuntimeError("earlier committed evidence changed after v3 freeze")

    panel_cache_path = args.cache_dir / (
        f"institutional_v3_panel_{baseline_spec['specification_sha256'][:12]}.parquet"
    )
    print("[v3] loading/rebuilding frozen point-in-time monthly panel", flush=True)
    request = WRDSResearchRequest.create(
        "1990-01-01", "2025-12-31", "2025-01-01",
        max_universe=1000, min_price=5.0, min_market_cap_millions=100.0,
    )
    crsp, fundamentals, _, monthly_cache = load_or_fetch_private_inputs(request, args.cache_dir, refresh=False)
    if panel_cache_path.exists():
        research_panel = pd.read_parquet(panel_cache_path)
    else:
        panel = build_point_in_time_panel(crsp, fundamentals, request, RealModelSpec())
        research_panel = panel[
            (panel["date"] >= pd.Timestamp(spec.evaluation_start))
            & (panel["date"] < pd.Timestamp("2025-01-01"))
        ].copy()
        research_panel.to_parquet(panel_cache_path, index=False)
    permnos = research_panel["permno"].dropna().astype(int).unique().tolist()
    print(
        f"[v3] panel ready: {len(research_panel):,} company-months; "
        f"{len(permnos):,} selected securities",
        flush=True,
    )
    daily, daily_manifest = load_or_fetch_daily_crsp(
        permnos,
        DailyCRSPRequest(start="2009-01-01", end="2024-12-31"),
        args.cache_dir / "crsp_daily",
        refresh=args.refresh_daily,
    )
    print(f"[v3] daily panel ready: {len(daily):,} rows", flush=True)
    snapshot_cache_path = args.cache_dir / f"institutional_v3_snapshots_{computed_hash[:12]}.pkl"
    if snapshot_cache_path.exists() and not args.refresh_daily:
        with snapshot_cache_path.open("rb") as handle:
            snapshots = pickle.load(handle)
    else:
        snapshots = build_daily_snapshots(research_panel, daily, spec)
        with snapshot_cache_path.open("wb") as handle:
            pickle.dump(snapshots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[v3] risk/liquidity snapshots ready: {len(snapshots):,}", flush=True)
    development_snapshots = [
        snapshot for snapshot in snapshots
        if snapshot["date"] <= pd.Timestamp(spec.development_end)
    ]
    candidate_results = {}
    for candidate in spec.candidates:
        print(f"[v3] solving development candidate: {candidate.name}", flush=True)
        candidate_results[candidate.name] = run_candidate_backtest(
            development_snapshots, candidate, spec
        )
    print("[v3] development candidate grid solved", flush=True)
    selected_name, selection_table = select_candidate(candidate_results, spec)
    selected = next(candidate for candidate in spec.candidates if candidate.name == selected_name)
    monthly = run_candidate_backtest(snapshots, selected, spec)
    print(f"[v3] selected candidate replay complete: {selected_name}", flush=True)
    summary = summarize_v3(monthly, spec)
    diagnostics = factor_diagnostics(research_panel, daily, spec)

    primary = monthly[
        np.isclose(monthly["aum_usd"], spec.primary_aum_usd)
        & np.isclose(monthly["borrow_bps"], spec.primary_borrow_bps)
    ].copy()
    baseline_public = pd.read_csv("results/wrds_real_data/monthly_portfolio_returns.csv", parse_dates=["date"])
    baseline = baseline_public[
        (baseline_public["signal"] == "COMPOSITE")
        & (baseline_public["strategy"] == "continuous_long_short")
        & np.isclose(baseline_public["cost_bps"], 10.0)
        & (baseline_public["date"] >= pd.Timestamp(spec.temporal_assessment_start))
    ].copy()
    temporal = primary[primary["period"] == "temporal_assessment"].copy()
    observed, lower, upper = paired_sharpe_difference_ci(
        baseline, temporal,
        repetitions=spec.bootstrap_repetitions,
        block_months=spec.bootstrap_block_months,
        seed=spec.bootstrap_seed,
    )
    baseline_metrics = _performance_metrics(baseline.sort_values("date"))
    temporal_metrics = _performance_metrics(temporal.sort_values("date"))
    gates = {
        "matched_sharpe_improved": bool(temporal_metrics["sharpe"] > baseline_metrics["sharpe"]),
        "paired_bootstrap_ci_excludes_zero": bool(lower > 0),
        "turnover_not_above_baseline": bool(
            temporal_metrics["average_monthly_turnover"] <= baseline_metrics["average_monthly_turnover"]
        ),
        "maximum_drawdown_not_worse": bool(
            temporal_metrics["max_drawdown"] >= baseline_metrics["max_drawdown"]
        ),
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
    if all(gates.values()) and quality_pass:
        classification = "ROBUST_RETROSPECTIVE_INSTITUTIONAL_IMPROVEMENT"
    elif gates["matched_sharpe_improved"] and quality_pass:
        classification = "RETROSPECTIVE_IMPROVEMENT_NOT_BOOTSTRAP_CONFIRMED"
    else:
        classification = "NO_ROBUST_RETROSPECTIVE_IMPROVEMENT"
    receipt = {
        "schema": "equity-factor-institutional-v3-comparison.v1",
        "institutional_v3_sha256": computed_hash,
        "selected_candidate": selected_name,
        "selection_data_end": spec.development_end,
        "classification": classification,
        "alpha_classification": "NO_VALIDATED_ALPHA",
        "all_evidence_retrospective": True,
        "matched_comparison": {
            "baseline_10bps": baseline_metrics,
            "institutional_v3_primary": temporal_metrics,
            "sharpe_difference": observed,
            "sharpe_difference_ci_lower": lower,
            "sharpe_difference_ci_upper": upper,
        },
        "gates": gates,
    }
    quality = {
        "schema": "equity-factor-institutional-v3-quality.v1",
        "status": "PASS" if quality_pass else "FAIL",
        "checks": constraint_checks,
        "licensed_rows_committed": False,
    }
    if not quality_pass:
        raise RuntimeError(f"institutional-v3 quality gate failed: {constraint_checks}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.results_dir / "portfolio_summary.csv", index=False)
    selection_table.to_csv(args.results_dir / "purged_selection.csv", index=False)
    public_columns = [
        "date", "period", "candidate", "aum_usd", "borrow_bps", "gross_return",
        "net_return", "turnover", "benchmark_return", "universe_size", "long_count",
        "short_count", "gross_exposure", "net_exposure", "beta_exposure",
        "size_exposure", "maximum_sector_net_exposure", "maximum_absolute_weight",
        "effective_breadth", "median_spread_bps", "median_adv_usd", "spread_cost",
        "impact_cost", "borrow_cost", "total_cost", "maximum_adv_participation",
        "capacity_breaches", "risk_observations", "risk_universe", "solver",
        "solver_status",
    ]
    monthly[public_columns].to_csv(args.results_dir / "monthly_portfolio_returns.csv", index=False)
    for name, table in diagnostics.items():
        table.to_csv(args.results_dir / f"{name}.csv", index=False)
    (args.results_dir / "comparison_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))
    (args.results_dir / "quality_receipt.json").write_text(json.dumps(quality, indent=2))
    (args.results_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    manifest = {
        "schema": "equity-factor-institutional-v3-private-data-manifest.v1",
        "institutional_v3_sha256": computed_hash,
        "monthly_source": monthly_cache["source"],
        "daily_source": daily_manifest["source"],
        "daily_request": daily_manifest["request"],
        "daily_universe_sha256": daily_manifest["universe_sha256"],
        "daily_number_of_permnos": daily_manifest["number_of_permnos"],
        "daily_number_of_rows": daily_manifest["number_of_rows"],
        "daily_partition_count": daily_manifest["partition_count"],
        "daily_snapshot_count": len(snapshots),
        "licensed_rows_committed": False,
    }
    (args.results_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    _plot(args.results_dir, monthly, summary)
    _write_readme(args.results_dir, receipt, summary)
    public_hashes = {
        path.name: _hash(path)
        for path in sorted(args.results_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    (args.results_dir / "artifact_manifest.json").write_text(
        json.dumps({
            "schema": "equity-factor-institutional-v3-artifact-manifest.v1",
            "institutional_v3_sha256": computed_hash,
            "artifacts": public_hashes,
        }, indent=2)
    )
    print(json.dumps({
        "status": "PASS",
        "institutional_v3_sha256": computed_hash,
        "selected_candidate": selected_name,
        "classification": classification,
        "temporal_sharpe_primary": temporal_metrics["sharpe"],
        "temporal_cagr_primary": temporal_metrics["cagr"],
        "temporal_max_drawdown_primary": temporal_metrics["max_drawdown"],
        "sharpe_difference_vs_baseline": observed,
        "sharpe_difference_ci": [lower, upper],
        "licensed_rows_committed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
