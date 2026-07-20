#!/usr/bin/env python3
"""Run the frozen point-in-time CRSP/Compustat equity-factor study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_research_charts(
    output_dir: Path,
    hypotheses: pd.DataFrame,
    era_stability: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    """Write presentation charts from aggregate or portfolio-level evidence only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = hypotheses.sort_values("mean_ic")
    centers = ordered["mean_ic"].to_numpy(dtype=float) * 100.0
    lower = (ordered["mean_ic"] - ordered["bootstrap_ci_lower"]).to_numpy(dtype=float) * 100.0
    upper = (ordered["bootstrap_ci_upper"] - ordered["mean_ic"]).to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [
        "#1f6f5b" if value == "RETROSPECTIVELY_SUPPORTED" else "#6b7280"
        for value in ordered["classification"]
    ]
    ax.errorbar(
        centers, np.arange(len(ordered)), xerr=np.vstack([lower, upper]),
        fmt="none", ecolor="#9ca3af", capsize=4, linewidth=1.5,
    )
    ax.scatter(centers, np.arange(len(ordered)), c=colors, s=65, zorder=3)
    ax.axvline(0.0, color="#111827", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered)), ordered["factor"])
    ax.set_xlabel("Mean primary-horizon rank IC (%)")
    ax.set_title("Primary hypotheses with 95% moving-block intervals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "primary_hypothesis_ic.png", dpi=160)
    plt.close(fig)

    composite = monthly[
        (monthly["period"] == "retrospective")
        & (monthly["signal"] == "COMPOSITE")
        & (monthly["strategy"] == "continuous_long_short")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cost_bps, group in composite.groupby("cost_bps"):
        group = group.sort_values("date")
        wealth = (1.0 + group["net_return"].astype(float)).cumprod()
        ax.plot(group["date"], wealth, label=f"{cost_bps:g} bps")
    ax.axhline(1.0, color="#111827", linewidth=0.8)
    ax.set_ylabel("Growth of $1")
    ax.set_title("Composite continuous long-short: retrospective cost sensitivity")
    ax.legend(title="Cost per dollar traded")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "composite_cost_sensitivity.png", dpi=160)
    plt.close(fig)

    pivot = era_stability.pivot(index="factor", columns="era", values="mean_ic")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(pivot.to_numpy(dtype=float) * 100.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("Primary-horizon rank IC by calendar era (%)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            ax.text(
                column, row, f"{pivot.iloc[row, column] * 100.0:.1f}",
                ha="center", va="center", fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Mean IC (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "era_stability_heatmap.png", dpi=160)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    specification_id: str,
    request,
    coverage: dict,
    portfolio: pd.DataFrame,
    ic: pd.DataFrame,
    attribution: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> None:
    headline = portfolio[
        (portfolio["period"] == "retrospective")
        & (portfolio["signal"] == "COMPOSITE")
        & (portfolio["strategy"] == "continuous_long_short")
        & (portfolio["cost_bps"] == 10.0)
    ]
    if headline.empty:
        headline_text = "The retrospective composite portfolio is unavailable."
    else:
        row = headline.iloc[0]
        headline_text = (
            "The locked continuous-score composite long-short portfolio at 10 bps "
            f"recorded a {row['sharpe']:.2f} Sharpe, {row['cagr']:.2%} CAGR, and "
            f"{row['max_drawdown']:.2%} maximum drawdown across "
            f"{int(row['n_months'])} retrospective months."
        )
    supported = hypotheses[
        hypotheses["classification"] == "RETROSPECTIVELY_SUPPORTED"
    ]
    significance_text = (
        f"{len(supported)} of {len(hypotheses)} predeclared hypotheses pass every "
        "retrospective direction, Holm, block-bootstrap, hit-rate, era-stability, "
        "and monotonicity gate."
    )
    alpha = attribution[
        (attribution["period"] == "retrospective")
        & (attribution["signal"] == "COMPOSITE")
        & (attribution["strategy"] == "continuous_long_short")
        & (attribution["cost_bps"] == 10.0)
    ]
    alpha_text = (
        f"Retrospective FF5+momentum alpha is {alpha.iloc[0]['alpha_annualized']:.2%} "
        f"annualized (HAC t={alpha.iloc[0]['alpha_hac_t_stat']:.2f})."
        if not alpha.empty
        else "Factor attribution is unavailable."
    )
    text = f"""# Real WRDS results

This evidence bundle is generated from CRSP monthly stock data and Compustat
annual fundamentals under specification `{specification_id}`. All currently
available rows are retrospective. The prospective clock starts on
{request.holdout_start.isoformat()} and requires 36 never-inspected months.

{headline_text} {significance_text} {alpha_text}

## Evidence boundary

- Universe: point-in-time NYSE/AMEX/Nasdaq common stocks, filtered to price >= $5,
  market capitalization >= $100 million, and the largest 1,000 companies monthly.
- Returns include CRSP delisting returns.
- Fundamentals use Compustat preliminary/final availability dates when present
  and a conservative six-month fallback otherwise.
- The composite uses fixed equal weights; no return-fitted weights are used.
- Non-size signals are neutralized to SIC sector and log market capitalization.
- The primary implementation is a continuous-score, dollar-neutral portfolio;
  the original quintile portfolio remains a comparator.
- Transaction-cost cases are 0, 10, and 25 bps per dollar traded.
- The prospective boundary is {request.holdout_start.isoformat()}.
- Primary hypotheses use one declared horizon each and Holm correction; other
  horizons form a separately corrected exploratory family.
- Inference includes HAC tests, deterministic 12-month moving-block bootstrap
  intervals, calendar-era stability, and quintile monotonicity.
- The former 2024 holdout was inspected and is no longer called pristine. It is
  part of the retrospective audit. No prospective result exists yet.
- Licensed security-level rows remain under ignored `data/private/` and are not committed.

Coverage: {coverage['months']} months, {coverage['unique_securities']} distinct securities,
and an average monthly universe of {coverage['average_monthly_universe']:.0f}.

## Artifacts

- `specification.json` — frozen request and model choices
- `data_manifest.json` — aggregate row counts and hashes of ignored private inputs
- `coverage.json` — aggregate coverage diagnostics
- `quality_receipt.json` — duplicate, look-ahead, universe, and return-bound gates
- `headline_summary.json` — machine-readable conclusion and composite metrics
- `research_protocol.json` — locked hypotheses, gates, and prospective policy
- `hypothesis_tests.csv` — primary-hypothesis pass/fail ledger
- `era_stability.csv` — decade-by-decade primary IC evidence
- `quantile_diagnostics.csv` — monotonicity and spread diagnostics
- `ic_summary.csv` — Spearman IC, HAC/bootstrap inference, and Holm correction
- `portfolio_summary.csv` — retrospective, prospective, and full-period metrics
- `factor_attribution.csv` — FF5 plus momentum alpha and factor loadings
- `monthly_portfolio_returns.csv` — portfolio-level derived returns only
- `primary_hypothesis_ic.png` — primary IC estimates and bootstrap intervals
- `composite_cost_sensitivity.png` — 0/10/25-bps cumulative portfolio evidence
- `era_stability_heatmap.png` — calendar-era primary IC stability

These are historical research results, not live performance or investment advice.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1990-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--holdout-start", default="2025-01-01")
    parser.add_argument("--max-universe", type=int, default=1000)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-market-cap-millions", type=float, default=100.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/wrds_real_data"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--scope", choices=["retrospective", "full"], default="retrospective",
        help="Prospective rows are opened only after the protocol is frozen.",
    )
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from factors.real_model import (
        HYPOTHESIS_REGISTRY,
        RealModelSpec,
        build_point_in_time_panel,
        compute_era_stability,
        compute_hypothesis_summary,
        compute_ic_summary,
        compute_factor_attribution,
        compute_monthly_portfolios,
        compute_portfolio_summary,
        compute_quantile_diagnostics,
        coverage_summary,
        public_monthly_results,
        quality_receipt,
        specification_hash,
    )
    from factors.wrds_data import WRDSResearchRequest, load_or_fetch_private_inputs

    request = WRDSResearchRequest.create(
        args.start,
        args.end,
        args.holdout_start,
        max_universe=args.max_universe,
        min_price=args.min_price,
        min_market_cap_millions=args.min_market_cap_millions,
    )
    spec = RealModelSpec()
    specification_id = specification_hash(request, spec)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    specification_path = args.results_dir / "specification.json"
    frozen_at_utc = datetime.now(timezone.utc).isoformat()
    if specification_path.exists():
        try:
            prior = json.loads(specification_path.read_text(encoding="utf-8"))
            if prior.get("specification_sha256") == specification_id:
                frozen_at_utc = prior.get("frozen_at_utc", frozen_at_utc)
        except (OSError, json.JSONDecodeError):
            pass
    specification = {
        "schema": "equity-factor-real-data-specification.v2",
        "specification_sha256": specification_id,
        "frozen_at_utc": frozen_at_utc,
        "protocol_frozen_before_prospective_data": True,
        "prospective_start": request.holdout_start.isoformat(),
        "prospective_status": "NOT_STARTED",
        "historical_disclosure": (
            "All data through 2024 have been inspected and are retrospective. The "
            "former 2024 holdout is not presented as pristine. Only observations on "
            "or after 2025-01-01 that were unavailable when this protocol was frozen "
            "may enter the prospective ledger."
        ),
        "audit_history": (
            "A 2010-2024 audit exposed share-class continuity and short-holdout "
            "problems. Version 2 extends the historical replication, uses Compustat "
            "preliminary/final dates when available, adds the investment signal, "
            "separates primary from exploratory tests, and locks a future-only "
            "validation policy. Version-2 results remain retrospective."
        ),
        "request": request.public_dict(),
        "model": spec.public_dict(),
    }
    specification_path.write_text(
        json.dumps(specification, indent=2), encoding="utf-8"
    )

    crsp, fundamentals, fama_french, cache_info = load_or_fetch_private_inputs(
        request, args.cache_dir, refresh=args.refresh
    )
    panel = build_point_in_time_panel(crsp, fundamentals, request, spec)
    coverage = coverage_summary(panel)
    quality = quality_receipt(panel, request)
    if quality["status"] != "PASS":
        raise RuntimeError(f"point-in-time quality gate failed: {quality['checks']}")
    ic = compute_ic_summary(panel, spec)
    era_stability = compute_era_stability(panel, spec)
    quantile_diagnostics = compute_quantile_diagnostics(panel, spec)
    hypotheses = compute_hypothesis_summary(
        ic, era_stability, quantile_diagnostics
    )
    monthly = compute_monthly_portfolios(panel, spec)
    portfolio = compute_portfolio_summary(monthly)
    attribution = compute_factor_attribution(monthly, fama_french, spec)
    if args.scope == "retrospective":
        ic = ic[ic["period"] == "retrospective"].copy()
        quantile_diagnostics = quantile_diagnostics[
            quantile_diagnostics["period"] == "retrospective"
        ].copy()
        monthly = monthly[monthly["period"] == "retrospective"].copy()
        portfolio = portfolio[portfolio["period"] == "retrospective"].copy()
        attribution = attribution[
            attribution["period"] == "retrospective"
        ].copy()

    crsp_path = Path(cache_info["crsp_cache"])
    comp_path = Path(cache_info["compustat_cache"])
    ff_path = Path(cache_info["fama_french_cache"])
    manifest = {
        "schema": "equity-factor-private-data-manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "specification_sha256": specification_id,
        "licensed_rows_committed": False,
        "source": cache_info["source"],
        "crsp_rows": int(len(crsp)),
        "compustat_rows": int(len(fundamentals)),
        "fama_french_rows": int(len(fama_french)),
        "panel_rows": int(len(panel)),
        "private_input_hashes": {
            "crsp_monthly_sha256": _file_hash(crsp_path),
            "compustat_annual_sha256": _file_hash(comp_path),
            "fama_french_monthly_sha256": _file_hash(ff_path),
        },
    }
    (args.results_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.results_dir / "coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    (args.results_dir / "quality_receipt.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    protocol = {
        "schema": "equity-factor-research-protocol.v1",
        "specification_sha256": specification_id,
        "frozen_at_utc": frozen_at_utc,
        "hypotheses": HYPOTHESIS_REGISTRY,
        "primary_portfolio": {
            "signal": "COMPOSITE",
            "strategy": "continuous_long_short",
            "cost_bps": 10.0,
        },
        "retrospective_gates": [
            "expected IC direction",
            "Holm-adjusted p < 0.05 within the primary family",
            "95% moving-block bootstrap interval excludes zero",
            "monthly IC hit rate >= 55%",
            "expected direction in >= 75% of calendar eras",
            "quintile-return monotonicity Spearman >= 0.50",
        ],
        "prospective_policy": {
            "start": request.holdout_start.isoformat(),
            "minimum_months": spec.min_prospective_months,
            "status": "NOT_STARTED",
            "rule": (
                "Do not call any result prospective unless the observation was "
                "unavailable when this protocol hash was frozen. No parameter changes "
                "are permitted before the minimum window closes."
            ),
        },
    }
    (args.results_dir / "research_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    ic.to_csv(args.results_dir / "ic_summary.csv", index=False)
    hypotheses.to_csv(args.results_dir / "hypothesis_tests.csv", index=False)
    era_stability.to_csv(args.results_dir / "era_stability.csv", index=False)
    quantile_diagnostics.to_csv(
        args.results_dir / "quantile_diagnostics.csv", index=False
    )
    portfolio.to_csv(args.results_dir / "portfolio_summary.csv", index=False)
    attribution.to_csv(args.results_dir / "factor_attribution.csv", index=False)
    public_monthly_results(monthly).to_csv(
        args.results_dir / "monthly_portfolio_returns.csv", index=False
    )
    _write_research_charts(
        args.results_dir, hypotheses, era_stability, monthly
    )
    headline_portfolio = portfolio[
        (portfolio["signal"] == "COMPOSITE")
        & (portfolio["strategy"] == "continuous_long_short")
        & (portfolio["cost_bps"] == 10.0)
        & (portfolio["period"].isin(["retrospective", "prospective"]))
    ]
    headline_ic = ic[
        (ic["factor"] == "COMPOSITE")
        & (ic["horizon_months"] == 12)
        & (ic["period"].isin(["retrospective", "prospective"]))
    ]
    headline_alpha = attribution[
        (attribution["signal"] == "COMPOSITE")
        & (attribution["strategy"] == "continuous_long_short")
        & (attribution["cost_bps"] == 10.0)
        & (attribution["period"] == "retrospective")
    ]
    headline = {
        "schema": "equity-factor-headline.v2",
        "specification_sha256": specification_id,
        "classification": "NO_VALIDATED_ALPHA",
        "interpretation": (
            "All current evidence is retrospective. Hypotheses are evaluated with "
            "predeclared primary horizons, multiple-testing control, block bootstrap, "
            "era stability, and monotonicity gates. No prospective alpha claim is made."
        ),
        "prospective_validation_status": "NOT_STARTED",
        "primary_hypothesis_classifications": hypotheses[
            ["factor", "classification"]
        ].to_dict("records"),
        "composite_continuous_long_short_10bps": headline_portfolio.to_dict("records"),
        "composite_primary_twelve_month_ic": headline_ic.to_dict("records"),
        "retrospective_ff5_momentum_alpha": headline_alpha.to_dict("records"),
    }
    (args.results_dir / "headline_summary.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    _write_readme(
        args.results_dir, specification_id, request, coverage, portfolio, ic,
        attribution, hypotheses,
    )

    print(json.dumps({
        "status": "PASS",
        "scope": args.scope,
        "specification_sha256": specification_id,
        "coverage": coverage,
        "quality": quality,
        "public_artifacts": str(args.results_dir),
        "licensed_rows_committed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
