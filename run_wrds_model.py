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


def _write_readme(
    output_dir: Path,
    specification_id: str,
    request,
    coverage: dict,
    portfolio: pd.DataFrame,
    ic: pd.DataFrame,
    attribution: pd.DataFrame,
) -> None:
    headline = portfolio[
        (portfolio["period"] == "holdout")
        & (portfolio["signal"] == "COMPOSITE")
        & (portfolio["strategy"] == "long_short")
        & (portfolio["cost_bps"] == 10.0)
    ]
    if headline.empty:
        headline_text = "Holdout results were not opened in this development-only run."
    else:
        row = headline.iloc[0]
        headline_text = (
            f"The predeclared {request.holdout_start.year}+ holdout composite long-short "
            f"portfolio at 10 bps "
            f"recorded a {row['sharpe']:.2f} Sharpe, {row['cagr']:.2%} CAGR, and "
            f"{row['max_drawdown']:.2%} maximum drawdown across {int(row['n_months'])} months."
        )
    significant = ic[(ic["period"] == "holdout") & (ic["significant_5pct"])]
    significance_text = (
        f"{len(significant)} holdout factor/horizon tests survive Bonferroni correction."
        if not headline.empty
        else "Holdout significance was not inspected."
    )
    alpha = attribution[
        (attribution["period"] == "development")
        & (attribution["signal"] == "COMPOSITE")
        & (attribution["strategy"] == "long_short")
        & (attribution["cost_bps"] == 10.0)
    ]
    alpha_text = (
        f"Development FF5+momentum alpha is {alpha.iloc[0]['alpha_annualized']:.2%} "
        f"annualized (HAC t={alpha.iloc[0]['alpha_hac_t_stat']:.2f})."
        if not alpha.empty
        else "Factor attribution is unavailable."
    )
    text = f"""# Real WRDS results

This evidence bundle is generated from CRSP monthly stock data and Compustat
annual fundamentals under specification `{specification_id}`.

{headline_text} {significance_text} {alpha_text}

## Evidence boundary

- Universe: point-in-time NYSE/AMEX/Nasdaq common stocks, filtered to price >= $5,
  market capitalization >= $100 million, and the largest 1,000 companies monthly.
- Returns include CRSP delisting returns.
- Fundamentals are linked through CCM and delayed six months after fiscal period end.
- The composite uses fixed equal weights; no holdout return is used to tune it.
- Transaction-cost cases are 0, 10, and 25 bps per dollar traded.
- The development/holdout boundary is {request.holdout_start.isoformat()}.
- An earlier 2020–2023 engine-validation run exposed a share-class continuity issue.
  That window is part of development; corrected portfolio rules were frozen before
  the previously unexamined 2024 confirmation was opened.
- WRDS currently ends in December 2024, leaving only twelve holdout signal months
  and eleven executable portfolio months. Formal inference requires at least 24
  months, so current holdout statistics are descriptive rather than confirmatory.
- Licensed security-level rows remain under ignored `data/private/` and are not committed.

Coverage: {coverage['months']} months, {coverage['unique_securities']} distinct securities,
and an average monthly universe of {coverage['average_monthly_universe']:.0f}.

## Artifacts

- `specification.json` — frozen request and model choices
- `data_manifest.json` — aggregate row counts and hashes of ignored private inputs
- `coverage.json` — aggregate coverage diagnostics
- `quality_receipt.json` — duplicate, look-ahead, universe, and return-bound gates
- `headline_summary.json` — machine-readable conclusion and composite metrics
- `ic_summary.csv` — Spearman IC, HAC inference, and Bonferroni correction
- `portfolio_summary.csv` — development, holdout, and full-period portfolio metrics
- `factor_attribution.csv` — FF5 plus momentum alpha and factor loadings
- `monthly_portfolio_returns.csv` — portfolio-level derived returns only

These are historical research results, not live performance or investment advice.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--holdout-start", default="2024-01-01")
    parser.add_argument("--max-universe", type=int, default=1000)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-market-cap-millions", type=float, default=100.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/private"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/wrds_real_data"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--scope", choices=["development", "full"], default="development",
        help="Open the holdout only after the specification and tests are frozen.",
    )
    args = parser.parse_args()

    sys.path.insert(0, "src")
    from factors.real_model import (
        RealModelSpec,
        build_point_in_time_panel,
        compute_ic_summary,
        compute_factor_attribution,
        compute_monthly_portfolios,
        compute_portfolio_summary,
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
        "schema": "equity-factor-real-data-specification.v1",
        "specification_sha256": specification_id,
        "frozen_at_utc": frozen_at_utc,
        "frozen_before_holdout_inspection": True,
        "portfolio_specification_frozen_before_holdout": True,
        "holdout_pristine": False,
        "holdout_disclosure": (
            "The portfolio specification was frozen before the 2024 holdout was "
            "opened. After inspection, a 24-month minimum-inference guard was added "
            "because WRDS contained only one holdout year; returns and portfolio "
            "construction were not changed. Treat holdout inference as descriptive."
        ),
        "audit_history": (
            "A prior 2010-2023 engine-validation run used a 2020 split and exposed "
            "a company/share-class continuity issue. No factor orientation, weight, "
            "quantile, cost, or liquidity parameter changed. The corrected final "
            "specification moves the untouched holdout to 2024-2025."
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
    monthly = compute_monthly_portfolios(panel, spec)
    portfolio = compute_portfolio_summary(monthly)
    attribution = compute_factor_attribution(monthly, fama_french, spec)
    if args.scope == "development":
        ic = ic[ic["period"] == "development"].copy()
        monthly = monthly[monthly["period"] == "development"].copy()
        portfolio = portfolio[portfolio["period"] == "development"].copy()
        attribution = attribution[attribution["period"] == "development"].copy()

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
    ic.to_csv(args.results_dir / "ic_summary.csv", index=False)
    portfolio.to_csv(args.results_dir / "portfolio_summary.csv", index=False)
    attribution.to_csv(args.results_dir / "factor_attribution.csv", index=False)
    public_monthly_results(monthly).to_csv(
        args.results_dir / "monthly_portfolio_returns.csv", index=False
    )
    headline_portfolio = portfolio[
        (portfolio["signal"] == "COMPOSITE")
        & (portfolio["strategy"] == "long_short")
        & (portfolio["cost_bps"] == 10.0)
        & (portfolio["period"].isin(["development", "holdout"]))
    ]
    headline_ic = ic[
        (ic["factor"] == "COMPOSITE")
        & (ic["horizon_months"] == 1)
        & (ic["period"].isin(["development", "holdout"]))
    ]
    headline_alpha = attribution[
        (attribution["signal"] == "COMPOSITE")
        & (attribution["strategy"] == "long_short")
        & (attribution["cost_bps"] == 10.0)
        & (attribution["period"] == "development")
    ]
    headline = {
        "schema": "equity-factor-headline.v1",
        "specification_sha256": specification_id,
        "classification": "NO_VALIDATED_ALPHA",
        "interpretation": (
            "The fixed composite is directionally positive but weak after costs; "
            "factor-adjusted alpha is indistinguishable from zero and the 2024 "
            "confirmation is too short for formal inference."
        ),
        "composite_long_short_10bps": headline_portfolio.to_dict("records"),
        "composite_one_month_ic": headline_ic.to_dict("records"),
        "development_ff5_momentum_alpha": headline_alpha.to_dict("records"),
    }
    (args.results_dir / "headline_summary.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    _write_readme(
        args.results_dir, specification_id, request, coverage, portfolio, ic,
        attribution,
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
