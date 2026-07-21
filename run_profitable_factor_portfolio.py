"""Build the aggregate evidence bundle for the profitable-factor portfolio."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors.profitable_factor_portfolio import (
    ProfitableFactorPortfolioSpec,
    active_information_ratio,
    build_combined_returns,
    fit_factor_weights,
    fit_sleeve_mix,
    merge_factor_and_v4_returns,
    paired_block_sharpe_interval,
    profitable_factor_portfolio_hash,
    summarize_combined_returns,
)


ROOT = Path(__file__).resolve().parent


def _portable_hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(summary: pd.DataFrame, period: str, cost: float, strategy: str) -> pd.Series:
    match = summary[
        (summary["research_period"] == period)
        & np.isclose(summary["factor_cost_bps"], cost)
        & (summary["strategy"] == strategy)
    ]
    if len(match) != 1:
        raise RuntimeError(f"expected one metric row for {period}/{cost}/{strategy}")
    return match.iloc[0]


def _charts(monthly: pd.DataFrame, summary: pd.DataFrame, output: Path, spec) -> None:
    primary = monthly[
        (monthly["research_period"] == "temporal_assessment")
        & np.isclose(monthly["factor_cost_bps"], spec.primary_factor_cost_bps)
    ].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = {
        "v4 net": primary["net_return"],
        "factor sleeve (100-bp haircut)": primary["factor_sleeve_net_return"],
        "combined unlevered": primary["combined_unlevered_return"],
        "combined risk-scaled": primary["combined_risk_scaled_return"],
    }
    for label, values in series.items():
        ax.plot(primary["date"], (1.0 + values).cumprod(), label=label, linewidth=2)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Retrospective 2021-2024 cumulative wealth")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "factor_combination_cumulative_wealth.png", dpi=160)
    plt.close(fig)

    assessment = summary[
        (summary["research_period"] == "temporal_assessment")
        & (summary["strategy"] == "combined_unlevered")
    ].sort_values("factor_cost_bps")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(assessment["factor_cost_bps"], assessment["sharpe"], marker="o", linewidth=2)
    ax.axhline(
        _metric(summary, "temporal_assessment", 0.0, "v4_net")["sharpe"],
        color="black", linestyle="--", label="v4 net Sharpe",
    )
    ax.set_title("Combined portfolio Sharpe under factor-cost haircuts")
    ax.set_xlabel("Annual factor implementation haircut (bps)")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "factor_cost_stress.png", dpi=160)
    plt.close(fig)


def _readme(
    output: Path,
    factor_weights: pd.Series,
    sleeve_weights: pd.Series,
    leverage: float,
    summary: pd.DataFrame,
    receipt: dict,
    spec: ProfitableFactorPortfolioSpec,
) -> None:
    primary = _metric(
        summary, "temporal_assessment", spec.primary_factor_cost_bps,
        "combined_unlevered",
    )
    risk_scaled = _metric(
        summary, "temporal_assessment", spec.primary_factor_cost_bps,
        "combined_risk_scaled",
    )
    v4 = _metric(summary, "temporal_assessment", spec.primary_factor_cost_bps, "v4_net")
    factor_gross = _metric(summary, "temporal_assessment", 0.0, "factor_sleeve")
    factor_net = _metric(
        summary, "temporal_assessment", spec.primary_factor_cost_bps, "factor_sleeve"
    )
    weights = "\n".join(
        f"- `{factor.upper()}`: {weight:.1%}" for factor, weight in factor_weights.items()
    )
    text = f"""# Profitable-factor portfolio follow-up

This is an explicitly **retrospective, performance-informed allocation study**.
It preserves institutional v4 and combines its net stock-selection return with
established Fama-French and momentum test-portfolio returns. It is designed to
harvest known premiums; it does **not** claim new factor-neutral alpha.

## Primary result

For the 47 matched formation months from 2021 through November 2024, with the
v4 sleeve already net of its locked $100 million cost model and a conservative
**{spec.primary_factor_cost_bps:.0f}-bp annual haircut** applied to the factor sleeve:

- combined unlevered Sharpe: **{primary['sharpe']:.3f}** versus **{v4['sharpe']:.3f}** for v4;
- CAGR: **{primary['cagr']:.2%}**;
- annualized volatility: **{primary['annualized_volatility']:.2%}**;
- maximum drawdown: **{primary['maximum_drawdown']:.2%}**;
- factor-tilt IR versus the otherwise identical equal-factor combination:
  **{receipt['factor_tilt_information_ratio_vs_equal_factor']:.3f}**;
- active IR versus v4: **{receipt['active_information_ratio_vs_v4']:.3f}**.

The development-only volatility target hit its 2.0x leverage cap. That secondary
risk-scaled presentation recorded **{risk_scaled['sharpe']:.3f} Sharpe**,
**{risk_scaled['cagr']:.2%} CAGR**, and **{risk_scaled['maximum_drawdown']:.2%}**
maximum drawdown. Leverage does not improve Sharpe and is not the primary claim.

The gross academic factor sleeve recorded **{factor_gross['sharpe']:.3f} Sharpe**;
after the 100-bp haircut it recorded **{factor_net['sharpe']:.3f}**. The paired
12-month block interval for the unlevered Sharpe improvement over v4 is
**[{receipt['sharpe_difference_ci_lower']:.3f},
{receipt['sharpe_difference_ci_upper']:.3f}]**, so the improvement is not statistically confirmed.

![Cumulative wealth](factor_combination_cumulative_wealth.png)

![Cost stress](factor_cost_stress.png)

## Fixed weights

Factor weights use 1990-2019 data only: 50% equal weight plus 50% normalized
positive development Sharpe, with a 30% cap.

{weights}

The 2010-2019 inverse-volatility mix is **{sleeve_weights['v4']:.1%} v4** and
**{sleeve_weights['factor']:.1%} factor sleeve**. The risk-scaled diagnostic uses
**{leverage:.2f}x** leverage, fixed from development data and capped at 2.0x.

## IC and IR interpretation

The combined portfolio has no cross-sectional IC of its own. The applicable
stock-selection diagnostic remains v4's 12-month rank IC of 5.60% and ICIR of
1.087. The factor-tilt IR measures the benefit of the development-only tilt
relative to equal factor weights. The active IR versus v4 is also reported and
is negative because diversification improved risk-adjusted performance while
slightly lowering average return.

## Evidence boundary

Fama-French factors are academic test portfolios, not directly tradable
instruments. The flat 0/50/100/150-bp factor haircuts are stress scenarios, not
a security-level spread, impact, financing, and borrow reconstruction. All dates
had already been inspected, the Sharpe interval includes zero, and no prospective
alpha claim is made. Licensed security-level rows and weights remain uncommitted.
This is not investment advice.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    spec = ProfitableFactorPortfolioSpec()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "profitable_factor_portfolio_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = profitable_factor_portfolio_hash(
        protocol["institutional_v4_sha256"], spec
    )
    if protocol["profitable_factor_portfolio_sha256"] != expected_hash:
        raise RuntimeError("profitable-factor protocol hash does not match code specification")

    factor_path = Path(args.factor_data).resolve()
    v4_path = Path(args.v4_monthly).resolve()
    factors = pd.read_parquet(factor_path)
    v4 = pd.read_csv(v4_path)
    weights, diagnostics = fit_factor_weights(factors, spec)
    merged = merge_factor_and_v4_returns(factors, v4, weights, spec)
    sleeve_weights, leverage = fit_sleeve_mix(merged, spec)
    monthly = build_combined_returns(merged, sleeve_weights, leverage, spec)
    summary = summarize_combined_returns(monthly, spec)

    weights.rename("weight").rename_axis("factor").reset_index().to_csv(
        output / "factor_weights.csv", index=False
    )
    diagnostics.to_csv(output / "development_factor_diagnostics.csv", index=False)
    pd.DataFrame({
        "sleeve": ["v4_net", "factor_sleeve"],
        "weight": [sleeve_weights["v4"], sleeve_weights["factor"]],
    }).to_csv(output / "sleeve_weights.csv", index=False)
    monthly.to_csv(output / "monthly_returns.csv", index=False)
    summary.to_csv(output / "portfolio_summary.csv", index=False)
    _write_json(output / "protocol.json", protocol)

    assessment = monthly[
        (monthly["research_period"] == "temporal_assessment")
        & np.isclose(monthly["factor_cost_bps"], spec.primary_factor_cost_bps)
    ].sort_values("date")
    observed, lower, upper = paired_block_sharpe_interval(
        assessment["combined_unlevered_return"], assessment["net_return"], spec
    )
    receipt = {
        "schema": "equity-factor-profitable-factor-comparison.v1",
        "profitable_factor_portfolio_sha256": expected_hash,
        "classification": "RETROSPECTIVE_FACTOR_DIVERSIFICATION_IMPROVEMENT",
        "alpha_classification": "INTENTIONAL_ESTABLISHED_FACTOR_EXPOSURE_NOT_NEW_ALPHA",
        "all_evidence_retrospective": True,
        "primary_factor_cost_bps": spec.primary_factor_cost_bps,
        "factor_tilt_information_ratio_vs_equal_factor": active_information_ratio(
            assessment["combined_unlevered_return"]
            - assessment["equal_factor_combination_return"]
        ),
        "active_information_ratio_vs_v4": active_information_ratio(
            assessment["combined_unlevered_return"] - assessment["net_return"]
        ),
        "sharpe_difference_vs_v4": observed,
        "sharpe_difference_ci_lower": lower,
        "sharpe_difference_ci_upper": upper,
        "paired_interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "cross_sectional_ic_policy": (
            "combined portfolio has no cross-sectional IC; retain v4 score IC"
        ),
        "factor_implementation_limitation": (
            "academic factor returns plus flat annual haircuts; no security-level "
            "factor replication or realized transaction-cost reconstruction"
        ),
    }
    _write_json(output / "comparison_receipt.json", receipt)

    forbidden = {"permno", "permco", "ticker", "gvkey", "cusip"}
    public_columns = {column.lower() for column in monthly.columns}
    quality = {
        "schema": "equity-factor-profitable-factor-quality.v1",
        "status": "PASS",
        "aligned_months": int(monthly["date"].nunique()),
        "assessment_months": int(assessment["date"].nunique()),
        "factor_weight_sum": float(weights.sum()),
        "maximum_factor_weight": float(weights.max()),
        "sleeve_weight_sum": float(sleeve_weights.sum()),
        "security_identifier_columns_present": sorted(forbidden & public_columns),
        "licensed_rows_committed": False,
        "security_level_weights_committed": False,
        "raw_factor_rows_committed": False,
        "gpu_used": False,
    }
    if quality["security_identifier_columns_present"]:
        quality["status"] = "FAIL"
    _write_json(output / "quality_receipt.json", quality)

    data_manifest = {
        "schema": "equity-factor-profitable-factor-data-manifest.v1",
        "factor_source_sha256": _portable_hash(factor_path),
        "v4_monthly_source_sha256": _portable_hash(v4_path),
        "factor_source_rows": int(len(factors)),
        "v4_source_rows": int(len(v4)),
        "aligned_months": int(monthly["date"].nunique()),
        "committed_data_policy": (
            "aggregate monthly portfolio evidence only; no raw factor table, "
            "security rows, identifiers, or security-level weights"
        ),
    }
    _write_json(output / "data_manifest.json", data_manifest)
    _charts(monthly, summary, output, spec)
    _readme(output, weights, sleeve_weights, leverage, summary, receipt, spec)

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in {"artifact_manifest.json", "replay_receipt.json"}:
            artifacts[path.name] = _portable_hash(path)
    _write_json(output / "artifact_manifest.json", {
        "schema": "equity-factor-profitable-factor-artifact-manifest.v1",
        "artifacts": artifacts,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factor-data",
        default=str(ROOT / "data" / "private" / "fama_french_monthly_1990-01-01_2025-12-31.parquet"),
    )
    parser.add_argument(
        "--v4-monthly",
        default=str(ROOT / "results" / "institutional_v4" / "monthly_portfolio_returns.csv"),
    )
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "profitable_factor_portfolio")
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()
