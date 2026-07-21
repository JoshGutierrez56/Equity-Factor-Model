import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

import pandas as pd

from factors.portfolio_engineering import (
    PortfolioEngineeringSpec,
    engineering_specification_hash,
)


RESULTS = Path("results/portfolio_engineering")


def test_portfolio_engineering_bundle_is_complete_and_protocol_locked():
    required = {
        "README.md",
        "protocol.json",
        "comparison_receipt.json",
        "portfolio_summary.csv",
        "factor_attribution.csv",
        "portfolio_era_stability.csv",
        "engineering_diagnostics.csv",
        "monthly_portfolio_returns.csv",
        "baseline_integrity.json",
        "quality_receipt.json",
        "data_manifest.json",
        "engineered_cumulative_wealth.png",
        "engineered_sharpe_costs.png",
    }
    assert required.issubset({path.name for path in RESULTS.iterdir()})
    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "FROZEN_BEFORE_RETURN_EVALUATION"
    assert protocol["portfolio_engineering_sha256"] == engineering_specification_hash(
        protocol["baseline_specification_sha256"], PortfolioEngineeringSpec()
    )


def test_engineering_bundle_preserves_baseline_and_claim_boundary():
    baseline = json.loads(
        (RESULTS / "baseline_integrity.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (RESULTS / "comparison_receipt.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (RESULTS / "quality_receipt.json").read_text(encoding="utf-8")
    )
    assert baseline["exact_replay"] is True
    assert comparison["alpha_classification"] == "NO_VALIDATED_ALPHA"
    assert comparison["all_evidence_retrospective"] is True
    assert quality["status"] == "PASS"
    assert all(value == 0 for value in quality["checks"].values())


def test_engineering_public_artifacts_contain_no_security_identifiers():
    forbidden = {"permno", "permco", "gvkey", "ticker", "cusip", "conm"}
    for path in RESULTS.glob("*.csv"):
        columns = {column.lower() for column in pd.read_csv(path, nrows=5).columns}
        assert forbidden.isdisjoint(columns), path
    manifest = json.loads((RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["licensed_rows_committed"] is False
    assert all(len(value) == 64 for value in manifest["private_input_hashes"].values())


def test_primary_result_is_not_silently_replaced_by_better_comparator():
    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS / "portfolio_summary.csv")
    primary = protocol["primary_candidate"]
    match = summary[
        (summary["strategy"] == primary["strategy"])
        & (summary["cost_bps"] == primary["cost_bps"])
    ]
    assert len(match) == 1
    comparison = json.loads(
        (RESULTS / "comparison_receipt.json").read_text(encoding="utf-8")
    )
    assert comparison["primary_engineered_10bps"]["strategy"] == primary["strategy"]

