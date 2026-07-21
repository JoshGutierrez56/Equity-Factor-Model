import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

import pandas as pd

from factors.real_model import RealModelSpec, specification_hash
from factors.wrds_data import WRDSResearchRequest


RESULTS = Path("results/wrds_real_data")


def test_real_result_bundle_is_complete_and_self_consistent():
    required = {
        "README.md", "specification.json", "data_manifest.json", "coverage.json",
        "quality_receipt.json", "headline_summary.json", "ic_summary.csv", "portfolio_summary.csv",
        "factor_attribution.csv", "monthly_portfolio_returns.csv",
        "research_protocol.json", "hypothesis_tests.csv", "era_stability.csv",
        "quantile_diagnostics.csv", "primary_hypothesis_ic.png",
        "composite_cost_sensitivity.png", "era_stability_heatmap.png",
    }
    assert required.issubset({path.name for path in RESULTS.iterdir()})
    specification = json.loads((RESULTS / "specification.json").read_text(encoding="utf-8"))
    request = WRDSResearchRequest.create(**specification["request"])
    assert specification["specification_sha256"] == specification_hash(
        request, RealModelSpec()
    )
    assert specification["protocol_frozen_before_prospective_data"] is True
    assert specification["prospective_status"] == "NOT_STARTED"


def test_real_result_bundle_contains_no_security_level_identifiers():
    forbidden = {"permno", "permco", "gvkey", "ticker", "cusip", "conm"}
    for path in RESULTS.glob("*.csv"):
        columns = {column.lower() for column in pd.read_csv(path, nrows=5).columns}
        assert forbidden.isdisjoint(columns), path
    manifest = json.loads((RESULTS / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["licensed_rows_committed"] is False
    assert all(len(value) == 64 for value in manifest["private_input_hashes"].values())


def test_short_prospective_sample_is_never_labeled_formally_significant():
    ic = pd.read_csv(RESULTS / "ic_summary.csv")
    short = ic[(ic["period"] == "prospective") & (ic["n_months"] < 36)]
    assert not short["inference_eligible"].any()
    assert not short["significant_5pct"].any()
    attribution = pd.read_csv(RESULTS / "factor_attribution.csv")
    short_alpha = attribution[
        (attribution["period"] == "prospective") & (attribution["n_months"] < 36)
    ]
    assert not short_alpha["inference_eligible"].any()
    assert not short_alpha["alpha_significant_5pct"].any()


def test_quality_receipt_passes_all_hard_gates():
    receipt = json.loads((RESULTS / "quality_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert all(value == 0 for value in receipt["checks"].values())


def test_headline_does_not_claim_validated_alpha():
    headline = json.loads((RESULTS / "headline_summary.json").read_text(encoding="utf-8"))
    assert headline["classification"] == "NO_VALIDATED_ALPHA"
    assert headline["prospective_validation_status"] == "NOT_STARTED"
