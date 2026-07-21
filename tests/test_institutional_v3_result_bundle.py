import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")

import pandas as pd

from factors.institutional_v3 import InstitutionalV3Spec, institutional_specification_hash


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "institutional_v3"


def _portable_hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_protocol_hash_and_prior_artifacts_remain_frozen():
    protocol = json.loads((ROOT / "institutional_v3_protocol.json").read_text())
    baseline = json.loads((ROOT / "results/wrds_real_data/specification.json").read_text())
    engineering = json.loads((ROOT / "portfolio_engineering_protocol.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_REAL_RETURN_EVALUATION"
    assert protocol["institutional_v3_sha256"] == institutional_specification_hash(
        baseline["specification_sha256"],
        engineering["portfolio_engineering_sha256"],
        InstitutionalV3Spec(),
    )
    paths = {
        "baseline_portfolio_summary_sha256": ROOT / "results/wrds_real_data/portfolio_summary.csv",
        "engineering_portfolio_summary_sha256": ROOT / "results/portfolio_engineering/portfolio_summary.csv",
        "engineering_comparison_receipt_sha256": ROOT / "results/portfolio_engineering/comparison_receipt.json",
    }
    for name, path in paths.items():
        assert _portable_hash(path) == protocol["frozen_artifact_hashes"][name]


def test_result_bundle_passes_constraints_and_preserves_negative_verdict():
    quality = json.loads((RESULTS / "quality_receipt.json").read_text())
    comparison = json.loads((RESULTS / "comparison_receipt.json").read_text())
    assert quality["status"] == "PASS"
    assert all(value == 0 for value in quality["checks"].values())
    assert quality["licensed_rows_committed"] is False
    assert comparison["selected_candidate"] == "assertive"
    assert comparison["selection_data_end"] == "2019-12-31"
    assert comparison["classification"] == "NO_ROBUST_RETROSPECTIVE_IMPROVEMENT"
    assert comparison["alpha_classification"] == "NO_VALIDATED_ALPHA"
    assert comparison["matched_comparison"]["sharpe_difference"] < 0


def test_public_artifacts_are_aggregate_only_and_complete():
    monthly = pd.read_csv(RESULTS / "monthly_portfolio_returns.csv")
    forbidden = {"permno", "permco", "ticker", "cusip", "gvkey", "price", "ret"}
    assert not forbidden.intersection(column.lower() for column in monthly.columns)
    assert set(monthly["period"]) == {"development", "temporal_assessment"}
    assert len(monthly) == 179 * 9
    manifest = json.loads((RESULTS / "artifact_manifest.json").read_text())
    for name, expected_hash in manifest["artifacts"].items():
        path = RESULTS / name
        assert path.exists()
        assert _portable_hash(path) == expected_hash


def test_readme_links_to_the_frozen_evidence():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Institutional version-3 implementation" in readme
    assert "no robust retrospective" in readme.lower()
    assert "results/institutional_v3/" in readme
