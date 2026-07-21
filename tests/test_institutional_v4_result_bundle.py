from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors.institutional_v4 import InstitutionalV4Spec, institutional_v4_hash


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "institutional_v4"


def _hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def test_protocol_hash_and_retrospective_classification() -> None:
    protocol = json.loads((ROOT / "institutional_v4_protocol.json").read_text())
    assert protocol["status"] == "FROZEN_BEFORE_V4_RETURN_EVALUATION"
    assert protocol["institutional_v4_sha256"] == institutional_v4_hash(
        protocol["institutional_v3_sha256"], InstitutionalV4Spec()
    )
    assert "RETROSPECTIVE" in protocol["experiment_classification"]
    assert all(not row["performance_informed"] for row in protocol["amendment_history"])


def test_v4_bundle_passes_and_preserves_no_alpha_claim() -> None:
    receipt = json.loads((RESULTS / "comparison_receipt.json").read_text())
    quality = json.loads((RESULTS / "quality_receipt.json").read_text())
    assert receipt["classification"] == "RETROSPECTIVE_MULTI_METRIC_IMPROVEMENT"
    assert receipt["alpha_classification"] == "NO_VALIDATED_ALPHA"
    assert receipt["all_evidence_retrospective"] is True
    assert quality["status"] == "PASS"
    assert quality["licensed_rows_committed"] is False
    assert quality["security_level_weights_committed"] is False


def test_v4_metrics_improve_without_hiding_distinctiveness_caveat() -> None:
    comparison = pd.read_csv(RESULTS / "implementation_comparison.csv").set_index(
        "implementation"
    )
    assert comparison.loc["v4", "sharpe"] > comparison.loc["baseline", "sharpe"]
    assert comparison.loc["v4", "sharpe"] > comparison.loc["v3", "sharpe"]
    assert comparison.loc["v4", "average_monthly_turnover"] < comparison.loc[
        "baseline", "average_monthly_turnover"
    ]
    assert np.isclose(
        comparison.loc["v4", "overlay_information_ratio"],
        comparison.loc["v4", "sharpe"],
    )
    assert comparison.loc["v4", "factor_residual_information_ratio"] < 0.0
    receipt = json.loads((RESULTS / "comparison_receipt.json").read_text())
    assert receipt["matched_sharpe_difference_ci_lower"] < 0.0


def test_v4_score_improves_matched_twelve_month_ic_and_icir() -> None:
    ic = pd.read_csv(RESULTS / "ic_comparison.csv")
    sample = ic[(ic["period"] == "temporal_assessment") & (ic["horizon_months"] == 12)]
    sample = sample.set_index("score")
    assert sample.loc["V4_SCORE", "mean_rank_ic"] > sample.loc["COMPOSITE", "mean_rank_ic"]
    assert sample.loc["V4_SCORE", "icir"] > sample.loc["COMPOSITE", "icir"]


def test_public_monthly_results_contain_no_security_identifiers() -> None:
    monthly = pd.read_csv(RESULTS / "monthly_portfolio_returns.csv")
    forbidden = {"permno", "permco", "ticker", "gvkey", "cusip"}
    assert forbidden.isdisjoint({column.lower() for column in monthly.columns})
    assert not monthly.empty


def test_artifact_manifest_matches_portable_hashes() -> None:
    manifest = json.loads((RESULTS / "artifact_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert _hash(RESULTS / name) == expected


def test_readmes_link_v4_and_disclose_limitations() -> None:
    root = (ROOT / "README.md").read_text(encoding="utf-8")
    evidence = (RESULTS / "README.md").read_text(encoding="utf-8")
    assert "results/institutional_v4/" in root
    assert "not statistically confirmed" in root.lower()
    assert "NO VALIDATED ALPHA" in evidence
    assert "same number as net Sharpe" in evidence
