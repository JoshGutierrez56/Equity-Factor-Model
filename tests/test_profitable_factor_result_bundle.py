from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factors.profitable_factor_portfolio import (
    ProfitableFactorPortfolioSpec,
    profitable_factor_portfolio_hash,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "profitable_factor_portfolio"


def _hash(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def test_protocol_hash_and_claim_boundary() -> None:
    protocol = json.loads((ROOT / "profitable_factor_portfolio_protocol.json").read_text())
    assert protocol["status"] == "DOCUMENTED_AFTER_EXPLORATORY_ITERATION"
    assert "RETROSPECTIVE" in protocol["experiment_classification"]
    assert protocol["profitable_factor_portfolio_sha256"] == profitable_factor_portfolio_hash(
        protocol["institutional_v4_sha256"], ProfitableFactorPortfolioSpec()
    )


def test_primary_combination_improves_sharpe_but_not_claimed_as_alpha() -> None:
    summary = pd.read_csv(RESULTS / "portfolio_summary.csv")
    primary = summary[
        (summary["research_period"] == "temporal_assessment")
        & np.isclose(summary["factor_cost_bps"], 100.0)
    ].set_index("strategy")
    assert primary.loc["combined_unlevered", "sharpe"] > primary.loc["v4_net", "sharpe"]
    receipt = json.loads((RESULTS / "comparison_receipt.json").read_text())
    assert receipt["alpha_classification"] == (
        "INTENTIONAL_ESTABLISHED_FACTOR_EXPOSURE_NOT_NEW_ALPHA"
    )
    assert receipt["paired_interval_excludes_zero"] is False


def test_cost_stress_and_negative_active_ir_are_not_hidden() -> None:
    summary = pd.read_csv(RESULTS / "portfolio_summary.csv")
    assessment = summary[
        (summary["research_period"] == "temporal_assessment")
        & (summary["strategy"] == "combined_unlevered")
    ].set_index("factor_cost_bps")
    assert list(assessment.index.astype(float)) == [0.0, 50.0, 100.0, 150.0]
    assert assessment.loc[0.0, "sharpe"] > assessment.loc[150.0, "sharpe"]
    receipt = json.loads((RESULTS / "comparison_receipt.json").read_text())
    assert receipt["factor_tilt_information_ratio_vs_equal_factor"] > 0.0
    assert receipt["active_information_ratio_vs_v4"] < 0.0


def test_public_evidence_contains_no_security_identifiers() -> None:
    monthly = pd.read_csv(RESULTS / "monthly_returns.csv")
    forbidden = {"permno", "permco", "ticker", "gvkey", "cusip"}
    assert forbidden.isdisjoint({column.lower() for column in monthly.columns})
    quality = json.loads((RESULTS / "quality_receipt.json").read_text())
    assert quality["status"] == "PASS"
    assert quality["licensed_rows_committed"] is False
    assert quality["gpu_used"] is False


def test_artifact_manifest_matches_portable_hashes() -> None:
    manifest = json.loads((RESULTS / "artifact_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert _hash(RESULTS / name) == expected


def test_two_run_replay_passes() -> None:
    replay = json.loads((RESULTS / "replay_receipt.json").read_text())
    assert replay["status"] == "PASS"
    assert replay["independent_runs"] == 2
    assert replay["portable_hashes_match"] is True
    assert replay["first_artifact_manifest_sha256"] == replay[
        "second_artifact_manifest_sha256"
    ]


def test_readmes_link_bundle_and_disclose_academic_factor_limitation() -> None:
    root = (ROOT / "README.md").read_text(encoding="utf-8")
    evidence = (RESULTS / "README.md").read_text(encoding="utf-8")
    assert "results/profitable_factor_portfolio/" in root
    assert "academic test portfolios" in evidence.lower()
    assert "not statistically confirmed" in evidence.lower()
