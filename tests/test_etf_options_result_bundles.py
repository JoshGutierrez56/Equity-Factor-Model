import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_etf_result_bundle_is_complete() -> None:
    result = ROOT / "results/retail_etf_proxy"
    required = {
        "README.md", "monthly_returns.csv", "portfolio_summary.csv",
        "comparison_receipt.json", "protocol.json", "artifact_manifest.json",
        "cumulative_wealth.png", "cost_stress.png", "target_weights.csv",
    }
    assert required <= {path.name for path in result.iterdir()}
    receipt = json.loads((result / "comparison_receipt.json").read_text())
    assert receipt["assessment_months"] == 60
    assert receipt["gpu_used"] is False


def test_options_result_bundle_has_full_coverage_and_no_raw_contracts() -> None:
    result = ROOT / "results/spy_options_overlay"
    coverage = json.loads((result / "coverage_receipt.json").read_text())
    assert coverage["planned_observations"] == 59
    assert coverage["executable_observations"] == 59
    assert coverage["coverage_rate"] == 1.0
    assert coverage["raw_orats_rows_committed"] is False
    assert coverage["exact_contract_identities_committed"] is False
    assert coverage["token_logged_or_committed"] is False
    monthly = pd.read_csv(result / "monthly_overlay_returns.csv")
    forbidden = {"strike", "expiration", "entry_bid", "entry_ask", "token"}
    assert not forbidden.intersection(column.lower() for column in monthly.columns)


def test_public_results_do_not_contain_orats_token() -> None:
    token_names = ("ORATS_API_TOKEN", "ORATS_TOKEN")
    for directory in (ROOT / "results/retail_etf_proxy", ROOT / "results/spy_options_overlay"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv", ".txt"}:
                text = path.read_text(encoding="utf-8")
                assert "token=" not in text.lower()
                assert all(name not in text or path.name == "protocol.json" for name in token_names)
