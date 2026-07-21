"""Download public ETF history and build the retail proxy evidence bundle."""
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
import yfinance as yf

from factors.retail_etf_proxy import (
    RetailETFProxySpec,
    active_information_ratio,
    build_monthly_returns,
    fit_spy_beta,
    paired_sharpe_interval,
    summarize,
)


ROOT = Path(__file__).resolve().parent


def _hash(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_prices(cache: Path, refresh: bool = False) -> pd.DataFrame:
    spec = RetailETFProxySpec()
    if cache.exists() and not refresh:
        return pd.read_parquet(cache).set_index("date")
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        list(spec.tickers), start="2013-07-18", end="2026-07-21",
        auto_adjust=True, progress=False, group_by="column", threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no ETF data")
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.loc[:, list(spec.tickers)].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.rename_axis("date").reset_index().to_parquet(cache, index=False)
    return prices


def build(args: argparse.Namespace) -> None:
    spec = RetailETFProxySpec()
    protocol = ROOT / "retail_etf_proxy_protocol.json"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prices = download_prices(Path(args.cache).resolve(), args.refresh)
    monthly = build_monthly_returns(prices, spec)
    summary = summarize(monthly)
    beta = fit_spy_beta(monthly, spec)
    primary = monthly[
        np.isclose(monthly["cost_bps"], spec.primary_cost_bps)
        & (monthly["research_period"] == "primary_assessment")
    ].sort_values("date")
    observed, lower, upper = paired_sharpe_interval(
        primary["net_return"], primary["spy_return"], spec
    )
    ir_spy = active_information_ratio(primary["net_return"], primary["spy_return"])
    ir_equal = active_information_ratio(
        primary["net_return"], primary["equal_weight_return"]
    )

    monthly.to_csv(output / "monthly_returns.csv", index=False)
    summary.to_csv(output / "portfolio_summary.csv", index=False)
    pd.DataFrame({"ticker": spec.tickers, "weight": spec.weights}).to_csv(
        output / "target_weights.csv", index=False
    )
    _json(output / "protocol.json", json.loads(protocol.read_text(encoding="utf-8")))
    receipt = {
        "schema": "equity-factor-retail-etf-proxy-receipt.v1",
        "classification": "RETROSPECTIVE_EXECUTABLE_PROXY_TEST",
        "protocol_sha256": _hash(protocol),
        "fixed_development_spy_beta": beta,
        "assessment_months": int(len(primary)),
        "active_information_ratio_vs_spy": ir_spy,
        "active_information_ratio_vs_equal_weight_etfs": ir_equal,
        "sharpe_difference_vs_spy": observed,
        "sharpe_difference_ci_lower": lower,
        "sharpe_difference_ci_upper": upper,
        "paired_interval_excludes_zero": bool(lower > 0 or upper < 0),
        "raw_public_price_rows_committed": False,
        "gpu_used": False,
    }
    _json(output / "comparison_receipt.json", receipt)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, column in {
        "fixed factor-ETF proxy": "net_return",
        "SPY": "spy_return",
        "equal-weight factor ETFs": "equal_weight_return",
    }.items():
        ax.plot(primary["date"], (1 + primary[column]).cumprod(), label=label, linewidth=2)
    ax.set_title("Executable ETF proxy: retrospective 2021-2025 assessment")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "cumulative_wealth.png", dpi=160)
    plt.close(fig)

    cost_rows = summary[
        (summary["research_period"] == "primary_assessment")
        & (summary["strategy"] == "etf_proxy")
    ].sort_values("cost_bps")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(cost_rows["cost_bps"], cost_rows["sharpe"], marker="o", linewidth=2)
    ax.set_title("ETF proxy Sharpe under turnover-cost stress")
    ax.set_xlabel("One-way turnover cost (bps)")
    ax.set_ylabel("Annualized Sharpe")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "cost_stress.png", dpi=160)
    plt.close(fig)

    def metric(strategy: str) -> pd.Series:
        row = summary[
            (summary["research_period"] == "primary_assessment")
            & np.isclose(summary["cost_bps"], spec.primary_cost_bps)
            & (summary["strategy"] == strategy)
        ]
        return row.iloc[0]

    proxy, spy, equal = metric("etf_proxy"), metric("spy"), metric("equal_weight_etfs")
    readme = f"""# Retail factor-ETF proxy

This is the first executable translation of the academic profitable-factor
sleeve. The mapping and rules were frozen before ETF outcomes were downloaded.
It is a retrospective paper-trading candidate, not a live record or investment
recommendation.

## Primary 2021-2025 result

At the locked 10-bp one-way turnover cost, the fixed-weight ETF proxy recorded:

- CAGR: **{proxy['cagr']:.2%}**;
- annualized volatility: **{proxy['annualized_volatility']:.2%}**;
- Sharpe: **{proxy['sharpe']:.3f}**;
- maximum drawdown: **{proxy['maximum_drawdown']:.2%}**;
- active IR versus SPY: **{ir_spy:.3f}**;
- active IR versus equal-weight factor ETFs: **{ir_equal:.3f}**.

SPY recorded **{spy['sharpe']:.3f} Sharpe** and the equal-weight five-ETF
benchmark recorded **{equal['sharpe']:.3f}**. The paired 12-month moving-block
interval for the proxy-minus-SPY Sharpe difference is **[{lower:.3f}, {upper:.3f}]**.

![Cumulative wealth](cumulative_wealth.png)

![Cost stress](cost_stress.png)

## Fixed holdings

- SPY 21.98%
- IWM 12.01%
- VLUE 12.91%
- QUAL 35.91%
- MTUM 17.19%

QUAL combines the RMW and CMA academic weights and is therefore an imperfect
proxy for conservative investment. Returns use adjusted public price history,
monthly close-to-close rebalancing, fractional shares, no leverage, and no
shorting. Raw price rows remain in ignored local storage.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    artifacts = {
        path.name: _hash(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _json(output / "artifact_manifest.json", {
        "schema": "equity-factor-retail-etf-proxy-artifacts.v1", "artifacts": artifacts
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(ROOT / "data/private/retail_etf_adjusted_prices.parquet"))
    parser.add_argument("--output", default=str(ROOT / "results/retail_etf_proxy"))
    parser.add_argument("--refresh", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
