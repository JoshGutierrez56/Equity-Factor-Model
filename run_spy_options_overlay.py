"""Run the frozen ORATS SPY put-spread and collar overlay study."""
from __future__ import annotations

import argparse
import datetime as dt
from hashlib import sha256
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factors.spy_options_overlay import (
    OptionsOverlaySpec,
    OratsHistoricalClient,
    build_request_plan,
    exact_exit_quote,
    intrinsic_quote,
    leg_manifest,
    leg_pnl,
    overlay_information_ratio,
    select_overlay_legs,
    summarize_overlays,
)


ROOT = Path(__file__).resolve().parent


def _hash(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _price_on_or_before(daily: pd.DataFrame, date: str) -> float:
    cutoff = pd.Timestamp(date)
    match = daily.loc[daily.index <= cutoff, "SPY"].dropna()
    if match.empty:
        raise ValueError(f"No SPY price on or before {date}")
    return float(match.iloc[-1])


def prepare_plan(monthly_path: Path, daily_path: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    spec = OptionsOverlaySpec()
    monthly_all = pd.read_csv(monthly_path, parse_dates=["date"])
    primary = monthly_all[np.isclose(monthly_all["cost_bps"], 10.0)].sort_values("date").copy()
    daily = pd.read_parquet(daily_path).set_index("date")
    daily.index = pd.to_datetime(daily.index)
    month_prices = daily.groupby(daily.index.to_period("M"), sort=True).tail(1)
    spy_lookup = month_prices["SPY"].rename("spy_price")
    primary = primary.merge(spy_lookup, left_on="date", right_index=True, how="left", validate="one_to_one")
    development = primary[primary["research_period"] == "development"]
    beta = float(development[["net_return", "spy_return"]].cov().iloc[0, 1] / development["spy_return"].var(ddof=1))
    plan = build_request_plan(primary, spec)
    output.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output / "request_plan.csv", index=False)
    _json(output / "preflight_receipt.json", {
        "schema": "equity-factor-spy-options-preflight.v1",
        "planned_observations": int(len(plan)),
        "maximum_requests": spec.maximum_requests,
        "planned_maximum_requests": int(len(plan) * 2),
        "fixed_development_spy_beta": beta,
        "protocol_sha256": _hash(ROOT / "spy_options_overlay_protocol.json"),
        "status": "PASS",
    })
    return plan, daily, beta


def run_live(plan: pd.DataFrame, daily: pd.DataFrame, beta: float, cache: Path) -> tuple[pd.DataFrame, dict]:
    spec = OptionsOverlaySpec()
    client = OratsHistoricalClient(cache, spec)
    rows: list[dict] = []
    local_contracts: list[dict] = []
    cache_hits = 0
    for index, observation in plan.iterrows():
        entry_rows, entry_hash, entry_cached = client.strikes(
            observation.entry_date, spec.dte_min, spec.dte_max
        )
        cache_hits += int(entry_cached)
        legs = select_overlay_legs(entry_rows, spec)
        base = {
            "observation_id": observation.observation_id,
            "entry_date": observation.entry_date,
            "exit_date": observation.exit_date,
            "underlying_return": float(observation.underlying_return),
            "coverage_status": "entry_contracts_unavailable" if not legs else "pending_exit",
            "put_spread_overlay_return": np.nan,
            "collar_overlay_return": np.nan,
            "put_spread_total_return": np.nan,
            "collar_total_return": np.nan,
        }
        if not legs:
            rows.append(base)
            continue
        expiration = next(iter(legs.values())).expiration
        expiration_date = dt.date.fromisoformat(expiration)
        exit_date = dt.date.fromisoformat(observation.exit_date)
        if expiration_date <= exit_date:
            underlying_exit = _price_on_or_before(daily, expiration)
            exit_quotes = {name: intrinsic_quote(leg, underlying_exit) for name, leg in legs.items()}
            exit_hash = "intrinsic_settlement"
        else:
            remaining = (expiration_date - exit_date).days
            exit_rows, exit_hash, exit_cached = client.strikes(
                observation.exit_date,
                max(0, remaining - spec.exit_dte_tolerance),
                remaining + spec.exit_dte_tolerance,
            )
            cache_hits += int(exit_cached)
            exit_quotes = {name: exact_exit_quote(exit_rows, leg) for name, leg in legs.items()}
        if any(bid is None or ask is None for bid, ask in exit_quotes.values()):
            rows.append({**base, "coverage_status": "exact_exit_quote_unavailable"})
            continue
        pnl = {
            name: leg_pnl(leg, *exit_quotes[name], spec) for name, leg in legs.items()
        }
        scale = beta / float(observation.spy_entry_price)
        put_spread = scale * (pnl["long_put_25d"] + pnl["short_put_10d"])
        collar = scale * (pnl["long_put_25d"] + pnl["short_call_15d"])
        rows.append({
            **base,
            "coverage_status": "executable_exact_contracts",
            "put_spread_overlay_return": put_spread,
            "collar_overlay_return": collar,
            "put_spread_total_return": float(observation.underlying_return) + put_spread,
            "collar_total_return": float(observation.underlying_return) + collar,
        })
        local_contracts.append({
            "observation_id": observation.observation_id,
            "entry_response_sha256": entry_hash,
            "exit_response_sha256": exit_hash,
            "legs": leg_manifest(legs),
        })
        if (index + 1) % 5 == 0:
            print(f"ORATS overlay checkpoint: {index + 1}/{len(plan)} months")
    cache.mkdir(parents=True, exist_ok=True)
    _json(cache / "exact_contract_manifest.json", {"observations": local_contracts})
    monthly = pd.DataFrame(rows)
    executable = monthly[monthly["coverage_status"] == "executable_exact_contracts"].copy()
    run = {
        "network_requests_this_run": client.request_count,
        "cache_hits_this_run": cache_hits,
        "provider_response_sets_used": int(len(plan) * 2),
        "planned_observations": int(len(plan)),
        "executable_observations": int(len(executable)),
        "coverage_rate": float(len(executable) / len(plan)),
    }
    return monthly, run


def build(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    plan, daily, beta = prepare_plan(
        Path(args.monthly).resolve(), Path(args.daily_cache).resolve(), output
    )
    if args.plan_only:
        print(f"Preflight PASS: {len(plan)} observations, <= {len(plan) * 2} ORATS requests")
        return
    monthly, run = run_live(plan, daily, beta, Path(args.orats_cache).resolve())
    executable = monthly[monthly["coverage_status"] == "executable_exact_contracts"].copy()
    coverage_pass = run["coverage_rate"] >= 0.80
    monthly.to_csv(output / "monthly_overlay_returns.csv", index=False)
    _json(output / "protocol.json", json.loads((ROOT / "spy_options_overlay_protocol.json").read_text()))
    _json(output / "coverage_receipt.json", {
        "schema": "equity-factor-spy-options-coverage.v1",
        "planned_observations": run["planned_observations"],
        "executable_observations": run["executable_observations"],
        "coverage_rate": run["coverage_rate"],
        "provider_response_sets_used": run["provider_response_sets_used"],
        "coverage_threshold": 0.80,
        "coverage_status": "PASS" if coverage_pass else "INSUFFICIENT",
        "raw_orats_rows_committed": False,
        "exact_contract_identities_committed": False,
        "token_logged_or_committed": False,
        "gpu_used": False,
    })
    if executable.empty:
        raise RuntimeError("No executable exact-contract overlay observations")
    summary = summarize_overlays(executable)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    put_ir = overlay_information_ratio(executable, "put_spread_total_return")
    collar_ir = overlay_information_ratio(executable, "collar_total_return")
    _json(output / "comparison_receipt.json", {
        "schema": "equity-factor-spy-options-comparison.v1",
        "put_spread_information_ratio_vs_underlying": put_ir,
        "collar_information_ratio_vs_underlying": collar_ir,
        "primary_claim_allowed": bool(coverage_pass),
        "classification": "RETROSPECTIVE_DEFINED_RISK_OPTIONS_OVERLAY",
    })

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, column in {
        "ETF underlying": "underlying_return",
        "ETF + SPY put spread": "put_spread_total_return",
        "ETF + SPY collar": "collar_total_return",
    }.items():
        ax.plot(pd.to_datetime(executable["exit_date"]), (1 + executable[column]).cumprod(), label=label, linewidth=2)
    ax.set_title("Defined-risk SPY overlays on the factor-ETF proxy")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "overlay_cumulative_wealth.png", dpi=160)
    plt.close(fig)

    values = summary.set_index("strategy")
    underlying = values.loc["etf_underlying"]
    put = values.loc["etf_plus_put_spread"]
    collar = values.loc["etf_plus_collar"]
    readme = f"""# SPY options overlay on the retail factor-ETF proxy

This frozen retrospective study uses exact ORATS SPY contracts, executable
bid/ask prices, and defined-risk structures only. Raw licensed chains and exact
contract identities remain in ignored local storage.

## Coverage

- planned monthly observations: **{run['planned_observations']}**;
- executable exact-contract observations: **{run['executable_observations']}**;
- coverage: **{run['coverage_rate']:.1%}** ({'PASS' if coverage_pass else 'INSUFFICIENT'} versus the locked 80% threshold);
- ORATS entry/exit response sets used: **{run['provider_response_sets_used']}**.

## Results on covered months

- ETF underlying: **{underlying['sharpe']:.3f} Sharpe**, **{underlying['cagr']:.2%} CAGR**, **{underlying['maximum_drawdown']:.2%} max drawdown**;
- ETF + put spread: **{put['sharpe']:.3f} Sharpe**, **{put['cagr']:.2%} CAGR**, **{put['maximum_drawdown']:.2%} max drawdown**, **{put_ir:.3f} active IR**;
- ETF + collar: **{collar['sharpe']:.3f} Sharpe**, **{collar['cagr']:.2%} CAGR**, **{collar['maximum_drawdown']:.2%} max drawdown**, **{collar_ir:.3f} active IR**.

![Overlay cumulative wealth](overlay_cumulative_wealth.png)

Long options enter at the ask and exit at the bid; short options enter at the
bid and exit at the ask. Each trade includes a $0.65-per-contract fee. Contracts
use 35-49 DTE, target 42 DTE, fixed deltas, one same expiration, and exact-identity
exit matching. This is club research, not a recommendation to trade options.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    artifacts = {
        path.name: _hash(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _json(output / "artifact_manifest.json", {
        "schema": "equity-factor-spy-options-artifacts.v1", "artifacts": artifacts
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", default=str(ROOT / "results/retail_etf_proxy/monthly_returns.csv"))
    parser.add_argument("--daily-cache", default=str(ROOT / "data/private/retail_etf_adjusted_prices.parquet"))
    parser.add_argument("--orats-cache", default=str(ROOT / "data/private/orats_spy_overlay"))
    parser.add_argument("--output", default=str(ROOT / "results/spy_options_overlay"))
    parser.add_argument("--plan-only", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
