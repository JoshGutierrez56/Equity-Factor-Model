"""Defined-risk SPY options overlays for the executable factor-ETF proxy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from factors.profitable_factor_portfolio import performance_metrics
from factors.retail_etf_proxy import active_information_ratio


@dataclass(frozen=True)
class OptionsOverlaySpec:
    dte_min: int = 35
    dte_max: int = 49
    target_dte: int = 42
    minimum_open_interest: int = 1
    fee_per_contract_per_trade: float = 0.65
    contract_multiplier: int = 100
    maximum_requests: int = 120
    request_interval_seconds: float = 0.27
    exit_dte_tolerance: int = 2
    entry_start: str = "2021-01-01"
    final_exit: str = "2025-12-31"

    @property
    def fee_per_share_per_trade(self) -> float:
        return self.fee_per_contract_per_trade / self.contract_multiplier


@dataclass(frozen=True)
class SelectedLeg:
    name: str
    option_type: str
    side: str
    target_delta: float
    strike: float
    expiration: str
    dte: int
    provider_call_delta: float
    entry_bid: float
    entry_ask: float


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _side_values(row: dict[str, Any], option_type: str) -> tuple[float | None, float | None, float | None, float | None]:
    prefix = "put" if option_type == "put" else "call"
    return (
        _number(row.get(f"{prefix}BidPrice")),
        _number(row.get(f"{prefix}AskPrice")),
        _number(row.get(f"{prefix}OpenInterest")),
        _number(row.get(f"{prefix}Volume")),
    )


def _spread(bid: float | None, ask: float | None) -> float:
    if bid is None or ask is None or ask < bid or (bid + ask) <= 0:
        return float("inf")
    return float((ask - bid) / ((ask + bid) / 2.0))


def _select_leg(
    rows: list[dict[str, Any]],
    *,
    name: str,
    option_type: str,
    side: str,
    target_delta: float,
    minimum_open_interest: int,
) -> SelectedLeg | None:
    call_target = target_delta + 1.0 if option_type == "put" else target_delta
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for row in rows:
        delta = _number(row.get("delta"))
        strike = _number(row.get("strike"))
        dte = _number(row.get("dte"))
        expiration = str(row.get("expirDate") or "")[:10]
        bid, ask, oi, volume = _side_values(row, option_type)
        if None in (delta, strike, dte, bid, ask) or not expiration:
            continue
        if ask < bid or ask <= 0 or (side == "short" and bid <= 0):
            continue
        if oi is None or oi < minimum_open_interest:
            continue
        key = (
            abs(delta - call_target),
            _spread(bid, ask),
            -oi,
            -(volume or 0.0),
            strike,
        )
        candidates.append((key, row))
    if not candidates:
        return None
    row = min(candidates, key=lambda item: item[0])[1]
    bid, ask, _oi, _volume = _side_values(row, option_type)
    return SelectedLeg(
        name=name,
        option_type=option_type,
        side=side,
        target_delta=target_delta,
        strike=float(row["strike"]),
        expiration=str(row["expirDate"])[:10],
        dte=int(row["dte"]),
        provider_call_delta=float(row["delta"]),
        entry_bid=float(bid),
        entry_ask=float(ask),
    )


def select_overlay_legs(rows: list[dict[str, Any]], spec: OptionsOverlaySpec | None = None) -> dict[str, SelectedLeg]:
    """Select one same-expiry long put, short put, and short call."""
    spec = spec or OptionsOverlaySpec()
    by_expiration: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dte = _number(row.get("dte"))
        expiration = str(row.get("expirDate") or "")[:10]
        if dte is None or not (spec.dte_min <= dte <= spec.dte_max) or not expiration:
            continue
        by_expiration.setdefault(expiration, []).append(row)
    ranked_expirations = sorted(
        by_expiration,
        key=lambda expiration: (
            abs(float(by_expiration[expiration][0]["dte"]) - spec.target_dte),
            expiration,
        ),
    )
    leg_specs = (
        ("long_put_25d", "put", "long", -0.25),
        ("short_put_10d", "put", "short", -0.10),
        ("short_call_15d", "call", "short", 0.15),
    )
    for expiration in ranked_expirations:
        selected: dict[str, SelectedLeg] = {}
        for name, option_type, side, target in leg_specs:
            leg = _select_leg(
                by_expiration[expiration], name=name, option_type=option_type,
                side=side, target_delta=target,
                minimum_open_interest=spec.minimum_open_interest,
            )
            if leg is None:
                break
            selected[name] = leg
        if len(selected) == len(leg_specs):
            return selected
    return {}


def exact_exit_quote(rows: list[dict[str, Any]], leg: SelectedLeg) -> tuple[float | None, float | None]:
    for row in rows:
        expiration = str(row.get("expirDate") or "")[:10]
        strike = _number(row.get("strike"))
        if expiration == leg.expiration and strike is not None and abs(strike - leg.strike) <= 1e-9:
            bid, ask, _oi, _volume = _side_values(row, leg.option_type)
            if bid is None or ask is None or ask < bid:
                return None, None
            return bid, ask
    return None, None


def leg_pnl(leg: SelectedLeg, exit_bid: float, exit_ask: float, spec: OptionsOverlaySpec | None = None) -> float:
    spec = spec or OptionsOverlaySpec()
    fees = 2.0 * spec.fee_per_share_per_trade
    if leg.side == "long":
        return float(exit_bid - leg.entry_ask - fees)
    return float(leg.entry_bid - exit_ask - fees)


def build_request_plan(monthly: pd.DataFrame, spec: OptionsOverlaySpec | None = None) -> pd.DataFrame:
    spec = spec or OptionsOverlaySpec()
    frame = monthly.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for previous, current in zip(frame.iloc[:-1].to_dict("records"), frame.iloc[1:].to_dict("records")):
        entry = pd.Timestamp(previous["date"])
        exit_date = pd.Timestamp(current["date"])
        if entry < pd.Timestamp(spec.entry_start) or exit_date > pd.Timestamp(spec.final_exit):
            continue
        rows.append({
            "observation_id": f"SPY_{entry:%Y%m%d}_{exit_date:%Y%m%d}",
            "entry_date": entry.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "underlying_return": float(current["net_return"]),
            "spy_entry_price": float(previous["spy_price"]),
        })
    plan = pd.DataFrame(rows)
    if len(plan) != 59:
        raise ValueError(f"Expected 59 monthly overlay observations, got {len(plan)}")
    if len(plan) * 2 > spec.maximum_requests:
        raise ValueError("ORATS request cap exceeded by plan")
    return plan


class OratsHistoricalClient:
    """Minimal credential-safe ORATS historical strikes client with local cache."""

    def __init__(self, cache_dir: Path, spec: OptionsOverlaySpec | None = None) -> None:
        self.spec = spec or OptionsOverlaySpec()
        self.token = os.environ.get("ORATS_API_TOKEN") or os.environ.get("ORATS_TOKEN")
        if not self.token:
            raise ValueError("Set ORATS_API_TOKEN before a live options pull")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.request_count = 0
        self._last_request = 0.0

    def strikes(self, trade_date: str, dte_min: int, dte_max: int) -> tuple[list[dict[str, Any]], str, bool]:
        path = self.cache_dir / f"SPY_{trade_date}_{dte_min}_{dte_max}.json"
        if path.exists():
            body = json.loads(path.read_text(encoding="utf-8"))
            return body.get("data", []), sha256(path.read_bytes()).hexdigest(), True
        if self.request_count >= self.spec.maximum_requests:
            raise RuntimeError("ORATS request cap reached")
        elapsed = time.time() - self._last_request
        if elapsed < self.spec.request_interval_seconds:
            time.sleep(self.spec.request_interval_seconds - elapsed)
        response = self.session.get(
            "https://api.orats.io/datav2/hist/strikes",
            params={
                "ticker": "SPY", "tradeDate": trade_date,
                "dte": f"{dte_min},{dte_max}", "token": self.token,
            },
            timeout=30,
        )
        self._last_request = time.time()
        self.request_count += 1
        if response.status_code in (401, 403, 429):
            raise RuntimeError(f"Fatal ORATS HTTP status {response.status_code}")
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise RuntimeError("ORATS strikes response failed schema validation")
        text = json.dumps(body, separators=(",", ":"))
        if self.token in text:
            raise RuntimeError("Credential appeared in ORATS response")
        path.write_text(json.dumps(body), encoding="utf-8")
        return body["data"], sha256(path.read_bytes()).hexdigest(), False


def intrinsic_quote(leg: SelectedLeg, underlying_price: float) -> tuple[float, float]:
    if leg.option_type == "put":
        value = max(leg.strike - underlying_price, 0.0)
    else:
        value = max(underlying_price - leg.strike, 0.0)
    return float(value), float(value)


def summarize_overlays(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy, column in {
        "etf_underlying": "underlying_return",
        "etf_plus_put_spread": "put_spread_total_return",
        "etf_plus_collar": "collar_total_return",
    }.items():
        values = monthly[column].dropna()
        metrics = performance_metrics(values)
        downside = float(np.sqrt(12.0) * np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
        mean = float(values.mean() * 12.0)
        cutoff = float(values.quantile(0.05))
        expected_shortfall = float(values[values <= cutoff].mean())
        rows.append({
            "strategy": strategy,
            **metrics,
            "downside_deviation": downside,
            "sortino": mean / downside if downside > 1e-12 else np.nan,
            "monthly_var_5pct": cutoff,
            "monthly_expected_shortfall_5pct": expected_shortfall,
        })
    return pd.DataFrame(rows)


def overlay_information_ratio(monthly: pd.DataFrame, column: str) -> float:
    return active_information_ratio(monthly[column], monthly["underlying_return"])


def leg_manifest(legs: dict[str, SelectedLeg]) -> dict[str, dict[str, Any]]:
    return {name: asdict(leg) for name, leg in legs.items()}
