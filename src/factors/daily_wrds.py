"""Private CRSP daily data acquisition for institutional portfolio research.

Licensed security-level rows are cached only below ``data/private``. Public
artifacts may contain aggregate coverage, risk, liquidity, and cost statistics,
but never PERMNOs, tickers, prices, or security-level returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from factors.wrds_data import connect_wrds


@dataclass(frozen=True)
class DailyCRSPRequest:
    """Bounded contract for the private daily risk/liquidity panel."""

    start: str = "2009-01-01"
    end: str = "2024-12-31"
    maximum_permnos: int = 6000

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def permno_universe_hash(permnos: Iterable[int]) -> str:
    canonical = ",".join(str(int(value)) for value in sorted(set(permnos)))
    return sha256(canonical.encode("utf-8")).hexdigest()


def build_daily_query(year: int, permnos: Iterable[int]) -> str:
    """Build one bounded calendar-year query for already-selected securities."""
    values = sorted(set(int(value) for value in permnos))
    if not values:
        raise ValueError("daily CRSP query requires at least one PERMNO")
    if len(values) > 6000:
        raise ValueError("daily CRSP query is capped at 6,000 PERMNOs")
    identifier_list = ",".join(str(value) for value in values)
    return f"""
        SELECT permno, date, ret, ABS(prc) AS price, vol,
               bid, ask, bidlo, askhi, shrout
        FROM crsp.dsf
        WHERE date BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND permno IN ({identifier_list})
        ORDER BY date, permno
    """.strip()


def normalize_crsp_daily(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily rows and derive conservative liquidity proxies."""
    required = {
        "permno", "date", "ret", "price", "vol", "bid", "ask",
        "bidlo", "askhi", "shrout",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"daily CRSP rows missing columns: {sorted(missing)}")
    frame = rows.loc[:, sorted(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in required.difference({"date", "permno"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["permno"] = pd.to_numeric(frame["permno"], errors="coerce")
    frame = frame.dropna(subset=["permno", "date"]).copy()
    frame["permno"] = frame["permno"].astype(int)
    frame["price"] = frame["price"].abs()
    frame["dollar_volume"] = frame["price"] * frame["vol"].clip(lower=0.0)

    mid = (frame["ask"] + frame["bid"]) / 2.0
    quoted = ((frame["ask"] - frame["bid"]) / mid).where(
        (frame["ask"] > 0) & (frame["bid"] > 0) & (frame["ask"] >= frame["bid"])
    )
    range_mid = (frame["askhi"] + frame["bidlo"]) / 2.0
    range_proxy = ((frame["askhi"] - frame["bidlo"]) / range_mid).where(
        (frame["askhi"] > 0)
        & (frame["bidlo"] > 0)
        & (frame["askhi"] >= frame["bidlo"])
    )
    # The high-low fallback is deliberately downweighted: it is a daily range,
    # not a quoted spread. A two-basis-point floor prevents zero-cost claims.
    frame["spread_fraction"] = quoted.fillna(0.10 * range_proxy).clip(0.0002, 0.02)
    frame["ret"] = frame["ret"].replace([np.inf, -np.inf], np.nan)
    frame = frame.drop_duplicates(["permno", "date"], keep="last")
    return frame.sort_values(["date", "permno"]).reset_index(drop=True)


def load_or_fetch_daily_crsp(
    permnos: Iterable[int],
    request: DailyCRSPRequest | None = None,
    cache_dir: str | Path = "data/private/crsp_daily",
    refresh: bool = False,
    connection=None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load/fetch partitioned daily rows without committing licensed data."""
    request = request or DailyCRSPRequest()
    values = sorted(set(int(value) for value in permnos))
    if not values:
        raise ValueError("no PERMNOs supplied")
    if len(values) > request.maximum_permnos:
        raise ValueError("PERMNO universe exceeds frozen request cap")
    start = pd.Timestamp(request.start)
    end = pd.Timestamp(request.end)
    if start >= end:
        raise ValueError("daily request start must precede end")
    universe_hash = permno_universe_hash(values)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    years = range(start.year, end.year + 1)
    paths = {
        year: root / f"crsp_daily_{year}_{universe_hash[:12]}.parquet"
        for year in years
    }
    missing_years = [year for year, path in paths.items() if refresh or not path.exists()]
    owned = bool(missing_years) and connection is None
    db = connection or (connect_wrds() if missing_years else None)
    try:
        for year in missing_years:
            print(f"[daily-crsp] fetching {year} ({len(values):,} selected PERMNOs)", flush=True)
            rows = db.raw_sql(build_daily_query(year, values), date_cols=["date"])
            normalized = normalize_crsp_daily(rows)
            normalized = normalized[
                (normalized["date"] >= start) & (normalized["date"] <= end)
            ]
            normalized.to_parquet(paths[year], index=False)
            print(f"[daily-crsp] cached {year}: {len(normalized):,} rows", flush=True)
    finally:
        if owned and db is not None and hasattr(db, "close"):
            db.close()
    parts = [normalize_crsp_daily(pd.read_parquet(paths[year])) for year in years]
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)].copy()
    manifest = {
        "source": "live_wrds" if missing_years else "ignored_private_cache",
        "request": request.public_dict(),
        "universe_sha256": universe_hash,
        "number_of_permnos": len(values),
        "number_of_rows": int(len(daily)),
        "partition_count": len(paths),
        "licensed_rows_committed": False,
    }
    return daily, manifest
