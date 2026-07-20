"""Bounded WRDS acquisition for point-in-time equity-factor research.

Licensed security-level rows are returned to the caller and may only be cached
under ``data/private/``.  Public artifacts must contain aggregate diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WRDSResearchRequest:
    """Frozen, bounded input contract for the real-data model."""

    start: date
    end: date
    holdout_start: date
    max_universe: int = 1000
    min_price: float = 5.0
    min_market_cap_millions: float = 100.0
    accounting_lag_months: int = 6

    @classmethod
    def create(
        cls,
        start: str | date,
        end: str | date,
        holdout_start: str | date,
        max_universe: int = 1000,
        min_price: float = 5.0,
        min_market_cap_millions: float = 100.0,
        accounting_lag_months: int = 6,
    ) -> "WRDSResearchRequest":
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        holdout_date = pd.Timestamp(holdout_start).date()
        if start_date >= end_date:
            raise ValueError("start must be before end")
        if not start_date < holdout_date <= end_date:
            raise ValueError("holdout_start must fall inside the sample")
        if (end_date - start_date).days > 365.25 * 40:
            raise ValueError("research window is capped at 40 years")
        if not 100 <= int(max_universe) <= 5000:
            raise ValueError("max_universe must be between 100 and 5000")
        if min_price <= 0 or min_market_cap_millions <= 0:
            raise ValueError("liquidity thresholds must be positive")
        if not 3 <= int(accounting_lag_months) <= 12:
            raise ValueError("accounting lag must be between 3 and 12 months")
        return cls(
            start=start_date,
            end=end_date,
            holdout_start=holdout_date,
            max_universe=int(max_universe),
            min_price=float(min_price),
            min_market_cap_millions=float(min_market_cap_millions),
            accounting_lag_months=int(accounting_lag_months),
        )

    def public_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key in ("start", "end", "holdout_start"):
            output[key] = output[key].isoformat()
        return output


def build_crsp_monthly_query(request: WRDSResearchRequest) -> str:
    """Return a bounded CRSP query with delisting-adjusted returns."""
    buffer_start = (pd.Timestamp(request.start) - pd.DateOffset(months=18)).date()
    return f"""
        WITH monthly AS (
            SELECT msf.permno,
                   msf.permco,
                   msf.date,
                   ABS(msf.prc) AS price,
                   msf.shrout,
                   msf.ret,
                   delist.dlret,
                   names.ticker,
                   names.shrcd,
                   names.exchcd,
                   names.siccd
            FROM crsp.msf AS msf
            JOIN LATERAL (
                SELECT msenames.ticker,
                       msenames.shrcd,
                       msenames.exchcd,
                       msenames.siccd
                FROM crsp.msenames AS msenames
                WHERE msenames.permno = msf.permno
                  AND msf.date BETWEEN msenames.namedt AND msenames.nameendt
                ORDER BY msenames.namedt DESC
                LIMIT 1
            ) AS names ON TRUE
            LEFT JOIN LATERAL (
                SELECT msedelist.dlret
                FROM crsp.msedelist AS msedelist
                WHERE msedelist.permno = msf.permno
                  AND DATE_TRUNC('month', msedelist.dlstdt)
                      = DATE_TRUNC('month', msf.date)
                ORDER BY msedelist.dlstdt DESC
                LIMIT 1
            ) AS delist ON TRUE
            WHERE msf.date BETWEEN '{buffer_start.isoformat()}'
                               AND '{request.end.isoformat()}'
              AND names.shrcd IN (10, 11)
              AND names.exchcd IN (1, 2, 3)
        )
        SELECT permno, permco, date, price, shrout, ret, dlret,
               ticker, shrcd, exchcd, siccd
        FROM monthly
        ORDER BY date, permno
    """.strip()


def build_compustat_query(request: WRDSResearchRequest) -> str:
    """Return annual fundamentals linked to CRSP without future observations."""
    buffer_start = (pd.Timestamp(request.start) - pd.DateOffset(years=3)).date()
    return f"""
        SELECT CAST(link.lpermno AS INTEGER) AS permno,
               funda.gvkey,
               funda.datadate,
               funda.pdate,
               funda.fdate,
               funda.fyear,
               funda.at,
               funda.lt,
               funda.seq,
               funda.ceq,
               funda.txditc,
               funda.pstkrv,
               funda.pstkl,
               funda.pstk,
               funda.revt,
               funda.sale,
               funda.cogs,
               funda.ib
        FROM comp.funda AS funda
        JOIN crsp.ccmxpf_linktable AS link
          ON link.gvkey = funda.gvkey
         AND funda.datadate BETWEEN link.linkdt
                                AND COALESCE(link.linkenddt, funda.datadate)
         AND link.linktype IN ('LC', 'LU')
         AND link.linkprim IN ('P', 'C')
        WHERE funda.indfmt = 'INDL'
          AND funda.datafmt = 'STD'
          AND funda.popsrc = 'D'
          AND funda.consol = 'C'
          AND funda.datadate BETWEEN '{buffer_start.isoformat()}'
                                  AND '{request.end.isoformat()}'
          AND link.lpermno IS NOT NULL
        ORDER BY permno, funda.datadate
    """.strip()


def build_fama_french_query(request: WRDSResearchRequest) -> str:
    """Return bounded monthly Fama-French five factors plus momentum."""
    return f"""
        SELECT dateff, mktrf, smb, hml, rmw, cma, umd, rf
        FROM ff.fivefactors_monthly
        WHERE dateff BETWEEN '{request.start.isoformat()}'
                           AND '{request.end.isoformat()}'
        ORDER BY dateff
    """.strip()


def connect_wrds():
    """Open a WRDS connection using the user's normal local credential flow."""
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - live dependency
        raise RuntimeError("Install requirements-live.txt before using WRDS") from exc
    username = os.environ.get("WRDS_USERNAME")
    if username:
        return wrds.Connection(wrds_username=username, verbose=False)
    return wrds.Connection(verbose=False)


def normalize_crsp_monthly(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "permno", "permco", "date", "price", "shrout", "ret", "dlret",
        "ticker", "shrcd", "exchcd", "siccd",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"CRSP rows missing columns: {sorted(missing)}")
    frame = rows.loc[:, sorted(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ("price", "shrout", "ret", "dlret", "siccd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["ret"].notna() | frame["dlret"].notna()].copy()
    frame["ret"] = frame["ret"].fillna(0.0)
    frame["dlret"] = frame["dlret"].fillna(0.0)
    frame["total_ret"] = (1.0 + frame["ret"]) * (1.0 + frame["dlret"]) - 1.0
    frame["market_equity_millions"] = frame["price"].abs() * frame["shrout"] / 1000.0
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["permno", "permco", "date", "total_ret"])
    frame["permno"] = frame["permno"].astype(int)
    frame["permco"] = frame["permco"].astype(int)
    if frame.duplicated(["permno", "date"]).any():
        raise ValueError("duplicate CRSP security/month rows")
    return frame.sort_values(["date", "permno"]).reset_index(drop=True)


def normalize_compustat(rows: pd.DataFrame, lag_months: int = 6) -> pd.DataFrame:
    required = {
        "permno", "gvkey", "datadate", "fyear", "at", "lt", "seq", "ceq",
        "txditc", "pstkrv", "pstkl", "pstk", "revt", "sale", "cogs", "ib",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Compustat rows missing columns: {sorted(missing)}")
    optional_dates = {column for column in ("pdate", "fdate") if column in rows.columns}
    selected = sorted(required | optional_dates)
    frame = rows.loc[:, selected].copy()
    frame["datadate"] = pd.to_datetime(frame["datadate"], errors="raise")
    conservative_fallback = frame["datadate"] + pd.DateOffset(months=lag_months)
    if optional_dates:
        preliminary = pd.to_datetime(
            frame.get("pdate", pd.Series(pd.NaT, index=frame.index)), errors="coerce"
        ).where(lambda values: values >= frame["datadate"])
        final = pd.to_datetime(
            frame.get("fdate", pd.Series(pd.NaT, index=frame.index)), errors="coerce"
        ).where(lambda values: values >= frame["datadate"])
        reported = preliminary.fillna(final)
        frame["availability_date"] = reported.fillna(conservative_fallback)
        frame["availability_source"] = np.select(
            [preliminary.notna(), final.notna()],
            ["compustat_pdate", "compustat_fdate"],
            default="six_month_fallback",
        )
    else:
        frame["availability_date"] = conservative_fallback
        frame["availability_source"] = "six_month_fallback"
    for column in required.difference({"permno", "gvkey", "datadate"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["permno", "gvkey", "datadate"])
    frame["permno"] = frame["permno"].astype(int)
    frame = frame.sort_values(["permno", "availability_date", "datadate"])
    return frame.drop_duplicates(["permno", "availability_date"], keep="last").reset_index(drop=True)


def normalize_fama_french(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"dateff", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Fama-French rows missing columns: {sorted(missing)}")
    frame = rows.loc[:, sorted(required)].copy()
    frame["dateff"] = pd.to_datetime(frame["dateff"], errors="raise")
    for column in required.difference({"dateff"}):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["dateff", "mktrf", "rf"])
    if frame.duplicated("dateff").any():
        raise ValueError("duplicate Fama-French month rows")
    return frame.sort_values("dateff").reset_index(drop=True)


def fetch_wrds_inputs(
    request: WRDSResearchRequest,
    connection=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and normalize bounded CRSP and Compustat inputs."""
    owned = connection is None
    db = connection or connect_wrds()
    try:
        crsp = db.raw_sql(build_crsp_monthly_query(request), date_cols=["date"])
        fundamentals = db.raw_sql(
            build_compustat_query(request), date_cols=["datadate"]
        )
        fama_french = db.raw_sql(
            build_fama_french_query(request), date_cols=["dateff"]
        )
        return (
            normalize_crsp_monthly(crsp),
            normalize_compustat(fundamentals, request.accounting_lag_months),
            normalize_fama_french(fama_french),
        )
    finally:
        if owned and hasattr(db, "close"):
            db.close()


def load_or_fetch_private_inputs(
    request: WRDSResearchRequest,
    cache_dir: str | Path = "data/private",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Use ignored Parquet caches; never write licensed rows under results/."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    window = f"{request.start.isoformat()}_{request.end.isoformat()}"
    crsp_path = cache / f"crsp_monthly_{window}.parquet"
    comp_path = cache / (
        f"compustat_annual_{window}_lag{request.accounting_lag_months}m.parquet"
    )
    ff_path = cache / f"fama_french_monthly_{window}.parquet"
    if not refresh and crsp_path.exists() and comp_path.exists() and ff_path.exists():
        crsp = normalize_crsp_monthly(pd.read_parquet(crsp_path))
        comp = normalize_compustat(
            pd.read_parquet(comp_path).drop(columns=["availability_date"], errors="ignore"),
            request.accounting_lag_months,
        )
        fama_french = normalize_fama_french(pd.read_parquet(ff_path))
        source = "ignored_private_cache"
    else:
        crsp, comp, fama_french = fetch_wrds_inputs(request)
        crsp.to_parquet(crsp_path, index=False)
        comp.to_parquet(comp_path, index=False)
        fama_french.to_parquet(ff_path, index=False)
        source = "live_wrds"
    return crsp, comp, fama_french, {
        "source": source,
        "crsp_cache": str(crsp_path),
        "compustat_cache": str(comp_path),
        "fama_french_cache": str(ff_path),
    }
