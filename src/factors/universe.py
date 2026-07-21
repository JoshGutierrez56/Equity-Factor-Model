"""
universe.py
===========
Build and cache the equity research universe.

We use the S&P 500 as a liquid, well-known large-cap proxy for the
Russell 3000. All constituents are downloaded from Wikipedia (free,
no API key) and prices are pulled via yfinance.

For a production system this would be replaced with a proper index
membership file from MSCI/FTSE Russell with point-in-time constituent
data to avoid survivorship bias. That limitation is clearly documented
in the backtest output.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE = Path("data/raw/sp500_tickers.csv")
_WIKI  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers(use_cache: bool = True) -> List[str]:
    """
    Return the current S&P 500 ticker list.

    Parameters
    ----------
    use_cache : bool
        If True and cache exists, skip the Wikipedia request.

    Returns
    -------
    list of ticker strings (e.g. ["AAPL", "MSFT", ...])
    """
    if use_cache and _CACHE.exists():
        tickers = pd.read_csv(_CACHE)["ticker"].tolist()
        logger.info("Loaded %d tickers from cache", len(tickers))
        return tickers

    try:
        tables = pd.read_html(_WIKI)
        df     = tables[0]
        # Column is "Symbol" in current Wikipedia table
        col    = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = (df[col]
                   .str.replace(".", "-", regex=False)   # BRK.B → BRK-B
                   .dropna()
                   .tolist())
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(_CACHE, index=False)
        logger.info("Downloaded %d tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.error("Could not fetch S&P 500 list: %s — using fallback", exc)
        return _FALLBACK_TICKERS


def download_prices(
    tickers: List[str],
    start: str = "2015-01-01",
    end:   Optional[str] = None,
    cache_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Download adjusted close prices for the universe via yfinance.

    Returns
    -------
    pd.DataFrame  — rows = dates, columns = tickers (wide format)
    """
    import yfinance as yf

    cache_path = Path(cache_dir) / "prices.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info("Loaded prices from cache: %s", cache_path)
        return df

    logger.info("Downloading prices for %d tickers (%s → %s)...",
                len(tickers), start, end or "today")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.dropna(how="all", axis=1)
    prices.index = pd.DatetimeIndex(prices.index)
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    logger.info("Saved %d tickers × %d days to %s",
                prices.shape[1], prices.shape[0], cache_path)
    return prices


def download_market_cap(
    tickers: List[str],
    cache_dir: str = "data/raw",
) -> pd.Series:
    """
    Fetch current market cap for each ticker (used for Size factor).
    Returns a Series indexed by ticker, values in USD billions.
    """
    import yfinance as yf

    cache_path = Path(cache_dir) / "market_cap.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0).squeeze()

    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            mc   = getattr(info, "market_cap", None)
            if mc:
                caps[t] = mc / 1e9
        except Exception:
            pass

    s = pd.Series(caps, name="market_cap_bn")
    s.to_csv(cache_path)
    return s


# ── Fallback universe for offline use ────────────────────────────────
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM",  "XOM",  "V",    "UNH",   "TSLA", "MA",    "PG",
    "JNJ",  "COST", "HD",   "MRK",  "ABBV",  "CVX",  "CRM",   "BAC",
    "NFLX", "KO",   "PEP",  "TMO",  "ACN",   "MCD",  "WMT",   "ABT",
    "CSCO", "NKE",  "TXN",  "DHR",  "PM",    "NEE",  "CAT",   "ORCL",
    "WFC",  "COP",  "UPS",  "RTX",  "HON",   "LOW",  "AMGN",  "IBM",
    "SPGI", "GE",   "DE",   "INTU", "QCOM",  "PLD",  "AXP",   "AMAT",
    "NOW",  "ISRG", "LMT",  "SYK",  "GS",    "BLK",  "ELV",   "MDT",
    "GILD", "CI",   "ADI",  "BKNG", "MMC",   "CB",   "SO",    "DUK",
    "ZTS",  "C",    "REGN", "TGT",  "EOG",   "SLB",  "MO",    "F",
]
