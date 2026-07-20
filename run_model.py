#!/usr/bin/env python3
"""
run_model.py
============
Cross-sectional equity factor model pipeline.

Usage
-----
    # Full pipeline (downloads data via yfinance)
    python run_model.py

    # Quick run with top-50 equal weight (no CVXPY needed)
    python run_model.py --top-n 50

    # Custom date range and universe
    python run_model.py --start 2018-01-01 --end 2024-01-01 --top-n 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("factor_model")
OUT_DIR = Path("outputs")
SECTORS = ["Apartment", "Industrial", "Retail", "Office"]


def _configure_console() -> None:
    """Keep Unicode status output from crashing on legacy Windows consoles."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(args: argparse.Namespace) -> None:
    _configure_console()
    sys.path.insert(0, "src")
    from factors.universe   import get_sp500_tickers, download_prices, download_market_cap, _FALLBACK_TICKERS
    from factors.signals    import FactorBuilder
    from factors.scoring    import FactorScorer
    from factors.ic_analysis import ICAnalyzer
    from factors.backtest   import Backtester
    from factors.charts     import save_all

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("  Cross-Sectional Equity Factor Model")
    logger.info("  Universe: S&P 500 | Factors: MOM, LV, SIZE, VAL, QUAL")
    logger.info("=" * 62)

    # ── 1. Universe + prices ─────────────────────────────────────────
    logger.info("Loading universe and price data...")
    if args.offline:
        tickers = _FALLBACK_TICKERS
        logger.warning("Offline mode: using %d fallback tickers (no download)", len(tickers))
        prices = _synthetic_prices(tickers, args.start, args.end)
        cap_rng = np.random.default_rng(43)
        market_cap = pd.Series(
            cap_rng.lognormal(10, 1, size=len(tickers)), index=tickers
        )
    else:
        tickers    = get_sp500_tickers()
        prices     = download_prices(tickers, start=args.start, end=args.end)
        market_cap = download_market_cap(tickers)

    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)

    # SPY benchmark
    if not args.offline:
        try:
            import yfinance as yf
            spy_raw = yf.download("SPY", start=args.start, end=args.end,
                                  auto_adjust=True, progress=False)
            spy     = spy_raw["Close"].pct_change().dropna()
            if hasattr(spy.index, "tz") and spy.index.tz:
                spy.index = spy.index.tz_localize(None)
        except Exception:
            spy = returns.mean(axis=1)
    else:
        spy = returns.mean(axis=1)

    logger.info("Universe: %d tickers | %d trading days (%s → %s)",
                len(prices.columns), len(prices),
                prices.index[0].date(), prices.index[-1].date())

    # ── 2. Monthly rebalancing dates ─────────────────────────────────
    rebal_dates = pd.date_range(
        start=prices.index[252],    # need 1 year of history first
        end=prices.index[-22],
        freq="MS",                  # month start
    )
    rebal_dates = pd.DatetimeIndex([
        prices.index[prices.index.get_indexer([d], method="bfill")[0]]
        for d in rebal_dates
        if prices.index.get_indexer([d], method="bfill")[0] < len(prices)
    ])
    logger.info("Rebalancing dates: %d", len(rebal_dates))

    # ── 3. Build factors ──────────────────────────────────────────────
    logger.info("Building factor exposures...")
    builder = FactorBuilder(prices, market_cap=market_cap, market_ret=spy)
    panel   = builder.build_panel(rebal_dates)

    # ── 4. Normalise and score ────────────────────────────────────────
    logger.info("Normalising and scoring...")
    scorer = FactorScorer(panel)
    scored = scorer.process()

    # ── 5. IC analysis ────────────────────────────────────────────────
    logger.info("Computing IC analysis...")
    analyzer   = ICAnalyzer(scored, returns)
    ic_summary = analyzer.ic_summary(horizons=[5, 21, 63, 126])
    corr_df    = analyzer.factor_correlation()

    # IC decay for all factors + composite
    all_factors = list(set(
        col for df in scored.values() for col in df.columns
    ))
    decay_dict = {f: analyzer.ic_decay(factor=f) for f in all_factors}
    ic_series  = analyzer.compute_ic(factor="COMPOSITE", horizon=21)

    print("\n┌─ IC Summary (top rows) ──────────────────────────────────")
    print(ic_summary.to_string())
    print("└─────────────────────────────────────────────────────────\n")

    # ── 6. Walk-forward backtest ──────────────────────────────────────
    logger.info("Running walk-forward backtest...")
    bt = Backtester(returns, scored, benchmark=spy)
    results = bt.run(top_n=args.top_n)

    metrics = Backtester.metrics(results)
    print("\n┌─ Backtest Metrics ──────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("└─────────────────────────────────────────────────────────\n")

    # ── 7. Save outputs ───────────────────────────────────────────────
    save_all(decay_dict, ic_series, corr_df, ic_summary, results, OUT_DIR)
    pd.Series(metrics).to_csv(OUT_DIR / "tables" / "backtest_metrics.csv")

    logger.info("=" * 62)
    logger.info("  Complete.  Outputs: %s/", OUT_DIR)
    logger.info("=" * 62)


def _synthetic_prices(tickers, start, end):
    """Generate synthetic price data for offline demo."""
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, end=end)
    ret   = rng.normal(0.0004, 0.015, (len(dates), len(tickers)))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + ret, axis=0),
        index=dates, columns=tickers,
    )
    return prices


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Cross-Sectional Equity Factor Model")
    p.add_argument("--start",   default="2015-01-01")
    p.add_argument("--end",     default="2024-01-01")
    p.add_argument("--top-n",   type=int, default=50,
                   help="Top-N equal weight (avoids CVXPY requirement)")
    p.add_argument("--offline", action="store_true",
                   help="Use synthetic data (no internet required)")
    main(p.parse_args())
