"""
signals.py
==========
Five-factor signal construction from price and fundamental data.

Factors
-------
1. Momentum (MOM)
   12-1 month price return — excludes the most recent month to avoid
   short-term reversal. Standard Jegadeesh & Titman (1993) specification.

   MOM_t = (P_{t-21} / P_{t-252}) - 1

2. Low Volatility (LV)
   Negative of 252-day realized volatility. Stocks with lower vol
   earn higher risk-adjusted returns (Ang et al. 2006, Frazzini & Pedersen 2014).

   LV_t = -σ(r_{t-252:t})  [annualised]

3. Size (SMB proxy)
   Negative log market cap. Small caps earn a size premium (Fama & French 1993).
   Uses current market cap as approximation (historical not available from free APIs).

   SIZE_t = -log(MarketCap)

4. Value (VAL)
   Book-to-market proxy: 1/Price (earnings yield proxy using trailing-12-month
   return reversion). Note: a proper implementation requires historical P/B from
   a fundamental data provider (Simfin, Compustat). This uses price-only proxy.

   VAL_t = -P12M_return  (contrarian; reversal of 12-month price to mean-revert)

5. Quality (QMJ proxy)
   Gross profitability proxy: negative idiosyncratic volatility of returns
   vs SPY (lower residual vol → more stable, higher-quality business).

   QUAL_t = -σ(ε_{t-252:t})  where ε is from market model regression

All factors are computed cross-sectionally at each rebalancing date.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_NAMES = ["MOM", "LV", "SIZE", "VAL", "QUAL"]


class FactorBuilder:
    """
    Compute all five factor exposures from a price panel.

    Parameters
    ----------
    prices     : pd.DataFrame  — wide, rows=dates, cols=tickers, daily adj close
    market_cap : pd.Series, optional  — ticker → market cap (USD bn)
    market_ret : pd.Series, optional  — market (SPY) daily returns for QUAL factor
    """

    def __init__(
        self,
        prices:     pd.DataFrame,
        market_cap: Optional[pd.Series] = None,
        market_ret: Optional[pd.Series] = None,
    ) -> None:
        self.prices     = prices.copy()
        self.market_cap = market_cap
        self.market_ret = market_ret
        self._returns   = self.prices.pct_change().replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Individual factors
    # ------------------------------------------------------------------

    def momentum(self, date: pd.Timestamp) -> pd.Series:
        """12-1 month price momentum, annualised."""
        t_end   = self.prices.index.get_indexer([date], method="ffill")[0]
        t_skip  = max(t_end - 21,  0)   # t-1 month
        t_start = max(t_end - 252, 0)   # t-12 months

        if t_end <= t_start:
            return pd.Series(dtype=float)

        p_end   = self.prices.iloc[t_skip]
        p_start = self.prices.iloc[t_start]
        mom     = (p_end / p_start) - 1.0
        return mom.rename("MOM")

    def low_volatility(self, date: pd.Timestamp, window: int = 252) -> pd.Series:
        """Negative annualised 252-day realised volatility."""
        t_end   = self.prices.index.get_indexer([date], method="ffill")[0]
        t_start = max(t_end - window, 0)
        ret_win = self._returns.iloc[t_start:t_end]
        vol     = ret_win.std(ddof=1) * np.sqrt(252)
        return (-vol).rename("LV")

    def size(self) -> pd.Series:
        """Negative log market cap (small = positive exposure)."""
        if self.market_cap is None or self.market_cap.empty:
            logger.warning("No market cap data — SIZE factor set to zero.")
            return pd.Series(0.0, index=self.prices.columns, name="SIZE")
        ln_mc = np.log(self.market_cap.reindex(self.prices.columns).fillna(
            self.market_cap.median()
        ))
        return (-ln_mc).rename("SIZE")

    def value(self, date: pd.Timestamp) -> pd.Series:
        """
        Price-based value proxy: negative 12-month return (reversal / mean-reversion).

        Note: replace with P/B or E/P from Simfin / Compustat for a proper
        value factor. This proxy captures the long-horizon reversal documented
        by De Bondt & Thaler (1985).
        """
        t_end   = self.prices.index.get_indexer([date], method="ffill")[0]
        t_start = max(t_end - 252, 0)
        if t_end <= t_start:
            return pd.Series(dtype=float)
        r12     = (self.prices.iloc[t_end] / self.prices.iloc[t_start]) - 1
        return (-r12).rename("VAL")

    def quality(self, date: pd.Timestamp, window: int = 252) -> pd.Series:
        """
        Quality proxy: negative idiosyncratic volatility vs market.
        Lower residual vol → more stable business → higher quality score.
        """
        t_end   = self.prices.index.get_indexer([date], method="ffill")[0]
        t_start = max(t_end - window, 0)
        ret_win = self._returns.iloc[t_start:t_end].dropna(how="all")

        if self.market_ret is None:
            mkt = ret_win.mean(axis=1)  # equal-weight proxy if no SPY
        else:
            mkt = self.market_ret.reindex(ret_win.index).fillna(0)

        idio_vols = {}
        for ticker in ret_win.columns:
            stock = ret_win[ticker].dropna()
            mkt_a = mkt.reindex(stock.index)
            if len(stock) < 60:
                continue
            X = np.column_stack([np.ones(len(mkt_a)), mkt_a.values])
            try:
                coef, *_ = np.linalg.lstsq(X, stock.values, rcond=None)
                resid     = stock.values - X @ coef
                idio_vols[ticker] = float(resid.std(ddof=1) * np.sqrt(252))
            except np.linalg.LinAlgError:
                pass

        qual = pd.Series(idio_vols, name="QUAL")
        return (-qual.reindex(self.prices.columns)).rename("QUAL")

    # ------------------------------------------------------------------
    # Build full factor matrix at a single date
    # ------------------------------------------------------------------

    def build_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """
        Compute all five factors at a single cross-section date.

        Returns
        -------
        pd.DataFrame  — rows=tickers, cols=FACTOR_NAMES
        """
        factors = {
            "MOM":  self.momentum(date),
            "LV":   self.low_volatility(date),
            "SIZE": self.size(),
            "VAL":  self.value(date),
            "QUAL": self.quality(date),
        }
        return pd.DataFrame(factors)

    # ------------------------------------------------------------------
    # Walk-forward factor panel
    # ------------------------------------------------------------------

    def build_panel(
        self,
        rebal_dates: pd.DatetimeIndex,
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Compute factor exposures at each rebalancing date.

        Parameters
        ----------
        rebal_dates : DatetimeIndex of rebalancing dates

        Returns
        -------
        dict  — {date: pd.DataFrame (tickers × factors)}
        """
        panel = {}
        for i, dt in enumerate(rebal_dates):
            logger.debug("Building factors at %s (%d/%d)", dt.date(), i+1, len(rebal_dates))
            panel[dt] = self.build_date(dt)
        logger.info("Factor panel built: %d dates × %d factors",
                    len(panel), len(FACTOR_NAMES))
        return panel
