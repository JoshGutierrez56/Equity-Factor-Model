import numpy as np
import pandas as pd

from factors.retail_etf_proxy import (
    RetailETFProxySpec,
    build_monthly_returns,
    fit_spy_beta,
    month_end_prices,
)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2013-07-18", "2025-12-31")
    rng = np.random.default_rng(7)
    market = rng.normal(0.0003, 0.008, len(dates))
    data = {}
    for index, ticker in enumerate(RetailETFProxySpec().tickers):
        returns = market + rng.normal(0.00002 * index, 0.002, len(dates))
        data[ticker] = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(data, index=dates)


def test_month_ends_preserve_real_trading_dates() -> None:
    result = month_end_prices(_prices())
    assert all(date.day <= 31 for date in result.index)
    assert result.index[-1] == pd.Timestamp("2025-12-31")
    assert result.index.to_period("M").is_unique


def test_locked_weights_and_costs() -> None:
    spec = RetailETFProxySpec()
    assert np.isclose(spec.weight_series().sum(), 1.0)
    monthly = build_monthly_returns(_prices(), spec)
    gross = monthly[np.isclose(monthly.cost_bps, 0.0)].net_return.reset_index(drop=True)
    costly = monthly[np.isclose(monthly.cost_bps, 25.0)].net_return.reset_index(drop=True)
    assert (gross >= costly - 1e-15).all()
    assert fit_spy_beta(monthly, spec) > 0


def test_future_returns_do_not_change_development_beta() -> None:
    spec = RetailETFProxySpec()
    monthly = build_monthly_returns(_prices(), spec)
    original = fit_spy_beta(monthly, spec)
    changed = monthly.copy()
    mask = changed.date > pd.Timestamp(spec.beta_development_end)
    changed.loc[mask, "net_return"] *= -10
    assert np.isclose(original, fit_spy_beta(changed, spec))
