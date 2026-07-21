import numpy as np
import pandas as pd

from factors.spy_options_overlay import (
    OptionsOverlaySpec,
    build_request_plan,
    exact_exit_quote,
    leg_pnl,
    select_overlay_legs,
)


def _row(strike, delta, *, expiration="2021-03-19", dte=42):
    return {
        "ticker": "SPY", "tradeDate": "2021-02-05",
        "expirDate": expiration, "dte": dte, "strike": strike, "delta": delta,
        "putBidPrice": 2.0, "putAskPrice": 2.1,
        "callBidPrice": 2.2, "callAskPrice": 2.3,
        "putOpenInterest": 100, "callOpenInterest": 100,
        "putVolume": 10, "callVolume": 10,
    }


def test_selection_uses_same_expiry_and_fixed_deltas() -> None:
    rows = [_row(350, 0.75), _row(330, 0.90), _row(390, 0.15)]
    legs = select_overlay_legs(rows)
    assert set(legs) == {"long_put_25d", "short_put_10d", "short_call_15d"}
    assert len({leg.expiration for leg in legs.values()}) == 1
    assert np.isclose(legs["long_put_25d"].provider_call_delta, 0.75)


def test_exact_exit_requires_same_strike_and_expiration() -> None:
    rows = [_row(350, 0.75)]
    leg = select_overlay_legs(rows + [_row(330, 0.90), _row(390, 0.15)])["long_put_25d"]
    assert exact_exit_quote(rows, leg) == (2.0, 2.1)
    assert exact_exit_quote([_row(351, 0.75)], leg) == (None, None)


def test_executable_long_and_short_pnl_use_correct_quote_sides() -> None:
    rows = [_row(350, 0.75), _row(330, 0.90), _row(390, 0.15)]
    legs = select_overlay_legs(rows)
    fee = 2 * OptionsOverlaySpec().fee_per_share_per_trade
    assert np.isclose(leg_pnl(legs["long_put_25d"], 2.5, 2.6), 2.5 - 2.1 - fee)
    assert np.isclose(leg_pnl(legs["short_call_15d"], 1.5, 1.6), 2.2 - 1.6 - fee)


def test_request_plan_is_bounded_to_59_months() -> None:
    dates = pd.date_range("2020-12-31", "2025-12-31", freq="ME")
    monthly = pd.DataFrame({
        "date": dates,
        "net_return": 0.01,
        "spy_price": np.linspace(300, 500, len(dates)),
    })
    plan = build_request_plan(monthly)
    assert len(plan) == 59
    assert len(plan) * 2 <= OptionsOverlaySpec().maximum_requests
