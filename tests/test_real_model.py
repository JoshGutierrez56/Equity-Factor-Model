import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest

from factors.real_model import (
    RealModelSpec,
    build_point_in_time_panel,
    compute_factor_attribution,
    compute_ic_summary,
    compute_monthly_portfolios,
    compute_portfolio_summary,
    public_monthly_results,
    quality_receipt,
    specification_hash,
)
from factors.wrds_data import (
    WRDSResearchRequest,
    build_compustat_query,
    build_crsp_monthly_query,
    build_fama_french_query,
    normalize_crsp_monthly,
    normalize_fama_french,
)


def _request():
    return WRDSResearchRequest.create(
        "2020-01-01", "2021-12-31", "2021-01-01", max_universe=100
    )


def _mock_inputs(n_securities=120):
    dates = pd.date_range("2018-01-31", "2021-12-31", freq="ME")
    crsp_rows = []
    fundamental_rows = []
    for number in range(n_securities):
        permno = 10000 + number
        rng = np.random.default_rng(number)
        monthly = 0.002 + number / n_securities * 0.004 + rng.normal(0, 0.025, len(dates))
        price = 20.0 * np.cumprod(1.0 + monthly)
        for date, ret, current_price in zip(dates, monthly, price):
            crsp_rows.append({
                "permno": permno,
                "permco": permno,
                "date": date,
                "price": current_price,
                "shrout": 20_000 + number * 100,
                "ret": ret,
                "dlret": np.nan,
                "ticker": f"T{number:03d}",
                "shrcd": 10,
                "exchcd": 1 + number % 3,
                "siccd": 2000 + (number % 6) * 100,
            })
        for year in (2017, 2018, 2019, 2020):
            scale = 100 + number + (year - 2017) * 5
            fundamental_rows.append({
                "permno": permno,
                "gvkey": f"{permno}",
                "datadate": pd.Timestamp(year=year, month=12, day=31),
                "fyear": year,
                "at": scale * 2.0,
                "lt": scale * 0.8,
                "seq": scale * 1.2,
                "ceq": scale * 1.1,
                "txditc": 2.0,
                "pstkrv": np.nan,
                "pstkl": np.nan,
                "pstk": 0.0,
                "revt": scale * 1.5,
                "sale": scale * 1.5,
                "cogs": scale * (0.8 - number / n_securities * 0.1),
                "ib": scale * 0.1,
            })
    return pd.DataFrame(crsp_rows), pd.DataFrame(fundamental_rows)


def test_request_and_specification_are_bounded_and_deterministic():
    request = _request()
    spec = RealModelSpec()
    assert specification_hash(request, spec) == specification_hash(request, spec)
    with pytest.raises(ValueError):
        WRDSResearchRequest.create("2020-01-01", "2021-01-01", "2019-01-01")
    with pytest.raises(ValueError):
        WRDSResearchRequest.create("2020-01-01", "2021-01-01", "2020-06-01", max_universe=50)


def test_queries_include_delisting_and_point_in_time_link_contracts():
    crsp_query = build_crsp_monthly_query(_request())
    comp_query = build_compustat_query(_request())
    assert "crsp.msedelist" in crsp_query
    assert "(1.0" not in crsp_query  # return composition is validated in Python
    assert "names.shrcd IN (10, 11)" in crsp_query
    assert "crsp.ccmxpf_linktable" in comp_query
    assert "link.linktype IN ('LC', 'LU')" in comp_query
    assert "ff.fivefactors_monthly" in build_fama_french_query(_request())
    assert "password" not in (crsp_query + comp_query).lower()


def test_fama_french_rows_require_unique_months():
    rows = pd.DataFrame([{
        "dateff": "2020-01-31", "mktrf": 0.01, "smb": 0.0, "hml": 0.0,
        "rmw": 0.0, "cma": 0.0, "umd": 0.0, "rf": 0.001,
    }])
    normalized = normalize_fama_french(rows)
    assert normalized.loc[0, "mktrf"] == pytest.approx(0.01)


def test_delisting_return_is_compounded_not_added():
    rows = pd.DataFrame([{
        "permno": 1, "permco": 1, "date": "2020-01-31", "price": 10,
        "shrout": 1000, "ret": -0.10, "dlret": -0.50, "ticker": "AAA",
        "shrcd": 10, "exchcd": 1, "siccd": 2000,
    }])
    normalized = normalize_crsp_monthly(rows)
    assert normalized.loc[0, "total_ret"] == pytest.approx(-0.55)


def test_share_classes_are_consolidated_at_company_level():
    from factors.real_model import _primary_company_security

    rows = pd.DataFrame([
        {"permno": 1, "permco": 7, "date": "2020-01-31", "price": 10,
         "shrout": 1000, "ret": 0.10, "dlret": np.nan, "ticker": "AAA",
         "shrcd": 10, "exchcd": 1, "siccd": 2000},
        {"permno": 2, "permco": 7, "date": "2020-01-31", "price": 5,
         "shrout": 1000, "ret": -0.10, "dlret": np.nan, "ticker": "AAB",
         "shrcd": 11, "exchcd": 1, "siccd": 2000},
    ])
    company = _primary_company_security(rows)
    assert len(company) == 1
    assert company.loc[0, "market_equity_millions"] == pytest.approx(15.0)
    assert company.loc[0, "total_ret"] == pytest.approx((0.10 * 10 - 0.10 * 5) / 15)


def test_month_with_no_return_or_delisting_observation_is_not_invented_as_zero():
    rows = pd.DataFrame([{
        "permno": 1, "permco": 1, "date": "2020-01-31", "price": 10,
        "shrout": 1000, "ret": np.nan, "dlret": np.nan, "ticker": "AAA",
        "shrcd": 10, "exchcd": 1, "siccd": 2000,
    }])
    assert normalize_crsp_monthly(rows).empty


def test_panel_respects_accounting_lag_and_holdout_boundary():
    crsp, fundamentals = _mock_inputs()
    panel = build_point_in_time_panel(crsp, fundamentals, _request(), RealModelSpec())
    available = panel.dropna(subset=["availability_date"])
    assert (available["availability_date"] <= available["date"]).all()
    assert set(panel["period"]) == {"development", "holdout"}
    assert panel.loc[panel["period"] == "development", "date"].max() < pd.Timestamp("2021-01-01")
    assert panel.loc[panel["period"] == "holdout", "date"].min() >= pd.Timestamp("2021-01-01")
    assert quality_receipt(panel, _request())["status"] == "PASS"


def test_panel_has_real_fundamental_signals_and_sector_neutral_scores():
    crsp, fundamentals = _mock_inputs()
    panel = build_point_in_time_panel(crsp, fundamentals, _request(), RealModelSpec())
    assert panel["VALUE"].notna().any()
    assert panel["QUALITY"].notna().any()
    assert panel["COMPOSITE"].notna().mean() > 0.90
    by_date = panel.dropna(subset=["VALUE_SCORE"]).groupby("date")["VALUE_SCORE"].mean()
    assert (by_date.abs() < 1e-10).all()


def test_portfolios_apply_costs_and_publish_no_security_identifiers():
    crsp, fundamentals = _mock_inputs()
    spec = RealModelSpec(transaction_cost_bps=(0.0, 10.0, 25.0))
    panel = build_point_in_time_panel(crsp, fundamentals, _request(), spec)
    monthly = compute_monthly_portfolios(panel, spec)
    assert not monthly.empty
    same = monthly[
        (monthly["signal"] == "COMPOSITE")
        & (monthly["strategy"] == "long_only")
    ].pivot(index="date", columns="cost_bps", values="net_return")
    assert (same[0.0] >= same[10.0]).all()
    assert (same[10.0] >= same[25.0]).all()
    public = public_monthly_results(monthly)
    assert "permno" not in public.columns
    assert "permco" not in public.columns
    assert "ticker" not in public.columns


def test_ic_and_performance_outputs_are_period_labeled():
    crsp, fundamentals = _mock_inputs()
    spec = RealModelSpec()
    panel = build_point_in_time_panel(crsp, fundamentals, _request(), spec)
    ic = compute_ic_summary(panel, spec)
    portfolio = compute_portfolio_summary(compute_monthly_portfolios(panel, spec))
    assert {"development", "holdout", "full"}.issubset(set(ic["period"]))
    assert {"development", "holdout", "full"}.issubset(set(portfolio["period"]))
    assert portfolio["sharpe"].notna().any()
    short_holdout = ic[(ic["period"] == "holdout") & (ic["n_months"] < 24)]
    assert not short_holdout["inference_eligible"].any()
    assert not short_holdout["significant_5pct"].any()


def test_factor_attribution_is_aggregate_and_period_labeled():
    crsp, fundamentals = _mock_inputs()
    spec = RealModelSpec()
    panel = build_point_in_time_panel(crsp, fundamentals, _request(), spec)
    monthly = compute_monthly_portfolios(panel, spec)
    dates = pd.date_range("2020-02-29", "2022-01-31", freq="ME")
    ff = pd.DataFrame({
        "dateff": dates,
        "mktrf": np.linspace(-0.02, 0.03, len(dates)),
        "smb": 0.001,
        "hml": 0.002,
        "rmw": -0.001,
        "cma": 0.0005,
        "umd": 0.003,
        "rf": 0.0002,
    })
    attribution = compute_factor_attribution(monthly, ff, spec)
    assert not attribution.empty
    assert {"development", "holdout", "full"}.issubset(attribution["period"])
    assert "permno" not in attribution.columns
