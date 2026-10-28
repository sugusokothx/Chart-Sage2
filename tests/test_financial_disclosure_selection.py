from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from historical_data import JpxHistoricalProvider, LocalDataPaths, _deduplicate_table
from jquants_cache_builder import _append_existing, build_market_cap_from_financial_shares


ASOF = pd.Timestamp("2024-12-30")


def _statement(disclosed: str, time: str, disc_no: str, period_end: str, revenue: float, shares: float) -> dict:
    return {
        "disclosed_date": disclosed,
        "disclosed_time": time,
        "disc_no": disc_no,
        "doc_type": "2QFinancialStatements_Consolidated_JP",
        "current_period_end": period_end,
        "code": "1001",
        "total_revenue": revenue,
        "operating_income": revenue / 10,
        "net_income": revenue / 20,
        "total_assets": revenue * 2,
        "stockholders_equity": revenue,
        "shares_outstanding": shares,
    }


def _revision(disclosed: str, time: str, disc_no: str) -> dict:
    # Forecast revisions carry forecasts only; actuals and shares stay empty.
    return {
        "disclosed_date": disclosed,
        "disclosed_time": time,
        "disc_no": disc_no,
        "doc_type": "EarnForecastRevision",
        "current_period_end": "2025-03-31",
        "code": "1001",
        "forecast_sales": 999.0,
    }


def _provider(tmp_path: Path, rows: list[dict]) -> JpxHistoricalProvider:
    pd.DataFrame(rows).to_parquet(tmp_path / "financial_summary.parquet", index=False)
    dates = pd.bdate_range("2024-01-04", ASOF)
    pd.DataFrame({"date": dates, "code": "1001", "open": 500.0, "high": 500.0, "low": 500.0, "close": 500.0, "volume": 1000.0}).to_parquet(
        tmp_path / "prices.parquet", index=False
    )
    pd.DataFrame({"asof_date": ["2024-01-04"], "code": ["1001"], "market_cap": [1e9]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
    return JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))


def _selected(tmp_path: Path, rows: list[dict]) -> tuple[pd.Series, float]:
    provider = _provider(tmp_path, rows)
    panel = provider.load_fundamental_panel(pd.DataFrame({"code": ["1001"]}), {"asof_date": ASOF}).set_index("code")
    attention = provider.load_attention_inputs(pd.DataFrame({"code": ["1001"]}), ASOF, ASOF)
    return panel.loc["1001"], float(attention["shares_reported"].iloc[-1])


def test_same_day_forecast_revision_does_not_replace_the_statement(tmp_path: Path) -> None:
    rows = [
        _statement("2024-11-07", "15:00", "20241107000001", "2024-09-30", 100.0, 1_000_000),
        _revision("2024-11-07", "15:00", "20241107000002"),
    ]
    panel, shares = _selected(tmp_path, rows)
    assert panel["TotalRevenue"] == 100.0
    assert panel["StockholdersEquity"] == 100.0
    assert shares == 1_000_000


def test_later_forecast_revision_does_not_hide_the_statement(tmp_path: Path) -> None:
    rows = [
        _statement("2024-11-07", "15:00", "20241107000001", "2024-09-30", 100.0, 1_000_000),
        _revision("2024-12-10", "15:00", "20241210000001"),
    ]
    panel, shares = _selected(tmp_path, rows)
    assert panel["TotalRevenue"] == 100.0
    assert shares == 1_000_000


def test_same_day_late_filings_resolve_to_the_newest_period(tmp_path: Path) -> None:
    # A late filer discloses the quarter, then the older full year, together.
    rows = [
        _statement("2024-11-14", "13:30", "20241111000001", "2024-09-30", 300.0, 3_000_000),
        _statement("2024-11-14", "13:30", "20241111000002", "2024-03-31", 100.0, 1_000_000),
    ]
    panel, shares = _selected(tmp_path, rows)
    assert panel["TotalRevenue"] == 300.0
    assert shares == 3_000_000


@pytest.mark.parametrize("period_end,expected", [("2024-06-30", 200.0), ("2024-09-30", 250.0)])
def test_later_corrections_replace_only_their_own_period(tmp_path: Path, period_end: str, expected: float) -> None:
    rows = [
        _statement("2024-08-09", "15:00", "20240809000001", "2024-06-30", 150.0, 1_000_000),
        _statement("2024-11-08", "15:00", "20241108000001", "2024-09-30", 200.0, 1_000_000),
        # A correction filed in December, of the June quarter or of September.
        _statement("2024-12-02", "15:00", "20241202000001", period_end, 250.0, 1_000_000),
    ]
    panel, _ = _selected(tmp_path, rows)
    assert panel["TotalRevenue"] == expected


def test_dedup_keeps_distinct_disclosures_and_drops_repeats() -> None:
    first = _statement("2024-11-07", "15:00", "20241107000001", "2024-09-30", 100.0, 1_000_000)
    refetched = {**first, "total_revenue": 101.0}
    frame = pd.DataFrame([first, _revision("2024-11-07", "15:00", "20241107000002"), refetched])
    out = _deduplicate_table(frame, "financial_summary")
    assert sorted(out["disc_no"]) == ["20241107000001", "20241107000002"]
    assert out.loc[out["disc_no"].eq("20241107000001"), "total_revenue"].item() == 101.0


@pytest.mark.parametrize("existing_format", ["csv", "parquet"])
def test_append_existing_keeps_same_day_disclosures(tmp_path: Path, existing_format: str) -> None:
    existing = pd.DataFrame([
        _statement("2024-11-07", "15:00", "20241107000001", "2024-09-30", 100.0, 1_000_000),
        _revision("2024-11-07", "15:00", "20241107000002"),
    ])
    if existing_format == "csv":
        existing.to_csv(tmp_path / "financial_summary.csv", index=False)
    else:
        existing.to_parquet(tmp_path / "financial_summary.parquet", index=False)
    keys = ["disclosed_date", "code", "disc_no"]
    # Nothing new fetched: the cached same-day rows must survive as they are.
    path = _append_existing(tmp_path, "financial_summary", pd.DataFrame(columns=["disclosed_date", "code"]), keys, "parquet")
    assert len(pd.read_parquet(path)) == 2
    # A refetch of the statement replaces it, and only it.
    refetched = existing.iloc[[0]].assign(disclosed_date=pd.Timestamp("2024-11-07"), total_revenue=101.0)
    out = pd.read_parquet(_append_existing(tmp_path, "financial_summary", refetched, keys, "parquet"))
    assert len(out) == 2
    assert pd.to_numeric(out["total_revenue"]).max() == 101.0


def test_market_cap_shares_follow_the_newest_period() -> None:
    financials = pd.DataFrame([
        _statement("2024-11-14", "13:30", "20241111000001", "2024-09-30", 300.0, 3_000_000),
        _statement("2024-11-14", "13:30", "20241111000002", "2024-03-31", 100.0, 1_000_000),
        _revision("2024-11-14", "13:30", "20241111000003"),
    ])
    prices = pd.DataFrame({"date": pd.to_datetime(["2024-11-29"]), "code": ["1001"], "close": [500.0], "adjustment_close": [500.0]})
    snapshots = pd.DataFrame({"asof_date": pd.to_datetime(["2024-11-29"]), "code": ["1001"]})
    out = build_market_cap_from_financial_shares(prices, snapshots, financials)
    assert out["shares_outstanding"].item() == 3_000_000
    assert out["market_cap"].item() == pytest.approx(1.5e9)
