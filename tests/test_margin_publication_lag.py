from __future__ import annotations

from pathlib import Path

import behaviour_factor_runtime
import pandas as pd
import pytest

from historical_data import JpxHistoricalProvider, LocalDataPaths


CODES = ["1001", "1002", "1003"]
# Application dates are Fridays; the balance grows 1,000 a week so the week a
# signal used is readable from the value.
WEEKS = {"2024-03-15": 1000, "2024-03-22": 2000, "2024-03-29": 3000}


def _write_cache(data_dir: Path, *, holidays: tuple[str, ...] = (), published: dict[str, str] | None = None) -> None:
    days = [d for d in pd.bdate_range("2024-03-01", "2024-04-12") if d.strftime("%Y-%m-%d") not in holidays]
    pd.DataFrame(
        [
            {"date": d.date(), "code": code, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000}
            for d in days
            for code in CODES
        ]
    ).to_csv(data_dir / "prices.csv", index=False)
    pd.DataFrame({"asof_date": "2024-03-01", "code": CODES, "market_cap": [1e10, 2e10, 3e10]}).to_csv(
        data_dir / "market_cap.csv", index=False
    )
    rows = []
    for app, buy in WEEKS.items():
        for i, code in enumerate(CODES):
            row = {"date": app, "code": code, "margin_buy_balance": buy * (i + 1), "margin_sell_balance": 100}
            if published is not None:
                row["published_date"] = published.get(app)
            rows.append(row)
    pd.DataFrame(rows).to_csv(data_dir / "margin.csv", index=False)


def _latest_week(data_dir: Path, signal: str) -> set[pd.Timestamp]:
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=data_dir))
    result = behaviour_factor_runtime.run_behaviour_factor(
        pd.DataFrame({"code": CODES}),
        signal,
        config={"behaviour_loader": provider.load_behaviour_inputs, "verbose": False},
    )
    return set(pd.to_datetime(result["detail"]["credit_data_date"]))


@pytest.mark.parametrize(
    ("signal", "expected_week"),
    [
        ("2024-03-29", "2024-03-22"),  # Friday of application: not yet public
        ("2024-04-01", "2024-03-22"),  # Monday: still one day early
        ("2024-04-02", "2024-03-29"),  # Tuesday: published on the 2nd business day
    ],
)
def test_margin_week_is_used_only_once_published(tmp_path: Path, signal: str, expected_week: str) -> None:
    _write_cache(tmp_path)
    assert _latest_week(tmp_path, signal) == {pd.Timestamp(expected_week)}


def test_publication_lag_counts_trading_days(tmp_path: Path) -> None:
    # With Monday closed, the Friday balance is published on Wednesday.
    _write_cache(tmp_path, holidays=("2024-04-01",))
    assert _latest_week(tmp_path, "2024-04-02") == {pd.Timestamp("2024-03-22")}
    assert _latest_week(tmp_path, "2024-04-03") == {pd.Timestamp("2024-03-29")}


def test_published_date_overrides_the_lag_rule(tmp_path: Path) -> None:
    # A late PubDate wins over the rule; a missing one falls back to it.
    _write_cache(tmp_path, published={"2024-03-29": "2024-04-04"})
    assert _latest_week(tmp_path, "2024-04-03") == {pd.Timestamp("2024-03-22")}
    assert _latest_week(tmp_path, "2024-04-04") == {pd.Timestamp("2024-03-29")}


def test_margin_table_keeps_published_date(tmp_path: Path) -> None:
    _write_cache(tmp_path, published={"2024-03-29": "2024-04-02"})
    margin = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path)).table("margin")
    assert "published_date" in margin.columns
    assert "disclosed_date" not in margin.columns
    assert pd.api.types.is_datetime64_any_dtype(margin["published_date"])
