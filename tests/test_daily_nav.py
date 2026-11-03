from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_backtest_engine import BacktestConfig, _run_interim_stop_loss_checks
from historical_data import JpxHistoricalProvider, LocalDataPaths
from local_backtest_runner import _metric_row


DATES = pd.bdate_range("2023-08-01", "2023-09-29")
START = pd.Timestamp("2023-08-31")
END = pd.Timestamp("2023-09-29")
SPLIT = pd.Timestamp("2023-09-15")
GAP = pd.Timestamp("2023-09-07")


def _provider(tmp_path: Path) -> JpxHistoricalProvider:
    # 1001 rises 10 a day and splits 1:2 at an unchanged value; 1002 is flat
    # at 100 but has no row on GAP.
    rows = []
    for i, date in enumerate(DATES):
        close = 1000.0 + 10.0 * i
        rows.append({"date": date, "code": "1001", "close": close / 2 if date >= SPLIT else close, "adjustment_factor": 0.5 if date == SPLIT else 1.0})
        if date != GAP:
            rows.append({"date": date, "code": "1002", "close": 100.0, "adjustment_factor": 1.0})
    prices = pd.DataFrame(rows)
    for col in ["open", "high", "low"]:
        prices[col] = prices["close"]
    prices["volume"] = 1000.0
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    pd.DataFrame({"asof_date": ["2023-08-01"] * 2, "code": ["1001", "1002"], "market_cap": [1e9, 1e9]}).to_parquet(
        tmp_path / "market_cap.parquet", index=False
    )
    return JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))


def _interim(provider: JpxHistoricalProvider, holdings: dict[str, int], cost_basis: dict[str, float], frequency: str) -> tuple[pd.DataFrame, list[dict]]:
    nav_rows: list[dict] = []
    trades: list[dict] = []
    _run_interim_stop_loss_checks(
        provider=provider,
        holdings=holdings,
        cost_basis=cost_basis,
        cash=1000.0,
        start_exclusive=START,
        end_exclusive=END,
        cfg=BacktestConfig(stop_loss_check_frequency=frequency),
        nav_rows=nav_rows,
        trade_rows=trades,
        corporate_action_rows=[],
    )
    return pd.DataFrame(nav_rows).set_index("date"), trades


@pytest.mark.parametrize("frequency", ["monthly", "weekly", "daily"])
def test_every_trading_day_between_executions_is_marked(tmp_path: Path, frequency: str) -> None:
    nav, trades = _interim(_provider(tmp_path), {"1001": 10, "1002": 10}, {"1001": 1000.0, "1002": 100.0}, frequency)
    expected_dates = [d for d in DATES if START < d < END]
    assert list(nav.index) == expected_dates
    assert trades == []
    # 1001 is worth 10 * (1000 + 10 i) through the split, 1002 stays at 1000
    # (its missing day carries the previous close), plus 1000 cash.
    expected = [1000.0 + 1000.0 + 10.0 * (1000.0 + 10.0 * DATES.get_loc(d)) for d in expected_dates]
    assert nav["nav"].tolist() == pytest.approx(expected)
    assert (nav["turnover"] == 0.0).all()


def test_a_stop_loss_day_keeps_its_turnover(tmp_path: Path) -> None:
    # A cost basis far above the price triggers the stop on the first check.
    nav, trades = _interim(_provider(tmp_path), {"1001": 10}, {"1001": 5000.0}, "daily")
    assert [t["execution_date"] for t in trades] == [nav.index[0]]
    assert nav["turnover"].iloc[0] > 0
    assert (nav["turnover"].iloc[1:] == 0.0).all()
    assert (nav["nav"].iloc[1:] == nav["nav"].iloc[0]).all()


def test_cash_only_days_are_marked(tmp_path: Path) -> None:
    nav, _ = _interim(_provider(tmp_path), {}, {}, "monthly")
    assert len(nav) == len([d for d in DATES if START < d < END])
    assert (nav["nav"] == 1000.0).all()


def test_volatility_scales_multi_day_moves_by_the_square_root() -> None:
    # Moves over 1, 3 and 20 weekdays, each +-1% per sqrt(day): one day's
    # volatility is exactly 1% whatever the spacing.  Fri -> Mon is one day.
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-11", "2024-02-08", "2024-02-09"])
    days = np.array([1, 3, 20, 1])
    log_nav = np.concatenate([[0.0], np.cumsum(0.01 * np.sqrt(days) * np.array([1, -1, 1, -1]))])
    row = _metric_row(pd.Series(dates), pd.Series(100.0 * np.exp(log_nav)), "strategy")
    assert row["annualized_volatility"] == pytest.approx(0.01 * np.sqrt(252.0))
