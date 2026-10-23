from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from jquants_cache_builder import (
    _append_existing,
    _read_cache_table,
    _shard_path,
    _week_over_week_margin_change,
    _write_shard,
    load_or_download_margin_shards,
    normalize_jquants_margin,
)


def _balances(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"code": "7203", "date": pd.to_datetime(dates), "margin_buy_balance": values})


def test_weekly_rows_compare_with_previous_row_even_on_short_weeks() -> None:
    # 2020-07-22 (Wed) closed a holiday-shortened week, five days after Friday.
    frame = _balances(["2020-07-10", "2020-07-17", "2020-07-22", "2020-07-31"], [100, 130, 120, 160])
    change = _week_over_week_margin_change(frame, "margin_buy_balance")
    assert change.isna().iloc[0]
    assert change.iloc[1:].tolist() == [30, -10, 40]


def test_daily_rows_compare_with_the_balance_a_week_earlier() -> None:
    days = pd.bdate_range("2026-09-28", "2026-10-09")
    frame = pd.concat(
        [
            _balances(["2026-09-18"], [0]),  # last weekly row before the switch
            _balances([d.strftime("%Y-%m-%d") for d in days], [100 * (i + 1) for i in range(len(days))]),
        ],
        ignore_index=True,
    )
    change = _week_over_week_margin_change(frame, "margin_buy_balance").set_axis(frame["date"])
    # One week of daily rows is five steps of 100, not a single day's step.
    assert change[pd.Timestamp("2026-10-06")] == 500
    assert change[pd.Timestamp("2026-10-09")] == 500
    # The first daily week reaches back to the weekly row.
    assert change[pd.Timestamp("2026-09-28")] == 100


def test_change_keeps_input_order() -> None:
    frame = _balances(["2026-10-09", "2026-10-02"], [300, 100])
    assert _week_over_week_margin_change(frame, "margin_buy_balance").tolist()[0] == 200


class _FakeClient:
    def __init__(self, rows: dict[str, list[dict]]) -> None:
        self.rows = rows
        self.requested: list[str] = []

    def get_dataset(self, endpoint_name: str, params: dict, result_key: str) -> pd.DataFrame:
        self.requested.append(params["date"])
        return pd.DataFrame(self.rows.get(params["date"], []))


def _raw(day: str, long_vol: float) -> dict:
    return {"Date": day, "Code": "72030", "LongVol": long_vol, "ShrtVol": 10.0, "PubDate": None}


def _empty_shard(data_dir: Path, key: str, *, written: str) -> None:
    path = _shard_path(data_dir, "margin", key, "csv")
    _write_shard(normalize_jquants_margin(pd.DataFrame()), path, "csv")
    stamp = pd.Timestamp(written).timestamp()
    os.utime(path, (stamp, stamp))


def test_empty_shard_written_before_publication_is_refetched(tmp_path: Path) -> None:
    # Fetched on Saturday, before Tuesday's publication: the cache is empty.
    _empty_shard(tmp_path, "20260918", written="2026-09-19 12:00")
    client = _FakeClient({"20260918": [_raw("2026-09-18", 500.0)]})
    out = load_or_download_margin_shards(
        client, tmp_path, pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-18"), "csv", endpoint_name="weekly_margin_interest"
    )
    assert client.requested == ["20260918"]
    assert out["margin_buy_balance"].tolist() == [500.0]


def test_settled_empty_shard_is_trusted(tmp_path: Path) -> None:
    # A Thursday before the daily switch is empty for good.
    _empty_shard(tmp_path, "20260917", written="2026-10-01 12:00")
    client = _FakeClient({})
    out = load_or_download_margin_shards(
        client, tmp_path, pd.Timestamp("2026-09-17"), pd.Timestamp("2026-09-17"), "csv", endpoint_name="weekly_margin_interest"
    )
    assert client.requested == []
    assert out.empty


def test_normalize_reads_publication_date_and_amounts() -> None:
    raw = pd.DataFrame(
        [{"PubDate": "2026-09-28", "Date": "2026-09-25", "Code": "86970", "LongVol": 270800.0, "ShrtVol": 417000.0, "LongVal": 589568400.0, "ShrtVal": 838347467.0}]
    )
    out = normalize_jquants_margin(raw)
    row = out.iloc[0]
    assert row["code"] == "8697"
    assert row["published_date"] == pd.Timestamp("2026-09-28")
    assert row["margin_buy_value"] == 589568400.0
    assert row["margin_sell_value"] == 838347467.0


@pytest.mark.parametrize("save_format", ["csv", "parquet"])
def test_refetched_margin_row_replaces_cached_row(tmp_path: Path, save_format: str) -> None:
    first = pd.DataFrame(
        {"date": pd.to_datetime(["2026-09-25"]), "published_date": [pd.NaT], "code": ["7203"], "margin_buy_balance": [1.0], "margin_sell_balance": [1.0]}
    )
    second = first.assign(published_date=pd.to_datetime(["2026-09-28"]), margin_buy_balance=2.0)
    _append_existing(tmp_path, "margin", first, ["date", "code"], save_format)
    path = _append_existing(tmp_path, "margin", second, ["date", "code"], save_format)
    out = _read_cache_table(path)
    assert len(out) == 1
    assert float(out["margin_buy_balance"].iloc[0]) == 2.0
