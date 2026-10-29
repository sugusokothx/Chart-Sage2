from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_data import JpxHistoricalProvider, LocalDataPaths
from topix_pool_runtime import RunReport, ScreenRules, build_snapshot


ANCHOR = pd.Timestamp("2023-10-31")
DATES = pd.bdate_range("2023-06-01", "2023-12-29")
CODES = ["1001", "1002"]


def _prices(split: pd.Timestamp, adjusted_columns: bool = True) -> pd.DataFrame:
    # Both codes trade the same adjusted path; 1001 splits 1:3 at an
    # unchanged market value, with the cache adjusted to its end (December).
    adjusted = 300.0 * (1.0 + 0.01 * np.sin(np.arange(len(DATES))))
    frames = []
    for code in CODES:
        ratio = np.where(DATES < split, 3.0, 1.0) if code == "1001" else np.ones(len(DATES))
        frame = pd.DataFrame({"date": DATES, "code": code, "close": adjusted * ratio, "volume": 3000.0 / ratio})
        if adjusted_columns:
            frame["adjustment_close"] = adjusted
            frame["adjustment_volume"] = 3000.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _snapshot(prices: pd.DataFrame, tmp_path: Path, provider_panel: bool) -> pd.DataFrame:
    report = RunReport(mode="test", anchor_date=str(ANCHOR.date()))
    panel: pd.DataFrame | dict[str, pd.DataFrame] = prices
    if provider_panel:
        # The local provider hands build_snapshot a per-code dict, not a frame.
        prices.assign(open=prices["close"], high=prices["close"], low=prices["close"]).to_parquet(tmp_path / "prices.parquet", index=False)
        pd.DataFrame({"asof_date": ["2023-06-01"] * 2, "code": CODES, "market_cap": [1e11, 1e11]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
        provider = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))
        panel = provider.get_price_panel(CODES, "2023-06-01", str(ANCHOR.date()), report)
    caps = pd.DataFrame({"code": CODES, "market_cap_jpy": [1e11, 1e11], "market_cap_billion_jpy": [100.0, 100.0], "market_cap_rank": [1, 2]})
    snapshot = build_snapshot(CODES, {code: "x" for code in CODES}, caps, ANCHOR, ScreenRules(), panel, report, verbose=False)
    return snapshot.set_index("code")


@pytest.mark.parametrize("provider_panel", [False, True])
def test_a_split_in_the_lookback_does_not_inflate_volatility_or_volume(tmp_path: Path, provider_panel: bool) -> None:
    snapshot = _snapshot(_prices(split=pd.Timestamp("2023-10-02")), tmp_path, provider_panel)
    assert snapshot.loc["1001", "annualized_volatility"] == pytest.approx(snapshot.loc["1002", "annualized_volatility"])
    assert snapshot.loc["1001", "avg_daily_volume"] == pytest.approx(3000.0)
    assert snapshot.loc["1001", "close"] == pytest.approx(snapshot.loc["1002", "close"])


@pytest.mark.parametrize("provider_panel", [False, True])
def test_a_split_after_the_anchor_keeps_volume_in_the_anchors_shares(tmp_path: Path, provider_panel: bool) -> None:
    snapshot = _snapshot(_prices(split=pd.Timestamp("2023-12-01")), tmp_path, provider_panel)
    # Before the split 1001 trades 1000 shares a day at three times the price.
    assert snapshot.loc["1001", "avg_daily_volume"] == pytest.approx(1000.0)
    assert snapshot.loc["1001", "close"] == pytest.approx(3.0 * snapshot.loc["1002", "close"])
    assert snapshot.loc["1001", "annualized_volatility"] == pytest.approx(snapshot.loc["1002", "annualized_volatility"])


def test_without_adjusted_columns_raw_series_are_used(tmp_path: Path) -> None:
    snapshot = _snapshot(_prices(split=pd.Timestamp("2023-10-02"), adjusted_columns=False), tmp_path, provider_panel=False)
    # The split day reads as a -67% return.
    assert snapshot.loc["1001", "annualized_volatility"] > 10 * snapshot.loc["1002", "annualized_volatility"]
