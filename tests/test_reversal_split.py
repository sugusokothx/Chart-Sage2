from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_backtest_engine import BacktestConfig, _factor_config, _price_loader_for_single
from historical_data import JpxHistoricalProvider, LocalDataPaths
from reversal_factor_runtime import run_reversal_factor


# 1001 splits 1:2 inside the reversal window at an unchanged market value.
ASOF = pd.Timestamp("2023-10-31")
SPLIT = pd.Timestamp("2023-10-02")
DATES = pd.bdate_range("2023-07-03", ASOF)


def _provider(tmp_path: Path, adjusted_columns: bool) -> JpxHistoricalProvider:
    pre = DATES < SPLIT
    prices = pd.DataFrame(
        {
            "date": DATES,
            "code": "1001",
            "close": np.where(pre, 1000.0, 500.0),
            "volume": np.where(pre, 1000.0, 2000.0),
            "adjustment_factor": np.where(DATES == SPLIT, 0.5, 1.0),
        }
    )
    for col in ["open", "high", "low"]:
        prices[col] = prices["close"]
    if adjusted_columns:
        for col in ["open", "high", "low", "close"]:
            prices[f"adjustment_{col}"] = 500.0
        prices["adjustment_volume"] = 2000.0
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    pd.DataFrame({"asof_date": ["2023-07-03"], "code": ["1001"], "market_cap": [1e9]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
    return JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))


def _reversal(provider: JpxHistoricalProvider, panel: bool) -> pd.Series:
    cfg = _factor_config(BacktestConfig(), "reversal", provider, ASOF)
    if not panel:
        cfg = {**cfg, "panel_price_loader": None, "price_loader": _price_loader_for_single(provider)}
    detail = run_reversal_factor(pd.DataFrame({"code": ["1001"]}), ASOF, config=cfg)["detail"]
    return detail.set_index("code").loc["1001"]


@pytest.mark.parametrize("panel", [True, False])
def test_a_split_in_the_window_is_not_a_reversal(tmp_path: Path, panel: bool) -> None:
    row = _reversal(_provider(tmp_path, adjusted_columns=True), panel)
    assert row["status"] == "ok"
    assert row["start_price_date"] < SPLIT < row["end_price_date"]
    assert row["reversal_factor"] == pytest.approx(0.0)


def test_without_adjusted_closes_raw_closes_are_used(tmp_path: Path) -> None:
    row = _reversal(_provider(tmp_path, adjusted_columns=False), panel=True)
    assert row["past_return"] == pytest.approx(-0.5)
