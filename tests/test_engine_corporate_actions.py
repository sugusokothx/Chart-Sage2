from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_backtest_engine import BacktestConfig, _execution_prices, _portfolio_value, _run_interim_stop_loss_checks
from historical_data import JpxHistoricalProvider, LocalDataPaths


# 1001 splits 1:2 on SPLIT at an unchanged market value: raw close 1000 -> 500.
SPLIT = pd.Timestamp("2023-09-15")
DATES = pd.bdate_range("2023-08-01", "2023-09-29")


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
        # Adjusted to the latest basis, so continuous through the split.
        for col in ["open", "high", "low", "close"]:
            prices[f"adjustment_{col}"] = 500.0
        prices["adjustment_volume"] = 2000.0
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    pd.DataFrame({"asof_date": ["2023-08-01"], "code": ["1001"], "market_cap": [1e9]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
    return JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))


@pytest.mark.parametrize("adjusted_columns", [True, False])
@pytest.mark.parametrize("execution_price", ["open", "close"])
def test_split_while_held_keeps_nav_and_cost_basis(tmp_path: Path, adjusted_columns: bool, execution_price: str) -> None:
    # Bought 100 shares at the real pre-split price.
    holdings, cost_basis = {"1001": 100}, {"1001": 1000.0}
    actions: list[dict] = []
    trades: list[dict] = []
    provider = _provider(tmp_path, adjusted_columns)
    _run_interim_stop_loss_checks(
        provider=provider,
        holdings=holdings,
        cost_basis=cost_basis,
        cash=0.0,
        start_exclusive=pd.Timestamp("2023-08-31"),
        end_exclusive=pd.Timestamp("2023-09-29"),
        cfg=BacktestConfig(execution_price=execution_price),
        nav_rows=[],
        trade_rows=trades,
        corporate_action_rows=actions,
    )
    # The split restates the holding once; it is neither a gain nor a stop-loss.
    assert trades == []
    assert holdings == {"1001": 200}
    assert cost_basis == {"1001": pytest.approx(500.0)}
    assert [row["date"] for row in actions] == [SPLIT]
    # The next rebalance values the restated holding at its execution price.
    prices = _execution_prices(provider, ["1001"], pd.Timestamp("2023-09-29"), execution_price)
    assert _portfolio_value(holdings, prices, 0.0) == pytest.approx(100_000.0)
