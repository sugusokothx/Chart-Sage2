from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behaviour_factor_runtime import DEFAULT_CONFIG, compute_latest_behaviour_rows, extract_universe_codes
from historical_data import JpxHistoricalProvider, LocalDataPaths


ASOF = pd.Timestamp("2024-04-05")
CODES = [f"{1001 + i}" for i in range(400)]


def _inputs(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(CODES)
    close = rng.uniform(100, 5000, n)
    mcap = rng.uniform(5e9, 5e11, n)
    # Balances as a share of market cap, converted to share counts.
    buy = rng.uniform(0.001, 0.05, n) * mcap / close
    return pd.DataFrame(
        {
            "code": CODES,
            "date": pd.Timestamp("2024-03-29"),
            "margin_buy_balance": buy,
            "margin_sell_balance": rng.uniform(0.1, 1.0, n) * buy,
            "margin_buy_change": rng.normal(0, 0.002, n) * mcap / close,
            "margin_sell_change": rng.normal(0, 0.001, n) * mcap / close,
            "market_cap_jpy": mcap,
            "avg_daily_trading_value": mcap / 200,
            "close": close,
        }
    )


def _factor(inputs: pd.DataFrame) -> pd.DataFrame:
    universe = extract_universe_codes(pd.DataFrame({"code": CODES}))
    return compute_latest_behaviour_rows(universe, inputs, ASOF, ASOF, DEFAULT_CONFIG).set_index("code")


def test_margin_value_ratios_use_jpy() -> None:
    inputs = _inputs()
    detail = _factor(inputs)
    expected = inputs.set_index("code").eval("margin_buy_balance * close / market_cap_jpy")
    np.testing.assert_allclose(detail["margin_buy_to_mcap"], expected.loc[detail.index])


def test_a_split_does_not_change_the_score() -> None:
    # 1:10 split: ten times the shares at a tenth of the price, same value.
    inputs = _inputs()
    split = inputs.copy()
    first = split["code"].eq(CODES[0])
    for col in ["margin_buy_balance", "margin_sell_balance", "margin_buy_change", "margin_sell_change"]:
        split.loc[first, col] *= 10
    split.loc[first, "close"] /= 10
    before, after = _factor(inputs), _factor(split)
    # credit_pressure takes log1p of share counts, so it moves by ~1/shares.
    np.testing.assert_allclose(after["credit_residual_factor"], before["credit_residual_factor"], atol=1e-5)


def test_cheap_stocks_and_large_share_counts_do_not_dominate() -> None:
    inputs = _inputs()
    # A 5-yen stock with an ordinary margin share of its market cap, and a
    # mega-cap whose share-count change is huge but small relative to its size.
    cheap = inputs["code"].eq(CODES[0])
    shares = ["margin_buy_balance", "margin_sell_balance", "margin_buy_change", "margin_sell_change"]
    inputs.loc[cheap, shares] *= inputs.loc[cheap, "close"].item() / 5
    inputs.loc[cheap, "close"] = 5.0
    big = inputs["code"].eq(CODES[1])
    inputs.loc[big, ["market_cap_jpy", *shares]] *= 1000
    detail = _factor(inputs)
    assert detail["credit_residual_factor"].abs().max() < 3.5
    # Extreme components are capped like credit_pressure.
    inputs.loc[cheap, "margin_buy_balance"] *= 1000
    assert _factor(inputs)["credit_residual_factor"].abs().max() < 3.5


def test_without_close_the_value_components_drop_out() -> None:
    detail = _factor(_inputs().drop(columns="close"))
    assert detail["margin_buy_to_mcap"].isna().all()
    assert detail["credit_change"].isna().all()
    assert detail["credit_residual_factor"].notna().all()


@pytest.mark.parametrize("holiday", [False, True])
def test_provider_values_balances_at_the_application_date_close(tmp_path: Path, holiday: bool) -> None:
    days = pd.bdate_range("2024-03-01", "2024-04-12")
    if holiday:
        # No trading on the application Friday: use the Thursday close.
        days = days[days != pd.Timestamp("2024-03-29")]
    prices = pd.DataFrame({"date": days, "code": "1001"})
    prices["close"] = 100.0 + np.arange(len(prices))
    for col in ["open", "high", "low"]:
        prices[col] = prices["close"]
    prices["volume"] = 1000
    prices.to_parquet(tmp_path / "prices.parquet", index=False)
    pd.DataFrame({"asof_date": ["2024-03-01"], "code": ["1001"], "market_cap": [1e10]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
    pd.DataFrame(
        {"date": pd.to_datetime(["2024-03-22", "2024-03-29"]), "code": "1001", "margin_buy_balance": [1000, 2000], "margin_sell_balance": [100, 100]}
    ).to_parquet(tmp_path / "margin.parquet", index=False)
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))
    out = provider.load_behaviour_inputs(pd.DataFrame({"code": ["1001"]}), pd.Timestamp("2024-03-01"), ASOF).set_index("date")
    close = prices.set_index("date")["close"]
    assert out.loc["2024-03-22", "close"] == close.loc["2024-03-22"]
    assert out.loc["2024-03-29", "close"] == close.loc[:"2024-03-29"].iloc[-1]
