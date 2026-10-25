from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attention_factor_runtime import run_attention_factor
from historical_data import JpxHistoricalProvider, LocalDataPaths
from jquants_cache_builder import _fins_summary_usecol, normalize_jquants_statements


# 1001 splits 1:2 on SPLIT; 1002 is its twin without a split, trading the
# same post-split volume every day.  On a correct basis both codes show the
# same avol and turnover at any date.
SPLIT = pd.Timestamp("2023-09-01")
DATES = pd.bdate_range("2022-01-03", "2024-03-29")
CODES = pd.DataFrame({"code": ["1001", "1002"]})


def _prices() -> pd.DataFrame:
    pre = DATES < SPLIT
    split = pd.DataFrame(
        {
            "date": DATES,
            "code": "1001",
            "close": np.where(pre, 1000.0, 500.0),
            "volume": np.where(pre, 1000.0, 2000.0),
            "adjustment_factor": np.where(DATES == SPLIT, 0.5, 1.0),
            "adjustment_close": 500.0,
            "adjustment_volume": 2000.0,
        }
    )
    twin = split.assign(code="1002", close=500.0, volume=2000.0, adjustment_factor=1.0)
    out = pd.concat([split, twin], ignore_index=True)
    for col in ["open", "high", "low"]:
        out[col] = out["close"]
        out[f"adjustment_{col}"] = out["adjustment_close"]
    return out


def _financials() -> pd.DataFrame:
    # 1001 discloses 1,000,000 shares for the June quarter before the split and
    # 2,000,000 for the September quarter after it.
    rows = [
        ("2023-08-10", "2023-06-30", 1_000_000, 2_000_000),
        ("2023-11-10", "2023-09-30", 2_000_000, 2_000_000),
    ]
    return pd.DataFrame(
        [
            {"disclosed_date": disclosed, "current_period_end": period_end, "code": code, "shares_outstanding": shares}
            for disclosed, period_end, split_shares, twin_shares in rows
            for code, shares in [("1001", split_shares), ("1002", twin_shares)]
        ]
    )


def _cache(tmp_path: Path, prices: pd.DataFrame | None = None, financials: pd.DataFrame | None = None) -> Path:
    (prices if prices is not None else _prices()).to_parquet(tmp_path / "prices.parquet", index=False)
    pd.DataFrame(
        {"asof_date": ["2022-01-03"] * 2, "code": ["1001", "1002"], "market_cap": [1e9, 1e9]}
    ).to_parquet(tmp_path / "market_cap.parquet", index=False)
    (financials if financials is not None else _financials()).to_parquet(tmp_path / "financial_summary.parquet", index=False)
    return tmp_path


def _attention(data_dir: Path, asof: str) -> pd.DataFrame:
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=data_dir))
    result = run_attention_factor(CODES, asof, config={"attention_loader": provider.load_attention_inputs, "verbose": False})
    return result["detail"].set_index("code")


def _assert_twins_match(detail: pd.DataFrame) -> None:
    assert detail["status"].eq("ok").all()
    for col in ["avol", "turnover"]:
        assert detail.loc["1001", col] == pytest.approx(detail.loc["1002", col])


def test_split_inside_window_does_not_inflate_avol(tmp_path: Path) -> None:
    # Raw volume doubles at the split, so raw avol would read about 1.5.
    _assert_twins_match(_attention(_cache(tmp_path), "2024-03-29"))


def test_split_after_period_end_rescales_reported_shares(tmp_path: Path) -> None:
    # The latest disclosure counts pre-split shares; October trades post-split.
    detail = _attention(_cache(tmp_path), "2023-10-31")
    _assert_twins_match(detail)
    assert detail.loc["1001", "shares_outstanding"] == pytest.approx(2_000_000)


def test_split_after_asof_matches_the_adjusted_basis(tmp_path: Path) -> None:
    # Adjusted August volume already counts post-split shares.
    _assert_twins_match(_attention(_cache(tmp_path), "2023-08-31"))


def test_disclosed_date_is_the_basis_without_period_end(tmp_path: Path) -> None:
    detail = _attention(_cache(tmp_path, financials=_financials().drop(columns="current_period_end")), "2023-10-31")
    _assert_twins_match(detail)


def test_period_end_is_the_basis_when_the_split_precedes_disclosure(tmp_path: Path) -> None:
    # Disclosed after the split, but the count is for the June period end.
    financials = _financials()
    financials.loc[financials["disclosed_date"].eq("2023-08-10"), "disclosed_date"] = "2023-09-15"
    _assert_twins_match(_attention(_cache(tmp_path, financials=financials), "2023-10-31"))


def test_unreliable_adjustments_fall_back_to_raw_prices(tmp_path: Path) -> None:
    # An adjusted/raw ratio jump with no corporate action marks the code unreliable.
    prices = _prices()
    jump = prices["code"].eq("1001") & prices["date"].ge(SPLIT)
    prices.loc[prices["code"].eq("1001"), "adjustment_factor"] = 1.0
    prices.loc[jump, ["adjustment_close", "adjustment_open", "adjustment_high", "adjustment_low"]] = 50.0
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=_cache(tmp_path, prices=prices)))
    out = provider.load_attention_inputs(CODES, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-10-31"))
    split = out.loc[out["code"].eq("1001")]
    np.testing.assert_array_equal(split["volume"], split["raw_volume"])
    assert split["shares_to_adjusted"].eq(1.0).all()


def test_statements_keep_the_period_end() -> None:
    raw = pd.DataFrame({"DiscDate": ["2023-08-10"], "Code": ["10010"], "CurPerEn": ["2023-06-30"], "ShOutFY": ["1000000"]})
    out = normalize_jquants_statements(raw)
    assert out["current_period_end"].tolist() == [pd.Timestamp("2023-06-30")]
    # The bulk CSV reader keeps only listed columns.
    assert _fins_summary_usecol("CurPerEn")
