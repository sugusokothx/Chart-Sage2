from __future__ import annotations

import attention_factor_runtime
import behaviour_factor_runtime
import dual_ma_factor_runtime
import factor_composite
import fundamental_factor_runtime
import pandas as pd
import pytest
import residual_momentum_factor_runtime
import reversal_factor_runtime
import topix_pool_runtime

from historical_data import JpxHistoricalProvider, LocalDataPaths
from runtime.historical_data import normalize_code


def test_normalize_code_handles_plain_code() -> None:
    assert normalize_code("7203") == "7203"


def test_normalize_code_handles_ticker_suffix() -> None:
    assert normalize_code("7203.T") == "7203"


def test_normalize_code_handles_five_digit_variant() -> None:
    assert normalize_code("72030") == "7203"


def test_normalize_code_returns_none_for_invalid_input() -> None:
    assert normalize_code("ABC") is None
    assert normalize_code(None) is None


def test_normalize_code_handles_alphanumeric_code() -> None:
    assert normalize_code("135A") == "135A"
    assert normalize_code("135a") == "135A"


def test_normalize_code_handles_alphanumeric_five_char_variant() -> None:
    assert normalize_code("135A0") == "135A"


def test_normalize_code_handles_alphanumeric_ticker_suffix() -> None:
    assert normalize_code("135A.T") == "135A"
    assert normalize_code("135A0.T") == "135A"


def test_normalize_code_rejects_non_common_share_variant() -> None:
    # 25935 is Ito En's preferred share; 2593 is its common stock.
    assert normalize_code("25935") is None
    assert normalize_code("25935.T") is None
    assert normalize_code(25935) is None
    assert normalize_code(25935.0) is None
    assert normalize_code("135A9") is None


def test_normalize_code_rejects_letter_in_numeric_only_position() -> None:
    assert normalize_code("ABCD") is None
    assert normalize_code("1A2") is None


# Every module keeps its own copy of the normalizer; they must agree on
# well-formed codes so a code survives the provider -> factor -> composite path.
NORMALIZERS = {
    "historical_data": normalize_code,
    "topix_pool_runtime": topix_pool_runtime._normalize_code,
    "factor_composite": factor_composite.normalize_code,
    "attention": attention_factor_runtime.normalize_jp_code,
    "behaviour": behaviour_factor_runtime.normalize_jp_code,
    "dual_ma": dual_ma_factor_runtime.normalize_jp_code,
    "fundamental": fundamental_factor_runtime.normalize_jp_code,
    "residual_momentum": residual_momentum_factor_runtime.normalize_jp_code,
    "reversal": reversal_factor_runtime.normalize_jp_code,
}

WELL_FORMED_CODES = [
    ("7203", "7203"),
    ("7203.T", "7203"),
    ("72030", "7203"),
    ("135A", "135A"),
    ("135a", "135A"),
    ("135A0", "135A"),
    ("135A.T", "135A"),
    ("135A0.T", "135A"),
]


@pytest.mark.parametrize("module", sorted(NORMALIZERS))
@pytest.mark.parametrize(("raw", "expected"), WELL_FORMED_CODES)
def test_normalizers_agree_on_well_formed_codes(module: str, raw: str, expected: str) -> None:
    assert NORMALIZERS[module](raw) == expected


NON_COMMON_SHARE_CODES = ["25935", "25935.T", "13019", "72031", "135A9"]


@pytest.mark.parametrize("module", sorted(NORMALIZERS))
@pytest.mark.parametrize("raw", NON_COMMON_SHARE_CODES)
def test_normalizers_reject_non_common_share_codes(module: str, raw: str) -> None:
    assert NORMALIZERS[module](raw) is None


def test_preferred_share_prices_do_not_overwrite_common_stock(tmp_path) -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-12-30", "2024-12-30", "2024-12-30"],
            "code": ["25930", "25935", "72030"],
            "open": [3000.0, 2000.0, 2800.0],
            "high": [3000.0, 2000.0, 2800.0],
            "low": [3000.0, 2000.0, 2800.0],
            "close": [3000.0, 2000.0, 2800.0],
            "volume": [100, 10, 1000],
        }
    )
    # Sort order would decide the survivor if 25935 collapsed onto 2593.
    for order in (prices, prices.iloc[::-1]):
        order.to_csv(tmp_path / "prices.csv", index=False)
        table = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path)).table("prices")
        assert sorted(table["code"].dropna()) == ["2593", "7203"]
        assert float(table.loc[table["code"] == "2593", "close"].iloc[0]) == 3000.0


@pytest.mark.parametrize(
    "to_ticker",
    [
        attention_factor_runtime.to_yfinance_ticker,
        behaviour_factor_runtime.to_yfinance_ticker,
        dual_ma_factor_runtime.to_yfinance_ticker,
        fundamental_factor_runtime.to_yfinance_ticker,
        residual_momentum_factor_runtime.to_yfinance_ticker,
    ],
)
def test_yfinance_ticker_handles_alphanumeric_code(to_ticker) -> None:
    assert to_ticker("135A") == "135A.T"
    assert to_ticker("135A0") == "135A.T"
    assert to_ticker("7203") == "7203.T"
