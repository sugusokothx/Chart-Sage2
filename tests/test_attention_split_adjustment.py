from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import factor_cache
from attention_factor_runtime import run_attention_factor
from historical_data import JpxHistoricalProvider, LocalDataPaths
from jquants_cache_builder import (
    _fins_summary_usecol,
    load_or_download_price_shards,
    normalize_jquants_prices,
    normalize_jquants_statements,
)


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
    # The source explicitly identifies these as unadjusted June share counts.
    financials = _financials()
    financials.loc[financials["disclosed_date"].eq("2023-08-10"), "disclosed_date"] = "2023-09-15"
    financials["shares_basis_date"] = financials["current_period_end"]
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


class _BulkPricesClient:
    def __init__(self, prices: pd.DataFrame) -> None:
        self.raw = prices.drop(columns=[c for c in prices if c.startswith("adjustment_") and c != "adjustment_factor"])
        self.requests: list[dict] = []

    def get_dataset(self, endpoint_name: str, params: dict, result_key: str) -> pd.DataFrame:
        self.requests.append(params)
        dates = self.raw["date"]
        return self.raw.loc[dates.between(pd.Timestamp(params["from"]), pd.Timestamp(params["to"]))].copy()


@pytest.mark.parametrize("split_date,ratio", [("2023-09-01", 2), ("2023-09-15", 5)])
@pytest.mark.parametrize("save_format", ["csv", "parquet"])
def test_monthly_bulk_cache_uses_one_adjustment_basis(tmp_path: Path, split_date: str, ratio: int, save_format: str) -> None:
    prices = _prices()
    split_code = prices["code"].eq("1001")
    pre = prices["date"].lt(split_date)
    prices.loc[split_code, "volume"] = np.where(pre[split_code], 2000.0 / ratio, 2000.0)
    for col in ["open", "high", "low", "close"]:
        prices.loc[split_code, col] = np.where(pre[split_code], 500.0 * ratio, 500.0)
    prices.loc[split_code, "adjustment_factor"] = np.where(prices.loc[split_code, "date"].eq(split_date), 1.0 / ratio, 1.0)
    financials = _financials()
    financials.loc[financials["code"].eq("1001") & financials["current_period_end"].eq("2023-06-30"), "shares_outstanding"] = 2_000_000 / ratio
    client = _BulkPricesClient(prices)
    # Assemble once before the split, then extend using those already-cached
    # months. The older shards must be rebased when September is appended.
    start = pd.Timestamp("2022-01-01")
    load_or_download_price_shards(client, tmp_path, start, pd.Timestamp("2023-08-31"), save_format)
    assembled = load_or_download_price_shards(client, tmp_path, start, pd.Timestamp("2024-03-31"), save_format)
    split = assembled.loc[assembled["code"].eq("1001")]
    np.testing.assert_allclose(pd.to_numeric(split["adjustment_volume"]), 2000.0)
    np.testing.assert_allclose(pd.to_numeric(split["adjustment_close"]), 500.0)
    _cache(tmp_path, financials=financials)
    if save_format == "csv":
        (tmp_path / "prices.parquet").unlink()
        assembled.to_csv(tmp_path / "prices.csv", index=False)
    else:
        assembled.to_parquet(tmp_path / "prices.parquet", index=False)
    for asof in ["2023-08-31", "2023-10-31", "2024-03-29"]:
        _assert_twins_match(_attention(tmp_path, asof))
    requested = len(client.requests)
    reloaded = load_or_download_price_shards(client, tmp_path, start, pd.Timestamp("2024-03-31"), save_format)
    assert len(client.requests) == requested
    np.testing.assert_allclose(pd.to_numeric(reloaded["adjustment_volume"]), pd.to_numeric(assembled["adjustment_volume"]))


def test_legacy_monthly_main_cache_is_repaired_on_read(tmp_path: Path) -> None:
    raw = _BulkPricesClient(_prices()).raw
    legacy = pd.concat([normalize_jquants_prices(month) for _, month in raw.groupby(raw["date"].dt.to_period("M"))], ignore_index=True)
    # Existing caches have no provenance column and are reused without running
    # the cache builder. They must still get the same cross-month correction.
    legacy = legacy.drop(columns=["adjustment_source"], errors="ignore")
    _cache(tmp_path, prices=legacy)
    _assert_twins_match(_attention(tmp_path, "2023-10-31"))


def test_supplied_adjustments_on_a_later_basis_are_preserved(tmp_path: Path) -> None:
    # An API cache can already include a split beyond its last stored date.
    prices = _prices()
    for col in ["open", "high", "low", "close"]:
        prices[f"adjustment_{col}"] /= 3.0
    prices["adjustment_volume"] *= 3.0
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=_cache(tmp_path, prices=prices)))
    out = provider.load_attention_inputs(CODES, DATES.min(), DATES.max())
    np.testing.assert_allclose(out["volume"], 6000.0)
    np.testing.assert_allclose(out["close"], 500.0 / 3.0)
    np.testing.assert_allclose(out["shares_outstanding"], 6_000_000)


def _post_period_financials(reported: float) -> pd.DataFrame:
    financials = _financials()
    first = financials["disclosed_date"].eq("2023-08-10")
    financials.loc[first, "disclosed_date"] = "2023-09-15"
    financials.loc[first & financials["code"].eq("1001"), "shares_outstanding"] = reported
    return financials


@pytest.mark.parametrize("reported,basis", [(1_000_000, "2023-06-30"), (2_000_000, "2023-09-15")])
@pytest.mark.parametrize("save_format", ["csv", "parquet"])
def test_explicit_share_basis_avoids_double_adjustment(tmp_path: Path, reported: float, basis: str, save_format: str) -> None:
    financials = _post_period_financials(reported)
    financials["shares_basis_date"] = basis
    # A further split outside the stored history is already in the supplied
    # adjusted prices; it still has to be applied exactly once to the shares.
    prices = _prices()
    prices["adjustment_volume"] *= 3.0
    for col in ["open", "high", "low", "close"]:
        prices[f"adjustment_{col}"] /= 3.0
    _cache(tmp_path, prices=prices, financials=financials)
    if save_format == "csv":
        (tmp_path / "financial_summary.parquet").unlink()
        financials.to_csv(tmp_path / "financial_summary.csv", index=False)
    detail = _attention(tmp_path, "2023-10-31")
    _assert_twins_match(detail)
    assert detail.loc["1001", "shares_outstanding"] == pytest.approx(6_000_000)


@pytest.mark.parametrize("reported", [1_000_000, 2_000_000])
@pytest.mark.parametrize("adjusted_columns", [True, False])
def test_unknown_share_basis_is_not_guessed_or_filled_from_cap(tmp_path: Path, reported: float, adjusted_columns: bool) -> None:
    prices = _prices()
    if not adjusted_columns:
        prices = _BulkPricesClient(prices).raw
    _cache(tmp_path, prices=prices, financials=_post_period_financials(reported))
    # The cap cache may itself be derived from the ambiguous financial shares.
    pd.DataFrame({"asof_date": ["2023-09-30"], "code": ["1001"], "market_cap": [reported * 500],
                  "shares_outstanding": [reported], "market_cap_source": ["fins_summary_shares_x_adjustment_close"]}).to_parquet(tmp_path / "market_cap.parquet", index=False)
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))
    out = provider.load_attention_inputs(CODES, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-10-31"))
    split = out.loc[out["code"].eq("1001")]
    assert split["shares_reported"].eq(reported).all()
    assert split["shares_outstanding"].isna().all()
    assert split["shares_to_adjusted"].isna().all()
    assert split["shares_basis_status"].eq("ambiguous").all()
    detail = _attention(tmp_path, "2023-10-31")
    assert detail.loc["1001", "status"] == "ambiguous_shares_basis"
    assert pd.isna(detail.loc["1001", "turnover"])
    assert pd.isna(detail.loc["1001", "attention_factor"])
    if adjusted_columns:
        assert detail.loc["1001", "avol"] == pytest.approx(detail.loc["1002", "avol"])
    # A later, unambiguous disclosure restores the calculation only once public.
    if adjusted_columns:
        _assert_twins_match(_attention(tmp_path, "2023-11-30"))


def test_unknown_basis_is_detected_without_daily_factors(tmp_path: Path) -> None:
    prices = _prices().drop(columns="adjustment_factor")
    detail = _attention(_cache(tmp_path, prices=prices, financials=_post_period_financials(2_000_000)), "2023-10-31")
    assert detail.loc["1001", "status"] == "ambiguous_shares_basis"
    assert pd.isna(detail.loc["1001", "attention_factor"])


@pytest.mark.parametrize("daily_factors", [True, False])
def test_rounded_volumes_do_not_look_like_a_split_before_disclosure(tmp_path: Path, daily_factors: bool) -> None:
    prices = _prices()
    # Rounding creates small daily ratio changes before the actual split.
    prices["adjustment_volume"] += np.arange(len(prices)) % 2 * 0.4
    if not daily_factors:
        prices = prices.drop(columns="adjustment_factor")
    detail = _attention(_cache(tmp_path, prices=prices), "2023-08-31")
    assert detail["status"].eq("ok").all()
    assert detail["turnover"].notna().all()


def test_offsetting_actions_before_disclosure_still_leave_basis_unknown(tmp_path: Path) -> None:
    prices = _prices()
    reverse = prices["code"].eq("1001") & prices["date"].ge("2023-09-08")
    prices.loc[reverse, "volume"] /= 2
    for col in ["open", "high", "low", "close"]:
        prices.loc[reverse, col] *= 2
    prices.loc[prices["code"].eq("1001") & prices["date"].eq("2023-09-08"), "adjustment_factor"] = 2.0
    prices = normalize_jquants_prices(_BulkPricesClient(prices).raw)
    detail = _attention(_cache(tmp_path, prices=prices, financials=_post_period_financials(1_000_000)), "2023-10-31")
    assert detail.loc["1001", "status"] == "ambiguous_shares_basis"
    assert pd.isna(detail.loc["1001", "turnover"])


@pytest.mark.parametrize("factor", ["attention", "dual_ma", "residual_momentum", "quality", "value", "behaviour"])
def test_old_factor_cache_is_not_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factor: str) -> None:
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=tmp_path))
    def lookup(name: str):
        return factor_cache.build_factor_cache_lookup(
            tmp_path / "factors", factor_name=name, anchor_date=pd.Timestamp("2023-10-31"),
            universe=CODES, factor_config={}, provider=provider,
        )

    result = {"minimal": pd.DataFrame({"code": ["1001"], "factor_value": [1.0]}), "summary": {}}
    with monkeypatch.context() as old:
        old.setattr(factor_cache, "FACTOR_CACHE_VERSIONS", {})
        for name in [factor, "reversal"]:
            factor_cache.save_factor_cache(lookup(name), result)
    assert factor_cache.load_factor_cache(lookup(factor)) is None
    assert factor_cache.load_factor_cache(lookup("reversal")) is not None
    factor_cache.save_factor_cache(lookup(factor), result)
    assert factor_cache.load_factor_cache(lookup(factor)) is not None


def test_statement_normalization_keeps_explicit_share_basis() -> None:
    raw = pd.DataFrame({"DiscDate": ["2023-09-15"], "Code": ["10010"], "CurPerEn": ["2023-06-30"],
                        "ShOutFY": ["2000000"], "shares_basis_date": ["2023-09-15"]})
    out = normalize_jquants_statements(raw)
    assert out["shares_basis_date"].tolist() == [pd.Timestamp("2023-09-15")]
    assert _fins_summary_usecol("shares_basis_date")
