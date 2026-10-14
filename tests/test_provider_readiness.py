from __future__ import annotations

from pathlib import Path

import pytest

from historical_data import JpxHistoricalProvider, LocalDataPaths
from provider_readiness import check_provider_readiness


SYNTHETIC_DATA = Path(__file__).resolve().parents[1] / "examples" / "synthetic_data"
ANCHOR = "2024-03-29"
# The synthetic cache has no margin or financial_summary table, so only the
# price-driven factors can be fully satisfied.
PRICE_FACTORS = ("dual_ma", "reversal", "residual_momentum")


def _provider() -> JpxHistoricalProvider:
    return JpxHistoricalProvider(LocalDataPaths(data_dir=SYNTHETIC_DATA))


def _status(report, check: str) -> str:
    return {c.check: c.status for c in report.checks}[check]


def test_reference_provider_interface_is_ready() -> None:
    report = check_provider_readiness(_provider())
    assert report.ready, report.summary()
    assert report.anchor_date is None
    assert not any(c.check.startswith("data:") for c in report.checks)


def test_reference_provider_with_synthetic_data_is_ready_for_price_factors() -> None:
    report = check_provider_readiness(_provider(), ANCHOR, factors=PRICE_FACTORS)
    assert report.ready, report.summary()
    assert _status(report, "data:get_universe") == "pass"
    assert _status(report, "data:price_frame") == "pass"
    # Synthetic prices carry no adjustment columns: a warning, not a blocker.
    assert [c.check for c in report.warnings()] == ["table:prices:recommended_columns"]


def test_missing_tables_block_only_the_factors_that_need_them() -> None:
    report = check_provider_readiness(_provider(), ANCHOR, factors=("behaviour",))
    assert not report.ready
    assert [c.check for c in report.blocking()] == ["data:load_behaviour_inputs"]
    assert "margin" in report.blocking()[0].detail


class _NoTableProvider(JpxHistoricalProvider):
    table = None  # type: ignore[assignment]


def test_missing_method_is_blocking_and_skips_its_data_probe() -> None:
    report = check_provider_readiness(_NoTableProvider(LocalDataPaths(data_dir=SYNTHETIC_DATA)), ANCHOR, factors=())
    blocking = {c.check: c.status for c in report.blocking()}
    assert blocking["method:table"] == "fail"
    assert blocking["table:prices"] == "skip"


class _PositionalOnlyPriceFrame(JpxHistoricalProvider):
    def price_frame(self, codes, start_date, end_date):  # type: ignore[override]
        return super().price_frame(codes, start_date, end_date)


def test_signature_mismatch_with_engine_call_shape_is_reported() -> None:
    report = check_provider_readiness(_PositionalOnlyPriceFrame(LocalDataPaths(data_dir=SYNTHETIC_DATA)), factors=())
    check = next(c for c in report.checks if c.check == "method:price_frame")
    assert check.status == "fail"
    assert "start=_" in check.detail


class _FiveCharCodeProvider(JpxHistoricalProvider):
    def get_universe(self, asof_date, report):  # type: ignore[override]
        return [f"{code}0" for code in super().get_universe(asof_date, report)]


def test_unnormalized_codes_are_blocking() -> None:
    report = check_provider_readiness(_FiveCharCodeProvider(LocalDataPaths(data_dir=SYNTHETIC_DATA)), ANCHOR, factors=())
    assert _status(report, "data:code_format") == "fail"
    assert not report.ready


class _ProxyMarketCapProvider(JpxHistoricalProvider):
    market_cap_basis = "trading_value_proxy"


def test_market_cap_proxy_is_a_warning() -> None:
    report = check_provider_readiness(_ProxyMarketCapProvider(LocalDataPaths(data_dir=SYNTHETIC_DATA)), ANCHOR, factors=PRICE_FACTORS)
    assert report.ready
    assert "attr:market_cap_basis" in [c.check for c in report.warnings()]


def test_report_frame_and_summary() -> None:
    report = check_provider_readiness(_provider(), ANCHOR, factors=PRICE_FACTORS)
    frame = report.to_frame()
    assert list(frame.columns) == ["check", "level", "status", "used_by", "detail"]
    assert set(frame["status"]) <= {"pass", "fail", "skip"}
    assert "ready=True" in report.summary()


def test_unknown_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown factors"):
        check_provider_readiness(_provider(), factors=("momentum",))
