"""Measure how far a historical data provider satisfies ``run_backtest()``.

The backtest engine duck-types its provider: ``JpxHistoricalProvider`` is the
reference implementation, but any object exposing the same methods and table
shapes can drive a backtest.  This module lists those expectations in one place
so an alternative provider can be measured before a full run.

Unlike ``historical_backtest_engine.validate_provider_readiness`` (a fail-fast
guard that the reference caches are non-empty), this check reports every gap at
once: interface checks need no data, and data probes run when an anchor date is
given.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

from historical_data import SCHEMAS, normalize_code
from topix_pool_runtime import RunReport


STANDARD_FACTORS = ("quality", "value", "dual_ma", "reversal", "attention", "behaviour", "residual_momentum")

# Provider methods each factor runtime receives from historical_backtest_engine._factor_config.
FACTOR_LOADERS: dict[str, tuple[str, ...]] = {
    "quality": ("load_fundamental_panel",),
    "value": ("load_fundamental_panel",),
    "dual_ma": ("load_price_history",),
    "reversal": ("load_price_history",),
    "attention": ("load_attention_inputs",),
    "behaviour": ("load_behaviour_inputs",),
    "residual_momentum": ("load_close_prices",),
}

_ARG = None
_LOADER_CALL = ((_ARG,) * 5, {})

# Representative call shapes taken from the engine, the screen and the factor
# runtimes; a provider method must bind every one of them.
_CALL_SHAPES: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
    "table": [(("prices",), {"copy": False}), (("universe",), {"required": False, "copy": False})],
    "get_universe": [((_ARG,) * 2, {})],
    "get_sector_map": [((_ARG,) * 3, {})],
    "get_market_caps": [((_ARG,) * 3, {})],
    "price_frame": [((_ARG,) * 3, {}), ((_ARG,), {"start": _ARG, "end": _ARG})],
    "load_index_close": [((_ARG,) * 4, {})],
    "load_price_history": [_LOADER_CALL],
    "load_attention_inputs": [_LOADER_CALL],
    "load_behaviour_inputs": [_LOADER_CALL],
    "load_fundamental_panel": [((_ARG,) * 2, {})],
    "load_close_prices": [((_ARG,) * 4, {})],
}

_CORE_METHODS: dict[str, tuple[str, str]] = {
    "table": ("required", "rebalance calendar, execution lag, corporate actions, regime"),
    "get_universe": ("required", "screen"),
    "get_sector_map": ("required", "screen, sector labels"),
    "get_market_caps": ("required", "screen"),
    "price_frame": ("required", "screen, execution prices, stop loss"),
    "load_index_close": ("recommended", "benchmark NAV, market risk gate"),
}

MARKET_CAP_COLUMNS = ("code", "market_cap_jpy", "market_cap_billion_jpy", "market_cap_rank")
PRICE_FRAME_COLUMNS = ("date", "code", "open", "close", "volume")
RECOMMENDED_PRICE_COLUMNS = ("adjustment_open", "adjustment_close", "adjustment_factor", "trading_value")


@dataclass
class CheckResult:
    check: str
    level: str  # "required" or "recommended"
    status: str  # "pass", "fail" or "skip"
    used_by: str
    detail: str = ""


@dataclass
class ReadinessReport:
    provider_name: str
    anchor_date: str | None
    factors: tuple[str, ...]
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when no required check failed or was skipped."""
        return not self.blocking()

    def blocking(self) -> list[CheckResult]:
        return [c for c in self.checks if c.level == "required" and c.status != "pass"]

    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.level == "recommended" and c.status != "pass"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks], columns=["check", "level", "status", "used_by", "detail"])

    def summary(self) -> str:
        counts = self.to_frame().groupby(["level", "status"]).size().to_dict() if self.checks else {}
        lines = [
            f"provider={self.provider_name} anchor_date={self.anchor_date} ready={self.ready}",
            "counts: " + ", ".join(f"{level}/{status}={n}" for (level, status), n in sorted(counts.items())),
        ]
        for title, rows in (("blocking", self.blocking()), ("warnings", self.warnings())):
            for c in rows:
                lines.append(f"[{title}] {c.check} ({c.status}; {c.used_by}): {c.detail}")
        return "\n".join(lines)


def _error_text(exc: Exception, limit: int = 200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _required_methods(factors: Iterable[str]) -> dict[str, tuple[str, str]]:
    methods = dict(_CORE_METHODS)
    for factor in factors:
        for name in FACTOR_LOADERS.get(factor, ()):
            used_by = methods.get(name, ("", ""))[1]
            label = f"factor:{factor}"
            methods[name] = ("required", f"{used_by}, {label}" if used_by else label)
    return methods


def _check_interface(provider: Any, methods: dict[str, tuple[str, str]]) -> list[CheckResult]:
    historical = getattr(provider, "supports_historical", False) is True
    results = [
        CheckResult(
            "attr:supports_historical",
            "required",
            "pass" if historical else "fail",
            "screen",
            "" if historical else "the screen reads price_frame() only when supports_historical is True",
        )
    ]
    for name, (level, used_by) in methods.items():
        method = getattr(provider, name, None)
        check = f"method:{name}"
        if not callable(method):
            results.append(CheckResult(check, level, "fail", used_by, "method is missing"))
            continue
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            results.append(CheckResult(check, level, "pass", used_by, "signature not inspectable"))
            continue
        mismatched = []
        for args, kwargs in _CALL_SHAPES[name]:
            try:
                signature.bind(*args, **kwargs)
            except TypeError as exc:
                shape = ", ".join(["_"] * len(args) + [f"{k}=_" for k in kwargs])
                mismatched.append(f"{name}({shape}): {exc}")
        status = "fail" if mismatched else "pass"
        results.append(CheckResult(check, level, status, used_by, "; ".join(mismatched)))
    return results


def _missing(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return sorted(set(columns) - set(frame.columns))


def _probe_prices_table(provider: Any, asof: pd.Timestamp) -> list[CheckResult]:
    used_by = _CORE_METHODS["table"][1]
    try:
        prices = provider.table("prices", copy=False)
    except Exception as exc:
        return [CheckResult("table:prices", "required", "fail", used_by, _error_text(exc))]
    missing = _missing(SCHEMAS["prices"], prices)
    if missing:
        return [CheckResult("table:prices", "required", "fail", used_by, f"missing columns {missing}")]
    dates = pd.to_datetime(prices["date"], errors="coerce")
    rows = int(dates.le(asof).sum())
    results = [
        CheckResult(
            "table:prices",
            "required",
            "pass" if rows else "fail",
            used_by,
            f"{rows} rows on or before anchor date, latest={dates.max().date() if dates.notna().any() else None}",
        )
    ]
    missing_recommended = _missing(RECOMMENDED_PRICE_COLUMNS, prices)
    results.append(
        CheckResult(
            "table:prices:recommended_columns",
            "recommended",
            "fail" if missing_recommended else "pass",
            "corporate actions, adjusted execution prices, regime turnover",
            f"missing {missing_recommended}; engine falls back to raw open/close and close*volume" if missing_recommended else "",
        )
    )
    return results


def _probe_universe_table(provider: Any) -> CheckResult:
    used_by = "regime active universe"
    try:
        universe = provider.table("universe", required=False, copy=False)
    except Exception as exc:
        return CheckResult("table:universe", "recommended", "fail", used_by, _error_text(exc))
    missing = _missing(SCHEMAS["universe"], universe)
    if universe.empty or missing:
        return CheckResult("table:universe", "recommended", "fail", used_by, f"empty or missing columns {missing}; regime uses every code")
    return CheckResult("table:universe", "recommended", "pass", used_by)


def _probe_frame(check: str, level: str, used_by: str, call: Any, columns: Iterable[str] = ()) -> CheckResult:
    try:
        value = call()
    except Exception as exc:
        return CheckResult(check, level, "fail", used_by, _error_text(exc))
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        return CheckResult(check, level, "fail", used_by, f"returned {type(value).__name__}, expected DataFrame/Series")
    if value.empty:
        return CheckResult(check, level, "fail", used_by, "returned no rows")
    missing = _missing(columns, value) if isinstance(value, pd.DataFrame) else []
    if missing:
        return CheckResult(check, level, "fail", used_by, f"missing columns {missing}")
    detail = f"{len(value)} rows"
    if isinstance(value, pd.DataFrame):
        # Placeholder loaders may return the requested codes with no values.
        value_cols = [col for col in value.columns if col not in {"code", "date", "asof_date"}]
        empty_cols = [col for col in value_cols if value[col].isna().all()]
        if empty_cols:
            detail += f"; {len(empty_cols)}/{len(value_cols)} value columns are all-null, e.g. {empty_cols[:5]}"
    return CheckResult(check, level, "pass", used_by, detail)


def _probe_data(
    provider: Any,
    asof: pd.Timestamp,
    factors: tuple[str, ...],
    methods: dict[str, tuple[str, str]],
    index_code: str,
    sample_size: int,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    report = RunReport(mode="provider_readiness", anchor_date=str(asof.date()))
    callable_methods = {name for name in methods if callable(getattr(provider, name, None))}

    def skip(check: str, level: str, used_by: str, reason: str) -> None:
        results.append(CheckResult(check, level, "skip", used_by, reason))

    if "table" in callable_methods:
        results.extend(_probe_prices_table(provider, asof))
        results.append(_probe_universe_table(provider))
    else:
        skip("table:prices", "required", _CORE_METHODS["table"][1], "table() is missing")

    codes: list[str] = []
    if "get_universe" in callable_methods:
        try:
            codes = list(provider.get_universe(asof, report))
        except Exception as exc:
            results.append(CheckResult("data:get_universe", "required", "fail", "screen", _error_text(exc)))
        else:
            results.append(
                CheckResult("data:get_universe", "required", "pass" if codes else "fail", "screen", f"{len(codes)} codes")
            )
            unnormalized = [c for c in codes if normalize_code(c) != c]
            results.append(
                CheckResult(
                    "data:code_format",
                    "required",
                    "fail" if unnormalized else "pass",
                    "every code join (engine normalizes to 4 characters)",
                    f"{len(unnormalized)} codes differ from normalize_code(), e.g. {unnormalized[:5]}" if unnormalized else "",
                )
            )
    sample = codes[:sample_size]
    if not sample:
        for name in ("get_sector_map", "get_market_caps", "price_frame"):
            skip(f"data:{name}", "required", _CORE_METHODS[name][1], "no universe codes to probe with")
    else:
        if "get_sector_map" in callable_methods:
            try:
                sectors = provider.get_sector_map(sample, asof, report)
            except Exception as exc:
                results.append(CheckResult("data:get_sector_map", "required", "fail", "screen", _error_text(exc)))
            else:
                covered = sum(1 for code in sample if code in (sectors or {}))
                results.append(
                    CheckResult(
                        "data:get_sector_map",
                        "required",
                        "pass" if covered else "fail",
                        "screen, sector labels",
                        f"{covered}/{len(sample)} sample codes have a sector",
                    )
                )
        if "get_market_caps" in callable_methods:
            result = _probe_frame(
                "data:get_market_caps",
                "required",
                "screen",
                lambda: provider.get_market_caps(sample, asof, report),
                MARKET_CAP_COLUMNS,
            )
            results.append(result)
        if "price_frame" in callable_methods:
            result = _probe_frame(
                "data:price_frame",
                "required",
                _CORE_METHODS["price_frame"][1],
                lambda: provider.price_frame(sample, asof - pd.Timedelta(days=60), asof),
                PRICE_FRAME_COLUMNS,
            )
            results.append(result)

    basis = getattr(provider, "market_cap_basis", "market_cap")
    results.append(
        CheckResult(
            "attr:market_cap_basis",
            "recommended",
            "pass" if basis == "market_cap" else "fail",
            "screen market-cap rules, size neutralization",
            "" if basis == "market_cap" else f"market caps are a proxy ({basis}); absolute JPY thresholds are not comparable",
        )
    )

    if "load_index_close" in callable_methods:
        result = _probe_frame(
            f"data:load_index_close:{index_code}",
            "recommended",
            _CORE_METHODS["load_index_close"][1],
            lambda: provider.load_index_close(index_code, asof - pd.Timedelta(days=60), asof, {}),
        )
        results.append(result)

    universe_frame = pd.DataFrame({"code": sample})
    config = {"asof_date": asof, "rebalance_date": asof}
    start = asof - pd.Timedelta(days=120)
    loader_calls = {
        "load_price_history": lambda: provider.load_price_history(universe_frame, start, asof, None, config),
        "load_attention_inputs": lambda: provider.load_attention_inputs(universe_frame, start, asof, None, config),
        "load_behaviour_inputs": lambda: provider.load_behaviour_inputs(universe_frame, start, asof, None, config),
        "load_fundamental_panel": lambda: provider.load_fundamental_panel(universe_frame, config),
        "load_close_prices": lambda: provider.load_close_prices(sample, start, asof, config),
    }
    probed: set[str] = set()
    for factor in factors:
        for name in FACTOR_LOADERS.get(factor, ()):
            if name in probed:
                continue
            probed.add(name)
            level, used_by = methods[name]
            if name not in callable_methods:
                skip(f"data:{name}", level, used_by, f"{name}() is missing")
            elif not sample:
                skip(f"data:{name}", level, used_by, "no universe codes to probe with")
            else:
                results.append(_probe_frame(f"data:{name}", level, used_by, loader_calls[name]))
    return results


def check_provider_readiness(
    provider: Any,
    anchor_date: str | pd.Timestamp | None = None,
    *,
    factors: Iterable[str] = STANDARD_FACTORS,
    index_code: str = "TOPIX",
    sample_size: int = 20,
) -> ReadinessReport:
    """Report which ``run_backtest()`` expectations a provider meets.

    ``factors`` limits factor-loader checks to the factors a backtest enables.
    Without ``anchor_date`` only the interface is checked and no data is read.
    """

    factor_tuple = tuple(factors)
    unknown = sorted(set(factor_tuple) - set(FACTOR_LOADERS))
    if unknown:
        raise ValueError(f"Unknown factors {unknown}; expected a subset of {list(STANDARD_FACTORS)}")
    methods = _required_methods(factor_tuple)
    asof = pd.Timestamp(anchor_date).normalize() if anchor_date is not None else None
    report = ReadinessReport(
        provider_name=str(getattr(provider, "provider_name", type(provider).__name__)),
        anchor_date=str(asof.date()) if asof is not None else None,
        factors=factor_tuple,
    )
    report.checks.extend(_check_interface(provider, methods))
    if asof is not None:
        report.checks.extend(_probe_data(provider, asof, factor_tuple, methods, index_code, sample_size))
    return report
