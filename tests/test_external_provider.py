from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import local_backtest_runner as runner
from local_backtest_runner import TeeLogger, _external_provider_spec, load_external_provider, parse_provider_args


PROVIDER_SOURCE = '''
from dataclasses import dataclass, field

import pandas as pd

CLOSED = []


@dataclass
class RecordingProvider:
    db_path: str
    table_start: str | None = None
    table_end: str | None = None
    provider_name: str = "recording"

    def table(self, name, *, required=True, copy=True):
        dates = pd.bdate_range("2024-01-02", "2024-06-28")
        return pd.DataFrame({"date": dates, "code": "1001", "close": 100.0})

    def close(self):
        CLOSED.append(self.db_path)
'''

REQUIRED = {"request_start": pd.Timestamp("2023-01-01"), "request_end": pd.Timestamp("2024-06-30")}


@pytest.fixture
def provider_file(tmp_path: Path) -> Path:
    path = tmp_path / "private_provider.py"
    path.write_text(PROVIDER_SOURCE, encoding="utf-8")
    return path


def test_provider_args_are_key_value_pairs() -> None:
    assert parse_provider_args(["db_path=/data/x.duckdb", "query=a=b"]) == {"db_path": "/data/x.duckdb", "query": "a=b"}
    with pytest.raises(ValueError):
        parse_provider_args(["db_path"])


def test_cli_options_override_the_config_provider_section() -> None:
    config = {"provider": {"module": "cfg.py:P", "args": {"db_path": "cfg.duckdb", "table_start": "2020-01-01"}}}
    assert _external_provider_spec(config, None, {"db_path": "cli.duckdb"}) == {
        "module": "cfg.py:P",
        "args": {"db_path": "cli.duckdb", "table_start": "2020-01-01"},
    }
    assert _external_provider_spec(config, "cli.py:Q", None)["module"] == "cli.py:Q"
    assert _external_provider_spec({}, None, None) is None
    with pytest.raises(ValueError):
        _external_provider_spec({}, None, {"db_path": "x"})


def test_load_from_a_file_fills_the_history_window(provider_file: Path) -> None:
    provider = load_external_provider(f"{provider_file}:RecordingProvider", {"db_path": "x.duckdb"}, required=REQUIRED, logger=TeeLogger())
    assert (provider.db_path, provider.table_start, provider.table_end) == ("x.duckdb", "2023-01-01", "2024-06-30")
    explicit = load_external_provider(
        f"{provider_file}:RecordingProvider", {"db_path": "x.duckdb", "table_start": "2022-01-01"}, required=REQUIRED, logger=TeeLogger()
    )
    assert explicit.table_start == "2022-01-01"


def test_load_from_an_importable_module(provider_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(provider_file.parent))
    monkeypatch.delitem(sys.modules, "private_provider", raising=False)
    provider = load_external_provider("private_provider:RecordingProvider", {"db_path": "y"}, required=REQUIRED, logger=TeeLogger())
    assert provider.db_path == "y"


@pytest.mark.parametrize(
    "spec,error",
    [("missing.py:RecordingProvider", FileNotFoundError), ("no_class_name", ValueError), ("{file}:Missing", AttributeError)],
)
def test_bad_provider_specs_fail_clearly(provider_file: Path, spec: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        load_external_provider(spec.format(file=provider_file), {}, required=REQUIRED, logger=TeeLogger())


def test_a_provider_that_is_not_ready_stops_the_run(provider_file: Path, tmp_path: Path) -> None:
    # RecordingProvider lacks the engine's loaders, so readiness blocks it.
    with pytest.raises(RuntimeError, match="not ready"):
        runner.run_local_backtest(
            start="2024-03-01",
            end="2024-05-31",
            api_file=tmp_path / "missing_api.txt",
            data_dir=tmp_path / "cache",
            runs_dir=tmp_path / "runs",
            provider_module=f"{provider_file}:RecordingProvider",
            provider_args={"db_path": "x.duckdb"},
        )
    assert not (tmp_path / "cache").exists()


def test_run_uses_the_external_provider_without_the_jquants_cache(
    provider_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(runner, "check_external_provider", lambda provider, required, bt_config, logger: calls.setdefault("checked", provider))
    monkeypatch.setattr(runner, "_prepare_local_cache_provider", lambda **kwargs: pytest.fail("J-Quants cache path used"))

    def fake_run_backtest(rebalance_dates, *, provider, rules, config, initial_capital):
        calls["provider"], calls["dates"] = provider, rebalance_dates
        return {}

    def fake_save(*, result, run_dir, config_used, required, cache_status):
        calls["config_used"], calls["cache_status"] = config_used, cache_status
        return {}

    monkeypatch.setattr(runner, "run_backtest", fake_run_backtest)
    monkeypatch.setattr(runner, "save_backtest_outputs", fake_save)
    runner.run_local_backtest(
        start="2024-03-01",
        end="2024-05-31",
        api_file=tmp_path / "missing_api.txt",
        data_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        provider_module=f"{provider_file}:RecordingProvider",
        provider_args={"db_path": "x.duckdb"},
    )
    provider = calls["provider"]
    assert calls["checked"] is provider
    assert [d.strftime("%Y-%m") for d in calls["dates"]] == ["2024-03", "2024-04", "2024-05"]
    assert calls["cache_status"]["data_source"] == "external_provider"
    assert calls["config_used"]["cli"]["data_dir"] is None
    assert calls["config_used"]["cli"]["provider_args"] == {"db_path": "x.duckdb"}
    assert sys.modules[type(provider).__module__].CLOSED == ["x.duckdb"]
    assert not (tmp_path / "cache").exists()
