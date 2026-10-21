from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from historical_data import JpxHistoricalProvider, LocalDataPaths


SYNTHETIC_DATA = Path(__file__).resolve().parents[1] / "examples" / "synthetic_data"
CODES = pd.DataFrame({"code": ["1001", "1002"]})
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-29")


def _csv_cache(tmp_path: Path) -> Path:
    for name in ["prices", "market_cap"]:
        shutil.copy(SYNTHETIC_DATA / f"{name}.csv", tmp_path / f"{name}.csv")
    return tmp_path


def _attention_inputs(data_dir: Path) -> pd.DataFrame:
    provider = JpxHistoricalProvider(LocalDataPaths(data_dir=data_dir))
    return provider.load_attention_inputs(CODES, START, END)


def test_csv_cache_derives_shares_from_market_cap(tmp_path: Path) -> None:
    out = _attention_inputs(_csv_cache(tmp_path))
    assert pd.api.types.is_float_dtype(out["shares_outstanding"])
    close = pd.to_numeric(out["close"])
    np.testing.assert_allclose(out["shares_outstanding"] * close, out["market_cap"])


def test_csv_and_parquet_caches_agree(tmp_path: Path) -> None:
    csv_dir, parquet_dir = tmp_path / "csv", tmp_path / "parquet"
    csv_dir.mkdir()
    parquet_dir.mkdir()
    _csv_cache(csv_dir)
    for name in ["prices", "market_cap"]:
        frame = pd.read_csv(csv_dir / f"{name}.csv", dtype={"code": str})
        frame.to_parquet(parquet_dir / f"{name}.parquet", index=False)
    from_csv = _attention_inputs(csv_dir)
    from_parquet = _attention_inputs(parquet_dir)
    for col in ["market_cap", "shares_outstanding"]:
        np.testing.assert_allclose(from_csv[col], from_parquet[col])


def test_csv_shares_outstanding_column_is_numeric(tmp_path: Path) -> None:
    cache = _csv_cache(tmp_path)
    caps = pd.read_csv(cache / "market_cap.csv", dtype=str)
    caps["shares_outstanding"] = "1000000"
    caps.to_csv(cache / "market_cap.csv", index=False)
    out = _attention_inputs(cache)
    assert out["shares_outstanding"].eq(1_000_000.0).all()
