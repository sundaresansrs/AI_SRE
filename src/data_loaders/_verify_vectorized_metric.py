from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from .metric_loader import _extract_service_metric, _read_gaia_csv, load_metric
from .run_truth import is_anomaly_for, is_anomaly_for_batch, load_run_truth

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]
SOURCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gaia"
    / "MicroSS"
    / "metric"
    / "extracted"
    / "metric"
    / "dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-08-01_2021-08-31.csv"
)


def load_metric_rowbyrow_reference(file_path: Path) -> pd.DataFrame:
    df = _read_gaia_csv(file_path)
    if set(df.columns) != {"timestamp", "value"}:
        unexpected = list(df.columns)
        raise ValueError(f"Unexpected metric columns for {file_path}: {unexpected}")

    service_name, metric_name = _extract_service_metric(file_path)
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Encountered null timestamps in metric file: {file_path}")

    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()

    rows = []
    for row_ts, value in zip(ts, pd.to_numeric(df["value"], errors="coerce")):
        if pd.isna(value):
            raise ValueError(f"Encountered non-numeric metric values in {file_path}")
        is_anomaly, fault_type = is_anomaly_for(service_name, row_ts)
        if not is_anomaly and str(service_name).strip() not in known_services:
            is_anomaly = False
            fault_type = "normal"
        elif not is_anomaly:
            is_anomaly = False
            fault_type = "normal"
        rows.append(
            {
                "timestamp": row_ts,
                "service_name": service_name,
                "metric_name": metric_name,
                "value": float(value),
                "is_anomaly": is_anomaly,
                "fault_type": fault_type,
            }
        )

    out = pd.DataFrame(rows)
    return out[TARGET_COLUMNS]


def verify_equivalence() -> bool:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Test metric file not found at: {SOURCE_FILE}")

    print(f"Testing equivalence on: {SOURCE_FILE.name}")

    # Warm cache
    load_run_truth()

    # Time old row-by-row
    print("Running OLD (row-by-row is_anomaly_for loop)...")
    t0 = time.perf_counter()
    df_old = load_metric_rowbyrow_reference(SOURCE_FILE)
    t_old = time.perf_counter() - t0
    print(f"OLD finished: {len(df_old)} rows in {t_old:.4f}s ({len(df_old)/t_old:.0f} rows/s)")

    # Time new vectorized
    print("Running NEW (vectorized is_anomaly_for_batch)...")
    t1 = time.perf_counter()
    df_new = load_metric(SOURCE_FILE)
    t_new = time.perf_counter() - t1
    print(f"NEW finished: {len(df_new)} rows in {t_new:.4f}s ({len(df_new)/t_new:.0f} rows/s)")

    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")

    # Assert exact frame equality
    assert_frame_equal(df_old, df_new, check_dtype=True)
    print("ASSERTION PASSED: df_old and df_new are identical (including column dtypes, values, and index)!")
    return True


if __name__ == "__main__":
    verify_equivalence()
