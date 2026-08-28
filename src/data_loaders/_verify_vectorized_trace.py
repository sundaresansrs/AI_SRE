from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from .run_truth import is_anomaly_for, load_run_truth
from .trace_loader import TARGET_COLUMNS, _read_gaia_csv, load_trace

SAMPLE_ROWS = 100_000
SOURCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gaia"
    / "MicroSS"
    / "trace"
    / "extracted"
    / "trace"
    / "trace_table_dbservice1_2021-07.csv"
)


def load_trace_rowbyrow_reference(file_path: Path, nrows: int | None = None) -> pd.DataFrame:
    df = _read_gaia_csv(file_path, nrows=nrows)
    required = {"timestamp", "service_name", "status_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace file missing required columns {sorted(missing)}: {file_path}")

    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()

    rows = []
    for ts, service, value in zip(
        pd.to_datetime(df["timestamp"], errors="coerce"),
        df["service_name"].astype(str),
        pd.to_numeric(df["status_code"], errors="coerce"),
    ):
        if pd.isna(ts):
            continue
        is_anomaly, fault_type = is_anomaly_for(service, ts)
        if not is_anomaly and str(service).strip() not in known_services and pd.notna(value) and value >= 400:
            is_anomaly = True
            fault_type = "http_error"
        elif not is_anomaly:
            is_anomaly = False
            fault_type = "normal"
        rows.append(
            {
                "timestamp": ts,
                "service_name": service,
                "metric_name": "request_status_code",
                "value": float(value),
                "is_anomaly": is_anomaly,
                "fault_type": fault_type,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No valid trace rows were produced from {file_path}")
    return out[TARGET_COLUMNS]


def verify_equivalence() -> bool:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Test trace file not found at: {SOURCE_FILE}")

    print(f"Testing equivalence on: {SOURCE_FILE.name} (first {SAMPLE_ROWS} rows)")

    # Warm cache
    load_run_truth()

    # Time old row-by-row
    print("Running OLD (row-by-row is_anomaly_for loop)...")
    t0 = time.perf_counter()
    df_old = load_trace_rowbyrow_reference(SOURCE_FILE, nrows=SAMPLE_ROWS)
    t_old = time.perf_counter() - t0
    print(f"OLD finished: {len(df_old)} rows in {t_old:.4f}s ({len(df_old)/t_old:.0f} rows/s)")

    # Time new vectorized
    print("Running NEW (vectorized is_anomaly_for_batch)...")
    t1 = time.perf_counter()
    df_new = load_trace(SOURCE_FILE, nrows=SAMPLE_ROWS)
    t_new = time.perf_counter() - t1
    print(f"NEW finished: {len(df_new)} rows in {t_new:.4f}s ({len(df_new)/t_new:.0f} rows/s)")

    speedup = t_old / t_new if t_new > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")

    # Assert exact frame equality
    try:
        assert_frame_equal(df_old, df_new, check_dtype=True)
        print("ASSERTION PASSED: df_old and df_new are identical (including column dtypes, values, and index)!")
        return True
    except AssertionError as err:
        print(f"ASSERTION FAILED: {err}")
        # Check differing rows for diagnosis
        diff_mask = (
            (df_old["timestamp"] != df_new["timestamp"])
            | (df_old["service_name"] != df_new["service_name"])
            | (df_old["metric_name"] != df_new["metric_name"])
            | (df_old["value"] != df_new["value"])
            | (df_old["is_anomaly"] != df_new["is_anomaly"])
            | (df_old["fault_type"] != df_new["fault_type"])
        )
        print(f"Number of differing rows: {diff_mask.sum()}")
        if diff_mask.any():
            print("Sample diff rows (OLD):")
            print(df_old[diff_mask].head(10))
            print("Sample diff rows (NEW):")
            print(df_new[diff_mask].head(10))
        raise


if __name__ == "__main__":
    verify_equivalence()
