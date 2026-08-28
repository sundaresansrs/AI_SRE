from __future__ import annotations

import csv
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from .business_loader import (
    TARGET_COLUMNS,
    _extract_business_timestamp,
    _fallback_fault_type,
    _normalize_business_frame,
    _repair_business_csv,
)
from .run_truth import is_anomaly_for, is_anomaly_for_batch, load_run_truth

SAMPLE_ROWS = 50_000
SOURCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gaia"
    / "MicroSS"
    / "business"
    / "extracted"
    / "business"
    / "business_table_2021-08.csv"
)


def _load_sample() -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as directory:
        source_prefix = Path(directory) / "business_prefix.csv"
        repaired_path = Path(directory) / "business_prefix_repaired.csv"
        with SOURCE_FILE.open("r", encoding="utf-8", errors="replace", newline="") as source:
            with source_prefix.open("w", encoding="utf-8", newline="") as prefix:
                for index, line in enumerate(source):
                    prefix.write(line)
                    if index >= 110_000:
                        break

        previous_progress = os.environ.get("GAIA_REPAIR_PROGRESS")
        os.environ["GAIA_REPAIR_PROGRESS"] = "0"
        try:
            _repair_business_csv(source_prefix, repaired_path)
        finally:
            if previous_progress is None:
                os.environ.pop("GAIA_REPAIR_PROGRESS", None)
            else:
                os.environ["GAIA_REPAIR_PROGRESS"] = previous_progress

        with repaired_path.open("r", encoding="utf-8", newline="") as repaired:
            rows = []
            reader = csv.DictReader(repaired)
            for row in reader:
                rows.append(row)
                if len(rows) == SAMPLE_ROWS:
                    break

    if len(rows) != SAMPLE_ROWS:
        raise RuntimeError(f"Expected {SAMPLE_ROWS} sample rows, found {len(rows)}")
    return pd.DataFrame(rows)


def _load_business_chunk_rowbyrow_reference(
    df: pd.DataFrame, path: Path, known_services: set[str]
) -> pd.DataFrame:
    if "datetime" in df.columns:
        date_timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        date_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"Expected a datetime-like column in business file: {path}")

    rows = []
    for dt, service, message in zip(date_timestamps, df["service"].astype(str), df["message"].fillna("")):
        ts = _extract_business_timestamp(message)
        if pd.isna(ts):
            ts = dt
        if pd.isna(ts):
            raise ValueError(f"Encountered null timestamps in business file: {path}")
        is_anomaly, fault_type = is_anomaly_for(service, ts)
        if not is_anomaly and str(service).strip() not in known_services:
            is_anomaly, fault_type = _fallback_fault_type(message)
        elif not is_anomaly:
            is_anomaly = False
            fault_type = "normal"
        rows.append(
            {
                "timestamp": ts,
                "service_name": service,
                "metric_name": "business_event",
                "value": 1.0,
                "is_anomaly": is_anomaly,
                "fault_type": fault_type,
            }
        )

    return pd.DataFrame(rows)[TARGET_COLUMNS]


def _services_with_overlapping_windows(truth: pd.DataFrame) -> set[str]:
    overlapping_services = set()
    for service_name, windows in truth.groupby("service_name", sort=False):
        ordered = windows.sort_values("start", kind="mergesort")
        previous_max_end = ordered["end"].cummax().shift()
        if (ordered["start"] <= previous_max_end).any():
            overlapping_services.add(str(service_name))
    return overlapping_services


def main() -> int:
    sample = _load_sample()
    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()

    started = time.perf_counter()
    reference = _load_business_chunk_rowbyrow_reference(sample, SOURCE_FILE, known_services)
    reference_seconds = time.perf_counter() - started

    started = time.perf_counter()
    vectorized = _normalize_business_frame(sample, SOURCE_FILE, known_services)
    vectorized_seconds = time.perf_counter() - started

    try:
        assert_frame_equal(reference, vectorized, check_dtype=True)
    except AssertionError as error:
        print(error)
        truth = truth.copy()
        overlapping_services = _services_with_overlapping_windows(truth)
        difference_mask = (
            (reference["is_anomaly"] != vectorized["is_anomaly"])
            | (reference["fault_type"] != vectorized["fault_type"])
        )
        differing = reference.index[difference_mask]
        print(f"differing_rows={len(differing)}")
        differing_services = reference.loc[difference_mask, "service_name"].astype(str)
        overlap_difference_mask = differing_services.isin(overlapping_services)
        print(f"overlap_window_differences={int(overlap_difference_mask.sum())}")
        print(f"non_overlap_window_differences={int((~overlap_difference_mask).sum())}")
        print(f"old_rowbyrow_seconds={reference_seconds:.6f}")
        print(f"new_vectorized_seconds={vectorized_seconds:.6f}")
        print(f"speedup={reference_seconds / vectorized_seconds:.2f}x")
        for index in differing:
            service_name = reference.at[index, "service_name"]
            print(
                f"service_name={service_name} "
                f"timestamp={reference.at[index, 'timestamp']} "
                f"old_fault_type={reference.at[index, 'fault_type']} "
                f"new_fault_type={vectorized.at[index, 'fault_type']} "
                f"overlapping_service={str(service_name) in overlapping_services}"
            )
        print("EQUIVALENCE TEST FAILED")
        return 1

    print(f"EQUIVALENCE TEST PASSED")
    print(f"row_count={len(sample)}")
    print(f"old_rowbyrow_seconds={reference_seconds:.6f}")
    print(f"new_vectorized_seconds={vectorized_seconds:.6f}")
    print(f"speedup={reference_seconds / vectorized_seconds:.2f}x")
    print(f"batch_lookup_rows={len(is_anomaly_for_batch(sample['service'].astype(str), pd.to_datetime(sample['datetime'])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())