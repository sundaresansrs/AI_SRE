from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .run_truth import is_anomaly_for, is_anomaly_for_batch, load_run_truth

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]


def _read_gaia_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Repair GAIA trace rows where the closing quote is emitted on its own line."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as tmp:
            temp_path = tmp.name
            prev_line = None
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as src:
                for line in src:
                    if line.strip() == '"':
                        if prev_line is not None:
                            tmp.write(prev_line.rstrip("\r\n") + '"\n')
                            prev_line = None
                        continue
                    if prev_line is not None:
                        tmp.write(prev_line)
                    prev_line = line
                if prev_line is not None:
                    tmp.write(prev_line)
        kwargs = {"low_memory": False}
        if nrows is not None:
            kwargs["nrows"] = nrows
        return pd.read_csv(temp_path, **kwargs)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_trace(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a GAIA trace CSV and normalize it to the standard schema.

    Run-log fault windows are authoritative when they exist, but we still fall back to the status
    code heuristic for cases like small unit tests that do not accompany the full GAIA run log.
    """
    path = Path(file_path)
    df = _read_gaia_csv(path, nrows=nrows)
    required = {"timestamp", "service_name", "status_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trace file missing required columns {sorted(missing)}: {path}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    valid_mask = ts.notna()
    if not valid_mask.any():
        raise ValueError(f"No valid trace rows were produced from {path}")

    df = df.loc[valid_mask].copy()
    ts = ts.loc[valid_mask]
    services = df["service_name"].astype(str)
    values = pd.to_numeric(df["status_code"], errors="coerce").astype("float64")

    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()
    is_known = services.str.strip().isin(known_services)

    anomaly_df = is_anomaly_for_batch(services, ts)
    truth_anomaly = anomaly_df["is_anomaly"].astype(bool)
    truth_fault_type = anomaly_df["fault_type"].astype(object)

    http_fallback_mask = (~truth_anomaly) & (~is_known) & (values.notna()) & (values >= 400)

    is_anomaly = truth_anomaly | http_fallback_mask
    fault_type = np.where(
        truth_anomaly,
        truth_fault_type,
        np.where(http_fallback_mask, "http_error", "normal"),
    )

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "service_name": services.astype(object),
            "metric_name": pd.Series("request_status_code", index=df.index, dtype=object),
            "value": values,
            "is_anomaly": is_anomaly.astype(bool),
            "fault_type": pd.Series(fault_type, index=df.index, dtype=object),
        },
        columns=TARGET_COLUMNS,
    ).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"No valid trace rows were produced from {path}")
    if nrows is not None:
        return out.head(nrows)
    return out
