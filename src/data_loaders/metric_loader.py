from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import pandas as pd

from .run_truth import is_anomaly_for, is_anomaly_for_batch, load_run_truth

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]


def _read_gaia_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Repair GAIA metric rows where the closing quote is emitted on its own line."""
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


def _extract_service_metric(file_path: str | Path) -> tuple[str, str]:
    name = Path(file_path).stem
    match = re.match(
        r"^(?P<service>.+?)_(?:\d+(?:\.\d+)+)_(?P<metric>.+)_(?:\d{4}-\d{2}-\d{2})_(?:\d{4}-\d{2}-\d{2})$",
        name,
    )
    if match is None:
        raise ValueError(f"Could not parse metric file name: {file_path}")
    service = match.group("service")
    metric = match.group("metric")
    return service, metric


def load_metric(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a GAIA metric CSV and normalize it to the standard schema.

    Metric rows have both a service name and a timestamp, so they can be matched to the GAIA
    run-log fault windows exactly the same way as business and trace rows.
    """
    path = Path(file_path)
    df = _read_gaia_csv(path, nrows=nrows)
    if set(df.columns) != {"timestamp", "value"}:
        unexpected = list(df.columns)
        raise ValueError(f"Unexpected metric columns for {path}: {unexpected}")

    service_name, metric_name = _extract_service_metric(path)
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Encountered null timestamps in metric file: {path}")

    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()

    values = pd.to_numeric(df["value"], errors="coerce")
    if values.isna().any():
        raise ValueError(f"Encountered non-numeric metric values in {path}")

    services = pd.Series(service_name, index=df.index, dtype=object)
    anomaly_df = is_anomaly_for_batch(services, ts)

    is_anomaly = anomaly_df["is_anomaly"].astype(bool)
    fault_type = anomaly_df["fault_type"].astype(object)
    if str(service_name).strip() not in known_services:
        is_anomaly = pd.Series(False, index=df.index, dtype=bool)
        fault_type = pd.Series("normal", index=df.index, dtype=object)

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "service_name": services,
            "metric_name": pd.Series(metric_name, index=df.index, dtype=object),
            "value": values.astype("float64"),
            "is_anomaly": is_anomaly,
            "fault_type": fault_type,
        },
        columns=TARGET_COLUMNS,
    )
    if nrows is not None:
        return out.head(nrows)
    return out
