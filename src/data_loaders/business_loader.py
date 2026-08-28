from __future__ import annotations

import os
import re
import tempfile
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .run_truth import is_anomaly_for, is_anomaly_for_batch, load_run_truth

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]


def _repair_business_csv(path: Path, output_path: Path) -> None:
    progress = os.getenv("GAIA_REPAIR_PROGRESS", "1") == "1"
    line_count = 0
    input_quote_count = 0
    output_quote_count = 0
    standalone_quote_lines = 0
    started = pd.Timestamp.now().timestamp()
    prev_line = None

    if progress:
        print(f"repair start input={path} output={output_path}", flush=True)
    with open(output_path, "w", encoding="utf-8", newline="") as repaired:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as source:
            for line in source:
                line_count += 1
                input_quote_count += line.count('"')
                if line.strip() == '"':
                    standalone_quote_lines += 1
                    if prev_line is not None:
                        repaired_line = prev_line.rstrip("\r\n") + '"\n'
                        repaired.write(repaired_line)
                        output_quote_count += repaired_line.count('"')
                        prev_line = None
                    continue
                if prev_line is not None:
                    repaired.write(prev_line)
                    output_quote_count += prev_line.count('"')
                prev_line = line
                if progress and line_count % 100_000 == 0:
                    elapsed = pd.Timestamp.now().timestamp() - started
                    print(
                        f"repair lines={line_count:,} elapsed_s={elapsed:.1f} "
                        f"lines_per_sec={line_count / elapsed:,.0f}",
                        flush=True,
                    )
            if prev_line is not None:
                repaired.write(prev_line)
                output_quote_count += prev_line.count('"')

    if progress:
        elapsed = pd.Timestamp.now().timestamp() - started
        print(
            f"repair complete lines={line_count:,} elapsed_s={elapsed:.1f} "
            f"lines_per_sec={line_count / elapsed:,.0f} input_quotes={input_quote_count:,} "
            f"output_quotes={output_quote_count:,} standalone_quote_lines={standalone_quote_lines:,} "
            f"output_quotes_even={output_quote_count % 2 == 0}",
            flush=True,
        )


def _read_gaia_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Repair GAIA CSV rows where the closing quote is emitted on its own line.

    The real export occasionally emits a bare closing quote on a standalone line. We repair that
    pattern while streaming to avoid materializing the entire multi-million-row CSV in memory.
    """
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
        if temp_path is None:
            return pd.DataFrame()
        kwargs = {"low_memory": False}
        if nrows is not None:
            kwargs["nrows"] = nrows
        return pd.read_csv(temp_path, **kwargs)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def _iter_gaia_csv(path: Path, chunksize: int):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False) as tmp:
            temp_path = tmp.name
        print(f"business repair begin file={path}", flush=True)
        _repair_business_csv(path, Path(temp_path))
        print(f"business repair end file={path}", flush=True)
        with open(temp_path, "r", encoding="utf-8", newline="") as repaired:
            print(f"business csv parse begin file={path}", flush=True)
            reader = csv.DictReader(repaired)
            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) == chunksize:
                    print(f"business chunk parsed file={path} rows={len(rows)}", flush=True)
                    yield pd.DataFrame(rows)
                    rows = []
            if rows:
                print(f"business chunk parsed file={path} rows={len(rows)} final=true", flush=True)
                yield pd.DataFrame(rows)
            print(f"business csv parse end file={path}", flush=True)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def _fallback_fault_type(message: str) -> tuple[bool, str]:
    text = str(message).lower()
    if re.search(r"permission denied|permission failed|access denied|forbidden|unauthorized", text):
        return True, "permission_error"
    if re.search(r"retry failed|failed|error|exception|fatal|unable|reject", text):
        return True, "error"
    if "warning" in text:
        return True, "warning"
    if re.search(r"timeout|timed out", text):
        return True, "timeout"
    if "memory" in text:
        return True, "memory_pressure"
    return False, "normal"


def _extract_business_timestamp(message: str) -> pd.Timestamp | pd.NaT:
    text = str(message)
    match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[.,](\d{1,6}))?", text)
    if not match:
        return pd.NaT
    date_part = match.group(1)
    fraction = match.group(2) or ""
    if fraction:
        fraction = fraction.ljust(6, "0")
        return pd.to_datetime(f"{date_part}.{fraction}")
    return pd.to_datetime(date_part)


def _normalize_business_frame(df: pd.DataFrame, path: Path, known_services: set[str]) -> pd.DataFrame:
    if "datetime" in df.columns:
        date_timestamps = pd.to_datetime(df["datetime"], errors="coerce")
    elif "timestamp" in df.columns:
        date_timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError(f"Expected a datetime-like column in business file: {path}")

    messages = df["message"].fillna("").astype(str)
    services = df["service"].astype(str)
    extracted = messages.str.extract(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[.,](\d{1,6}))?",
        expand=True,
    )
    fractions = extracted[1].fillna("").str.ljust(6, "0")
    extracted_timestamps = pd.to_datetime(
        extracted[0] + np.where(extracted[1].notna(), "." + fractions, ""),
        errors="coerce",
    )
    timestamps = extracted_timestamps.fillna(date_timestamps)
    if timestamps.isna().any():
        raise ValueError(f"Encountered null timestamps in business file: {path}")

    anomaly_results = is_anomaly_for_batch(services, timestamps)

    permission_mask = messages.str.contains(
        r"permission denied|permission failed|access denied|forbidden|unauthorized",
        case=False,
        na=False,
        regex=True,
    )
    error_mask = messages.str.contains(
        r"retry failed|failed|error|exception|fatal|unable|reject",
        case=False,
        na=False,
        regex=True,
    )
    warning_mask = messages.str.contains("warning", case=False, na=False, regex=False)
    timeout_mask = messages.str.contains(r"timeout|timed out", case=False, na=False, regex=True)
    memory_mask = messages.str.contains("memory", case=False, na=False, regex=False)
    fallback_anomaly = np.select(
        [permission_mask, error_mask, warning_mask, timeout_mask, memory_mask],
        [True, True, True, True, True],
        default=False,
    ).astype(bool)
    fallback_fault_type = np.select(
        [permission_mask, error_mask, warning_mask, timeout_mask, memory_mask],
        ["permission_error", "error", "warning", "timeout", "memory_pressure"],
        default="normal",
    )

    unknown_service = ~services.str.strip().isin(known_services)
    use_fallback = ~anomaly_results["is_anomaly"] & unknown_service
    anomaly = anomaly_results["is_anomaly"].to_numpy(copy=True)
    fault_type = anomaly_results["fault_type"].astype(object).to_numpy(copy=True)
    anomaly[use_fallback.to_numpy()] = fallback_anomaly[use_fallback.to_numpy()]
    fault_type[use_fallback.to_numpy()] = fallback_fault_type[use_fallback.to_numpy()]
    fault_type[~anomaly_results["is_anomaly"].to_numpy() & ~unknown_service.to_numpy()] = "normal"

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "service_name": services,
            "metric_name": "business_event",
            "value": 1.0,
            "is_anomaly": anomaly,
            "fault_type": fault_type,
        },
        index=df.index,
    )[TARGET_COLUMNS]


def load_business(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a GAIA business CSV and normalize it to the standardized schema.

    Run-truth windows are used first when available, because the GAIA run log is the canonical
    source of injected fault intervals. A local heuristic fallback preserves compatibility with the
    repository's synthetic tests when no matching run-log window exists.
    """
    path = Path(file_path)
    df = _read_gaia_csv(path, nrows=nrows)

    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()
    out = _normalize_business_frame(df, path, known_services)
    if nrows is not None:
        return out.head(nrows)
    return out


def load_business_chunks(file_path: str | Path, chunksize: int = 100_000):
    path = Path(file_path)
    truth = load_run_truth()
    known_services = set(truth["service_name"].astype(str).str.strip()) if not truth.empty else set()
    for df in _iter_gaia_csv(path, chunksize):
        print(f"business normalize begin file={path} rows={len(df)}", flush=True)
        result = _normalize_business_frame(df, path, known_services)
        print(f"business normalize end file={path} rows={len(result)}", flush=True)
        yield result
