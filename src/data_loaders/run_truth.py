from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "gaia" / "MicroSS" / "run" / "extracted" / "run"


def _normalize_fault_type(value: str) -> str:
    cleaned = str(value).strip().strip("[]")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return cleaned.lower() if cleaned else "normal"


@lru_cache(maxsize=1)
def load_run_truth() -> pd.DataFrame:
    """Load the GAIA run/fault-injection log as the authoritative anomaly source.

    The run log records service-specific fault windows such as "[memory_anomalies] ... start at
    2021-07-01 11:44:26.882752 and lasts 600 seconds". We convert those into service + time-range
    windows used by the other loaders to derive is_anomaly and fault_type.
    """
    if not RUN_DIR.exists():
        return pd.DataFrame(columns=["service_name", "fault_type", "start", "end"])

    frames: list[pd.DataFrame] = []
    for path in sorted(RUN_DIR.glob("run_table_*.csv")):
        df = pd.read_csv(path, low_memory=False)
        if "service" not in df.columns or "message" not in df.columns:
            continue
        subset = df[["service", "message"]].copy()
        subset["service_name"] = subset["service"].astype(str).str.strip()
        subset["fault_type"] = "normal"
        subset["start"] = pd.NaT
        subset["end"] = pd.NaT

        for idx, row in subset.iterrows():
            msg = str(row["message"])
            tag_match = re.search(r"\[([^\]]+)\]", msg)
            if tag_match is None:
                continue
            fault_tag = tag_match.group(1).strip()
            if "normal memory freed" in msg.lower():
                continue
            time_match = re.search(r"start at\s+(.+?)\s+and lasts\s+([0-9.]+)\s+seconds", msg, flags=re.IGNORECASE)
            if time_match is None:
                continue
            start = pd.to_datetime(time_match.group(1), errors="coerce")
            duration = float(time_match.group(2))
            if pd.isna(start):
                continue
            end = start + pd.to_timedelta(duration, unit="s")
            subset.at[idx, "fault_type"] = _normalize_fault_type(fault_tag)
            subset.at[idx, "start"] = start
            subset.at[idx, "end"] = end

        valid = subset[subset["fault_type"] != "normal"].copy()
        if valid.empty:
            continue
        frames.append(valid[["service_name", "fault_type", "start", "end"]].copy())

    if not frames:
        return pd.DataFrame(columns=["service_name", "fault_type", "start", "end"])

    truth = pd.concat(frames, ignore_index=True)
    truth = truth.dropna(subset=["start", "end"])
    return truth


def is_anomaly_for(service_name: str, timestamp: pd.Timestamp) -> tuple[bool, str]:
    """Return the active fault using the most recently started window on overlap."""
    truth = load_run_truth()
    if truth.empty:
        return False, "normal"
    service_rows = truth[truth["service_name"] == str(service_name)]
    if service_rows.empty:
        return False, "normal"

    match = service_rows[(timestamp >= service_rows["start"]) & (timestamp <= service_rows["end"])]
    if match.empty:
        return False, "normal"

    fault_type = match.sort_values("start", kind="mergesort").iloc[-1]["fault_type"]
    return True, fault_type


def is_anomaly_for_batch(services: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    """Vectorized lookup using the most recently started active window on overlap.

    When multiple fault windows are active for a service, the window with the latest
    start time wins. This makes nested or overlapping faults prefer the more specific
    fault over a longer enclosing window.
    """
    if len(services) != len(timestamps):
        raise ValueError("services and timestamps must have the same length")
    if not services.index.equals(timestamps.index):
        raise ValueError("services and timestamps must have the same index")

    result = pd.DataFrame(
        {"is_anomaly": False, "fault_type": "normal"},
        index=services.index,
    )
    truth = load_run_truth()
    if truth.empty or services.empty:
        return result

    left = pd.DataFrame(
        {
            "_position": range(len(services)),
            "service_name": services.astype(str).to_numpy(),
            "timestamp": pd.to_datetime(timestamps, errors="coerce").to_numpy(),
        }
    )
    valid_left = left[left["timestamp"].notna()].copy()
    if valid_left.empty:
        return result

    right = truth[["service_name", "fault_type", "start", "end"]].copy()
    right["service_name"] = right["service_name"].astype(str)
    right["start"] = pd.to_datetime(right["start"], errors="coerce")
    right["end"] = pd.to_datetime(right["end"], errors="coerce") + pd.to_timedelta(1, unit="ns")
    right = right.dropna(subset=["start", "end"])
    if right.empty:
        return result

    # Convert overlapping truth windows into disjoint segments whose winner is the
    # latest-started active window. This lets merge_asof fall back to an enclosing
    # window when a more recent nested window has already ended.
    segments = []
    for service_name, windows in right.groupby("service_name", sort=False):
        boundaries = sorted(set(windows["start"]) | set(windows["end"]))
        for segment_start, segment_end in zip(boundaries, boundaries[1:]):
            active = windows[(windows["start"] <= segment_start) & (windows["end"] > segment_start)]
            if active.empty:
                continue
            winner = active.sort_values("start", kind="mergesort").iloc[-1]
            segments.append(
                {
                    "service_name": service_name,
                    "fault_type": winner["fault_type"],
                    "start": segment_start,
                    "end": segment_end,
                }
            )
    right = pd.DataFrame(segments, columns=["service_name", "fault_type", "start", "end"])
    if right.empty:
        return result

    valid_left = valid_left.sort_values(["timestamp", "service_name"], kind="mergesort")
    right = right.sort_values(["start", "service_name"], kind="mergesort")
    matched = pd.merge_asof(
        valid_left,
        right,
        left_on="timestamp",
        right_on="start",
        by="service_name",
        direction="backward",
    )
    matched_mask = matched["end"].notna() & (matched["timestamp"] < matched["end"])
    matched = matched.loc[matched_mask]
    if matched.empty:
        return result

    result.iloc[matched["_position"].to_numpy(), 0] = True
    result.iloc[matched["_position"].to_numpy(), 1] = matched["fault_type"].to_numpy()
    return result
