from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data_loaders.run_truth import RUN_DIR, load_run_truth

SERVICES = {"webservice1", "webservice2"}
TAG_PATTERN = re.compile(r"\[([^\]]+)\]")
TIME_PATTERN = re.compile(
    r"start at\s+(.+?)\s+and lasts\s+([0-9.]+)\s+seconds",
    flags=re.IGNORECASE,
)


def format_timestamp(value: object) -> str:
    return pd.Timestamp(value).isoformat(sep=" ") if pd.notna(value) else "NaT"


def print_truth_windows(truth: pd.DataFrame) -> None:
    print("=== PARSED TRUTH WINDOWS ===")
    filtered = truth[truth["service_name"].isin(SERVICES)].sort_values(
        ["service_name", "start"], kind="mergesort"
    )
    for index, row in filtered.iterrows():
        duration = row["end"] - row["start"]
        print(
            f"truth_index={index} service_name={row['service_name']} "
            f"fault_type={row['fault_type']} start={format_timestamp(row['start'])} "
            f"end={format_timestamp(row['end'])} duration={duration}"
        )


def find_overlaps(truth: pd.DataFrame) -> dict[str, list[tuple[pd.Series, pd.Series, pd.Timedelta]]]:
    overlaps: dict[str, list[tuple[pd.Series, pd.Series, pd.Timedelta]]] = {}
    filtered = truth[truth["service_name"].isin(SERVICES)].sort_values(
        ["service_name", "start"], kind="mergesort"
    )
    for service_name, windows in filtered.groupby("service_name", sort=False):
        service_overlaps = []
        rows = [row for _, row in windows.iterrows()]
        for first, second in combinations(rows, 2):
            overlap_start = max(first["start"], second["start"])
            overlap_end = min(first["end"], second["end"])
            if overlap_start <= overlap_end:
                service_overlaps.append((first, second, overlap_end - overlap_start))
        overlaps[service_name] = service_overlaps
    return overlaps


def print_overlaps(overlaps: dict[str, list[tuple[pd.Series, pd.Series, pd.Timedelta]]]) -> None:
    print("=== OVERLAPPING WINDOW PAIRS ===")
    for service_name in sorted(SERVICES):
        pairs = overlaps.get(service_name, [])
        print(f"service_name={service_name} overlap_pair_count={len(pairs)}")
        for pair_number, (first, second, overlap_duration) in enumerate(pairs, start=1):
            first_duration = first["end"] - first["start"]
            second_duration = second["end"] - second["start"]
            print(
                f"pair={pair_number} "
                f"A[type={first['fault_type']} start={format_timestamp(first['start'])} "
                f"end={format_timestamp(first['end'])} duration={first_duration}] "
                f"B[type={second['fault_type']} start={format_timestamp(second['start'])} "
                f"end={format_timestamp(second['end'])} duration={second_duration}] "
                f"overlap_duration={overlap_duration}"
            )


def raw_matched_messages() -> None:
    print("=== RAW MATCHED RUN-TRUTH MESSAGES ===")
    total = 0
    for path in sorted(RUN_DIR.glob("run_table_*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if "service" not in frame.columns or "message" not in frame.columns:
            continue
        for row_number, row in frame.iterrows():
            service_name = str(row["service"]).strip()
            if service_name not in SERVICES:
                continue
            message = str(row["message"])
            tag_match = TAG_PATTERN.search(message)
            time_match = TIME_PATTERN.search(message)
            if tag_match is None or time_match is None:
                continue
            if "normal memory freed" in message.lower():
                continue
            start = pd.to_datetime(time_match.group(1), errors="coerce")
            if pd.isna(start):
                continue
            duration = float(time_match.group(2))
            total += 1
            print(
                f"file={path.name} raw_row={row_number + 2} service_name={service_name} "
                f"parsed_fault_tag={tag_match.group(1).strip()} "
                f"parsed_start={format_timestamp(start)} duration_seconds={duration} "
                f"message={message}"
            )
    print(f"raw_matched_message_count={total}")


def print_classification(
    truth: pd.DataFrame,
    overlaps: dict[str, list[tuple[pd.Series, pd.Series, pd.Timedelta]]],
) -> None:
    print("=== OVERLAP CLASSIFICATION ===")
    for service_name in sorted(SERVICES):
        pairs = overlaps.get(service_name, [])
        near_total = 0
        partial = 0
        other = 0
        for first, second, overlap_duration in pairs:
            first_duration = first["end"] - first["start"]
            second_duration = second["end"] - second["start"]
            overlap_ratio = overlap_duration / min(first_duration, second_duration)
            if overlap_ratio >= 0.95:
                near_total += 1
            elif overlap_duration > pd.Timedelta(0):
                partial += 1
            else:
                other += 1
        print(
            f"service_name={service_name} total_pairs={len(pairs)} "
            f"near_total_overlap_pairs={near_total} partial_overlap_pairs={partial} "
            f"other_pairs={other}"
        )
    print(
        "read=Near-total means overlap is at least 95% of the shorter window; "
        "partial means genuine positive-duration intersection below 95%. "
        "Inspect the printed durations and starts to distinguish sequential injections "
        "from duplicate or parsing-generated windows."
    )


def main() -> None:
    truth = load_run_truth()
    overlaps = find_overlaps(truth)
    print_truth_windows(truth)
    print_overlaps(overlaps)
    raw_matched_messages()
    print_classification(truth, overlaps)


if __name__ == "__main__":
    main()
