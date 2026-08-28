from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .business_loader import load_business, load_business_chunks
from .metric_loader import load_metric
from .trace_loader import load_trace

TARGET_COLUMNS = ["timestamp", "service_name", "metric_name", "value", "is_anomaly", "fault_type"]
BUSINESS_CHUNKSIZE = 50_000
ARROW_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns")),
        pa.field("service_name", pa.string()),
        pa.field("metric_name", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("is_anomaly", pa.bool_()),
        pa.field("fault_type", pa.string()),
    ]
)
# Known non-numeric GAIA metadata CSV files to skip.
# An exhaustive scan across 100% of all 10,817 metric files confirmed that these
# two Redis server configuration path exports are the only files with non-numeric (empty)
# value columns across the entire metric dataset.
SKIP_NON_NUMERIC_METRIC_FILES = {
    "redis_0.0.0.3_redis_info_server_config_file_2021-07-01_2021-07-15.csv",
    "redis_0.0.0.3_redis_info_server_config_file_2021-07-15_2021-07-31.csv",
}


def _source_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*.csv") if path.is_file())


def _normalise_for_parquet(frame: pd.DataFrame, source_file: Path) -> pd.DataFrame | None:
    if list(frame.columns) != TARGET_COLUMNS:
        raise ValueError(f"Unexpected columns from {source_file}: {list(frame.columns)}")
    if frame.empty or len(frame) == 0:
        print(f"SKIPPED (empty): {source_file.name} - 0 rows after filtering", flush=True)
        return None

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise").astype("float64")
    if frame["timestamp"].isna().any():
        raise ValueError(f"Null timestamps were produced from {source_file}")
    if frame["is_anomaly"].isna().any():
        raise ValueError(f"Null anomaly flags were produced from {source_file}")
    frame["is_anomaly"] = frame["is_anomaly"].astype("bool")
    return frame


def merge_gaia(
    repo_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    source_root = repo_root / "data" / "raw" / "gaia" / "MicroSS"
    output_path = output_path or repo_root / "data" / "processed" / "gaia_unified.parquet"
    loaders = {
        "business": load_business,
        "metric": load_metric,
        "trace": load_trace,
    }
    source_counts = {source: 0 for source in loaders}
    anomaly_counts = {source: 0 for source in loaders}
    file_counts = {}
    total_rows = 0
    writer = None
    temporary_path = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{output_path.stem}_", suffix=".parquet", dir=output_path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        for source, loader in loaders.items():
            files = _source_files(source_root / source)
            if not files:
                raise FileNotFoundError(f"No CSV files found in {source_root / source}")
            file_counts[source] = len(files)
            print(f"{source}: {len(files)} CSV files", flush=True)

            for index, source_file in enumerate(files, start=1):
                if source == "metric" and source_file.name in SKIP_NON_NUMERIC_METRIC_FILES:
                    print(
                        f"metric file SKIPPED (non-numeric metadata, not telemetry): {source_file.name}",
                        flush=True,
                    )
                    continue

                print(f"{source} file begin index={index}/{len(files)} path={source_file}", flush=True)
                frames = load_business_chunks(source_file, chunksize=BUSINESS_CHUNKSIZE) if source == "business" else [loader(source_file)]
                file_rows = 0
                for frame in frames:
                    print(f"{source} frame received index={index} rows={len(frame)}", flush=True)
                    normalized_frame = _normalise_for_parquet(frame, source_file)
                    if normalized_frame is None:
                        print(f"Skipping empty file: {source_file.name}", flush=True)
                        continue
                    frame = normalized_frame
                    print(f"{source} frame normalized index={index} rows={len(frame)}", flush=True)
                    table = pa.Table.from_pandas(
                        frame[TARGET_COLUMNS], schema=ARROW_SCHEMA, preserve_index=False, safe=True
                    )
                    print(f"{source} arrow table ready index={index} rows={len(frame)}", flush=True)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_path, ARROW_SCHEMA)
                    writer.write_table(table)
                    print(f"{source} parquet write complete index={index} rows={len(frame)}", flush=True)

                    rows = len(frame)
                    file_rows += rows
                    source_counts[source] += rows
                    anomaly_counts[source] += int(frame["is_anomaly"].sum())
                    total_rows += rows

                if index == 1 or index == len(files) or index % 100 == 0:
                    print(f"  {index}/{len(files)} files, {file_rows} rows", flush=True)
                print(f"{source} file end index={index}/{len(files)} rows={file_rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if temporary_path is None or total_rows == 0:
        raise ValueError("No GAIA rows were written")
    os.replace(temporary_path, output_path)

    return {
        "output": str(output_path),
        "file_counts": file_counts,
        "row_counts": source_counts,
        "anomaly_counts": anomaly_counts,
        "total_rows": total_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all extracted GAIA CSVs into one Parquet file.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = merge_gaia(args.repo_root.resolve(), args.output.resolve() if args.output else None)

    report["anomaly_percentages"] = {
        source: (100 * report["anomaly_counts"][source] / report["row_counts"][source])
        for source in report["row_counts"]
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()