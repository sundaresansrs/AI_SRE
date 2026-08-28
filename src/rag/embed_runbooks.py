"""Generate sentence-transformer embeddings for chunked VOID runbooks."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "runbook_chunks.parquet"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "runbook_embeddings.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_ROWS = 34
EXPECTED_DIM = 384


def main() -> None:
    started_at = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    chunks = pd.read_parquet(INPUT_PATH)
    required_columns = {"chunk_id", "chunk_text"}
    missing_columns = required_columns - set(chunks.columns)
    if missing_columns:
        raise ValueError(f"Input is missing required columns: {sorted(missing_columns)}")
    if len(chunks) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} input rows, found {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    vectors = model.encode(
        chunks["chunk_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError(
            f"Expected embedding shape {(EXPECTED_ROWS, EXPECTED_DIM)}, found {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("At least one embedding contains NaN or infinite values")
    if np.any(np.all(vectors == 0, axis=1)):
        raise ValueError("At least one embedding is all zero")

    output = pd.DataFrame({
        "chunk_id": chunks["chunk_id"].tolist(),
        "embedding": vectors.tolist(),
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)

    persisted = pd.read_parquet(OUTPUT_PATH)
    persisted_vectors = np.asarray(persisted["embedding"].tolist(), dtype=np.float32)
    if len(persisted) != EXPECTED_ROWS or persisted_vectors.shape != (EXPECTED_ROWS, EXPECTED_DIM):
        raise ValueError("Persisted embedding output has an unexpected shape")
    if not np.isfinite(persisted_vectors).all() or np.any(np.all(persisted_vectors == 0, axis=1)):
        raise ValueError("Persisted output contains an invalid embedding")

    runtime_seconds = time.perf_counter() - started_at
    print(f"Model name: {MODEL_NAME}")
    print(f"Embedding dim: {EXPECTED_DIM}")
    print(f"Final array shape: {persisted_vectors.shape}")
    print(f"First 5 values of first embedding: {persisted_vectors[0][:5].tolist()}")
    print(f"Device used: {device}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Total runtime seconds: {runtime_seconds:.2f}")


if __name__ == "__main__":
    main()