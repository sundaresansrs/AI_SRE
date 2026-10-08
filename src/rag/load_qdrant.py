"""Load embedded runbook chunks into a local Qdrant collection and smoke-test it."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "runbook_chunks.parquet"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "runbook_embeddings.parquet"
COLLECTION_NAME = "runbook_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    health = client.get_collections()
    print(f"Qdrant REST/API health check: OK ({len(health.collections)} collection(s) visible)")

    chunks = pd.read_parquet(CHUNKS_PATH)
    embeddings = pd.read_parquet(EMBEDDINGS_PATH)
    merged = chunks.merge(embeddings, on="chunk_id", how="inner", validate="one_to_one")
    expected_points = len(chunks)
    if expected_points == 0 or len(embeddings) != expected_points:
        raise ValueError("Chunks and embeddings must be non-empty and have the same number of rows")
    if len(merged) != expected_points:
        raise ValueError(f"Expected {expected_points} joined rows, found {len(merged)}")

    vectors = np.asarray(merged["embedding"].tolist(), dtype=np.float32)
    if vectors.shape != (expected_points, VECTOR_SIZE):
        raise ValueError(f"Expected vector shape {(expected_points, VECTOR_SIZE)}, found {vectors.shape}")
    if not np.isfinite(vectors).all() or np.any(np.all(vectors == 0, axis=1)):
        raise ValueError("Input embeddings contain NaN/infinite values or an all-zero vector")

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    print(f"Collection creation: OK ({COLLECTION_NAME}, size={VECTOR_SIZE}, distance=COSINE)")

    points = [
        models.PointStruct(
            id=index,
            vector=vector.tolist(),
            payload={
                "chunk_id": row.chunk_id,
                "source_file": row.source_file,
                "chunk_text": row.chunk_text,
            },
        )
        for index, row in merged.iterrows()
        for vector in [vectors[index]]
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)

    point_count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"Final point count: {point_count}")
    if point_count != expected_points:
        raise ValueError(f"Expected {expected_points} points after upsert, found {point_count}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    query_vector = model.encode(
        ["database connection timeout"],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0].tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3,
        with_payload=True,
    ).points

    print('\nTest query: "database connection timeout"')
    print(f"Model name: {MODEL_NAME}; device: {device}")
    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        preview = " ".join(str(payload.get("chunk_text", "")).split()[:15]) + "..."
        print(f"{rank}. id={result.id}, score={result.score:.6f}")
        print(f"   source_file={payload.get('source_file')}")
        print(f"   chunk_text_preview={preview}")


if __name__ == "__main__":
    main()