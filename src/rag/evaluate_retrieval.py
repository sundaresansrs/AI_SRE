"""Evaluate top-k retrieval against a small hand-labeled runbook test set."""

import sys
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_NAME = "runbook_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
VECTOR_SIZE = 384

TEST_SET = [
    (
        "Kubernetes kube-proxy/kubelet version mismatch breaks networking",
        [
            "2025_circleci_kubelet-kubeproxy-version-mismatch_chunk0",
            "2025_github_k8s-upgrade-kube-proxy-iptables_chunk0",
        ],
    ),
    (
        "DNS configuration change breaks name resolution",
        [
            "2024_pagerduty_dns-config-change-container-cluster_chunk0",
            "2024_github_dns-cascading-outage_chunk0",
            "2025_aws_dynamodb-dns-enactor-race-condition_chunk0",
        ],
    ),
    (
        "Removing cache nodes overloads a downstream cluster",
        ["2022_slack_cache-node-removal-cascade_chunk0"],
    ),
    (
        "BGP prefix withdrawal ordering breaks routing",
        ["2022_cloudflare_bgp-prefix-ordering-outage_chunk0"],
    ),
    (
        "Schema migration locks a database table during queries",
        [
            "2017_gocardless_migration-lock-blocks-queries_chunk0",
            "2021_github_mysql-replica-deadlock-cascade_chunk0",
        ],
    ),
    (
        "Missing exponential backoff causes a retry storm",
        ["2013_spotify_cascading-failure-no-backoff_chunk0"],
    ),
    (
        "Thundering herd reconnect exhausts memory",
        ["2016_discord_thundering-herd-reconnect_chunk0"],
    ),
    (
        "Database capacity reduction removes headroom, causing an outage under load",
        [
            "2016_buildkite_db-capacity-downgrade-cascade_chunk0",
            "2020_duo_queue-overload-insufficient-capacity_chunk0",
        ],
    ),
    (
        "File descriptor limit crashes a database proxy",
        ["2022_github_proxysql-limitnofile-systemd-cap_chunk0"],
    ),
    (
        "Mistyped command removes far more servers than intended",
        ["2017_amazon_s3-typo-outage_chunk0"],
    ),
    (
        "Kubernetes upgrade removes network policies",
        ["2025_datadog_cilium-k8s-outage_chunk0"],
    ),
    (
        "New load balancer feature triggers a latent bug in downstream services",
        ["2022_google_load-balancer-latent-bug-europe_chunk0"],
    ),
]

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def retrieve(query: str, k: int = 5) -> list[tuple[str, float, str]]:
    """Return the top-k matching chunk ID, score, and source filename."""
    model = _get_model()
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    if len(query_vector) != VECTOR_SIZE:
        raise ValueError(f"Expected query vector size {VECTOR_SIZE}, found {len(query_vector)}")

    client = QdrantClient(url=QDRANT_URL)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=k,
        with_payload=True,
    ).points
    return [
        (
            str(result.payload.get("chunk_id")),
            float(result.score),
            str(result.payload.get("source_file")),
        )
        for result in results
    ]


def main() -> None:
    precisions: list[float] = []
    successes: list[int] = []
    missed_queries: list[str] = []

    print(f"Model: {MODEL_NAME}")
    print(f"Qdrant collection: {COLLECTION_NAME}")
    print(f"Test queries: {len(TEST_SET)}")

    for query, ground_truth in TEST_SET:
        retrieved = retrieve(query, k=5)
        truth = set(ground_truth)
        hits = [item for item in retrieved if item[0] in truth]
        misses = [item for item in retrieved if item[0] not in truth]
        precision = len(hits) / 5
        success = int(bool(hits))
        precisions.append(precision)
        successes.append(success)
        if not success:
            missed_queries.append(query)

        print(f'\nQuery: "{query}"')
        print(f"Ground truth: {ground_truth}")
        print("Retrieved top 5:")
        for rank, (chunk_id, score, source_file) in enumerate(retrieved, start=1):
            status = "HIT" if chunk_id in truth else "MISS"
            print(f"  {rank}. {chunk_id} | score={score:.6f} | source_file={source_file} | {status}")
        print(f"Hits: {[item[0] for item in hits]}")
        print(f"Misses: {[item[0] for item in misses]}")
        print(f"Precision@5: {precision:.4f}")
        print(f"Success@5: {success}")

    mean_precision = sum(precisions) / len(precisions)
    hit_rate = sum(successes) / len(successes)
    print("\n=== Aggregate Results ===")
    print(f"Mean Precision@5: {mean_precision:.4f}")
    print(f"Mean Success@5 (Hit Rate@5): {hit_rate:.4f}")
    result = "PASS" if hit_rate >= 0.75 else "FAIL"
    print(f"PASS/FAIL (Hit Rate@5 >= 0.75): {result}")
    if result == "FAIL":
        print("Queries missed entirely:")
        for query in missed_queries:
            print(f"  - {query}")


if __name__ == "__main__":
    main()