"""
Self-contained benchmark for remex (formerly polar-embed).

Uses synthetic embeddings that mimic real embedding characteristics
(clustered structure, anisotropy) so benchmarks run without
sentence-transformers or faiss-cpu.

Run:
    python bench/benchmark.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from remex import Quantizer
from remex.rotation import haar_rotation


def make_clustered_embeddings(n, d, n_clusters=20, seed=42):
    """Synthetic embeddings with cluster structure mimicking real models."""
    rng = np.random.default_rng(seed)
    # Cluster centers on a unit sphere
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Points around each center with some spread
    labels = rng.integers(0, n_clusters, size=n)
    spread = 0.3  # controls tightness
    vecs = centers[labels] + spread * rng.standard_normal((n, d)).astype(np.float32)
    # Normalize to unit sphere
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def exact_knn(corpus, queries, k):
    """Brute-force exact inner-product search."""
    scores = queries @ corpus.T
    return np.argsort(-scores, axis=1)[:, :k]


def recall_at_k(pred, truth, k):
    """Fraction of true top-k found in predicted top-k."""
    hits = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k]) & set(t[:k]))
    return hits / (len(pred) * k)


def run_benchmark():
    d = 384
    n_queries = 200

    print("=" * 70)
    print("remex benchmark")
    print("=" * 70)

    for n_corpus in [1_000, 5_000, 10_000, 50_000]:
        print(f"\n--- Corpus: {n_corpus:,} vectors, d={d}, queries={n_queries} ---")
        corpus = make_clustered_embeddings(n_corpus, d, n_clusters=20, seed=42)
        queries = make_clustered_embeddings(n_queries, d, n_clusters=20, seed=99)

        # Ground truth
        truth10 = exact_knn(corpus, queries, 10)
        truth100 = exact_knn(corpus, queries, 100)

        header = f"{'Method':<35s} {'Comp':>6s} {'MSE':>10s} {'R@10':>7s} {'R@100':>7s} {'Enc ms':>8s} {'Qry ms':>8s}"
        print(header)
        print("-" * len(header))

        for bits in [2, 3, 4, 8]:
            pq = Quantizer(d=d, bits=bits)

            # Encode
            t0 = time.perf_counter()
            compressed = pq.encode(corpus)
            enc_ms = (time.perf_counter() - t0) * 1000

            mse = pq.mse(corpus[:min(1000, n_corpus)])
            ratio = compressed.compression_ratio

            # Search (batch)
            t0 = time.perf_counter()
            pred, _ = pq.search_batch(compressed, queries, k=100)
            search_ms = (time.perf_counter() - t0) * 1000

            r10 = recall_at_k(pred, truth10, 10)
            r100 = recall_at_k(pred, truth100, 100)

            label = f"remex {bits}-bit (oblivious)"
            print(f"{label:<35s} {ratio:>5.1f}x {mse:>10.4f} {r10:>7.3f} {r100:>7.3f} {enc_ms:>7.0f} {search_ms:>7.0f}")

        # Two-stage search (Matryoshka)
        pq = Quantizer(d=d, bits=4)
        compressed = pq.encode(corpus)

        all_pred = []
        t0 = time.perf_counter()
        for q in queries:
            idx, _ = pq.search_twostage(compressed, q, k=10, candidates=200)
            all_pred.append(idx)
        search_ms = (time.perf_counter() - t0) * 1000
        pred = np.array(all_pred)
        r10 = recall_at_k(pred, truth10, 10)

        label = "remex 4-bit (two-stage)"
        print(f"{label:<35s} {compressed.compression_ratio:>5.1f}x {'':>10s} {r10:>7.3f} {'':>7s} {'':>8s} {search_ms:>7.0f}")

    # Distribution analysis
    print(f"\n--- Post-rotation distribution analysis (n=10000, d={d}) ---")
    corpus = make_clustered_embeddings(10_000, d, seed=42)
    R = haar_rotation(d, seed=42)
    norms = np.linalg.norm(corpus, axis=1)
    unit = corpus / norms[:, None]
    rotated = unit @ R.T

    sigma_per_dim = np.std(rotated, axis=0)
    kurtosis = np.mean(rotated**4, axis=0) / np.mean(rotated**2, axis=0)**2

    print(f"  Expected σ (1/√d):     {1/np.sqrt(d):.6f}")
    print(f"  Actual σ (global):     {np.std(rotated):.6f}")
    print(f"  Per-dim σ (mean±std):  {np.mean(sigma_per_dim):.4f} ± {np.std(sigma_per_dim):.4f}")
    print(f"  Kurtosis (Gaussian=3): {np.mean(kurtosis):.2f} ± {np.std(kurtosis):.2f}")
    print(f"  σ range:               [{np.min(sigma_per_dim):.4f}, {np.max(sigma_per_dim):.4f}]")


if __name__ == "__main__":
    run_benchmark()
