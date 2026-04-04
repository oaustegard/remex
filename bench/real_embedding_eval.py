"""
Real embedding evaluation for remex (formerly polar-embed, Issue #1).

Encodes text with a real transformer model, then benchmarks:
- remex recall@k at various bit widths
- FAISS Product Quantization baseline
- Exact brute-force ground truth

Reports recall@10 and recall@100 plus compression ratios.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_real_embeddings(n_corpus=10000, n_queries=500, model_name="all-MiniLM-L6-v2"):
    """Encode text into real embeddings using a sentence transformer."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    d = model.get_sentence_embedding_dimension()
    print(f"  dimension: {d}")

    # Generate diverse text from simple templates
    topics = [
        "machine learning", "climate change", "quantum physics", "cooking recipes",
        "space exploration", "music theory", "ancient history", "economics",
        "biology", "philosophy", "sports", "technology", "medicine", "art",
        "mathematics", "psychology", "politics", "literature", "travel", "education",
    ]
    templates = [
        "An introduction to {t} and its impact on society",
        "The latest developments in {t} research",
        "How {t} affects daily life",
        "A comprehensive guide to understanding {t}",
        "The history and evolution of {t}",
        "Common misconceptions about {t}",
        "Why {t} matters for the future",
        "The relationship between {t} and innovation",
        "Practical applications of {t} in modern world",
        "Key concepts everyone should know about {t}",
        "Challenges and opportunities in {t}",
        "The role of {t} in solving global problems",
        "Expert perspectives on {t}",
        "Emerging trends in {t}",
        "How to get started with {t}",
    ]

    total = n_corpus + n_queries
    sentences = []
    rng = np.random.default_rng(42)
    for i in range(total):
        t = topics[i % len(topics)]
        tmpl = templates[i % len(templates)]
        # Add variation with index to make sentences unique
        sentences.append(f"{tmpl.format(t=t)} (variant {i})")

    print(f"Encoding {total} sentences...")
    t0 = time.time()
    all_embs = model.encode(sentences, show_progress_bar=True, batch_size=256)
    print(f"  encoded in {time.time()-t0:.1f}s")

    corpus = all_embs[:n_corpus].astype(np.float32)
    queries = all_embs[n_corpus:].astype(np.float32)
    return corpus, queries, d


def make_random_baseline(n_corpus, n_queries, d):
    """Random unit vectors — the 'easy' case for comparison."""
    rng = np.random.default_rng(123)
    corpus = rng.standard_normal((n_corpus, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((n_queries, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return corpus, queries


def exact_knn(corpus, queries, k):
    """Brute-force exact inner product search."""
    scores = queries @ corpus.T
    topk = np.argsort(-scores, axis=1)[:, :k]
    return topk


def recall_at_k(pred, truth, k):
    """Recall@k: fraction of true top-k found in predicted top-k."""
    assert pred.shape == truth.shape
    recalls = []
    for p, t in zip(pred, truth):
        recalls.append(len(set(p[:k]) & set(t[:k])) / k)
    return np.mean(recalls)


def benchmark_remex(corpus, queries, d, bits_list, k_list):
    """Run remex at various bit widths, report recall."""
    from remex import Quantizer

    results = []
    for bits in bits_list:
        pq = Quantizer(d=d, bits=bits)

        t0 = time.time()
        compressed = pq.encode(corpus)
        encode_time = time.time() - t0

        mse = pq.mse(corpus)
        ratio = compressed.compression_ratio

        max_k = max(k_list)
        t0 = time.time()
        all_pred = []
        for q in queries:
            idx, _ = pq.search(compressed, q, k=max_k)
            all_pred.append(idx)
        search_time = time.time() - t0
        pred = np.array(all_pred)

        results.append({
            "method": f"remex-{bits}bit",
            "bits": bits,
            "mse": mse,
            "compression_ratio": ratio,
            "encode_time": encode_time,
            "search_time": search_time,
            "pred": pred,
        })
    return results


def benchmark_faiss_pq(corpus, queries, d, m_list, k_list):
    """Run FAISS Product Quantization as baseline."""
    import faiss

    results = []
    for m in m_list:
        nbits_per_sub = 8  # standard FAISS PQ
        index = faiss.IndexPQ(d, m, nbits_per_sub)

        t0 = time.time()
        index.train(corpus)
        index.add(corpus)
        encode_time = time.time() - t0

        max_k = max(k_list)
        t0 = time.time()
        _, pred = index.search(queries, max_k)
        search_time = time.time() - t0

        # Compression: m bytes per vector (8-bit codes) vs d*4 bytes float32
        ratio = (d * 4) / m

        # MSE
        recon = index.sa_decode(index.sa_encode(corpus))
        mse = float(np.mean(np.sum((corpus - recon) ** 2, axis=1)))

        results.append({
            "method": f"FAISS-PQ(m={m})",
            "m": m,
            "mse": mse,
            "compression_ratio": ratio,
            "encode_time": encode_time,
            "search_time": search_time,
            "pred": pred,
        })
    return results


def analyze_embedding_distribution(corpus, d):
    """Check if post-rotation coordinates are Gaussian (the key assumption)."""
    from remex.rotation import haar_rotation

    R = haar_rotation(d, seed=42)
    norms = np.linalg.norm(corpus, axis=1)
    unit = corpus / np.maximum(norms, 1e-8)[:, None]
    rotated = unit @ R.T

    # Check Gaussianity of coordinates
    sigma_expected = 1.0 / np.sqrt(d)
    sigma_actual = np.std(rotated, axis=0)
    kurtosis = np.mean(rotated**4, axis=0) / np.mean(rotated**2, axis=0)**2

    print("\n=== Post-Rotation Distribution Analysis ===")
    print(f"  Expected σ (N(0,1/d)):  {sigma_expected:.6f}")
    print(f"  Actual σ (mean±std):    {np.mean(sigma_actual):.6f} ± {np.std(sigma_actual):.6f}")
    print(f"  Kurtosis (Gaussian=3):  {np.mean(kurtosis):.3f} ± {np.std(kurtosis):.3f}")
    print(f"  Norm mean±std:          {np.mean(norms):.3f} ± {np.std(norms):.3f}")

    # Check anisotropy of original embeddings
    cov_diag = np.var(corpus, axis=0)
    anisotropy = np.max(cov_diag) / (np.min(cov_diag) + 1e-10)
    print(f"  Original anisotropy:    {anisotropy:.1f}x (max_var/min_var)")

    return {
        "sigma_expected": sigma_expected,
        "sigma_actual_mean": float(np.mean(sigma_actual)),
        "kurtosis_mean": float(np.mean(kurtosis)),
        "anisotropy": float(anisotropy),
    }


def run_benchmark(corpus, queries, d, label, k_list=[10, 100]):
    """Full benchmark suite on a given embedding set."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  corpus: {corpus.shape}, queries: {queries.shape}")
    print(f"{'='*60}")

    # Distribution analysis
    dist = analyze_embedding_distribution(corpus, d)

    # Ground truth
    print("\nComputing exact kNN ground truth...")
    max_k = max(k_list)
    truth = exact_knn(corpus, queries, max_k)

    # remex at various bit widths
    bits_list = [2, 3, 4, 8]
    print("\n--- remex ---")
    pq_results = benchmark_remex(corpus, queries, d, bits_list, k_list)

    # FAISS PQ baselines (m = number of subquantizers)
    # m=48 → 48 bytes/vec, m=96 → 96 bytes/vec for d=384
    m_list = [d // 8, d // 4]  # coarse and fine
    print("\n--- FAISS Product Quantization ---")
    faiss_results = benchmark_faiss_pq(corpus, queries, d, m_list, k_list)

    # Report
    all_results = pq_results + faiss_results
    print(f"\n{'Method':<22s} {'MSE':>10s} {'Ratio':>8s} {'R@10':>8s} {'R@100':>8s} {'Enc(s)':>8s} {'Srch(s)':>8s}")
    print("-" * 80)
    for r in all_results:
        recalls = {}
        for k in k_list:
            recalls[k] = recall_at_k(r["pred"][:, :k], truth[:, :k], k)
        print(f"{r['method']:<22s} {r['mse']:>10.4f} {r['compression_ratio']:>7.1f}x "
              f"{recalls.get(10, 0):>7.3f} {recalls.get(100, 0):>7.3f} "
              f"{r['encode_time']:>7.2f}s {r['search_time']:>7.2f}s")

    return all_results, dist, truth


if __name__ == "__main__":
    N_CORPUS = 10000
    N_QUERIES = 500

    # Phase 1: Real embeddings
    corpus_real, queries_real, d = load_real_embeddings(N_CORPUS, N_QUERIES)
    real_results, real_dist, _ = run_benchmark(corpus_real, queries_real, d, "REAL EMBEDDINGS (all-MiniLM-L6-v2)")

    # Phase 2: Random baseline (same dimension)
    corpus_rand, queries_rand = make_random_baseline(N_CORPUS, N_QUERIES, d)
    rand_results, rand_dist, _ = run_benchmark(corpus_rand, queries_rand, d, "RANDOM UNIT VECTORS (d=384)")

    # Summary comparison
    print(f"\n{'='*60}")
    print("  COMPARISON: Real vs Random")
    print(f"{'='*60}")
    print(f"{'':>22s} {'Real R@10':>10s} {'Rand R@10':>10s} {'Gap':>8s}")
    print("-" * 55)
    for rr, rn in zip(real_results[:4], rand_results[:4]):  # remex only
        recall_real = recall_at_k(rr["pred"][:, :10], exact_knn(corpus_real, queries_real, 10), 10)
        recall_rand = recall_at_k(rn["pred"][:, :10], exact_knn(corpus_rand, queries_rand, 10), 10)
        gap = recall_real - recall_rand
        print(f"{rr['method']:<22s} {recall_real:>9.3f} {recall_rand:>9.3f} {gap:>+7.3f}")
