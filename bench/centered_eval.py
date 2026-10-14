"""What a caller-declared corpus mean is worth (`Quantizer(mean=...)`).

A centred code stores the offset from the mean, ``||x - mu||`` and
``mu . (x - mu)``. Search scores ``q.mu + mu.r + ||r|| (q-mu).u_hat / kappa``
(remex 2.0); the ``1.x`` column ranks by ``q . decode(x)`` instead, which is
what remex 1.x searched. The ``naive`` column subtracts the mean and keeps
the residual's own length with no correction at all.

The gain tracks how much of the corpus is a shared direction, so it is large
on raw inner-product embeddings and small on L2-normalised ones. Measure on
your own corpus before turning it on.

Usage:
    python bench/centered_eval.py            # synthetic + MiniLM if installed
    bash bench/fetch_specter2_cache.sh       # then:
    python bench/centered_eval.py --specter2
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import remex  # noqa: E402
from remex import Quantizer  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".specter2_cache")
BITS = (1, 2, 3, 4, 8)


def exact_topk(corpus, queries, k):
    return np.argsort(-(queries @ corpus.T), axis=1)[:, :k]


def recall_at_k(pred, truth, k):
    return sum(len(set(p[:k].tolist()) & set(t[:k].tolist()))
               for p, t in zip(pred, truth)) / (len(pred) * k)


def naive_centred(corpus, mu, bits, seed=42):
    """Subtract the mean, keep the residual's own length. The wrong half."""
    pq = Quantizer(d=corpus.shape[1], bits=bits, seed=seed)
    return pq.decode(pq.encode(corpus - mu)) + mu


def report(name, corpus, queries):
    truth = exact_topk(corpus, queries, 100)
    mu = remex.corpus_mean(corpus)
    shared = float(np.linalg.norm(mu) / np.linalg.norm(corpus, axis=1).mean())
    print(f"\n=== {name} ===")
    print(f"  n={len(corpus)} d={corpus.shape[1]} queries={len(queries)}"
          f"  ||mean|| / mean ||x|| = {shared:.3f}")
    print(f"  {'bits':>4} {'R@10 plain':>10} {'centred':>8} {'delta':>7}"
          f" | {'1.x':>7} {'naive':>7} | {'R@100 plain':>11} {'centred':>8} {'delta':>7}")
    for bits in BITS:
        plain = Quantizer(d=corpus.shape[1], bits=bits, seed=42)
        cen = Quantizer(d=corpus.shape[1], bits=bits, seed=42, mean=mu)
        pa = plain.search_batch(plain.encode(corpus), queries, k=100)[0]
        codes = cen.encode(corpus)
        pb = cen.search_batch(codes, queries, k=100)[0]
        # remex 1.x scored centred codes as q . decode(x); kept for comparison
        p1 = exact_topk(cen.decode(codes), queries, 100)
        pn = exact_topk(naive_centred(corpus, mu, bits), queries, 100)
        a10, b10, o10, n10 = (recall_at_k(p, truth, 10) for p in (pa, pb, p1, pn))
        a100, b100 = (recall_at_k(p, truth, 100) for p in (pa, pb))
        print(f"  {bits:>4} {a10:>10.3f} {b10:>8.3f} {b10 - a10:>+7.3f}"
              f" | {o10:>7.3f} {n10:>7.3f} | {a100:>11.3f} {b100:>8.3f} {b100 - a100:>+7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specter2", action="store_true",
                    help="also run the cached SPECTER2 partitions")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    report("synthetic gaussian, d=768",
           rng.standard_normal((9500, 768)).astype(np.float32),
           rng.standard_normal((500, 768)).astype(np.float32))

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from real_embedding_eval import load_real_embeddings
        corpus, queries, _ = load_real_embeddings()
        report("all-MiniLM-L6-v2", corpus, queries)
    except ImportError:
        print("\n[skip] all-MiniLM-L6-v2 — pip install -e '.[bench]'")

    if args.specter2:
        for part in ("broad", "narrow"):
            path = os.path.join(CACHE_DIR, f"specter2_nlp_{part}.npy")
            if not os.path.exists(path):
                print(f"\n[skip] {path} — run bench/fetch_specter2_cache.sh")
                continue
            X = np.load(path)
            perm = np.random.default_rng(99).permutation(len(X))
            report(f"SPECTER2 {part}", X[perm[500:]].astype(np.float32),
                   X[perm[:500]].astype(np.float32))


if __name__ == "__main__":
    main()
