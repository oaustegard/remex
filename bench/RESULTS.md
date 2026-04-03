# Real Embedding Evaluation Results

**Date**: 2026-04-02  
**Model**: all-MiniLM-L6-v2 (d=384)  
**Corpus**: 10,000 vectors | **Queries**: 500  

## Distribution Analysis (Post-Rotation)

| Metric | Real Embeddings | Random Vectors |
|--------|----------------|----------------|
| Expected σ (1/√d) | 0.051031 | 0.051031 |
| Actual σ (global) | 0.051018 | 0.051027 |
| Per-dim σ (mean±std) | 0.0437 ± 0.0056 | 0.0510 ± 0.0004 |
| Kurtosis (Gaussian=3) | 2.72 ± 0.45 | 2.98 ± 0.05 |
| Original anisotropy | 37M× | 1.1× |

**Key finding**: The rotation trick works — global σ matches theory perfectly. But per-dimension variance spans ~0.03–0.06, indicating residual structure the rotation doesn't fully eliminate.

## Recall Results: Real Embeddings

| Method | Compression* | MSE | R@10 | R@100 |
|--------|-------------|-----|------|-------|
| PolarQuant 2-bit | 16× | 0.1164 | 0.517 | 0.860 |
| PolarQuant 3-bit | 10.4× | 0.0341 | 0.599 | 0.897 |
| PolarQuant 4-bit | 7.8× | 0.0093 | 0.707 | 0.932 |
| PolarQuant 8-bit | 2.0× | 0.0000 | 0.974 | 0.995 |
| FAISS PQ (m=48) | 32× | 0.0636 | 0.618 | 0.877 |
| FAISS PQ (m=96) | 16× | 0.0341 | 0.816 | 0.946 |

*PolarQuant compression uses bit-packing for on-disk storage. In-memory indices use uint8 for fast search.

## Recall Results: Random Unit Vectors

| Method | MSE | R@10 | R@100 |
|--------|-----|------|-------|
| PolarQuant 2-bit | 0.1170 | 0.533 | 0.634 |
| PolarQuant 3-bit | 0.0344 | 0.740 | 0.799 |
| PolarQuant 4-bit | 0.0094 | 0.847 | 0.895 |
| PolarQuant 8-bit | 0.0000 | 0.986 | 0.991 |
| FAISS PQ (m=48) | 0.2875 | 0.280 | 0.390 |
| FAISS PQ (m=96) | 0.0863 | 0.484 | 0.587 |

## Analysis

### The Recall Gap
PolarQuant loses 14% R@10 at 3-4 bits on real vs random embeddings. This is **not** because the Gaussian assumption breaks — the global distribution fits perfectly. The gap comes from:

1. **Data clustering**: Real embeddings form tight clusters (20 topics). Within-cluster inner products are high, so quantization errors have outsized impact on ranking.
2. **Per-dimension heterogeneity**: σ varies 2× across dimensions (0.03–0.06). The uniform codebook over-quantizes low-variance dims and under-quantizes high-variance ones.

### FAISS PQ: Opposite Pattern
FAISS PQ dramatically *improves* on real embeddings vs random (0.618 vs 0.280 at m=48). It trains on the data, learning subspace structure that random vectors don't have. This is the fundamental trade-off: data-oblivious (PolarQuant) vs data-adaptive (FAISS PQ).

### Fitted Codebook: No Help
Fitting Lloyd-Max to the actual global σ instead of theoretical σ produced zero improvement (σ was already correct to 4 decimal places). Per-dimension codebooks are the next experiment.

### Practical Comparison at Matched Compression
| Method | Compression | R@10 |
|--------|------------|------|
| PolarQuant 3-bit | ~10× | 0.599 |
| FAISS PQ (m=96) | 16× | 0.816 |

FAISS wins convincingly at the compression points practitioners care about.

## Conclusions for Issue #1

**Acceptance criterion**: R@10 at 4-bit within 10% of random → **NOT MET** (0.707 vs 0.847 = 17% gap)

**But the framing was wrong**. Random vectors are the *hardest* case for FAISS PQ (no structure to exploit) and the *easiest* for PolarQuant (coordinates are already perfectly Gaussian). Real embeddings flip both directions.

**PolarQuant's lane**:
- Training-free, deterministic, portable (no index to ship)
- 8-bit is near-lossless (R@10=0.974) at 2× compression — good for caching
- 4-bit is serviceable (R@10=0.707) for coarse retrieval with reranking
- Not competitive with data-adaptive methods at aggressive compression

**Next steps**:
- [ ] Per-dimension codebooks (fit σ_j per coordinate)
- [ ] Bit-packing (issue #4) for honest compression numbers
- [ ] Larger/more diverse corpus (current 20-topic set may overstate clustering effect)
- [ ] Blog post: "QJL hurts retrieval" + honest PolarQuant positioning
