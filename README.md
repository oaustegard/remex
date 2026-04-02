# polar-embed

Retrieval-validated embedding compression. Compress vectors 8-15× with measured recall.

Based on the rotation + Lloyd-Max insight from [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026), focused on the use case that matters most: **embedding storage and retrieval for RAG**.

## Quick start

```python
from polarquant import PolarQuantizer

# Data-oblivious: no training data needed
pq = PolarQuantizer(d=384, bits=4)
compressed = pq.encode(embeddings)    # (n, 384) float32 → bit-packed
indices, scores = pq.search(compressed, query, k=10)

# Optional: calibrate with a sample for +1-3% recall
pq = PolarQuantizer(d=384, bits=4).calibrate(sample_vectors)
compressed = pq.encode(embeddings)

# Save/load (bit-packed format)
compressed.save("index.npz")
```

## How it works

1. **Random rotation** — A fixed orthogonal matrix makes coordinates approximately i.i.d. N(0, 1/d). This is the key trick: it makes quantization data-oblivious.
2. **Scalar quantization** — Each coordinate is independently quantized. Two modes:
   - **Data-oblivious** (default): Lloyd-Max codebook optimized for the theoretical N(0, 1/d) distribution. Zero training data needed.
   - **Calibrated**: Per-dimension k-means codebooks learned from a sample. Captures per-coordinate variance spread that the theoretical codebook can't.
3. **Bit-packing** — Indices stored at actual bit width (not uint8), giving honest compression ratios.

## Benchmarks

Tested on real embeddings from all-MiniLM-L6-v2 (d=384), 5k corpus, 200 queries.

### Compression ratios (bit-packed)

| Bits | Per vector (d=384) | vs float32 |
|------|-------------------|------------|
| 2 | 100 bytes | **15.4×** |
| 3 | 148 bytes | **10.4×** |
| 4 | 196 bytes | **7.8×** |

### Recall on real embeddings

| Method | R@10 | R@100 | MSE |
|--------|------|-------|-----|
| PolarQuant 4-bit (oblivious) | 0.826 | 0.977 | 0.0096 |
| PolarQuant 4-bit (calibrated, n=1000) | 0.838 | 0.981 | 0.0071 |
| PolarQuant 3-bit (oblivious) | 0.733 | 0.963 | 0.0345 |
| PolarQuant 3-bit (calibrated, n=1000) | 0.758 | 0.969 | 0.0226 |
| PolarQuant 8-bit (oblivious) | 0.987 | 0.998 | 0.0001 |
| FAISS PQ m=96 (16×, trained) | 0.863 | 0.966 | 0.0365 |
| FAISS PQ m=48 (32×, trained) | 0.718 | 0.939 | 0.0778 |

### Calibration: when to use it

Calibration learns per-dimension codebooks from a sample. It helps when the sample is large enough:

| Bits | Minimum sample | Benefit |
|------|---------------|---------|
| 4-bit | ≥750 vectors | +0.3% to +2.9% R@10 |
| 3-bit | ≥100 vectors | +1.3% to +3.8% R@10 |

Below these thresholds, the data-oblivious codebook performs better.

### Search performance (100k vectors, d=384)

| | First query (cold) | Subsequent queries |
|---|---|---|
| PolarQuant | 280-360ms | **6-9ms** |
| FAISS PQ m=96 | — | 4ms |

First search builds a cache; subsequent queries reuse it. Encode speed: ~20μs/vector (10-17× faster than FAISS PQ build time).

## Why polar-embed over TurboQuant?

TurboQuant adds QJL residual correction for unbiased inner product estimates. This helps KV cache attention but **hurts** retrieval — the variance from QJL outweighs the debiasing when only ranking matters.

## Why not FAISS Product Quantization?

FAISS PQ trains on your data and achieves higher recall at matched compression. Use FAISS when:
- You have a stable corpus and can afford training time
- Search latency at >50k vectors matters (FAISS has IVF for sublinear search)

Use polar-embed when:
- You want zero (or minimal) training — data-oblivious mode just works
- Your corpus changes frequently (no retraining needed)
- You need fast encode (20μs/vec vs 200μs+/vec for FAISS)
- 8-bit near-lossless caching (R@10=0.987 at 4× compression)

## Install

```bash
pip install -e ".[dev]"   # development
pip install -e ".[bench]" # + faiss-cpu, sentence-transformers
pytest                     # 49 tests
```

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT
