# remex

Retrieval-validated embedding compression. 2-16x smaller vectors with measured recall.

Based on the rotation + Lloyd-Max scalar quantization insight from [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026), focused on the use case that matters most: **embedding storage and retrieval for RAG systems**.

## Quick start

```python
from remex import Quantizer

# Compress embeddings — no training data needed
pq = Quantizer(d=384, bits=4)            # d = your embedding dimension
compressed = pq.encode(embeddings)       # (n, 384) float32 → compressed
indices, scores = pq.search(compressed, query, k=10)

# Save/load (bit-packed on disk)
compressed.save("index.npz")
from remex import CompressedVectors
loaded = CompressedVectors.load("index.npz")
```

The quantizer is fully determined by `(d, bits, seed)` — no training, no fitting, no index to ship.

## How it works

Three steps, each with a clear purpose:

1. **Random rotation** — A fixed orthogonal matrix ([Haar-distributed](https://arxiv.org/abs/math-ph/0609050) via QR decomposition) transforms any embedding distribution so that coordinates become approximately i.i.d. N(0, 1/d). This is the key insight from TurboQuant: it makes quantization **data-oblivious**, meaning no training data is required.

2. **[Lloyd-Max](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) scalar quantization** — Each coordinate is independently quantized using optimal boundaries for the N(0, 1/d) distribution. The codebook is computed from the theoretical Gaussian CDF, not from data. This produces the minimum mean-squared-error scalar quantizer for Gaussian inputs.

3. **Bit-packing** — Indices are stored at their actual bit width (not wasteful uint8), giving honest compression ratios. A 4-bit codebook uses 4 bits per coordinate on disk.

Norms are stored separately as float32, preserving inner-product ranking up to quantization error.

**Why not QJL?** TurboQuant includes a QJL (quantized [Johnson-Lindenstrauss](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)) residual correction stage for unbiased inner product estimation. We omit it because QJL adds variance that hurts retrieval — when only ranking order matters (not absolute scores), the MSE-optimal rotation + Lloyd-Max stage empirically dominates.

## [Matryoshka](https://arxiv.org/abs/2205.13147) bit precision

An n-bit quantized index's top k bits are a valid k-bit code. remex exploits this: **encode once at full bit-width, search at any lower precision** by right-shifting indices. Centroid tables are precomputed for all bit levels.

This enables two-stage coarse-to-fine retrieval from a single encoded representation:

```python
pq = Quantizer(d=384, bits=8)
compressed = pq.encode(corpus)

# Two-stage: coarse ADC (Asymmetric Distance Computation) scan at reduced bits,
# then full-precision rerank
indices, scores = pq.search_twostage(
    compressed, query, k=10,
    candidates=200,          # coarse pass returns 200 candidates
    coarse_precision=4,      # coarse scan at 4-bit (default: bits-2)
)
```

The nesting incurs a small penalty vs independently optimized codebooks: ~1.2% at 4-bit, up to ~10% at 2-bit. In practice this matters little for the coarse stage, which only needs to identify the right neighborhood.

## Benchmarks

### Recall vs bit level (synthetic, d=384, 10k corpus, 200 queries)

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 8-bit | 4.0x | 0.0000 | 0.987 | 0.991 |
| remex 4-bit | 7.8x | 0.0094 | 0.850 | 0.895 |
| remex 3-bit | 10.4x | 0.0343 | 0.719 | 0.800 |
| remex 2-bit | 15.4x | 0.1171 | 0.538 | 0.634 |

### Real embeddings (all-MiniLM-L6-v2, d=384, 10k corpus, 500 queries)

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 8-bit | 2.0x | 0.0000 | 0.974 | 0.995 |
| remex 4-bit | 7.8x | 0.0093 | 0.707 | 0.932 |
| remex 3-bit | 10.4x | 0.0341 | 0.599 | 0.897 |
| remex 2-bit | 16x | 0.1164 | 0.517 | 0.860 |
| FAISS PQ (m=96, trained) | 16x | 0.0341 | 0.816 | 0.946 |
| FAISS PQ (m=48, trained) | 32x | 0.0636 | 0.618 | 0.877 |

### Scaling with corpus size (synthetic, 4-bit)

| Corpus | R@10 | R@100 | Encode (ms) | Search (ms) |
|--------|------|-------|-------------|-------------|
| 1k | 0.880 | 0.930 | 12 | 4 |
| 5k | 0.862 | 0.905 | 63 | 13 |
| 10k | 0.850 | 0.895 | 134 | 21 |
| 50k | 0.839 | 0.872 | 689 | 140 |

Full benchmark details and distribution sensitivity analysis in [`bench/RESULTS.md`](bench/RESULTS.md).

## When to use remex / when not to

### Use remex when

- **You want zero training.** The quantizer is deterministic and portable — just `(d, bits, seed)`. No codebook to train, no index to ship, no retraining when your corpus changes.
- **You need fast encode.** Encoding is ~20μs/vector (rotation + searchsorted). Adding new vectors never requires retraining.
- **8-bit caching is enough.** At 8-bit (4x compression), R@10 = 0.974 on real embeddings. Near-lossless and much cheaper than float32.
- **You want coarse retrieval + reranking.** 4-bit R@10=0.707 is enough for a first pass if you rerank the top candidates with a cross-encoder or full-precision search.

### Do not use remex when

- **You need high recall at aggressive compression on real data.** At 4-bit, FAISS PQ (m=96) achieves R@10=0.816 vs remex's 0.707 on real embeddings. Data-adaptive methods exploit structure that data-oblivious methods cannot.
- **Your embeddings form very tight clusters.** When cluster spread σ < 0.05, 4-bit R@10 drops to 0.53 (from 0.85 at normal spread). Quantization errors flip rankings among near-identical vectors. 8-bit is much more robust (R@10 stays above 0.95).
- **You need sublinear search.** remex is brute-force only. For >100k vectors, consider FAISS IVF, HNSW, or similar ANN indices. remex's compact encoding can feed into an external ANN index.

### Distribution sensitivity (10k corpus, 4-bit, varying cluster tightness)

| Cluster spread (σ) | 2-bit R@10 | 4-bit R@10 | 8-bit R@10 |
|-------------------|-----------|-----------|-----------|
| 0.01 (very tight) | 0.163 | 0.533 | 0.954 |
| 0.05 | 0.478 | 0.831 | 0.984 |
| 0.10 | 0.532 | 0.846 | 0.987 |
| 0.30 (typical) | 0.538 | 0.850 | 0.987 |
| 1.00 (diffuse) | 0.525 | 0.848 | 0.984 |

**Detection**: If your 4-bit R@10 is significantly below 0.80 on a held-out set, your embeddings likely have tight clusters. Use 8-bit, or switch to a data-adaptive method.

## Compression ratios

Honest packed sizes (bit-packed on disk, d=384):

| Bits | Bytes per vector | vs float32 | File size per 10k vectors |
|------|-----------------|------------|--------------------------|
| 2 | 100 | **15.4x** | 0.93 MB |
| 3 | 148 | **10.4x** | 1.42 MB |
| 4 | 196 | **7.8x** | 1.83 MB |
| 8 | 388 | **4.0x** | 3.61 MB |

Float32 baseline: 1,536 bytes/vector (15.36 MB per 10k vectors).

In-memory, indices are stored as uint8 for fast search. The `PackedVectors` class keeps them bit-packed in memory too, using 2-4x less RAM for sub-byte widths.

## API reference

### `Quantizer(d, bits=4, seed=42)`

Main quantizer class (formerly `PolarQuantizer`, which remains available as a deprecated alias).

- **`d`** — Vector dimension (must match your embeddings).
- **`bits`** — Bits per coordinate: 1-4 or 8. Sweet spot is 3-4. Use 8 for near-lossless.
- **`seed`** — Random seed for the rotation matrix. Same seed = same quantizer.

#### Methods

**`encode(X)`** — Quantize `(n, d)` float32 array. Returns `CompressedVectors`.

**`decode(compressed, precision=None)`** — Reconstruct `(n, d)` float32 from compressed. Optional `precision` (1 to bits) for Matryoshka decode.

**`search(compressed, query, k=10, precision=None)`** — Find k nearest neighbors by approximate inner product. Caches a dequantized float32 matrix for fast repeated queries. Returns `(indices, scores)`.

**`search_batch(compressed, queries, k=10, precision=None)`** — Batch version of `search()` using matrix multiplication for better throughput. Returns `(indices, scores)` where both are `(n_queries, k)`.

**`search_adc(compressed, query, k=10, precision=None, chunk_size=4096)`** — Memory-efficient search via [ADC (Asymmetric Distance Computation)](https://ieeexplore.ieee.org/document/5432202/) lookup-table scoring. No float32 cache — peak memory is `chunk_size * d * 4` bytes (~6 MB). Slower per-query but uses ~5x less RAM. Returns `(indices, scores)`.

**`search_twostage(compressed, query, k=10, candidates=500, coarse_precision=None)`** — Two-stage Matryoshka retrieval: ADC coarse scan (no cache) then full-precision rerank on candidates only. Memory-efficient: only the small candidate set is dequantized. Returns `(indices, scores)`.

**`mse(X, precision=None)`** — Mean per-vector reconstruction error (L2 squared).

### `CompressedVectors`

Container for quantized data. Created by `Quantizer.encode()`. Stores indices as uint8 in memory for fast search/decode.

#### Properties

- **`n`** — Number of vectors.
- **`nbytes`** — Bit-packed size in bytes (honest compression).
- **`nbytes_unpacked`** — In-memory size (uint8 indices + float32 norms).
- **`compression_ratio`** — `(n * d * 4) / nbytes`.
- **`resident_bytes`** — Actual RAM including any active caches.

#### Methods

- **`save(path)`** / **`load(path)`** — Save/load to `.npz` with bit-packed indices.
- **`save_arrow(path)`** / **`load_arrow(path)`** — Save/load to Arrow IPC (Feather v2) format. Requires `pyarrow`.
- **`subset(idx)`** — Return a new `CompressedVectors` with only the given row indices.
- **`drop_cache()`** — Free the dequantized float32 cache to reclaim memory.

### `PackedVectors`

Memory-efficient packed storage. Keeps indices bit-packed in memory, unpacking on demand. Uses 2-4x less RAM than `CompressedVectors` for sub-byte widths.

```python
from remex import PackedVectors

packed = PackedVectors.from_compressed(compressed)  # pack in memory
packed = PackedVectors.from_rows(rows, norms, d=384, bits=4)  # from DB rows

# ADC and two-stage search work directly on PackedVectors
indices, scores = pq.search_adc(packed, query, k=10)
indices, scores = pq.search_twostage(packed, query, k=10)

# Matryoshka precision reduction
packed_2bit = packed.at_precision(2)

# Convert back if needed
compressed = packed.to_compressed()
```

Cached `search()` is not supported on `PackedVectors` — use `search_adc()` or `search_twostage()`, or convert with `to_compressed()`.

### `GPUSearcher` (optional)

GPU-accelerated search wrapper. Requires CuPy or PyTorch with CUDA. Falls back to NumPy.

```python
from remex.gpu import GPUSearcher

searcher = GPUSearcher(pq, compressed)
indices, scores = searcher.search(query, k=10)
indices, scores = searcher.search_adc(query, k=10)
indices, scores = searcher.search_twostage(query, k=10, candidates=200)
```

### Memory profiles (100k vectors, d=384, 8-bit)

| Strategy | Resident RAM | ms/query |
|----------|-------------|----------|
| `search()` (cached) | 192 MB | 3.9 |
| `search()` (cold) | 39 MB | 137 |
| `search_adc()` (no cache) | 39 MB | 152 |
| `search_twostage()` (no cache) | 39 MB | 152 |

Choose `search()` when latency matters and RAM is available. Choose `search_adc()` or `search_twostage()` when memory is constrained (serverless, edge, or very large corpora).

### Low-level utilities

```python
from remex import pack, unpack, packed_nbytes
from remex import lloyd_max_codebook, nested_codebooks
```

- **`pack(indices, bits)`** / **`unpack(packed, bits, n_values)`** — Bit-pack/unpack uint8 arrays.
- **`packed_nbytes(n_values, d, bits)`** — Compute packed byte count.
- **`lloyd_max_codebook(d, bits)`** — Generate optimal boundaries and centroids for N(0, 1/d).
- **`nested_codebooks(d, max_bits)`** — Build Matryoshka centroid tables for all bit levels 1..max_bits.

## vs TurboQuant

TurboQuant (Zandieh et al., ICLR 2026) adds QJL (quantized [Johnson-Lindenstrauss](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)) residual correction for unbiased inner product estimates. This is important for KV cache attention, where unbiased estimation matters. For **retrieval** (ranking by approximate inner product), the QJL variance hurts more than the debiasing helps. remex implements only the MSE-optimal rotation + Lloyd-Max stage, which empirically dominates for nearest-neighbor search.

## vs [FAISS Product Quantization](https://ieeexplore.ieee.org/document/5432202/)

| | remex | FAISS PQ |
|---|---|---|
| Training | None | Required (trains on corpus) |
| Recall at matched compression | Lower on real data | Higher (learns structure) |
| Encode speed | ~20μs/vec | ~200μs+/vec |
| Corpus updates | Re-encode only new vectors | Retrain or accept stale codebook |
| Index portability | Quantizer is `(d, bits, seed)` | Must ship trained index |
| Sublinear search | No (brute-force) | Yes (IVF, HNSW) |
| GPU support | NumPy/CuPy/PyTorch fallback | Native CUDA |

**Use FAISS when**: You have a stable, large corpus, need sublinear search, and can afford training time.

**Use remex when**: You want zero training, fast encode, frequently changing corpora, or near-lossless 8-bit caching (R@10=0.974 at 4x compression).

### vs scalar quantization (naive rounding)

Without the rotation step, scalar quantization on raw embeddings is catastrophically bad — embeddings are highly anisotropic (variance ratios of 10^7x across dimensions). The random rotation spreads information uniformly across coordinates, making scalar quantization viable.

At 3-bit, remex achieves 72-80% R@10 vs ~40% for naive scalar quantization on the same data.

## Installation

```bash
pip install remex                       # from PyPI (when published)
pip install -e ".[dev]"                 # development: + pytest, pytest-cov
pip install -e ".[bench]"              # benchmarking: + faiss-cpu, sentence-transformers
```

## Testing

```bash
pytest                                  # 126 tests (~6 min)
pytest tests/test_polar_embed.py -v     # core tests
pytest tests/test_matryoshka.py -v      # Matryoshka/nested codebook tests
pytest tests/test_adc_gpu.py -v         # ADC and GPU searcher tests
pytest tests/test_packed_vectors.py -v  # PackedVectors tests
```

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Jégou, Douze & Schmid (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1):117–128. [IEEE Xplore](https://ieeexplore.ieee.org/document/5432202/) — introduces Product Quantization, ADC (Asymmetric Distance Computation), and SDC for approximate nearest neighbor search.
- Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS 2022. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) — the nested/coarse-to-fine representation idea that inspires remex's bit-level nesting.
- Mezzadri (2007). *How to Generate Random Matrices from the Classical Compact Groups.* Notices of the AMS 54(5):592–604. [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050) — the QR-of-Gaussian method for Haar-distributed orthogonal matrices used in `remex/rotation.py`.
- Lloyd (1982). *Least Squares Quantization in PCM.* IEEE Trans. Information Theory 28(2):129–137. [IEEE Xplore](https://ieeexplore.ieee.org/document/1056489) — optimal scalar quantization (Lloyd-Max algorithm) for minimum MSE.

## License

MIT
