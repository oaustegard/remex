# remex

Retrieval-validated embedding compression. 2-16x smaller vectors with measured recall.

> Formerly known as **polar-embed**.

Based on the rotation + Lloyd-Max scalar quantization insight from [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026), focused on the use case that matters most: **embedding storage and retrieval for RAG systems**.

## Quick start

```python
from remex import Quantizer

# Compress embeddings — no training data needed
pq = Quantizer(d=384, bits=4)       # d = your embedding dimension
compressed = pq.encode(embeddings)        # (n, 384) float32 → compressed
indices, scores = pq.search(compressed, query, k=10)

# Save/load (bit-packed on disk)
compressed.save("index.npz")
from remex import CompressedVectors
loaded = CompressedVectors.load("index.npz")
```

## How it works

Three steps, each with a clear purpose:

1. **Random rotation** — A fixed orthogonal matrix (Haar-distributed) transforms any embedding distribution so that coordinates become approximately i.i.d. N(0, 1/d). This is the key insight from TurboQuant: it makes quantization data-oblivious, meaning no training data is required.

2. **Scalar quantization** — Each coordinate is independently quantized using Lloyd-Max optimal boundaries for the N(0, 1/d) distribution. Theoretical codebook. Zero training. Works on any embeddings.

3. **Bit-packing** — Indices are stored at their actual bit width (not wasteful uint8), giving honest compression ratios. A 4-bit codebook uses 4 bits per coordinate on disk.

Norms are stored separately as float32, so inner-product ranking is preserved exactly up to quantization error.

## Features

- **Data-oblivious compression**: No training, no fitting, no index to ship. The quantizer is fully determined by (dimension, bits, seed).
- **Matryoshka bit precision**: Encode once at full bit-width, search at any lower precision by right-shifting indices. Enables two-stage coarse-to-fine retrieval from a single representation.
- **Bit-packed serialization**: `save()`/`load()` uses true sub-byte packing. Compression ratios are honest.
- **Fast encode**: ~20us/vector. 10-17x faster than FAISS PQ index build.
- **Warm search caching**: First query builds a dequantized cache; subsequent queries on the same corpus reuse it (6-9ms vs 280-360ms cold start at 100k vectors).
- **Deterministic**: Same seed produces identical results across runs and platforms.

## Benchmarks

### Synthetic clustered embeddings (d=384, 20 clusters)

Tested with synthetic embeddings that mimic real model characteristics (cluster structure, unit-normalized). Ground truth is exact brute-force inner product.

#### 10k corpus, 200 queries

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 8-bit (oblivious) | 4.0x | 0.0000 | 0.987 | 0.991 |
| remex 4-bit (oblivious) | 7.8x | 0.0094 | 0.850 | 0.895 |
| remex 3-bit (oblivious) | 10.4x | 0.0343 | 0.719 | 0.800 |
| remex 2-bit (oblivious) | 15.4x | 0.1171 | 0.538 | 0.634 |

#### Scaling with corpus size (4-bit oblivious)

| Corpus | R@10 | R@100 | Encode | Search (200 queries) |
|--------|------|-------|--------|---------------------|
| 1k | 0.880 | 0.930 | 14ms | 15ms |
| 5k | 0.862 | 0.905 | 68ms | 29ms |
| 10k | 0.850 | 0.895 | 128ms | 58ms |
| 50k | 0.839 | 0.872 | 786ms | 269ms |

### Real embeddings (all-MiniLM-L6-v2, d=384)

From `bench/real_embedding_eval.py` using 10k corpus and 500 queries encoded by sentence-transformers:

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 8-bit | 2.0x | 0.0000 | 0.974 | 0.995 |
| remex 4-bit | 7.8x | 0.0093 | 0.707 | 0.932 |
| remex 3-bit | 10.4x | 0.0341 | 0.599 | 0.897 |
| remex 2-bit | 16x | 0.1164 | 0.517 | 0.860 |
| FAISS PQ (m=96, trained) | 16x | 0.0341 | 0.816 | 0.946 |
| FAISS PQ (m=48, trained) | 32x | 0.0636 | 0.618 | 0.877 |

### Compression ratios (bit-packed, d=384)

| Bits | Bytes per vector | vs float32 |
|------|-----------------|------------|
| 2 | 100 | **15.4x** |
| 3 | 148 | **10.4x** |
| 4 | 196 | **7.8x** |
| 8 | 388 | **4.0x** |

## Limitations and honest positioning

remex is not the best tool for every compression scenario. Here's where it falls short:

- **Real embeddings degrade more than synthetic.** On all-MiniLM-L6-v2 embeddings, 4-bit R@10 drops from ~0.85 (synthetic) to 0.707 (real). Real embeddings cluster by topic, and tight clusters amplify quantization errors in ranking. The Gaussian assumption holds globally but not per-dimension (sigma varies 2x across dimensions, from ~0.03 to ~0.06).

- **FAISS PQ wins at matched compression on real data.** At 16x compression, FAISS PQ (m=96) achieves R@10=0.816 vs remex 2-bit at 0.517. FAISS trains on your data and learns subspace structure. This is the fundamental data-oblivious vs data-adaptive trade-off.

- **Linear scan only (for now).** Search is brute-force dot product over all vectors. No sublinear indexing (IVF, HNSW). At >100k vectors, latency grows linearly. The two-stage architecture is a natural fit for adding partition-based coarse search, or for plugging remex's training-free encoding into an existing ANN index.

- **Matryoshka nesting penalty.** Nested codebooks are ~1.2% worse than independently optimized codebooks at 4-bit, but up to ~10% worse at 2-bit. For two-stage search, the coarse pass only needs to identify the right neighborhood (not rank precisely), so the penalty is less impactful in practice.

- **CPU only (for now).** All operations are NumPy on CPU. The hot path (matrix multiply + top-k) maps trivially to GPU via CuPy or PyTorch tensors — contributions welcome.

## API reference

### `Quantizer(d, bits=4, seed=42)`

Main quantizer class (formerly `PolarQuantizer`, which remains available as a deprecated alias).

- **`d`** — Vector dimension (must match your embeddings).
- **`bits`** — Bits per coordinate, 1-8. Sweet spot is 3-4. Use 8 for near-lossless.
- **`seed`** — Random seed for the rotation matrix. Same seed = same quantizer.

#### Methods

**`encode(X)`** — Quantize `(n, d)` float32 array. Returns `CompressedVectors`.

**`decode(compressed, precision=None)`** — Reconstruct `(n, d)` float32 from compressed. Optional `precision` (1 to bits) for Matryoshka decode.

**`search(compressed, query, k=10, precision=None)`** — Find k nearest neighbors by approximate inner product. Caches a dequantized float32 matrix for fast repeated queries. Returns `(indices, scores)`.

**`search_adc(compressed, query, k=10, precision=None, chunk_size=4096)`** — Memory-efficient search via lookup-table scoring. No float32 cache — peak memory is `chunk_size * d * 4` bytes (~6 MB). Slower per-query but uses 5x less RAM. Returns `(indices, scores)`.

**`search_twostage(compressed, query, k=10, candidates=500, coarse_precision=None)`** — Two-stage Matryoshka retrieval: ADC coarse scan (no cache) then full-precision rerank on candidates only. Memory-efficient: only the small candidate set is dequantized. Returns `(indices, scores)`.

**`mse(X, precision=None)`** — Mean per-vector reconstruction error (L2 squared).

### `CompressedVectors`

Container for quantized data. Created by `Quantizer.encode()`.

#### Properties

- **`n`** — Number of vectors.
- **`nbytes`** — Bit-packed size in bytes (honest compression).
- **`nbytes_unpacked`** — In-memory size (uint8 indices + float32 norms).
- **`compression_ratio`** — `(n * d * 4) / nbytes`.
- **`resident_bytes`** — Actual RAM including any active caches.

#### Methods

- **`save(path)`** — Save to `.npz` with bit-packed indices.
- **`load(path)`** — Class method. Load from `.npz`.
- **`subset(idx)`** — Return a new `CompressedVectors` with only the given row indices.
- **`drop_cache()`** — Free the dequantized float32 cache to reclaim memory.

### `GPUSearcher` (optional)

GPU-accelerated search wrapper. Requires CuPy or PyTorch with CUDA. Falls back to NumPy.

```python
from remex.gpu import GPUSearcher

searcher = GPUSearcher(pq, compressed)                # auto-detect backend
searcher = GPUSearcher(pq, compressed, backend="torch")  # explicit

indices, scores = searcher.search(query, k=10)           # cached, fast
indices, scores = searcher.search_adc(query, k=10)       # low-memory ADC
indices, scores = searcher.search_twostage(query, k=10, candidates=200)
```

### Memory profiles (100k vectors, d=384)

| Strategy | Resident RAM | R@10 | ms/query |
|---|---|---|---|
| `search()` (cached) | 192 MB | 0.981 | 2.6 |
| `search_adc()` (no cache) | 39 MB | 0.981 | 169 |
| `search_twostage()` 4→8 (200 cands) | 39 MB | 0.981 | 175 |
| `search_twostage()` 2→8 (200 cands) | 39 MB | 0.945 | 188 |

Choose `search()` when latency matters and RAM is available. Choose `search_adc()` or `search_twostage()` when memory is constrained (e.g. serverless, edge, or very large corpora).

### Low-level utilities

```python
from remex import pack, unpack, packed_nbytes
from remex import lloyd_max_codebook, nested_codebooks
```

- **`pack(indices, bits)`** / **`unpack(packed, bits, n_values)`** — Bit-pack/unpack uint8 arrays.
- **`packed_nbytes(n_values, d, bits)`** — Compute packed byte count.
- **`lloyd_max_codebook(d, bits)`** — Generate optimal boundaries and centroids for N(0, 1/d).
- **`nested_codebooks(d, max_bits)`** — Build Matryoshka centroid tables for all bit levels 1..max_bits.

## Comparison with alternatives

### vs TurboQuant (Zandieh et al.)

TurboQuant adds QJL (quantized Johnson-Lindenstrauss) residual correction for unbiased inner product estimates. This matters for KV cache attention where unbiasedness is critical, but **hurts retrieval** — the variance from QJL outweighs the debiasing when only ranking order matters. remex implements only the MSE-optimal rotation + Lloyd-Max stage, which empirically dominates for nearest-neighbor search.

### vs FAISS Product Quantization

| | remex | FAISS PQ |
|---|---|---|
| Training | None | Required (trains on corpus) |
| Recall at matched compression | Lower on real data | Higher (learns structure) |
| Encode speed | ~20us/vec | ~200us+/vec |
| Corpus updates | Re-encode only new vectors | Retrain or accept stale codebook |
| Index portability | Quantizer is (d, bits, seed) | Must ship trained index |
| Sublinear search | No (brute-force) | Yes (IVF, HNSW) |
| GPU support | No | Yes |

**Use FAISS when**: You have a stable, large corpus, need sublinear search, and can afford training time.

**Use remex when**: You want zero training, fast encode, frequently changing corpora, or near-lossless 8-bit caching (R@10=0.974 at 4x compression).

### vs scalar quantization (naive rounding)

Without the rotation step, scalar quantization on raw embeddings is catastrophically bad — embeddings are highly anisotropic (variance ratios of 10^7x across dimensions). The random rotation is what makes scalar quantization viable: it spreads information uniformly across coordinates.

At 3-bit, remex achieves 72-80% R@10 vs ~40% for naive scalar quantization on the same data (tested in `tests/test_polar_embed.py::TestRetrieval::test_beats_naive_at_3bit`).

## Installation

```bash
pip install remex                       # from PyPI (when published)
pip install -e ".[dev]"                  # development: + pytest, pytest-cov
pip install -e ".[bench]"               # benchmarking: + faiss-cpu, sentence-transformers
```

Run the test suite:

```bash
pytest                                   # 88 tests
pytest -v                                # verbose output
python bench/benchmark.py                # self-contained benchmark (no extra deps)
python bench/real_embedding_eval.py      # real embeddings (requires bench deps)
```

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

## License

MIT
