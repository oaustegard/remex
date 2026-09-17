# remex

Retrieval-validated embedding compression. 2-16x smaller vectors with measured recall.

Based on the rotation + Lloyd-Max scalar quantization insight from [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026), focused on the use case that matters most to most AI Engineers: **embedding storage and retrieval for RAG systems**.

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

The quantizer is fully determined by `(d, bits, seed, rotation, normalize, scale)` — no training, no fitting, no index to ship. `rotation` defaults to `"haar"` and is part of the encoding exactly as `seed` is; every container records it, and decoding against the wrong one raises rather than returning wrong-but-plausible vectors. `normalize` and `scale` select the pipeline described below versus [scalar mode](#scalar-mode-codes-as-hash-keys), and default to the former.

## How it works

Three steps, each with a clear purpose:

1. **Random rotation** — A fixed orthogonal matrix transforms any embedding distribution so that coordinates become approximately i.i.d. N(0, 1/d). This is the key insight from TurboQuant: it makes quantization **data-oblivious**, meaning no training data is required.

   Two constructions are available, selected by `rotation=`:

   | | construction | d=768 | d=1536 | d=3072 |
   |---|---|--:|--:|--:|
   | `"haar"` *(default)* | Householder QR, O(d³) | 1.27 s | 10.44 s | 116.29 s |
   | `"rht"` | randomized Hadamard, O(d² log d) | 0.034 s | 0.242 s | 1.00 s |
   | | | **38×** | **43×** | **116×** |

   *Building the rotation matrix only, min of 2, single-core Xeon @ 2.10 GHz.
   Not `Quantizer.__init__` as a whole, which also builds the codebook.*

   `"haar"` is [Haar-distributed](https://arxiv.org/abs/math-ph/0609050);
   `"rht"` is the randomized Hadamard transform — the standard
   incoherence-processing rotation, and what the coordinates-become-Gaussian
   argument actually needs.

   It measures indistinguishable from Haar on retrieval recall (−0.0001 ± 0.0013, pooled over 3 corpora × 6 bit widths × 5 seeds). So it is a build-time option, not a quality improvement — which is why the default has not moved. It needs an even `d`; odd dimensions raise and must use `"haar"`.

   Since the next release, `"rht"` is applied in operator form: permute, flip signs, and run a block fast Walsh–Hadamard transform per row, without building the d×d matrix. A small C kernel is compiled on first use; with no C compiler, or `REMEX_NO_NATIVE=1`, a NumPy implementation gives the same bits more slowly (a warning is logged). The operator's codes are identical across x86, ARM and Apple Silicon. Against the dense matrix, `Quantizer` construction at d=3072 takes 6–11% of the time, and batch encode 41–67% (measured through the public API on GitHub runners; [details](https://github.com/oaustegard/experiments/tree/main/rht-operator-native)). Rows are split across threads above `REMEX_PARALLEL_MIN` input floats (default 2^15); `REMEX_NUM_THREADS` caps the thread count. The Mojo port rebuilds the same matrix byte-for-byte off the same PCG64 stream and encodes through it, so its codes match Python's in all but about 1e-6 of coordinates.

2. **[Lloyd-Max](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) scalar quantization** — Each coordinate is independently quantized using optimal boundaries for the N(0, 1/d) distribution. The codebook is computed from the theoretical Gaussian CDF, not from data. This produces the minimum mean-squared-error scalar quantizer for Gaussian inputs.

3. **Bit-packing** — Indices are stored at their actual bit width (not wasteful uint8), giving honest compression ratios. A 4-bit codebook uses 4 bits per coordinate on disk.

4. **Optional centering** — `Quantizer(mean=...)` encodes each vector's offset from a corpus mean you supply, and solves for the stored length so the reconstruction keeps the original vector's norm. Both halves are one feature: subtracting a mean and keeping the residual's own length *loses* recall at every bit width. The correction rides in the norms column that already exists, so a centred index costs no extra bytes per vector beyond the one stored mean.

   remex never measures the mean — the caller declares it, the way it declares `scale` in scalar mode — and `remex.corpus_mean(X)` is the blessed way to compute one. Whether it pays depends on how much of your corpus is a shared direction: on SPECTER2 (raw inner product) it is worth +0.03 to +0.08 R@10, on L2-normalised all-MiniLM-L6-v2 +0.005 to +0.018, and on isotropic Gaussian vectors nothing. Measure with [`bench/centered_eval.py`](bench/centered_eval.py) before turning it on.

5. **Reconstruction-length correction** — Norms are stored separately as float32, but the direction they multiply is a quantized one whose own length is not 1: at 2-bit it measures 0.89-0.97 and varies per vector. Multiplying by the stored norm alone therefore reconstructs a vector about 1% off in length, per vector, which reorders any neighbours closer together than that. `renorm=True` (the default) divides the length out. It is read off the codes, so nothing extra is stored and no format changes; pass `renorm=False` to reproduce the previous behaviour.

   The gain is set by how tightly packed the corpus is rather than by the size of the bias, which measures the same on Gaussian data as on real embeddings. On all-MiniLM-L6-v2 it is worth +0.21 R@10 at 4-bit; on isotropic Gaussian vectors it is worth nothing. See [`bench/norm_correction_eval.py`](bench/norm_correction_eval.py).

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
| remex 8-bit | 4.0x | 0.0001 | 0.992 | 0.998 |
| remex 4-bit | 7.8x | 0.0097 | 0.919 | 0.977 |
| remex 3-bit | 10.4x | 0.0351 | 0.862 | 0.958 |
| remex 2-bit | 15.4x | 0.1218 | 0.759 | 0.928 |
| remex 1-bit | 29.5x | 0.4050 | 0.635 | 0.880 |
| FAISS PQ (m=96, 8-bit, trained) | 16.0x | 0.0341 | 0.584 | 0.897 |
| FAISS PQ (m=48, 8-bit, trained) | 32.0x | 0.0636 | 0.424 | 0.845 |

Re-measured 2026-09-08 with `renorm=True`. Without it the remex rows read
0.971 / 0.709 / 0.597 / 0.502 / 0.635 at R@10. The FAISS rows were re-run at
the same time on faiss 1.15.0: their MSE reproduces the previously published
values exactly, their recall does not (0.584 here against 0.816 before at
m=96), so the difference is in PQ codebook training rather than in the
harness. Both FAISS scoring paths — `index.search` and reconstruct-then-score
— agree with each other here.

### Centering (`Quantizer(mean=...)`), R@10 plain → centred

| corpus | ‖mean‖ / mean ‖x‖ | 1-bit | 2-bit | 3-bit | 4-bit | 8-bit |
|---|---|---|---|---|---|---|
| SPECTER2 broad, 9.5k, d=768 | 0.92 | 0.637 → 0.686 | 0.773 → 0.852 | 0.864 → 0.913 | 0.917 → 0.949 | 0.994 → 0.996 |
| SPECTER2 narrow, 9.5k, d=768 | 0.92 | 0.670 → 0.706 | 0.789 → 0.858 | 0.869 → 0.920 | 0.922 → 0.958 | 0.993 → 0.998 |
| all-MiniLM-L6-v2, 10k, d=384 | 0.51 | 0.635 → 0.609 | 0.759 → 0.777 | 0.862 → 0.868 | 0.919 → 0.924 | 0.992 → 0.992 |
| synthetic gaussian, 9.5k | 0.01 | 0.258 → 0.254 | 0.540 → 0.542 | 0.732 → 0.728 | 0.860 → 0.858 | 0.990 → 0.989 |

The second column is what predicts the gain: how long the corpus mean is
against a typical vector. Near zero, centering does nothing. Reproduce with
`python bench/centered_eval.py --specter2`, which also prints the naive arm —
subtracting the mean without restoring the full length — that loses 0.03 to
0.18 R@10 and is why the two halves ship as one feature.

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

- **You need the smallest possible index.** remex spends bits per coordinate, so at a fixed byte budget a trained product quantizer packs more dimensions per byte. remex 4-bit is 196 B/vector against FAISS PQ's 96 B at m=96; the recall comparison above favours remex, but it is not a like-for-like size.
- **Your embeddings form extremely tight clusters.** Tight neighbourhoods are where quantization error flips rankings. `renorm=True` repairs most of this — at cluster spread σ=0.01 it takes 4-bit R@10 from 0.100 to 0.527 and 8-bit from 0.815 to 0.956 — but σ=0.01 is still the regime where 4-bit is not enough on its own. Go to 8-bit, or rerank.
- **You need sublinear search at high recall.** The flat scan is exhaustive; `IVFCoarseIndex` (below) gives a sublinear coarse tier but is approximate by construction and earns its keep only above ~10M vectors. For anything more demanding, consider FAISS IVF, HNSW, or similar — remex's compact encoding can feed an external ANN index.

### Distribution sensitivity (10k corpus, d=384, 200 queries, varying cluster tightness)

R@10, `renorm=False` → `renorm=True`. The correction is worth most exactly
where remex was weakest, and nothing where neighbours are already far apart.

| Cluster spread (σ) | rank-10→50 score gap | 2-bit | 4-bit | 8-bit |
|---|---|---|---|---|
| 0.01 (very tight) | 0.002 | 0.048 → 0.159 | 0.100 → 0.527 | 0.815 → 0.956 |
| 0.05 | 0.038 | 0.449 → 0.525 | 0.774 → 0.846 | 0.976 → 0.985 |
| 0.10 | 0.112 | 0.597 → 0.617 | 0.868 → 0.885 | 0.988 → 0.989 |
| 0.30 (typical) | 0.163 | 0.537 → 0.544 | 0.851 → 0.853 | 0.984 → 0.985 |
| 1.00 (diffuse) | 0.163 | 0.541 → 0.544 | 0.866 → 0.864 | 0.989 → 0.990 |

The middle column is the median relative score gap between the 10th and 50th
true neighbour — the distance a ~1% reconstruction-length error has to cross
to reorder anything. It, not the bit width, predicts the size of the gain.

**Detection**: If your 4-bit R@10 is significantly below 0.80 on a held-out
set, your embeddings likely have tight clusters. Use 8-bit, or rerank.

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

### `Quantizer(d, bits=4, seed=42, rotation="haar", normalize=True, scale=None, renorm=True, mean=None)`

Main quantizer class (formerly `PolarQuantizer`, which remains available as a deprecated alias).

- **`d`** — Vector dimension (must match your embeddings).
- **`bits`** — Bits per coordinate: 1-4 or 8. Sweet spot is 3-4. Use 8 for near-lossless.
- **`seed`** — Random seed for the rotation matrix. Same seed = same quantizer.
- **`rotation`** — `"haar"` (default), `"rht"`, or `"none"` (the identity — see [scalar mode](#scalar-mode-codes-as-hash-keys)). Part of the encoding exactly as `seed` is: every container records it, a file written before rotations were recorded resolves to `"haar"`, and decoding against the wrong one raises.
- **`normalize`** — `True` (default) factors each vector into unit direction plus a stored norm, as described above. `False` selects [scalar mode](#scalar-mode-codes-as-hash-keys): quantize the coordinates directly, store no norms. Also part of the encoding — a mismatch raises.
- **`scale`** — Scalar mode only (default `1.0`): the coordinate standard deviation the Lloyd-Max cells are cut for. The normalizing path derives it from the unit sphere as `1/sqrt(d)` and rejects an explicit value.
- **`mean`** — `None` (default) encodes whole vectors. A `(d,)` array turns on centered mode: codes become offsets from that mean, and the reconstruction is scaled to the original vector's length. Part of the encoding like `rotation` — every container records it and decoding against a different one raises. Requires `renorm=True`, and is rejected in scalar mode. Use `remex.corpus_mean(X)` to compute one; remex will not measure it for you.
- **`renorm`** — `True` (default) divides out the decoded direction's length so a reconstruction has the norm that was stored for it. Unlike `rotation` and `normalize` this is *not* part of the encoding: it changes how codes are read, never what they are, so it is not recorded in any container and the same file decodes under either setting. `False` reproduces the previous behaviour. No effect at 1-bit, where every decoded direction has the same length.

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
- **`has_norms`** — `False` for scalar-mode containers, whose `norms` is `None`. Both size properties account for the absent column.

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

### `IVFCoarseIndex` (sublinear coarse-tier scan)

Inverted-file index over the coarse Matryoshka tier. Lets you visit
only `nprobe` of `2**n_bits` cells per query, replacing the
bandwidth-bound flat coarse scan in two-stage retrieval. **Stays
data-oblivious** — no k-means, no training, no fitting.

```python
from remex import IVFCoarseIndex, Quantizer

pq = Quantizer(d=768, bits=8, seed=42)
compressed = pq.encode(corpus)              # CompressedVectors or PackedVectors

# Mode 1: random-hyperplane LSH (SimHash). Pure data-oblivious — works
# on any embedding distribution. Determined by (d, n_bits, seed).
ivf = IVFCoarseIndex(pq, compressed, n_bits=12, mode="lsh", seed=0)

# Mode 2: sign of the first n_bits post-rotation coords. Free given
# the existing rotation (these bits are already MSBs of the encoded
# indices). Cell balance depends on rotated coords being ~i.i.d.
# Gaussian, which is checked by bench/specter2_eval.py.
ivf = IVFCoarseIndex(pq, compressed, n_bits=12, mode="rotated_prefix")

# Stage-1 only — top-K candidates from the visited cells, ADC scored
indices, scores = ivf.search_coarse(query, k=500, nprobe=8, precision=1)

# End-to-end: IVF coarse + full-precision rerank
indices, scores = ivf.search_twostage(
    query, k=10, candidates=500, nprobe=8, coarse_precision=1
)
```

Multi-probe is by Hamming distance from the query's hash code: the
`nprobe` cells with the lowest Hamming distance to `q_hash` are
visited (ties broken by cell ID). Setting `nprobe = 2**n_bits`
recovers a flat scan; the index is exact in that limit and tests
verify byte-identical agreement with `Quantizer.search_adc` /
`Quantizer.search_twostage`.

#### When IVF wins, when flat-scan wins

IVF is for the regime where stage-1 latency is the bottleneck (≥ tens
of millions of vectors). The trade-off is recall vs latency:

| `nprobe` / `n_cells` | Pool scanned | Recall vs flat | Speedup |
|---|---|---|---|
| 1 / 2^b | ~1/2^b of corpus | low — only same-cell neighbors | up to ~2^b |
| ~5–25% | ~5–25% of corpus | typical 0.85–0.95 R@10 | 4–20× |
| 100% | full corpus | 1.0 (bit-identical to flat) | 0.95–1.0× |

Flat-scan wins when:

- Corpus < ~10M vectors. Stage-1 is already < 50 ms; the IVF index
  overhead and per-query hash cost don't pay back.
- Recall@K must equal flat-scan exactly. IVF is approximate by
  construction — vectors in unvisited cells are missed.
- Embeddings are deeply mixed and queries are uniformly distributed
  in angle, so cells don't capture meaningful neighborhoods.

Bridge-edge preservation (cross-FoS / cross-partition recall) is
benchmarked explicitly in `bench/specter2_eval.py` — running broad +
narrow SPECTER2 partitions concatenated and reporting how many of the
flat-scan top-K cross-partition hits the IVF top-K preserves at each
`nprobe`. Both hash modes are content-based (hyperplane signs on the
rotated representation), so they don't partition by FoS — but at very
low `nprobe` cross-partition hits drop simply because pool size
shrinks.

#### Memory cost (excluding the corpus)

| Component | Bytes |
|---|---|
| `cell_ids` | `2 * n` |
| `sorted_idx` | `8 * n` |
| `cell_offsets` | `8 * (2**n_bits + 1)` |
| `hyperplanes` (lsh only) | `4 * n_bits * d` |

For 100M vectors at `n_bits=12`: ~960 MB index overhead vs ~9.6 GB
1-bit coarse memory — about 10% surcharge for ~5–20× stage-1 speedup
at moderate `nprobe`.

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
- **`lloyd_max_codebook(d, bits, sigma=None)`** — Generate optimal boundaries and centroids for N(0, sigma); `sigma=None` is the unit-sphere value `1/sqrt(d)`.
- **`nested_codebooks(d, max_bits, sigma=None)`** — Build Matryoshka centroid tables for all bit levels 1..max_bits.

## Scalar mode: codes as hash keys

`Quantizer(normalize=False)` skips the unit-sphere factorization and quantizes the (optionally rotated) coordinates directly. No norms are computed, none are stored, and the codes are the whole output.

```python
import numpy as np
from remex import Quantizer

# Enumerated algebraic values, not embeddings: heavy-tailed, and full of
# near-parallel families that a unit-norm factorization would collapse.
values = np.array([[c, c, 2 * c, -c] for c in (0.1, 0.7, 1.3, 2.0)])

pq = Quantizer(d=4, bits=4, normalize=False, rotation="none", scale=1.0)
codes = pq.encode(values).indices          # (n, 4) uint8, and no norms array
keys = [bytes(row) for row in codes]       # exact-match hash keys

coarse = codes >> 2                        # 2-bit keys: coarser, more collisions
```

Use it when the codes themselves are the product — hash keys for a join, bucketing, dedup — rather than an approximation of a direction. Two properties of the default pipeline work against that:

- **Unit-norm factorization collapses constant-direction families.** Every `c * ones(d)` shares one direction code, differing only in the norm that scalar mode does not store. On roughly isotropic embeddings that is harmless; on enumerated values those families are enormous, and the shared code is a mega-bucket rather than a key.
- **float32 range.** The normalizing path casts input to float32 and stores float32 norms (for byte-identical [`.pq` parity with the Mojo port](#mojo-port-polarquant)). Values past ~3.4e38 become `inf` there. Scalar mode works in float64 and saturates at the outermost cell instead.

What is unchanged: Lloyd-Max cell shaping, Matryoshka nesting (right-shift a code for a coarser one — a per-query collision/recall dial), and determinism, which is what makes a code usable as an exact-match key at all. `rotation="none"` strengthens that last point: with no rotation matmul in the way, a code is a bare `searchsorted` of the input value, identical across BLAS builds and thread counts, and coordinate *j* of the code depends only on coordinate *j* of the input.

**You own the range conditioning.** remex stays data-oblivious and will not measure your data to pick cells. Bound your inputs or push heavy tails through `arcsinh`/`log`, then set `scale` to the spread you conditioned them to; anything past `±3 * scale` lands in an outermost cell.

Scalar-mode containers carry `norms is None`, which is how every serializer marks them: no `norms` entry in `.npz`, no `norms` column in Arrow, and the no-norms flag (byte 18, bit 0) in `.pq`. Mixing the two modes raises rather than silently rescaling every coordinate. The Mojo port always normalizes, so it does not read scalar-mode `.pq` files and `save_params` rejects a scalar quantizer.

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

## Mojo port (`polarquant`)

**Checkout-only — the Mojo sources are deliberately excluded from the PyPI
distribution** (`.mojo` files are not pip-installable and Mojo is not a
declarable dependency), so `pip install remex` will not contain them. Clone the
repository to build it.

A standalone Mojo CLI binary lives in [`remex/mojo/`](remex/mojo/). It
mirrors the encode + ADC search path with no Python runtime
dependency, reading `.npy` corpus files directly and writing a small
binary `.pq` container that the Python library can load via
`remex.load_pq()` (and vice versa via `remex.save_pq()`).

```bash
cd remex/mojo
mojo build -I . polarquant.mojo -o polarquant
./polarquant encode corpus.npy --bits 4 --seed 42 -o corpus.pq
./polarquant search corpus.pq query.npy --k 10 --seed 42
```

For bit-identical encoding to Python (matching rotations and
codebook), use `--params P.bin` after dumping with
`remex.save_params(quantizer, P)`. See
[`remex/mojo/README.md`](remex/mojo/README.md) for build, test, and
benchmark instructions.

## References

- Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Jégou, Douze & Schmid (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1):117–128. [IEEE Xplore](https://ieeexplore.ieee.org/document/5432202/) — introduces Product Quantization, ADC (Asymmetric Distance Computation), and SDC for approximate nearest neighbor search.
- Kusupati et al. (2022). *Matryoshka Representation Learning.* NeurIPS 2022. [arXiv:2205.13147](https://arxiv.org/abs/2205.13147) — the nested/coarse-to-fine representation idea that inspires remex's bit-level nesting.
- Mezzadri (2007). *How to Generate Random Matrices from the Classical Compact Groups.* Notices of the AMS 54(5):592–604. [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050) — the QR-of-Gaussian method for Haar-distributed orthogonal matrices used in `remex/rotation.py`.
- Lloyd (1982). *Least Squares Quantization in PCM.* IEEE Trans. Information Theory 28(2):129–137. [IEEE Xplore](https://ieeexplore.ieee.org/document/1056489) — optimal scalar quantization (Lloyd-Max algorithm) for minimum MSE.

## License

MIT
