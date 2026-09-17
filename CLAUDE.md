# CLAUDE.md

## Project overview

**remex** (formerly polar-embed) is a Python library for retrieval-validated embedding compression. It implements random orthogonal rotation + Lloyd-Max scalar quantization (from TurboQuant, Zandieh et al. ICLR 2026) to compress embedding vectors 2-16x with measured recall, optimized for nearest-neighbor retrieval in RAG systems.

Key differentiator: **data-oblivious** — no training required. The codes a quantizer writes are fully determined by `(dimension, bits, seed, rotation, normalize, scale)`, with one opt-in exception: `mean` (see design decision 4). `renorm` is a further constructor argument but deliberately not part of that tuple: it changes how codes are read, not what they are. `renorm` is a seventh constructor argument but deliberately not part of that tuple: it changes how codes are read, not what they are.

## Architecture

```
remex/
├── __init__.py       # Public API, version
├── core.py           # Quantizer, CompressedVectors (main classes)
├── codebook.py       # Lloyd-Max codebooks + Matryoshka nested tables
├── ivf.py            # IVFCoarseIndex — coarse-tier IVF, data-oblivious
├── packing.py        # Bit-packing for sub-byte storage (1-8 bit)
├── rotation.py       # Haar (QR), randomized Hadamard (dense + RHTOperator), identity
├── _native.py        # C kernel for RHTOperator, compiled on first use; NumPy fallback
└── gpu.py            # Optional GPU backend (CuPy/PyTorch/NumPy)

tests/
├── test_polar_embed.py   # Core: rotation, codebook, quantizer, retrieval, packing
├── test_matryoshka.py    # Nested codebooks, precision parameter, two-stage search, subset
├── test_adc_gpu.py       # ADC search, memory accounting, GPUSearcher (numpy fallback)
├── test_ivf.py           # IVFCoarseIndex: cell ID, multi-probe, recall, packed interop
├── test_packed_vectors.py # PackedVectors creation, unpacking, ADC, serialization
├── test_coverage_gaps.py  # Edge cases, save/load all bits, subset search
├── test_scalar_mode.py   # normalize=False: hash-key properties, rotation="none", mode guard
├── test_norm_correction.py # renorm: decoded length, per-precision, path agreement, recall
└── test_centered_mode.py   # mean=: length restoration, guards, path agreement, npz

bench/
├── benchmark.py          # Self-contained benchmark (no external deps)
├── real_embedding_eval.py  # Real embeddings benchmark (needs sentence-transformers, faiss)
├── norm_correction_eval.py # renorm off/on: cluster-spread sweep + cached SPECTER2
├── centered_eval.py        # mean= off/on, plus the naive arm that loses recall
└── RESULTS.md            # Benchmark results with distribution sensitivity analysis
```

### Data flow

```
float32 embeddings
    → normalize (store norms separately)
    → rotate (R @ x; "rht" applies RHTOperator instead of a matrix)
    → quantize (searchsorted into Lloyd-Max boundaries → uint8 indices)
    → CompressedVectors (indices + norms)

Search:
    → rotate query (R @ q)
    → score via matmul (cached dequant) or ADC (lookup table over indices)
    → top-k selection
```

**Scalar mode** (`Quantizer(normalize=False)`) drops the first step and the
norms with it — codes are Lloyd-Max indices of the raw coordinates, cut for
a caller-declared `scale` instead of the unit sphere's `1/sqrt(d)`, in
float64 rather than float32. It exists for use cases where the codes *are*
the product (exact-match hash keys, joins, bucketing) rather than an
approximation of a direction; `rotation="none"` pairs with it to make a code
a bare `searchsorted`. `norms is None` is what every container and
serializer uses to record the mode, and mixing modes raises.

### Three search strategies

| Method | Memory | Speed | When to use |
|--------|--------|-------|-------------|
| `search()` | High (caches n*d*4 float32) | Fast (matmul) | Repeated queries, RAM available |
| `search_adc()` | Low (uint8 indices only) | Slower (table lookup) | Memory-constrained, serverless |
| `search_twostage()` | Low (ADC coarse + small fine) | Medium | Best recall/memory trade-off |

### Sublinear coarse-tier scan: `IVFCoarseIndex`

Optional inverted-file index over the coarse Matryoshka tier. Visits
only `nprobe` of `2**n_bits` cells per query, replacing the
bandwidth-bound flat coarse scan in `search_twostage` for very large
corpora (≥ tens of millions of vectors). Two **data-oblivious** hash
modes (no k-means, no fitting):

- `mode='lsh'` — random-hyperplane SimHash, deterministic from
  `(d, n_bits, seed)`.
- `mode='rotated_prefix'` — sign of the first `n_bits` post-rotation
  coordinates; free given the existing rotation matrix.

Multi-probe is by Hamming distance from the query's hash. Setting
`nprobe = 2**n_bits` recovers a flat scan exactly (verified by
test_ivf.py). The latency-recall Pareto and cross-FoS bridge edge
preservation are benchmarked in `bench/specter2_eval.py`.

## Development

```bash
pip install -e ".[dev]"    # numpy, scipy, pytest, pytest-cov
pytest                      # 126 tests, ~6 min
pytest tests/test_adc_gpu.py -v  # just ADC/GPU tests, ~30s
```

### Running benchmarks

```bash
python bench/benchmark.py               # synthetic data, no extra deps
pip install -e ".[bench]"               # for real embedding benchmarks
python bench/real_embedding_eval.py     # needs sentence-transformers + faiss-cpu
python bench/norm_correction_eval.py    # renorm off/on; add --specter2 for the cache
python bench/centered_eval.py           # mean= off/on; add --specter2 for the cache

# SPECTER2 (allenai/specter2_base, d=768) — encoding the transformer takes
# ~50 min on CPU per 10k papers. Skip the encode by pulling a precomputed
# cache from GH release:
bash bench/fetch_specter2_cache.sh      # ~45 MB, restores .specter2_cache/
python bench/specter2_eval.py --cached  # then run the bench against the cache
```

## Code conventions

- **NumPy-only core**: No PyTorch/CuPy dependency in `remex/core.py`. GPU support is opt-in via `remex/gpu.py`. The one compiled piece, `remex/_native.py`, is optional: without a C compiler (or with `REMEX_NO_NATIVE=1`) `RHTOperator` uses a NumPy path that performs the same operations in the same order and returns the same bits. `tests/test_rht_operator.py` enforces that equality.
- **No training**: Fully data-oblivious. The quantizer is determined by `(d, bits, seed, rotation, normalize, scale)` alone. Scalar mode keeps this: `scale` is a value the caller *declares*, never one remex measures off the data.
- **The rotation is part of the encoding**: every persisted container (`.pq` byte 17, `.npz` `rotation` key,
  Arrow `b"rotation"` metadata) records which rotation wrote it, and an absent record means `"haar"` — the
  frozen historical value, never the live default. That rule is what makes the default safe to change; see
  `bench/gates/rotation_identity_gate.py`.
- **So is the mode**: a scalar-mode container (`normalize=False`) has `norms is None`, and every serializer
  records that by *omitting* the norms column (`.npz`/Arrow) or setting the no-norms flag (`.pq` byte 18,
  bit 0). Decoding or searching across the two modes raises, like a rotation mismatch.
- **Honest compression**: `nbytes` property uses bit-packed sizes, not uint8. Benchmark tables report packed compression ratios.
- **Deterministic**: Same `(d, bits, seed)` must produce identical results across runs. `rotation="none"` and `rotation="rht"` strengthen this to bit-identical across machines, because no matmul enters the encoding (`"rht"` uses only gathers, multiplies and pairwise add/sub, built with `-ffp-contract=off`). `"haar"` still depends on the BLAS kernel. Codebook boundaries are midpoints of the float32 centroids for the same reason: float64 midpoints inherited ulp differences from `scipy.stats.norm` across NumPy SIMD levels.
- **Test thresholds**: Recall tests use conservative bounds (e.g. 2-bit R@10 >= 0.3, not exact values) because recall depends on random data.

## Key design decisions

1. **Norms stored separately as float32** — preserves inner-product ranking up to quantization error. This is why 8-bit gives R@10=0.98+ despite "only" 4x compression.

2. **Matryoshka via right-shift** — An n-bit index's top k bits are a valid k-bit code. This enables two-stage search from a single encoding. The nesting penalty depends on which level you extract:
   - **1-bit**: 0% penalty (the MSB of an n-bit Lloyd-Max code *is* the sign bit, which is exactly the standalone 1-bit code). `tests/test_matryoshka.py::TestPrecisionOneBit::test_matryoshka_1bit_equals_standalone_1bit` enforces this bit-for-bit equality.
   - **4-bit**: ~1.2% recall penalty vs an independently-optimized 4-bit codebook.
   - **2-bit**: ~10% recall penalty (worst level — the inner Lloyd-Max boundaries don't align with sign-based partitioning).

3. **Reconstruction-length correction (`renorm=True`, default)** — `norms` is the length of the *original* vector, but it multiplies a quantized direction whose own length is not 1 (0.89-0.97 at 2-bit, varying per vector). `Quantizer._direction_lengths` reads that length off the codes and `_effective_norms` divides it out, so nothing is stored and no container format changes. Worth +0.21 R@10 at 4-bit on all-MiniLM-L6-v2 and +0.18 to +0.28 on SPECTER2; worth ~0.000 on isotropic Gaussian data, which is why the synthetic benchmarks never showed it. Nothing at 1-bit, where every decoded direction has the same length.

   Three things to keep in mind when touching scoring code:
   - **Every path that multiplies by norms must go through `_effective_norms`.** `core.py` (decode, search, search_adc, search_twostage, search_batch), `ivf.py` and `gpu.py` all do. `tests/test_ivf.py` and `tests/test_adc_gpu.py` assert those paths agree with `Quantizer.search`, so missing one fails loudly rather than silently returning a different ranking.
   - **It is per precision.** A Matryoshka level has its own centroid table and therefore its own direction lengths; the cache on the container is keyed by precision for that reason.
   - **The Mojo port does not implement it yet.** `mojo/src/quantizer.mojo` still multiplies raw `norms`, so `polarquant` search diverges from Python search. `.pq` encode parity is unaffected — the codes are identical.

4. **Centered mode (`mean=`, opt-in)** — encodes `x - mu` and solves at encode time for the stored length `m` such that `||mu + m*u_hat|| == ||x||`. That keeps the whole correction in the norms column, so a centered index costs no per-vector bytes; `_centred_lengths` does the quadratic. The two halves are inseparable: centering with the residual's own length loses 0.03-0.18 R@10, which `bench/centered_eval.py`'s naive column shows.

   - **The mean is caller-declared, never measured.** `remex.corpus_mean(X)` computes one, but `Quantizer` will not call it for you — same contract as `scale` in scalar mode, and what keeps "no training" honest.
   - **It is part of the encoding.** Containers carry it, `_check_mean` runs from `_resolve_centroids` like `_check_rotation`, and a mismatch raises rather than returning vectors shifted by the difference.
   - **Every scoring path needs `_query_offset`.** `q . mu` is constant across the corpus so it cannot reorder, but it must be added for scores to match `decode()`. `core.py`, `ivf.py` and `gpu.py` all do; `tests/test_centered_mode.py::TestIvfAndGpuAgree` is what makes a missed path loud, and it was verified to go red with the offsets removed.
   - **`.pq` refuses a centered container** — the format has no section for the mean, and a reader that dropped it would decode every vector shifted. `.npz` and Arrow round-trip it. The Mojo port is untouched for the same reason.
   - **Gain is corpus-dependent**, predicted by `||mean|| / mean ||x||`: 0.92 on SPECTER2 (+0.03 to +0.08 R@10), 0.51 on all-MiniLM-L6-v2 (+0.005 to +0.018), 0.01 on isotropic Gaussian (nothing). Not a default.

5. **ADC for memory efficiency** — The lookup table `(d, 2^bits)` is tiny (~6KB for 2-bit d=384). Chunked scoring keeps temporary allocation at ~6MB regardless of corpus size.

6. **GPU is a wrapper, not a fork** — `GPUSearcher` wraps `Quantizer + CompressedVectors` rather than replacing them. The core stays pure NumPy.

## Testing

- All tests in `tests/` directory, run with `pytest`
- Tests use `np.random.default_rng(seed)` for reproducibility
- ADC tests verify exact match with cached search (same top-k, same scores to rtol=1e-5)
- GPU tests run against numpy fallback backend (no GPU required in CI)
- Matryoshka tests cover nesting correctness, precision bounds, and two-stage recall

## Common tasks

**Adding a new search method**: Add to `Quantizer` in `core.py`, add corresponding method in `GPUSearcher` in `gpu.py`, add tests in `test_adc_gpu.py`.

**Changing codebook generation**: Modify `codebook.py`. Run `test_polar_embed.py::TestCodebook` and `test_matryoshka.py::TestNestedCodebooks` — they verify symmetry, monotonicity, and nesting properties.

**Changing bit-packing**: Modify `packing.py`. The `TestPacking` class in `test_polar_embed.py` runs roundtrip tests for all 1-8 bit widths.
