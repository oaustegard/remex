# CLAUDE.md

## Project overview

**remex** (formerly polar-embed) is a Python library for retrieval-validated embedding compression. It implements random orthogonal rotation + Lloyd-Max scalar quantization (from TurboQuant, Zandieh et al. ICLR 2026) to compress embedding vectors 2-16x with measured recall, optimized for nearest-neighbor retrieval in RAG systems.

Key differentiator: **data-oblivious** — no training required. The quantizer is fully determined by `(dimension, bits, seed)`.

## Architecture

```
remex/
├── __init__.py       # Public API, version
├── core.py           # Quantizer, CompressedVectors (main classes)
├── codebook.py       # Lloyd-Max codebooks + Matryoshka nested tables
├── packing.py        # Bit-packing for sub-byte storage (1-8 bit)
├── rotation.py       # Haar random orthogonal rotation via QR
└── gpu.py            # Optional GPU backend (CuPy/PyTorch/NumPy)

tests/
├── test_polar_embed.py   # Core: rotation, codebook, quantizer, retrieval, packing
├── test_matryoshka.py    # Nested codebooks, precision parameter, two-stage search, subset
├── test_adc_gpu.py       # ADC search, memory accounting, GPUSearcher (numpy fallback)
├── test_packed_vectors.py # PackedVectors creation, unpacking, ADC, serialization
└── test_coverage_gaps.py  # Edge cases, save/load all bits, subset search

bench/
├── benchmark.py          # Self-contained benchmark (no external deps)
├── real_embedding_eval.py  # Real embeddings benchmark (needs sentence-transformers, faiss)
└── RESULTS.md            # Benchmark results with distribution sensitivity analysis
```

### Data flow

```
float32 embeddings
    → normalize (store norms separately)
    → rotate (R @ x, random orthogonal matrix)
    → quantize (searchsorted into Lloyd-Max boundaries → uint8 indices)
    → CompressedVectors (indices + norms)

Search:
    → rotate query (R @ q)
    → score via matmul (cached dequant) or ADC (lookup table over indices)
    → top-k selection
```

### Three search strategies

| Method | Memory | Speed | When to use |
|--------|--------|-------|-------------|
| `search()` | High (caches n*d*4 float32) | Fast (matmul) | Repeated queries, RAM available |
| `search_adc()` | Low (uint8 indices only) | Slower (table lookup) | Memory-constrained, serverless |
| `search_twostage()` | Low (ADC coarse + small fine) | Medium | Best recall/memory trade-off |

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
```

## Code conventions

- **NumPy-only core**: No PyTorch/CuPy dependency in `remex/core.py`. GPU support is opt-in via `remex/gpu.py`.
- **No training**: Fully data-oblivious. The quantizer is determined by `(d, bits, seed)` alone.
- **Honest compression**: `nbytes` property uses bit-packed sizes, not uint8. Benchmark tables report packed compression ratios.
- **Deterministic**: Same `(d, bits, seed)` must produce identical results across runs.
- **Test thresholds**: Recall tests use conservative bounds (e.g. 2-bit R@10 >= 0.3, not exact values) because recall depends on random data.

## Key design decisions

1. **Norms stored separately as float32** — preserves inner-product ranking up to quantization error. This is why 8-bit gives R@10=0.98+ despite "only" 4x compression.

2. **Matryoshka via right-shift** — An n-bit index's top k bits are a valid k-bit code. This enables two-stage search from a single encoding, but incurs ~1.2% nesting penalty at 4-bit and ~10% at 2-bit vs independently optimized codebooks.

3. **ADC for memory efficiency** — The lookup table `(d, 2^bits)` is tiny (~6KB for 2-bit d=384). Chunked scoring keeps temporary allocation at ~6MB regardless of corpus size.

4. **GPU is a wrapper, not a fork** — `GPUSearcher` wraps `Quantizer + CompressedVectors` rather than replacing them. The core stays pure NumPy.

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
