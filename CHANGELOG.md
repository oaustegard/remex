# Changelog

## v0.7.0 — 2026-09-11

Five weeks since v0.6.0 (2026-08-04), from six pull requests.

Two of the entries below came from reading RSLM
([arXiv 2608.30384](https://arxiv.org/abs/2608.30384)) against this codebase in
September. RSLM stores a scale so that the reconstructed vector's norm is right;
checking why remex did not need one showed that it did. The decoded direction is
not unit length, so `norms * u_hat` had been about 1% long or short, by a
different amount per vector, in every release so far. RSLM's other main claim,
encoding a residual against a partition centroid, arrives here in the form the
data-oblivious design allows: one caller-supplied corpus mean. Both land without
a format change or a stored byte, and `renorm` is on by default, so upgrading
changes what `search` returns.

Scalar mode and `rotation="none"` are unrelated and merged in August.

### Added

- **Centered mode — `Quantizer(..., mean=...)`, opt-in**
  ([#81](https://github.com/oaustegard/remex/issues/81)). Encodes each
  vector's offset from a corpus mean the caller supplies, and solves at
  encode time for the stored length `m` such that `‖mu + m·u_hat‖ = ‖x‖`.
  Putting the correction in the norms column the container already has means
  a centered index costs no extra bytes per vector — only the one stored
  mean, `d × 4` bytes for the whole index.

  Both halves are one feature. Subtracting a mean and keeping the residual's
  own length loses 0.03 to 0.18 R@10 at every bit width measured;
  `bench/centered_eval.py` reports that naive arm alongside the real one.

  R@10, plain → centered:

  | corpus | ‖mean‖ / mean ‖x‖ | 1-bit | 2-bit | 4-bit |
  |---|---|---|---|---|
  | SPECTER2 broad, 9.5k | 0.92 | 0.637 → 0.686 | 0.773 → 0.852 | 0.917 → 0.949 |
  | SPECTER2 narrow, 9.5k | 0.92 | 0.670 → 0.706 | 0.789 → 0.858 | 0.922 → 0.958 |
  | all-MiniLM-L6-v2, 10k | 0.51 | 0.635 → 0.609 | 0.759 → 0.777 | 0.919 → 0.924 |
  | synthetic gaussian, 9.5k | 0.01 | 0.258 → 0.254 | 0.540 → 0.542 | 0.860 → 0.858 |

  The second column predicts the gain, and it is why this is opt-in rather
  than a default: near zero, centering does nothing, and at 1-bit on MiniLM
  it costs 0.026.

  remex never measures the mean. `remex.corpus_mean(X)` computes one and the
  caller passes it, the same contract `scale` has in scalar mode, which is
  what keeps the data-oblivious claim honest. The mean is part of the
  encoding: every container records it, and `_check_mean` refuses codes
  written against a different one the way `_check_rotation` already refuses a
  different rotation.

- **`remex.corpus_mean(X)`** — float64 accumulation, float32 result, so the
  value does not depend on how the rows were chunked.

- **`bench/centered_eval.py`** — plain, centered and naive-centered arms
  across synthetic, MiniLM and (under `--specter2`) the cached SPECTER2
  partitions. Prints `‖mean‖ / mean ‖x‖` next to the recall so the decision
  is visible before the sweep.

- **Reconstruction-length correction — `Quantizer(..., renorm=True)`, on by
  default** ([#81](https://github.com/oaustegard/remex/issues/81)). remex
  stores each vector's exact norm and a quantized unit direction, then
  reconstructs `norms * u_hat`. The decoded direction is not unit length: at
  2-bit it measures 0.89-0.97 and varies per vector, so every reconstruction
  came out about 1% off in length, by a different amount each time. That is
  enough to reorder any two neighbours closer together than 1%. `renorm`
  divides the length out. It is computed from the codes, so no bytes are
  stored, no container format changes, and a file written by either setting
  decodes under the other.

  Measured R@10, before -> after:

  | corpus | 2-bit | 3-bit | 4-bit | 8-bit |
  |---|---|---|---|---|
  | all-MiniLM-L6-v2, 10k | 0.502 -> 0.759 | 0.597 -> 0.862 | 0.709 -> 0.919 | 0.971 -> 0.992 |
  | SPECTER2 broad, 9.5k | 0.517 -> 0.773 | 0.611 -> 0.864 | 0.736 -> 0.917 | 0.974 -> 0.994 |
  | SPECTER2 narrow, 9.5k | 0.508 -> 0.789 | 0.602 -> 0.869 | 0.718 -> 0.922 | 0.967 -> 0.993 |
  | synthetic Gaussian, 9.5k | 0.542 -> 0.544 | 0.737 -> 0.739 | 0.858 -> 0.861 | - |

  How tightly packed the corpus is sets the size of the gain, not the size of
  the bias, which measures the same on Gaussian data as on real embeddings: a
  1% error only reorders neighbours within 1% of each other. That is why every
  synthetic benchmark in this repo missed it for six releases. 1-bit is
  unaffected, because there every decoded direction has the same length and a
  common factor cannot reorder anything.

  Reconstruction MSE moves the other way at the low bit widths (SPECTER2
  2-bit 58.3 -> 60.1, 1-bit 176.9 -> 197.6, unchanged from 3-bit up). The
  correction targets ranking rather than squared error, and `mse()` reports
  what `decode()` returns, so it reflects that.

  Prior art: RSLM ([arXiv 2608.30384](https://arxiv.org/abs/2608.30384)) makes
  the same correction and stores an explicit 2-byte scale for it. remex needs
  no stored scale because it reconstructs the direction before scoring.

- **`bench/norm_correction_eval.py`** — cluster-spread sweep with the
  correction off and on, plus the cached SPECTER2 partitions under
  `--specter2`. Reports the median rank-10-to-50 score gap alongside recall,
  which is the quantity that predicts the gain.

- **Scalar mode — `Quantizer(d, bits, normalize=False, scale=1.0)`**
  ([#77](https://github.com/oaustegard/remex/issues/77)). Quantizes the
  (optionally rotated) coordinates directly with the Lloyd-Max codebook:
  no unit-sphere factorization, no norms computed or stored, codes are the
  whole output. For use cases where the codes *are* the product — exact-match
  hash keys for a join, bucketing, dedup — the default pipeline works against
  you twice: unit-norm factorization maps every constant-direction family onto
  a single direction code, and the float32 cast that buys `.pq` parity with
  the Mojo port turns large values into `inf`. Scalar mode drops the first and
  works in float64, saturating at the outermost cell instead. Lloyd-Max cell
  shaping, Matryoshka nesting and determinism are unchanged.

- **`rotation="none"`** — the identity, on-disk code 2. Intended for scalar
  mode, where it makes a code a bare `searchsorted` of the input value: no
  matmul, so identical inputs give identical codes across BLAS builds and
  thread counts, and coordinate *j* of the code depends only on coordinate
  *j* of the input.

- **`sigma` on `lloyd_max_codebook` / `nested_codebooks`**, plus
  `coordinate_sigma(d, sigma)` — the coordinate spread the cells are cut for.
  `None` remains the unit-sphere `1/sqrt(d)`, bit-for-bit as before.

- **`has_norms`** on `CompressedVectors` / `PackedVectors`.

### Changed

- **`IVFCoarseIndex` and `GPUSearcher` scoring add `Quantizer._query_offset`.**
  `q · mu` is constant across the corpus and cannot reorder, but it has to be
  added for their scores to match `Quantizer.search`.
  `tests/test_centered_mode.py::TestIvfAndGpuAgree` covers both, and was
  verified to fail with the offsets removed.
- **`save_pq` raises on a centered container.** The `.pq` format has no
  section for the mean, and a reader that dropped it would decode every
  vector shifted. `.npz` and Arrow round-trip it.

- **`GPUSearcher._scale` takes a `precision` argument** and resolves norms
  through `Quantizer._effective_norms`, cached per precision on device.
  `IVFCoarseIndex` scoring does the same. Both previously read
  `compressed.norms` directly, which diverges from `Quantizer.search` once the
  correction is on; `tests/test_adc_gpu.py` and `tests/test_ivf.py` catch it.
- **`Quantizer._adc_score_packed` takes an optional `norms` argument**,
  defaulting to the container's own, so callers can pass effective norms.
- **README benchmark tables re-measured.** The real-embedding table now
  reports `renorm=True`. Its FAISS rows were re-run at the same time on faiss
  1.15.0: their MSE reproduces the previously published values exactly, their
  recall does not (0.584 against 0.816 at m=96), so the difference sits in PQ
  codebook training rather than in the harness. Flagged in the table rather
  than silently replaced.

- `norms` is now optional on `CompressedVectors` and `PackedVectors`, and its
  absence is how every serializer records scalar mode: no `norms` entry in
  `.npz`, no `norms` column in Arrow, and a no-norms flag at `.pq` byte 18
  bit 0 (previously reserved-and-zero, so existing files are unaffected).
  A scalar-mode `.pq` is *shorter* than an old reader's arithmetic expects, so
  such a reader — the Mojo port included — fails it as truncated rather than
  reading indices as norms. `save_params` rejects a scalar quantizer for the
  same reason.

- Decoding or searching codes with a quantizer in the other mode now raises
  `mode mismatch`, alongside the existing `rotation mismatch` check. The two
  codebooks differ by a factor of `scale * sqrt(d)`, so crossing them would
  otherwise rescale every reconstructed coordinate.

Scalar mode leaves the default path untouched: same codes, same files, same
`(d, bits, seed, rotation)` determinism. The reconstruction-length correction
above does change default *scoring*, though not the codes or the files.

### Known gaps

- The Mojo port (`mojo/src/quantizer.mojo`) still multiplies raw `norms`, so
  `polarquant` search diverges from Python search until it carries the same
  correction. Encode parity is unaffected: the codes are identical either way.

## v0.6.0 — 2026-08-04

Four months since v0.5.1 (2026-04-06), reconstructed from the 29 pull requests
merged in between. Entries below are grouped by the work they belong to rather
than listed per-PR; each links the PRs that carry the detail.

Three things dominate the release: a Mojo port of the quantizer that reaches
parity with the Python implementation, GPU kernels on Mojo 1.0 / Apple Metal,
and — landing last — construction that is two orders of magnitude faster plus a
rotation identity that is finally recorded on disk.

### Added

- **Mojo port of `Quantizer`** — encode and ADC search
  ([#36](https://github.com/oaustegard/remex/pull/36)), `decode()` and
  `PackedVectors` ([#44](https://github.com/oaustegard/remex/pull/44)),
  Matryoshka nested codebooks and `search_twostage`
  ([#43](https://github.com/oaustegard/remex/pull/43)), and `IVFCoarseIndex`
  at parity with Python ([#63](https://github.com/oaustegard/remex/pull/63)).
  A NumPy-bit-identical `--seed` and a deterministic Haar in Python
  ([#46](https://github.com/oaustegard/remex/pull/46)) are what make "parity"
  checkable rather than asserted.

- **GPU / Metal execution.** Scaffolding for `--device gpu`
  ([#45](https://github.com/oaustegard/remex/pull/45)), then encode and ADC
  kernels on Mojo 1.0 and Apple Metal
  ([#65](https://github.com/oaustegard/remex/pull/65)), a persistent
  `GPUCorpus` for search with coalesced `Rᵀ` reads
  ([#66](https://github.com/oaustegard/remex/pull/66)), and `--device auto`
  for per-stage device routing
  ([#68](https://github.com/oaustegard/remex/pull/68)).

- **`IVFCoarseIndex`** — data-oblivious IVF over the Matryoshka coarse tier
  ([#58](https://github.com/oaustegard/remex/pull/58)), with `precision=1`
  extraction covered in tests, bench and docs
  ([#56](https://github.com/oaustegard/remex/pull/56)).

- **Opt-in randomized Hadamard rotation** (`rotation="rht"`), O(d² log d)
  instead of Haar's O(d³)
  ([#71](https://github.com/oaustegard/remex/pull/71)), implemented in Mojo
  byte-identically to Python
  ([#73](https://github.com/oaustegard/remex/pull/73)). Haar remains the
  default.

- **CI** ([#74](https://github.com/oaustegard/remex/pull/74)) — including tests
  for the documentation's own claims, which is what makes the rotation gate
  below mean anything.

- **Benchmark caches and results** — SPECTER2 embeddings fetcher
  ([#57](https://github.com/oaustegard/remex/pull/57)), n=10k 1-bit Matryoshka
  results ([#59](https://github.com/oaustegard/remex/pull/59)), narrowed cache
  with real IVF/bridge numbers
  ([#62](https://github.com/oaustegard/remex/pull/62)), a
  `gemini-embedding-001` cache
  ([#64](https://github.com/oaustegard/remex/pull/64)), SPECTER2 distribution
  analysis ([#32](https://github.com/oaustegard/remex/pull/32)), and Mojo
  bench results in `mojo/bench/RESULTS.md`
  ([#48](https://github.com/oaustegard/remex/pull/48),
  [#54](https://github.com/oaustegard/remex/pull/54)).

- **Research notes** — mixed-precision quantization via residual-error tail
  selection ([#35](https://github.com/oaustegard/remex/pull/35)), Matryoshka ×
  scalar quantization for byte-optimal retrieval
  ([#70](https://github.com/oaustegard/remex/pull/70)), and canonical
  references with ADC defined
  ([#33](https://github.com/oaustegard/remex/pull/33)).

### Packaging

- **The Mojo port is not in the PyPI distribution.** `.mojo` sources are not
  installable by pip and Mojo is not a declarable dependency, so
  `remex.mojo*` is now excluded from `packages.find`. Measured on the 0.6.0
  build: before the exclude the wheel carried **zero** `.mojo` files but did
  ship `remex/mojo/bench/compare.py` and two test-fixture builders — a package
  path present in the wheel with its actual content absent. After, both wheel
  and sdist contain zero `mojo` entries and eight `remex/*.py` modules. Run
  the Mojo and GPU paths from a checkout.

- **`__version__` is resolved from installed metadata.** It was the hardcoded
  literal `"0.5.0"` — stale even against the v0.5.1 already on PyPI — so a
  0.6.0 wheel reported 0.5.0 on import. Now `importlib.metadata.version`, with
  `"0.0.0+unknown"` for a bare checkout. Same fix remax made in its v0.1.0.

### Changed

- **The rotation is recorded in every container, and enforced on read**
  ([#74](https://github.com/oaustegard/remex/pull/74)). Containers recorded
  `(d, bits, n)` and sometimes `seed`, never the rotation, so a reader fell
  back on whatever `Quantizer`'s default happened to be at read time. Flipping
  that default would have decoded every stored index in a frame its codes were
  never written in — nothing raises, `search()` still returns k neighbours, and
  they are the wrong k. #73 made this reachable rather than theoretical by
  letting rht-encoded indexes exist. **An absent record resolves to `"haar"`,
  never to the live default**, so indexes written before this release keep
  decoding correctly.

- **`Quantizer.__init__` is 100–200× faster**
  ([#71](https://github.com/oaustegard/remex/pull/71)). `lloyd_max_codebook`
  called scipy's *scalar* `norm.cdf` / `norm.pdf` 307,200 times; the per-level
  loop vectorizes exactly, because `bounds` is materialised before any centroid
  moves, so every cell reads the same boundaries. Bit-identical output.

  | d | before | after | |
  |---|---|---|---|
  | 768 | 30.6 s | 0.22 s | 139× |
  | 1536 | ~45 s | 0.39 s | ~115× |
  | 3072 | ~358 s | 1.79 s | ~200× |

  *(with `rotation="rht"`; at the Haar default, d=3072 is 208 s → still
  dominated by the QR.)*

- **Mojo hot paths row-blocked at NB=8** — the rotation matvec
  ([#47](https://github.com/oaustegard/remex/pull/47)) and coarse-stage ADC
  scoring ([#52](https://github.com/oaustegard/remex/pull/52)) — plus a
  heap-based coarse top-k
  ([#51](https://github.com/oaustegard/remex/pull/51)) and a SIMD-vectorized
  encode hot path ([#37](https://github.com/oaustegard/remex/pull/37)).

### Reverted

- **The encode `Rᵀ` transpose** ([#67](https://github.com/oaustegard/remex/pull/67)),
  which cost 1.8× on Apple Metal. The `GPUCorpus` search work from the same
  line was kept.

## v0.5.1 — 2026-04-06

See the [release](https://github.com/oaustegard/remex/releases/tag/v0.5.1) and
the pull requests up to that tag. This file starts at v0.6.0; earlier history
was not reconstructed.
