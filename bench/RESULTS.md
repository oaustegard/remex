# Benchmark Results

**Date**: 2026-04-05 (recall, distribution, SPECTER2) · 2026-05-01 (Mojo twostage refresh, PR [#51](https://github.com/oaustegard/remex/pull/51)) · 2026-05-02 (IVF coarse-tier sweep + cross-FoS bridge, [#58](https://github.com/oaustegard/remex/pull/58) / [#60](https://github.com/oaustegard/remex/issues/60))
**Library version**: remex 0.5.1
**Hardware**: CPU (NumPy), no GPU
**Seeds**: corpus seed=42, query seed=99, quantizer seed=42

## Synthetic Benchmarks (d=384, 20 clusters, spread=0.3)

### Recall vs bit level (10k corpus, 200 queries)

| Method | Compression | MSE | R@10 | R@100 | Encode (ms) | Search (ms) |
|--------|------------|-----|------|-------|-------------|-------------|
| remex 2-bit | 15.4x | 0.1171 | 0.538 | 0.634 | 79 | 21 |
| remex 3-bit | 10.4x | 0.0343 | 0.719 | 0.800 | 153 | 22 |
| remex 4-bit | 7.8x | 0.0094 | 0.850 | 0.895 | 134 | 21 |
| remex 8-bit | 4.0x | 0.0000 | 0.987 | 0.991 | 238 | 21 |

### Scaling with corpus size (4-bit)

| Corpus | R@10 | R@100 | Encode (ms) | Search (ms) |
|--------|------|-------|-------------|-------------|
| 1k | 0.880 | 0.930 | 12 | 4 |
| 5k | 0.862 | 0.905 | 63 | 13 |
| 10k | 0.850 | 0.895 | 134 | 21 |
| 50k | 0.839 | 0.872 | 689 | 140 |

### Two-stage search (4-bit Matryoshka, 200 candidates)

| Corpus | R@10 | Search (ms) |
|--------|------|-------------|
| 1k | 0.880 | 343 |
| 5k | 0.861 | 1535 |
| 10k | 0.849 | 2894 |
| 50k | 0.837 | 14156 |

Two-stage recall matches single-stage to within 0.5%, validating the Matryoshka coarse-to-fine approach.

### Distribution sensitivity (10k corpus, 200 queries, varying cluster tightness)

| Spread (σ) | 2-bit R@10 | 3-bit R@10 | 4-bit R@10 | 8-bit R@10 |
|-----------|-----------|-----------|-----------|-----------|
| 0.01 | 0.163 | 0.331 | 0.533 | 0.954 |
| 0.05 | 0.478 | 0.693 | 0.831 | 0.984 |
| 0.10 | 0.532 | 0.727 | 0.846 | 0.987 |
| 0.30 | 0.538 | 0.719 | 0.850 | 0.987 |
| 0.50 | 0.540 | 0.717 | 0.859 | 0.986 |
| 1.00 | 0.525 | 0.720 | 0.848 | 0.984 |

**Key finding**: Very tight clusters (σ=0.01) severely degrade recall at all bit levels below 8. At 4-bit, R@10 drops from 0.85 to 0.53 — a 37% loss. At 8-bit, the degradation is only 3%. This is because tight clusters create near-identical vectors where small quantization errors flip rankings.

### Post-rotation distribution analysis (10k vectors, d=384)

| Metric | Value |
|--------|-------|
| Expected σ (1/√d) | 0.051031 |
| Actual σ (global) | 0.051031 |
| Per-dim σ (mean±std) | 0.0510 ± 0.0004 |
| Kurtosis (Gaussian=3) | 2.98 ± 0.05 |
| σ range | [0.0495, 0.0525] |

The rotation produces near-perfect N(0, 1/d) coordinates on synthetic data.

## Real Embedding Benchmarks (all-MiniLM-L6-v2, d=384)

From `bench/real_embedding_eval.py` using 10k corpus and 500 queries encoded by sentence-transformers:

| Method | Compression | MSE | R@10 | R@100 |
|--------|------------|-----|------|-------|
| remex 2-bit | 16x | 0.1164 | 0.517 | 0.860 |
| remex 3-bit | 10.4x | 0.0341 | 0.599 | 0.897 |
| remex 4-bit | 7.8x | 0.0093 | 0.707 | 0.932 |
| remex 8-bit | 2.0x | 0.0000 | 0.974 | 0.995 |
| FAISS PQ (m=96, trained) | 16x | 0.0341 | 0.816 | 0.946 |
| FAISS PQ (m=48, trained) | 32x | 0.0636 | 0.618 | 0.877 |

**Real vs synthetic gap**: 4-bit R@10 drops from 0.85 (synthetic) to 0.707 (real). Real embeddings cluster by topic, amplifying quantization errors within tight clusters.

## SPECTER2 Scientific Embeddings (allenai/specter2_base, d=768)

From `bench/specter2_eval.py` using 1k papers per partition, 500 queries (random split). Papers fetched from Semantic Scholar API, encoded locally with SPECTER2.

### Post-rotation distribution analysis

| Metric | Broad (NLP) | Narrow (Transformer) | Expected |
|--------|------------|---------------------|----------|
| Per-coord σ mean | 0.0140 | 0.0138 | 0.0361 |
| σ ratio (actual/expected) | 0.389 | 0.381 | 1.000 |
| Excess kurtosis | 0.005 | -0.050 | 0.000 |
| KS rejections (α=0.05) | 20/20 | 20/20 | ~1/20 |

The Gaussian assumption does not hold: per-coordinate σ is only 38% of expected. However, both broad and narrow partitions show identical deviation — domain specificity does not matter.

### Recall

| Bits | Compression | Broad R@10 | Broad R@100 | Narrow R@10 | Narrow R@100 |
|------|------------|-----------|------------|------------|-------------|
| 2 | 15.7x | 0.630 | 0.783 | 0.688 | 0.797 |
| 3 | 10.5x | 0.714 | 0.838 | 0.767 | 0.864 |
| 4 | 7.9x | 0.834 | 0.910 | 0.859 | 0.923 |
| 8 | 4.0x | 0.989 | 0.994 | 0.988 | 0.993 |

Despite the 61% σ deviation, 4-bit R@10 > 0.83 and 8-bit is essentially lossless. SPECTER2 4-bit recall (0.834) is notably higher than MiniLM (0.707), likely due to higher dimensionality (768 vs 384) providing more quantization budget.

See [docs/specter2-case-study.md](../docs/specter2-case-study.md) for full analysis.

### 1-bit Matryoshka extraction at scale (10k corpus, 100 queries)

From `bench/onebit_experiment.py` against the published [SPECTER2 NLP-broad 10k cache](https://github.com/oaustegard/claude-container-layers/releases/tag/specter2-nlp-broad-10k) (run `bash bench/fetch_specter2_cache.sh` once, then `ONEBIT_N=10000 python3 bench/onebit_experiment.py`).

#### Standalone bit sweep

| Bits | Compression | R@10 | R@100 |
|------|------------|------|-------|
| 1 | 30.7× | **0.635** | 0.694 |
| 2 | 15.7× | 0.501 | 0.628 |
| 3 | 10.5× | 0.595 | 0.732 |
| 4 | 7.9× | 0.731 | 0.820 |
| 8 | 4.0× | 0.971 | 0.984 |

**1-bit dominates 2-bit and 3-bit on R@10**, and beats 2-bit on R@100. The "valley" extends across 1–3 bits, not just 2-bit. Lloyd-Max wins back at 4-bit. This is consistent with Charikar's SimHash result: sign-bit hashing is asymptotically optimal for cosine similarity preservation, while Lloyd-Max optimizes per-coordinate MSE — a different (and at low bits, worse) objective for inner-product retrieval. The 2-bit and 3-bit Lloyd-Max boundaries land inside the dense Gaussian lobe of post-rotation coordinates, creating systematic ranking errors that the 1-bit sign-only comparison avoids.

#### Matryoshka extraction (encode @ 8-bit, search @ precision)

| Precision | R@10 | R@100 | Δ@10 vs standalone |
|-----------|------|-------|--------------------|
| 1 | 0.635 | 0.694 | **+0.000** (bit-for-bit identical) |
| 2 | 0.381 | 0.531 | −0.120 (~12% nesting penalty) |
| 4 | 0.756 | 0.835 | +0.025 (slightly better than standalone) |

The 1-bit Matryoshka extraction has zero nesting penalty because the MSB of an n-bit Lloyd-Max code *is* the sign bit, which is exactly the standalone 1-bit code. The invariant is enforced in `tests/test_matryoshka.py::TestPrecisionOneBit::test_matryoshka_1bit_equals_standalone_1bit`.

#### Two-stage retrieval: 1-bit coarse → 8-bit rerank

| Candidates | % of corpus | R@10 | R@100 |
|------------|------------|------|-------|
| 100 | 1.0% | 0.966 | 0.694 |
| 150 | 1.5% | **0.971** | 0.824 |
| 200 | 2.0% | 0.972 | 0.888 |
| 300 | 3.0% | 0.971 | 0.946 |

**At 1.5% candidate budget, R@10 matches the full-8-bit ceiling (0.971).** The 1-bit coarse pass narrows the field aggressively but retains the top-10 winners.

#### Two-stage retrieval: 2-bit coarse → 8-bit rerank (for comparison)

| Candidates | R@10 | R@100 | Gap vs 1-bit coarse |
|------------|------|-------|--------------------|
| 100 | 0.851 | 0.531 | −11.5 pp R@10 |
| 150 | 0.910 | 0.649 | −6.1 pp R@10 |
| 200 | 0.937 | 0.724 | −3.5 pp R@10 |
| 300 | 0.948 | 0.820 | −2.3 pp R@10 |

**2-bit coarse is dominated by 1-bit coarse at every candidate budget**, while requiring 2× the in-memory footprint. There is no scenario in this experiment where 2-bit is the right architectural choice.

#### Architectural takeaway for very-large-scale two-stage retrieval

For the Semantic Scholar use case (~100M SPECTER2 vectors, two-stage with in-memory coarse + RDS-resident 8-bit rerank), the recommended configuration is:

- **In-memory coarse tier**: 1-bit Matryoshka extracted from the 8-bit codes (right-shift to MSB). At d=768 this packs to **9.6 GB for 100M vectors**. The coarse score is `popcount(query_signs XOR coarse_signs)` — no Lloyd-Max lookup needed.
- **Rerank tier**: 8-bit Lloyd-Max indices stored in Postgres, fetched by ID for the top-K candidates returned by the coarse pass. At K~1.5% of corpus, R@10 matches the full-8-bit ceiling.
- **What not to do**: 2-bit coarse. Half the recall, twice the RAM, no upside.

#### A note on σ-ratio measurement

The 10k experiment reports σ ratio = 0.9993 using `np.std(rotated_flattened)` over the rotated `(n, d)` matrix. The existing 1k-paper section above reports σ ratio = 0.389 using per-coordinate σ mean. These are different statistics — they coincide for zero-mean-per-coord Gaussian samples but diverge if SPECTER2's post-rotation marginals have non-zero per-coordinate means. Reconciling the two metrics on the same 10k corpus is queued as a follow-up; meanwhile, the recall results above are based on identical rotation + Lloyd-Max code paths as `bench/specter2_eval.py`, so distribution-analysis discrepancies don't affect them.

### Coarse-tier IVF latency–recall sweep (10k corpus, 100 queries, 8-bit codes)

From `bench/specter2_eval.py --cached --only-ivf` against the published [SPECTER2 NLP-broad 10k cache](https://github.com/oaustegard/claude-container-layers/releases/tag/specter2-nlp-broad-10k) and [SPECTER2 NLP-narrow 10k cache](https://github.com/oaustegard/claude-container-layers/releases/tag/specter2-nlp-narrow-10k) (both fetched by `bash bench/fetch_specter2_cache.sh`). Container CPU, no GPU.

The benchmark sweeps `(mode, n_bits, nprobe)` for `IVFCoarseIndex` against the
flat-scan baseline (`Quantizer.search_adc` at `precision=1`). All numbers are
real SPECTER2 embeddings encoded at 8-bit, with the 1-bit Matryoshka coarse
tier as the IVF target — matching the SS deployment architecture from the
1-bit experiment above.

Column meanings:
- `pool%` — mean candidate corpus rows actually scanned by ADC, as % of corpus
- `R@10/coarse` — stage-1 candidate-set recall vs flat-scan stage-1 (top-10)
- `R@10/ts` — end-to-end Recall@10 after fine rerank vs exact KNN truth (flat baseline = 0.971)
- `speedup` — flat-scan coarse latency / IVF coarse latency
- `bridge` — IVF cross-partition top-K hits / flat baseline cross-partition hits (bridge sweep only)

#### SPECTER2 — Broad (NLP), single partition (n=9900 + 100 queries)

Flat-scan baseline: **36.16 ms/q coarse, R@10=0.9710**.

| mode | n_bits | n_cells | nonempty | nprobe | pool% | R@10/coarse | R@10/ts | lat (ms) | speedup |
|------|-------:|--------:|---------:|-------:|------:|------------:|--------:|---------:|--------:|
| rotated_prefix |  8 |  256 | 114 |    3 | 12.3% | 0.341 | 0.316 |  4.66 | 7.76× |
| rotated_prefix |  8 |  256 | 114 |   13 | 32.7% | 0.622 | 0.613 | 12.19 | 2.97× |
| rotated_prefix |  8 |  256 | 114 |   26 | 49.6% | 0.788 | 0.769 | 19.03 | 1.90× |
| rotated_prefix |  8 |  256 | 114 |   64 | 77.4% | 0.948 | 0.921 | 31.39 | 1.15× |
| rotated_prefix |  8 |  256 | 114 |  128 | 94.5% | 0.988 | 0.960 | 38.73 | 0.93× |
| rotated_prefix | 10 | 1024 | 338 |   10 |  9.8% | 0.386 | 0.361 |  3.90 | 9.26× |
| rotated_prefix | 10 | 1024 | 338 |  102 | 45.1% | 0.805 | 0.784 | 17.29 | 2.09× |
| rotated_prefix | 10 | 1024 | 338 |  256 | 71.2% | 0.930 | 0.902 | 28.32 | 1.28× |
| rotated_prefix | 10 | 1024 | 338 |  512 | 90.0% | 0.985 | 0.950 | 38.39 | 0.94× |
| rotated_prefix | 12 | 4096 | 484 |   41 | 19.6% | 0.525 | 0.503 |  7.72 | 4.68× |
| rotated_prefix | 12 | 4096 | 484 |  205 | 47.9% | 0.821 | 0.800 | 18.88 | 1.92× |
| rotated_prefix | 12 | 4096 | 484 | 1024 | 82.4% | 0.970 | 0.938 | 33.77 | 1.07× |
| lsh            |  8 |  256 | 133 |   13 | 40.0% | 0.464 | 0.446 | 15.28 | 2.37× |
| lsh            |  8 |  256 | 133 |   26 | 52.1% | 0.563 | 0.565 | 20.13 | 1.80× |
| lsh            |  8 |  256 | 133 |   64 | 79.6% | 0.819 | 0.803 | 32.91 | 1.10× |
| lsh            | 10 | 1024 | 392 |  102 | 48.2% | 0.580 | 0.567 | 18.19 | 1.99× |
| lsh            | 10 | 1024 | 392 |  256 | 75.0% | 0.820 | 0.792 | 27.44 | 1.32× |
| lsh            | 10 | 1024 | 392 |  512 | 92.6% | 0.951 | 0.934 | 33.75 | 1.07× |
| lsh            | 12 | 4096 | 947 |  205 | 28.6% | 0.403 | 0.392 | 10.89 | 3.32× |
| lsh            | 12 | 4096 | 947 |  410 | 47.0% | 0.617 | 0.603 | 16.71 | 2.16× |
| lsh            | 12 | 4096 | 947 | 1024 | 73.2% | 0.809 | 0.795 | 26.19 | 1.38× |
| lsh            | 12 | 4096 | 947 | 2048 | 90.3% | 0.938 | 0.916 | 31.55 | 1.15× |

Broad-only verdict (auto-generated by the bench script):

- **rotated_prefix**: best speedup at R@10 ≥ 0.922 → **1.07× at n_bits=12, nprobe=1024, pool=82.4%**
- **lsh**: best speedup at R@10 ≥ 0.922 → **1.07× at n_bits=10, nprobe=512, pool=92.6%**

#### SPECTER2 — Broad host + Narrow bridge (n=19900 + 100 queries)

Flat-scan baseline: **65.17 ms/q coarse, R@10=0.9700**, **bridge hits 1.61 of 10** cross-partition (i.e. 16% of broad queries' top-10 land in the narrow corpus — bridge edges are real and non-trivial).

| mode | n_bits | n_cells | nonempty | nprobe | pool% | R@10/coarse | R@10/ts | lat (ms) | speedup | bridge |
|------|-------:|--------:|---------:|-------:|------:|------------:|--------:|---------:|--------:|-------:|
| rotated_prefix |  8 |  256 |  118 |    3 | 13.4% | 0.348 | 0.336 |  9.07 | 7.18× | 1.18 |
| rotated_prefix |  8 |  256 |  118 |   13 | 33.6% | 0.623 | 0.624 | 22.86 | 2.85× | 1.06 |
| rotated_prefix |  8 |  256 |  118 |   26 | 51.4% | 0.779 | 0.767 | 34.90 | 1.87× | 1.01 |
| rotated_prefix |  8 |  256 |  118 |   64 | 79.0% | 0.943 | 0.915 | 52.76 | 1.24× | 0.98 |
| rotated_prefix | 10 | 1024 |  355 |   10 | 10.3% | 0.391 | 0.381 |  6.97 | 9.35× | 1.16 |
| rotated_prefix | 10 | 1024 |  355 |  102 | 46.4% | 0.802 | 0.781 | 31.40 | 2.08× | 1.02 |
| rotated_prefix | 10 | 1024 |  355 |  256 | 72.5% | 0.927 | 0.896 | 47.72 | 1.37× | 0.96 |
| rotated_prefix | 10 | 1024 |  355 |  512 | 90.5% | 0.986 | 0.948 | 59.83 | 1.09× | 0.99 |
| rotated_prefix | 12 | 4096 |  516 |   41 | 20.2% | 0.522 | 0.512 | 13.31 | 4.90× | 1.04 |
| rotated_prefix | 12 | 4096 |  516 |  205 | 49.4% | 0.817 | 0.794 | 33.49 | 1.95× | 1.00 |
| rotated_prefix | 12 | 4096 |  516 | 1024 | 83.1% | 0.969 | 0.936 | 53.94 | 1.21× | 0.99 |
| lsh            |  8 |  256 |  159 |   13 | 31.9% | 0.441 | 0.438 | 21.24 | 3.07× | 1.01 |
| lsh            |  8 |  256 |  159 |   26 | 45.7% | 0.544 | 0.551 | 30.67 | 2.12× | 1.07 |
| lsh            |  8 |  256 |  159 |   64 | 71.8% | 0.802 | 0.784 | 47.40 | 1.38× | 1.04 |
| lsh            | 10 | 1024 |  501 |  102 | 41.4% | 0.550 | 0.548 | 27.23 | 2.39× | 1.07 |
| lsh            | 10 | 1024 |  501 |  256 | 68.5% | 0.806 | 0.783 | 45.31 | 1.44× | 1.03 |
| lsh            | 10 | 1024 |  501 |  512 | 89.8% | 0.952 | 0.928 | 59.57 | 1.09× | 1.01 |
| lsh            | 12 | 4096 | 1347 |  410 | 40.3% | 0.585 | 0.578 | 26.55 | 2.46× | 1.06 |
| lsh            | 12 | 4096 | 1347 | 1024 | 66.1% | 0.796 | 0.783 | 44.18 | 1.48× | 1.04 |
| lsh            | 12 | 4096 | 1347 | 2048 | 85.8% | 0.932 | 0.907 | 57.38 | 1.14× | 0.96 |

Bridge-sweep verdict (auto-generated):

- **rotated_prefix**: best speedup at R@10 ≥ 0.921 → **1.21× at n_bits=12, nprobe=1024, pool=83.1%**
- **lsh**: best speedup at R@10 ≥ 0.921 → **1.09× at n_bits=10, nprobe=512, pool=89.8%**

#### Findings

1. **Speedups are marginal at 10k.** Best end-to-end-recall-preserving speedup is 1.07–1.21×. This is consistent with the README/CLAUDE.md positioning: IVF wins at "≥ tens of millions of vectors" where the flat coarse scan is memory-bandwidth-bound. At 10k × d=768 × 1-bit packed = ~1 MB the coarse memory fits in L2 cache and the per-query IVF lookup overhead (~3 ms) absorbs most of the savings.
2. **Bridge edges are preserved.** The `bridge` column hovers tightly around 1.00 across both modes and all `nprobe` values that meet the recall floor. The architectural concern from #53 — *"does `rotated_prefix` accidentally re-introduce field-of-study partitioning by content-derived hashing?"* — is answered empirically: **no**. Even at aggressive `nprobe=10/1024` (10% pool, R@10 collapse to 0.38), `rotated_prefix` still pulls 1.16× the flat-baseline cross-partition rate. There is no scenario in this sweep where IVF systematically silos by field of study.
3. **`rotated_prefix` and `lsh` are roughly tied on the recall–latency Pareto.** `rotated_prefix` has the edge at the high-recall end (1.21× vs 1.09× at R@10 ≥ 0.95×flat in the bridge sweep) and is free to build (no Gaussian hyperplane materialization). `lsh` spreads cells more uniformly (1347 vs 516 nonempty at 12-bit on the bridge corpus) and edges out at low-recall, high-speedup operating points. Both are within 0.05 R@10 of each other at any matched `pool%`.
4. **At low `nprobe`, recall collapses fast.** Below ~30% pool, R@10/ts drops to 0.13–0.50. The "free" 5–10× speedups at the top of each sweep are all in the unusable regime. This is expected for a data-oblivious hash with a long-tail cell distribution.
5. **The bridge sweep latency baseline doubles** (36 → 65 ms/q) when the corpus doubles (10k → 20k), confirming the coarse scan is bandwidth-bound and IVF would help at scale.

Recommendation for ≤ 100k corpora: **use `search_twostage` directly, skip the IVF index.** Re-evaluate at ~10M+ vectors where the bandwidth ceiling actually bites.

#### Reproduce

```bash
bash bench/fetch_specter2_cache.sh                    # ~90 MB, both partitions
python bench/specter2_eval.py --cached --only-ivf     # ~5 min on container CPU
```

Knobs: `--ivf-bits` (default 8, the 8-bit Lloyd-Max codes that the 1-bit Matryoshka coarse tier is extracted from), `--ivf-coarse-precision` (default 1, the right-shifted Matryoshka level used for stage-1 ADC), `--ivf-candidates` (default 500, stage-1 pool size handed to fine rerank), `--ivf-queries` (default 100), `-n / --num-vectors` (default 10000, per partition).

## Research investigations

- [docs/research/hybrid-precision-quantization.md](../docs/research/hybrid-precision-quantization.md) — Ported OjaKV&rsquo;s &ldquo;keep high-residual vectors at higher precision&rdquo; trick to TurboQuant-style scalar quantization. Verdict: FILE. Residual distribution is near-flat under Haar rotation + Lloyd-Max (top 10% of vectors carry only 10.8% of error mass), and cross-tier score merging is miscalibrated on anisotropic embeddings (e.g. SPECTER2). `search_twostage()` remains the right memory/recall trade-off.

## Memory Profiles (100k vectors, d=384, 8-bit)

| Strategy | Resident RAM | ms/query |
|----------|-------------|----------|
| CompressedVectors (no cache) | 38.8 MB | — |
| CompressedVectors (cached) | 192.4 MB | 3.9 |
| CompressedVectors (cold) | 38.8 MB | 137 |
| PackedVectors (always packed) | 38.8 MB | — |
| search_adc() (no cache) | 38.8 MB | 152 |
| search_twostage() (no cache) | 38.8 MB | 152 |

## Compression Ratios (bit-packed, d=384)

| Bits | Bytes per vector | vs float32 | File size per 10k vectors |
|------|-----------------|------------|--------------------------|
| 2 | 100 | 15.4x | 0.93 MB |
| 3 | 148 | 10.4x | 1.42 MB |
| 4 | 196 | 7.8x | 1.83 MB |
| 8 | 388 | 4.0x | 3.61 MB |

Float32 baseline: 1,536 bytes/vector, 15.36 MB per 10k vectors.

## Mojo port (`polarquant`) vs NumPy

The Mojo port moved to [oaustegard/remex-mojo](https://github.com/oaustegard/remex-mojo)
in 1.0. Its Mojo-vs-NumPy results, build matrix and reproduce commands live in
that repository's `bench/RESULTS.md`. Headline at d=384, n=10k, 4 bits: encode
1.27×, ADC search 6.0×, two-stage search 5.5× faster than NumPy.

## Reproducibility

All benchmarks can be reproduced with:

```bash
python bench/benchmark.py               # synthetic data (no extra deps)
pip install -e ".[bench]"               # for real embedding benchmarks
python bench/real_embedding_eval.py     # needs sentence-transformers + faiss-cpu
```

Seeds: corpus generation uses `np.random.default_rng(42)`, queries use seed 99, quantizer uses seed 42.
