# Case Study: SPECTER2 Scientific Embeddings

How well does remex's Gaussian assumption hold for real scientific paper embeddings, and does it matter?

## Background

remex's Lloyd-Max quantizer assumes that after random orthogonal rotation, each coordinate of a unit-normalized embedding follows N(0, 1/&radic;d). This is guaranteed for isotropic distributions and approximately true for many real embeddings. But scientific document embeddings like [SPECTER2](https://huggingface.co/allenai/specter2_base) (768-d) are trained on citation graphs and encode domain-specific structure that may violate this assumption.

This case study measures the actual post-rotation distribution of SPECTER2 embeddings and quantifies the recall impact.

## Setup

We fetched paper titles and abstracts from the [Semantic Scholar API](https://api.semanticscholar.org/) and encoded them with the `allenai/specter2_base` model (768-d). Two partitions test whether domain breadth affects the distribution:

| Partition | Query | Purpose |
|-----------|-------|---------|
| Broad | "natural language processing" | Diverse NLP field |
| Narrow | "transformer attention mechanism" | Tight subfield |

1,000 papers per partition, each with title + abstract. Embeddings are L2-normalized, rotated with remex's Haar rotation (seed=42), then analyzed.

Reproduce with:
```bash
pip install transformers torch
python bench/specter2_eval.py -n 1000 --plots    # first run: ~15 min (API + encoding)
python bench/specter2_eval.py -n 1000 --cached    # subsequent: seconds
```

## Finding 1: The Gaussian assumption does not hold

After rotation, per-coordinate &sigma; should be 1/&radic;768 = 0.0361. Instead, SPECTER2 coordinates have &sigma; &asymp; 0.014 &mdash; only **38% of the expected value**.

| Metric | Broad (NLP) | Narrow (Transformer) | Expected |
|--------|------------|---------------------|----------|
| Per-coord &sigma; mean | 0.0140 | 0.0138 | 0.0361 |
| &sigma; ratio (actual/expected) | 0.389 | 0.381 | 1.000 |
| &sigma; range | [0.010, 0.021] | [0.010, 0.020] | ~[0.034, 0.038]&ast; |
| Excess kurtosis | 0.005 | -0.050 | 0.000 |
| KS rejections (&alpha;=0.05) | 20/20 | 20/20 | ~1/20 |

&ast;*Synthetic d=384 range from bench/benchmark.py for comparison.*

The KS test rejects the Gaussian hypothesis on every tested coordinate (20/20). The coordinates are Gaussian in *shape* (kurtosis &asymp; 0) but the variance is wrong &mdash; Lloyd-Max boundaries are set too wide, wasting quantization levels on probability mass that doesn't exist.

### Why does this happen?

SPECTER2 embeddings have extremely concentrated norms (mean=21.7, std=0.13). After L2 normalization to the unit sphere, the vectors occupy a narrow band rather than being spread isotropically. The Haar rotation isotropizes the *directions* but can't fix the fact that the effective variance per coordinate is much lower than 1/d when the original embedding has strong anisotropy concentrated in a low-dimensional subspace.

## Finding 2: Broad vs narrow fields are identical

| Metric | Broad | Narrow | Difference |
|--------|-------|--------|------------|
| &sigma; deviation from Gaussian | 61.1% | 61.9% | 0.8 pp |
| KS statistic (mean) | 0.442 | 0.483 | +0.041 |
| Norm std | 0.140 | 0.122 | -0.018 |

Domain specificity does **not** meaningfully change the post-rotation distribution. Whether you're compressing a broad NLP corpus or a narrow subfield, the Gaussian mismatch is the same. This is good news: you don't need to worry about per-domain tuning.

## Finding 3: Recall is still strong despite the mismatch

| Bits | Compression | Broad R@10 | Narrow R@10 |
|------|------------|-----------|------------|
| 2 | 15.7x | 0.630 | 0.688 |
| 3 | 10.5x | 0.714 | 0.767 |
| 4 | 7.9x | 0.834 | 0.859 |
| 8 | 4.0x | 0.989 | 0.988 |

Despite a 61% &sigma; deviation, **4-bit achieves R@10 > 0.83** and **8-bit is essentially lossless** (0.989). For comparison, remex on synthetic isotropic data (d=384) achieves 4-bit R@10 = 0.85.

The SPECTER2 4-bit recall (0.834) is notably higher than the MiniLM result (0.707) from `bench/real_embedding_eval.py`, despite SPECTER2 having worse Gaussian fit. This likely reflects SPECTER2's higher dimensionality (768 vs 384) providing more quantization budget per unit of information.

### Mapping to the sensitivity table

remex's [distribution sensitivity table](../bench/RESULTS.md#distribution-sensitivity-10k-corpus-200-queries-varying-cluster-tightness) characterizes recall vs cluster spread &sigma;. The SPECTER2 per-coordinate &sigma; values (0.010&ndash;0.021) correspond to the moderate-spread regime in that table, predicting 4-bit R@10 in the 0.83&ndash;0.85 range &mdash; exactly what we observe.

## Recommendations for SPECTER2 users

1. **8-bit is the safe default.** At 4x compression with R@10=0.989, there is virtually no recall cost. For 200M Semantic Scholar abstracts at 768-d, this reduces storage from 576 GB (float32) to 144 GB.

2. **4-bit works for coarse retrieval.** R@10=0.834 is good enough for a first-pass retriever feeding a cross-encoder reranker. At 7.9x compression, the same 200M corpus fits in ~72 GB.

3. **No per-domain tuning needed.** Broad and narrow fields produce identical distributions. A single `Quantizer(d=768, bits=4)` works across all of Semantic Scholar.

4. **The Gaussian mismatch is real but tolerable.** A distribution-aware Lloyd-Max codebook (fitted to the actual &sigma; &asymp; 0.014 rather than the theoretical 0.036) could recover 1&ndash;3% recall at 4-bit. This is a potential future optimization but not critical given the already-strong results.

## Reproducing these results

```bash
# Full pipeline: fetch from S2 API + encode with SPECTER2 + analyze
pip install transformers torch
python bench/specter2_eval.py -n 1000 --plots

# Distribution analysis only (skip recall benchmark)
python bench/specter2_eval.py -n 1000 --cached --skip-recall

# Larger corpus (slower encoding)
python bench/specter2_eval.py -n 10000 --plots
```

Results are cached in `bench/.specter2_cache/`. Subsequent runs with `--cached` complete in seconds.
