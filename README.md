# PolarQuant

Data-oblivious vector quantization for embedding compression. Compress embedding indices to 3-4 bits per dimension with strong retrieval recall and zero calibration.

Based on the rotation + Lloyd-Max insight from [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026).

## How it works

1. **Random rotation** — A fixed orthogonal matrix transforms any input distribution so that coordinates become approximately i.i.d. N(0, 1/d). This is the key trick: it makes the quantizer data-oblivious.
2. **Lloyd-Max scalar quantization** — Each coordinate is independently quantized using a codebook optimized for the known post-rotation Gaussian distribution. No per-vector or per-dataset calibration needed.

The result: near-optimal MSE distortion within a ~2.7x constant factor of the information-theoretic lower bound, at any bit-width.

## Usage

```python
from polarquant import PolarQuantizer

# Initialize (precomputes rotation matrix + codebook)
pq = PolarQuantizer(d=768, bits=4)

# Encode your embeddings
compressed = pq.encode(embeddings)  # (n, 768) float32 → CompressedVectors

# Search
indices, scores = pq.search(compressed, query_vector, k=10)

# Or decode for downstream use
reconstructed = pq.decode(compressed)  # (n, 768) float32

# Save/load
compressed.save("index.npz")
```

## Why not the full TurboQuant?

TurboQuant adds a QJL (Quantized Johnson-Lindenstrauss) residual correction that makes inner product estimates provably unbiased. Our experiments show this matters for KV cache attention (where softmax amplifies bias) but **hurts** retrieval recall — the extra noise from QJL dequantization outweighs the debiasing benefit when only ranking matters.

At 4-bit, d=256:

| Method | Recall@10 |
|---|---|
| PolarQuant (rotation + Lloyd-Max) | **0.86** |
| TurboQuant Prod (LM + QJL) | 0.68 |
| Naive minmax | 0.78 |

## Install

```bash
pip install -e ".[dev]"   # development
pytest                     # run tests
```

## References

- Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zandieh, A., Daliri, M., & Han, I. (2024). *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* AAAI 2025. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)

## License

MIT
