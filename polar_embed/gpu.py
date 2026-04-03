"""Optional GPU acceleration for polar-embed search.

Moves the hot search data to GPU and runs scoring there. Falls back
to NumPy when no GPU backend is available.

Supported backends:
  - **cupy**: Drop-in NumPy replacement on CUDA. Preferred.
  - **torch**: PyTorch tensors on CUDA. Widely installed.
  - **numpy**: CPU fallback (default).

Usage::

    from polar_embed import PolarQuantizer
    from polar_embed.gpu import GPUSearcher

    pq = PolarQuantizer(d=384, bits=4)
    compressed = pq.encode(corpus)

    # Move to GPU — auto-detects CuPy or PyTorch
    searcher = GPUSearcher(pq, compressed)
    indices, scores = searcher.search(query, k=10)

    # Explicit backend
    searcher = GPUSearcher(pq, compressed, backend="torch")

    # Two-stage with GPU coarse + CPU fine
    indices, scores = searcher.search_twostage(query, k=10, candidates=200)

Design notes:
  - GPUSearcher holds GPU-resident copies of indices, norms, centroids,
    and the rotation matrix. The CompressedVectors object is not modified.
  - search() uses the cached dequantized representation (like PolarQuantizer.search).
  - search_adc() uses lookup-table scoring on GPU (like PolarQuantizer.search_adc).
  - search_twostage() uses ADC for coarse (low memory) and materialized
    decode for fine reranking (tiny candidate set).
  - All results are returned as numpy arrays on CPU.
"""

import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_BACKENDS = {}


def _try_cupy():
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def _try_torch():
    try:
        import torch
        if torch.cuda.is_available():
            return torch
    except ImportError:
        pass
    return None


def available_backends():
    """Return list of available GPU backends, best first."""
    backends = []
    if _try_cupy() is not None:
        backends.append("cupy")
    if _try_torch() is not None:
        backends.append("torch")
    backends.append("numpy")
    return backends


def _detect_backend(requested: Optional[str] = None) -> str:
    """Resolve backend name, defaulting to best available."""
    if requested is not None:
        if requested == "numpy":
            return "numpy"
        if requested == "cupy":
            if _try_cupy() is None:
                raise ImportError("CuPy is not installed or no CUDA device.")
            return "cupy"
        if requested == "torch":
            if _try_torch() is None:
                raise ImportError(
                    "PyTorch is not installed or no CUDA device."
                )
            return "torch"
        raise ValueError(f"Unknown backend: {requested!r}")

    avail = available_backends()
    return avail[0]  # best available (cupy > torch > numpy)


# ---------------------------------------------------------------------------
# Backend-specific array ops
# ---------------------------------------------------------------------------

class _NumpyOps:
    """CPU reference implementation — same semantics as GPU backends."""

    def __init__(self):
        self.xp = np

    def to_device(self, arr: np.ndarray):
        return arr

    def to_numpy(self, arr):
        return np.asarray(arr)

    def matmul(self, A, B):
        return A @ B

    def matvec(self, A, v):
        return A @ v

    def topk(self, scores, k):
        """Return (indices, scores) of top-k, sorted descending."""
        if k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def gather_sum(self, table, indices):
        """table: (d, n_levels), indices: (n, d) → (n,) sum of lookups."""
        dim_idx = np.arange(table.shape[0])
        return table[dim_idx, indices].sum(axis=1)


class _CuPyOps:
    """CuPy GPU backend."""

    def __init__(self):
        import cupy as cp
        self.xp = cp
        self._cp = cp

    def to_device(self, arr: np.ndarray):
        return self._cp.asarray(arr)

    def to_numpy(self, arr):
        return self._cp.asnumpy(arr)

    def matmul(self, A, B):
        return A @ B

    def matvec(self, A, v):
        return A @ v

    def topk(self, scores, k):
        cp = self._cp
        if k >= len(scores):
            idx = cp.argsort(-scores)
        else:
            idx = cp.argpartition(-scores, k)[:k]
            idx = idx[cp.argsort(-scores[idx])]
        return idx, scores[idx]

    def gather_sum(self, table, indices):
        cp = self._cp
        dim_idx = cp.arange(table.shape[0])
        return table[dim_idx, indices].sum(axis=1)


class _TorchOps:
    """PyTorch CUDA backend."""

    def __init__(self):
        import torch
        self._torch = torch
        self.device = torch.device("cuda")

    @property
    def xp(self):
        return self._torch

    def to_device(self, arr: np.ndarray):
        return self._torch.from_numpy(arr).to(self.device)

    def to_numpy(self, arr):
        return arr.cpu().numpy()

    def matmul(self, A, B):
        return self._torch.matmul(A, B)

    def matvec(self, A, v):
        return self._torch.mv(A, v)

    def topk(self, scores, k):
        torch = self._torch
        actual_k = min(k, len(scores))
        vals, idx = torch.topk(scores, actual_k)
        return idx, vals

    def gather_sum(self, table, indices):
        """table: (d, n_levels) float32, indices: (n, d) int64 → (n,)."""
        torch = self._torch
        d, n_levels = table.shape
        n = indices.shape[0]
        # Flatten to 1D gather, then reshape and sum
        # offset each dimension: flat_idx = indices + dim_offset * n_levels
        dim_offsets = torch.arange(d, device=table.device) * n_levels  # (d,)
        flat_table = table.reshape(-1)  # (d * n_levels,)
        flat_idx = indices + dim_offsets.unsqueeze(0)  # (n, d)
        gathered = flat_table[flat_idx.reshape(-1)].reshape(n, d)  # (n, d)
        return gathered.sum(dim=1)


def _make_ops(backend: str):
    if backend == "numpy":
        return _NumpyOps()
    elif backend == "cupy":
        return _CuPyOps()
    elif backend == "torch":
        return _TorchOps()
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# GPUSearcher
# ---------------------------------------------------------------------------

class GPUSearcher:
    """GPU-accelerated searcher wrapping a PolarQuantizer + CompressedVectors.

    Transfers quantization data to GPU once on construction. Subsequent
    search calls run on GPU with results returned as numpy arrays.

    Args:
        quantizer: A PolarQuantizer instance.
        compressed: CompressedVectors to search over.
        backend: "cupy", "torch", "numpy", or None for auto-detect.
    """

    def __init__(self, quantizer, compressed, backend: Optional[str] = None):
        self.backend_name = _detect_backend(backend)
        self.ops = _make_ops(self.backend_name)
        self._pq = quantizer
        self._compressed = compressed

        # Move static data to device
        self._R = self.ops.to_device(quantizer.R.astype(np.float32))
        self._norms = self.ops.to_device(compressed.norms)

        # Indices: keep as int type appropriate for backend
        if self.backend_name == "torch":
            # PyTorch gather needs int64
            self._indices = self.ops.to_device(
                compressed.indices.astype(np.int64)
            )
        else:
            self._indices = self.ops.to_device(compressed.indices)

        # Centroids for full precision
        self._centroids = self.ops.to_device(
            quantizer.centroids.astype(np.float32)
        )

        # Nested centroid tables for Matryoshka precision levels
        self._nested = {}
        for prec, c in quantizer._nested.items():
            self._nested[prec] = self.ops.to_device(c.astype(np.float32))

        self._calibrated = quantizer.calibrated
        self._bits = quantizer.bits
        self._d = quantizer.d
        self._n = compressed.n

        # Cached dequantized representation (built on first search())
        self._x_hat_rot_gpu = None

    @property
    def resident_bytes_gpu(self) -> int:
        """Estimated GPU memory usage in bytes."""
        # indices + norms + rotation matrix + centroids
        idx_bytes = self._n * self._d  # uint8 or int64
        if self.backend_name == "torch":
            idx_bytes *= 8  # int64
        norm_bytes = self._n * 4
        R_bytes = self._d * self._d * 4
        cent_bytes = sum(
            len(c) * 4 if hasattr(c, '__len__') else 0
            for c in self._nested.values()
        )
        cache_bytes = 0
        if self._x_hat_rot_gpu is not None:
            cache_bytes = self._n * self._d * 4
        return idx_bytes + norm_bytes + R_bytes + cent_bytes + cache_bytes

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cached search on GPU (full precision).

        First call builds the dequantized float32 representation on GPU.
        Subsequent calls reuse it — just a matvec + topk.

        Returns numpy arrays.
        """
        ops = self.ops
        q = ops.to_device(np.asarray(query, dtype=np.float32))
        q_rot = ops.matvec(self._R, q)

        if self._x_hat_rot_gpu is None:
            self._x_hat_rot_gpu = self._build_x_hat_rot()

        scores = ops.matvec(self._x_hat_rot_gpu, q_rot) * self._norms
        idx, vals = ops.topk(scores, k)
        return ops.to_numpy(idx), ops.to_numpy(vals)

    def search_adc(
        self,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Memory-efficient ADC search on GPU.

        No float32 cache — builds a (d, 2^bits) lookup table per query,
        then gathers + sums over indices.

        Returns numpy arrays.
        """
        ops = self.ops
        q = ops.to_device(np.asarray(query, dtype=np.float32))
        q_rot = ops.matvec(self._R, q)

        centroids, indices = self._resolve_gpu(precision)

        # Build ADC table: (d, n_levels)
        if self._calibrated and precision is None:
            # centroids: (d, n_levels)
            table = centroids * q_rot.unsqueeze(1) if self.backend_name == "torch" \
                else centroids * q_rot[:, None]
        else:
            # centroids: (n_levels,)
            if self.backend_name == "torch":
                table = ops.xp.outer(q_rot, centroids)
            else:
                table = ops.xp.outer(q_rot, centroids)

        scores = ops.gather_sum(table, indices) * self._norms
        idx, vals = ops.topk(scores, k)
        return ops.to_numpy(idx), ops.to_numpy(vals)

    def search_twostage(
        self,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        coarse_precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Two-stage search: ADC coarse on GPU + precise rerank.

        Stage 1: ADC lookup-table scan at coarse_precision (low memory).
        Stage 2: Dequantize only the candidate rows, rerank at full precision.

        Returns numpy arrays.
        """
        ops = self.ops

        if coarse_precision is None:
            coarse_precision = max(1, self._bits - 2)

        q = ops.to_device(np.asarray(query, dtype=np.float32))
        q_rot = ops.matvec(self._R, q)
        coarse_k = min(candidates, self._n)

        # Stage 1: ADC coarse
        coarse_centroids, coarse_indices = self._resolve_gpu(coarse_precision)

        if self.backend_name == "torch":
            coarse_table = ops.xp.outer(q_rot, coarse_centroids)
        else:
            coarse_table = ops.xp.outer(q_rot, coarse_centroids)

        coarse_scores = ops.gather_sum(coarse_table, coarse_indices) * self._norms

        # Top candidates (no sort needed, just partition)
        coarse_idx, _ = ops.topk(coarse_scores, coarse_k)

        # Stage 2: dequantize candidates at full precision, rerank
        fine_indices = self._indices[coarse_idx]  # (candidates, d)

        if self._calibrated:
            if self.backend_name == "torch":
                dim_idx = ops.xp.arange(self._d, device=self._centroids.device)
                X_hat_cand = self._centroids[dim_idx, fine_indices]
            else:
                dim_idx = ops.xp.arange(self._d)
                X_hat_cand = self._centroids[dim_idx, fine_indices]
        else:
            X_hat_cand = self._centroids[fine_indices]

        fine_scores = ops.matvec(X_hat_cand, q_rot) * self._norms[coarse_idx]
        rerank_idx, rerank_scores = ops.topk(fine_scores, k)

        original_idx = coarse_idx[rerank_idx]
        return ops.to_numpy(original_idx), ops.to_numpy(rerank_scores)

    def drop_cache(self):
        """Free the GPU-resident dequantized cache."""
        self._x_hat_rot_gpu = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_x_hat_rot(self):
        """Dequantize indices to float32 on GPU."""
        ops = self.ops
        if self._calibrated:
            if self.backend_name == "torch":
                dim_idx = ops.xp.arange(self._d, device=self._centroids.device)
            else:
                dim_idx = ops.xp.arange(self._d)
            return self._centroids[dim_idx, self._indices]
        else:
            return self._centroids[self._indices]

    def _resolve_gpu(self, precision: Optional[int]):
        """Get (centroids, indices) on GPU for given precision."""
        if precision is None or precision == self._bits:
            return self._centroids, self._indices

        if self._calibrated:
            raise ValueError(
                "Matryoshka precision is not available in calibrated mode."
            )
        if precision < 1 or precision > self._bits:
            raise ValueError(
                f"precision must be 1-{self._bits}, got {precision}"
            )

        centroids = self._nested[precision]
        shift = self._bits - precision
        if self.backend_name == "torch":
            indices = self._indices >> shift
        else:
            indices = self._indices >> shift

        return centroids, indices
