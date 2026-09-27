"""remex: Retrieval-validated embedding compression.

Compress embeddings 2-16x with proven recall. Based on the rotation + Lloyd-Max
scalar quantization insight from TurboQuant (Zandieh et al., ICLR 2026,
arXiv:2504.19874). Implements the MSE-optimal stage which empirically
outperforms the full TurboQuant Prod variant for nearest-neighbor retrieval.

Supports Matryoshka bit precision: encode once at full bit-width, then
search at any lower precision via right-shifting indices. Enables two-stage
coarse-to-fine retrieval from a single encoded representation.

Formerly known as polar-embed.
"""

import warnings

from remex.core import (
    Quantizer, CompressedVectors, PackedVectors, corpus_mean,
    anisotropy, AnisotropyWarning,
)
from remex.codebook import (
    coordinate_sigma, lloyd_max_codebook, nested_codebooks,
)
from remex.ivf import IVFCoarseIndex
from remex.packing import pack, unpack, packed_nbytes
from remex.pq_format import save_pq, load_pq, save_params

try:  # resolved from installed metadata, never hardcoded
    from importlib.metadata import version as _pkg_version, PackageNotFoundError

    __version__ = _pkg_version("remex")
except PackageNotFoundError:  # bare source checkout, not installed
    __version__ = "0.0.0+unknown"
__all__ = [
    "corpus_mean", "anisotropy", "AnisotropyWarning",
    "Quantizer", "CompressedVectors", "PackedVectors",
    "PolarQuantizer",  # deprecated alias
    "IVFCoarseIndex",
    "coordinate_sigma", "lloyd_max_codebook", "nested_codebooks",
    "pack", "unpack", "packed_nbytes",
    "save_pq", "load_pq", "save_params",
]


def __getattr__(name):
    if name == "PolarQuantizer":
        warnings.warn(
            "PolarQuantizer has been renamed to Quantizer. "
            "The PolarQuantizer alias is deprecated and will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Quantizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
