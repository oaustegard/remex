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

from remex.core import Quantizer, CompressedVectors
from remex.codebook import lloyd_max_codebook, nested_codebooks
from remex.packing import pack, unpack, packed_nbytes

__version__ = "0.5.0"
__all__ = [
    "Quantizer", "CompressedVectors",
    "PolarQuantizer",  # deprecated alias
    "lloyd_max_codebook", "nested_codebooks",
    "pack", "unpack", "packed_nbytes",
]

# Deprecated alias — will be removed in a future release
PolarQuantizer = Quantizer
