"""PolarQuant: Data-oblivious vector quantization for embedding compression.

Based on the rotation + Lloyd-Max scalar quantization insight from TurboQuant
(Zandieh et al., ICLR 2026, arXiv:2504.19874). Implements the MSE-optimal
stage which empirically outperforms the full TurboQuant Prod variant for
nearest-neighbor retrieval tasks.
"""

from polarquant.core import PolarQuantizer, CompressedVectors
from polarquant.codebook import lloyd_max_codebook

__version__ = "0.1.0"
__all__ = ["PolarQuantizer", "CompressedVectors", "lloyd_max_codebook"]
