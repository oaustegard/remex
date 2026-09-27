"""`.pq` binary file format used by the Mojo CLI port.

Layout (little-endian):
    bytes 0-1   : magic 'PQ' (0x50, 0x51)
    byte 2      : reserved (0)
    byte 3      : version (=1)
    bytes 4-7   : d (u32)
    bytes 8-15  : n (u64)
    byte 16     : bits (u8)
    byte 17     : rotation code (u8) — 0 = haar, 1 = rht, 2 = none
    byte 18     : flags (u8) — bit 0 set = scalar mode, no norms section
    bytes 19-31 : reserved (13 zero bytes)
    bytes 32+   : packed_indices (length = packed_nbytes(n*d, bits))
    then        : norms (n × float32, little-endian), absent in scalar mode

This is a minimal alternative to the Python `.npz` format that is trivially
parseable from Mojo without unzip/numpy-header machinery. See
`src/pq_format.mojo` in oaustegard/remex-mojo.

The flags byte lives in what byte 18 already was — reserved and zero — so
every file written before it existed reads back as "has norms", which they
all do. In the other direction a scalar-mode file is *shorter* than an old
reader's arithmetic expects, so it fails the length check as
"truncated .pq data" rather than silently reading indices as norms. The
Mojo port is one such reader: it does not implement scalar mode yet.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from remex.packing import SUPPORTED_BITS, pack, unpack, packed_nbytes
from remex.rotation import (
    ROTATION_CODES, rotation_from_code, validate_rotation,
)

if TYPE_CHECKING:
    from remex.core import CompressedVectors, Quantizer

PQ_HEADER_BYTES = 32
PQ_VERSION = 1
PQ_MAGIC = b"PQ\x00\x01"

PARAMS_HEADER_BYTES = 16
PARAMS_VERSION = 1
PARAMS_MAGIC = b"PR\x00\x01"

#: Byte 18, bit 0: the norms section is absent (scalar mode).
PQ_FLAG_NO_NORMS = 0x01


# Rotations the Mojo port can rebuild from `(d, seed)` alone, mapped to the
# `polarquant` flag that selects each one. `--params` reads R straight out of
# the file and is rotation-agnostic, but a caller dumping params is usually
# also exercising the seed path, and a rotation Mojo has never heard of would
# fail there without saying why. Keep in step with `src/rotation.mojo` in
# oaustegard/remex-mojo.
MOJO_ROTATIONS = {"haar": "--rotation haar", "rht": "--rotation rht"}


def save_params(path: str | Path, quantizer: "Quantizer") -> None:
    """Dump (R, boundaries, centroids) so a Mojo binary can mirror this Quantizer.

    Used to verify bit-identical encode output between Python and Mojo.

    The Mojo CLI consumes this two ways. ``--params P`` loads R from the file
    and works for any rotation. ``--seed S --rotation K`` rebuilds R in Mojo
    from the seed instead; that path needs a construction Mojo actually
    implements, which is what the check below is for.
    """
    if not getattr(quantizer, "normalize", True):
        raise ValueError(
            "save_params does not support scalar-mode quantizers "
            "(normalize=False). The Mojo encoder always normalizes and "
            "always writes norms, so it would encode different codes from "
            "these params, and a parity check against it would compare two "
            "different pipelines."
        )

    rotation = getattr(quantizer, "rotation", "haar")
    if rotation not in MOJO_ROTATIONS:
        raise ValueError(
            f"save_params supports rotation in {sorted(MOJO_ROTATIONS)}, got "
            f"{rotation!r}. The Mojo port has no construction for it, so "
            f"`polarquant --seed` would encode against a different rotation "
            f"than this Quantizer and the codes would not match."
        )

    d = int(quantizer.d)
    bits = int(quantizer.bits)
    R = np.ascontiguousarray(quantizer.R, dtype=np.float32)
    boundaries = np.ascontiguousarray(quantizer.boundaries, dtype=np.float32)
    centroids = np.ascontiguousarray(quantizer.centroids, dtype=np.float32)
    if R.shape != (d, d):
        raise ValueError(f"R shape {R.shape} != (d, d) = ({d}, {d})")
    if boundaries.size != (1 << bits) - 1:
        raise ValueError("boundaries size mismatch")
    if centroids.size != (1 << bits):
        raise ValueError("centroids size mismatch")

    header = bytearray(PARAMS_HEADER_BYTES)
    header[0:4] = PARAMS_MAGIC
    header[4:8] = struct.pack("<I", d)
    header[8] = bits & 0xFF
    header[9] = ROTATION_CODES[rotation]
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(R.tobytes())
        f.write(boundaries.tobytes())
        f.write(centroids.tobytes())


def save_pq(path: str | Path, compressed: "CompressedVectors") -> None:
    """Serialize a CompressedVectors to the .pq binary format.

    A scalar-mode container (``norms is None``) sets the no-norms flag and
    writes no norms section. Such a file is readable by ``load_pq`` but not
    yet by the Mojo port.
    """
    if getattr(compressed, "mean", None) is not None:
        raise ValueError(
            "the .pq format cannot carry a centered container yet: it has "
            "no section for the corpus mean, and a reader that dropped the "
            "mean would decode every vector shifted. Use .npz or Arrow, "
            "which both round-trip it."
        )

    n, d, bits = int(compressed.n), int(compressed.d), int(compressed.bits)
    packed = pack(compressed.indices.ravel(), bits)
    expected_packed = packed_nbytes(n, d, bits)
    if packed.nbytes != expected_packed:
        raise ValueError(
            f"packed indices size mismatch: got {packed.nbytes}, "
            f"expected {expected_packed}"
        )

    header = bytearray(PQ_HEADER_BYTES)
    header[0:4] = PQ_MAGIC
    header[4:8] = struct.pack("<I", d)
    header[8:16] = struct.pack("<Q", n)
    header[16] = bits & 0xFF
    header[17] = ROTATION_CODES[validate_rotation(compressed.rotation)]
    if compressed.norms is None:
        header[18] = PQ_FLAG_NO_NORMS

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(packed.tobytes())
        if compressed.norms is not None:
            norms = np.ascontiguousarray(compressed.norms, dtype=np.float32)
            f.write(norms.tobytes())


def load_pq(path: str | Path) -> "CompressedVectors":
    """Read a .pq file and return a CompressedVectors."""
    from remex.core import CompressedVectors

    raw = Path(path).read_bytes()
    if len(raw) < PQ_HEADER_BYTES:
        raise ValueError(".pq file too small")
    if raw[0:2] != b"PQ":
        raise ValueError("bad .pq magic")
    if raw[3] != PQ_VERSION:
        raise ValueError(f"unsupported .pq version: {raw[3]}")

    (d,) = struct.unpack("<I", raw[4:8])
    (n,) = struct.unpack("<Q", raw[8:16])
    bits = raw[16]
    # Byte 17 was reserved-and-zero before this field existed, and 0 is
    # haar, so a pre-field file resolves to LEGACY_ROTATION for free.
    rotation = rotation_from_code(raw[17])
    flags = raw[18]
    unknown_flags = flags & ~PQ_FLAG_NO_NORMS
    if unknown_flags:
        raise ValueError(
            f"unknown .pq flag bits 0x{unknown_flags:02x} in header; this "
            f"file was written by a newer remex."
        )
    has_norms = not (flags & PQ_FLAG_NO_NORMS)
    if bits not in SUPPORTED_BITS:
        raise ValueError(
            f"bits={bits} is not supported. Use one of "
            f"{list(SUPPORTED_BITS)}."
        )

    expected_packed = packed_nbytes(n, d, bits)
    norms_bytes = n * 4 if has_norms else 0
    expected_total = PQ_HEADER_BYTES + expected_packed + norms_bytes
    if len(raw) < expected_total:
        raise ValueError("truncated .pq data")

    packed = np.frombuffer(
        raw, dtype=np.uint8,
        count=expected_packed,
        offset=PQ_HEADER_BYTES,
    )
    if has_norms:
        norms = np.frombuffer(
            raw, dtype=np.float32,
            count=n,
            offset=PQ_HEADER_BYTES + expected_packed,
        ).copy()
    else:
        norms = None

    indices = unpack(packed, bits, n * d).reshape(n, d)
    return CompressedVectors(indices.copy(), norms, d, bits, rotation)
