"""Bit-packing for sub-byte quantization indices.

Packs uint8 index arrays into minimal bits for storage compression.
Supports 1-8 bits per value.

Packing ratios:
  1-bit: 8 values per byte (8:1)
  2-bit: 4 values per byte (4:1)
  3-bit: 8 values per 3 bytes (2.67:1)
  4-bit: 2 values per byte (2:1)
  5-8 bit: 1 value per byte (1:1)
"""

import numpy as np


def pack(indices: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack uint8 indices into minimal-width byte array.

    Args:
        indices: (...) uint8 array with values in [0, 2^bits).
        bits: Bits per value (1-8).

    Returns:
        Packed uint8 byte array.
    """
    flat = indices.ravel().astype(np.uint8)
    n = len(flat)

    if bits == 8:
        return flat.copy()
    elif bits == 4:
        # 2 values per byte: high nibble | low nibble
        padded = _pad_to_multiple(flat, 2)
        return (padded[0::2] << 4) | padded[1::2]
    elif bits == 2:
        # 4 values per byte
        padded = _pad_to_multiple(flat, 4)
        return (
            (padded[0::4] << 6)
            | (padded[1::4] << 4)
            | (padded[2::4] << 2)
            | padded[3::4]
        )
    elif bits == 1:
        # Use numpy's built-in packbits (MSB first)
        return np.packbits(flat)
    elif bits == 3:
        # 8 values → 24 bits → 3 bytes
        padded = _pad_to_multiple(flat, 8)
        groups = padded.reshape(-1, 8)
        out = np.empty((len(groups), 3), dtype=np.uint8)
        # Byte 0: [v0:3][v1:3][v2:2hi]
        out[:, 0] = (groups[:, 0] << 5) | (groups[:, 1] << 2) | (groups[:, 2] >> 1)
        # Byte 1: [v2:1lo][v3:3][v4:3][v5:1hi]
        out[:, 1] = (
            ((groups[:, 2] & 1) << 7)
            | (groups[:, 3] << 4)
            | (groups[:, 4] << 1)
            | (groups[:, 5] >> 2)
        )
        # Byte 2: [v5:2lo][v6:3][v7:3]
        out[:, 2] = ((groups[:, 5] & 3) << 6) | (groups[:, 6] << 3) | groups[:, 7]
        return out.ravel()
    else:
        # 5, 6, 7 bits: generic bitstream packing
        return _pack_generic(flat, bits)


def unpack(packed: np.ndarray, bits: int, n_values: int) -> np.ndarray:
    """
    Unpack byte array into uint8 indices.

    Args:
        packed: Packed byte array from pack().
        bits: Bits per value (must match what was used for packing).
        n_values: Number of original values to unpack.

    Returns:
        (n_values,) uint8 array.
    """
    if bits == 8:
        return packed[:n_values].copy()
    elif bits == 4:
        out = np.empty(len(packed) * 2, dtype=np.uint8)
        out[0::2] = (packed >> 4) & 0x0F
        out[1::2] = packed & 0x0F
        return out[:n_values]
    elif bits == 2:
        out = np.empty(len(packed) * 4, dtype=np.uint8)
        out[0::4] = (packed >> 6) & 0x03
        out[1::4] = (packed >> 4) & 0x03
        out[2::4] = (packed >> 2) & 0x03
        out[3::4] = packed & 0x03
        return out[:n_values]
    elif bits == 1:
        return np.unpackbits(packed)[:n_values]
    elif bits == 3:
        groups = packed.reshape(-1, 3)
        out = np.empty((len(groups), 8), dtype=np.uint8)
        b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]
        out[:, 0] = (b0 >> 5) & 0x07
        out[:, 1] = (b0 >> 2) & 0x07
        out[:, 2] = ((b0 & 0x03) << 1) | ((b1 >> 7) & 0x01)
        out[:, 3] = (b1 >> 4) & 0x07
        out[:, 4] = (b1 >> 1) & 0x07
        out[:, 5] = ((b1 & 0x01) << 2) | ((b2 >> 6) & 0x03)
        out[:, 6] = (b2 >> 3) & 0x07
        out[:, 7] = b2 & 0x07
        return out.ravel()[:n_values]
    else:
        return _unpack_generic(packed, bits, n_values)


def packed_nbytes(n_values: int, d: int, bits: int) -> int:
    """Compute packed byte count for n_values vectors of dimension d."""
    total_values = n_values * d
    if bits == 8:
        return total_values
    elif bits == 4:
        return (total_values + 1) // 2
    elif bits == 2:
        return (total_values + 3) // 4
    elif bits == 1:
        return (total_values + 7) // 8
    elif bits == 3:
        return ((total_values + 7) // 8) * 3
    else:
        return (total_values * bits + 7) // 8


def _pad_to_multiple(arr: np.ndarray, multiple: int) -> np.ndarray:
    """Pad array to a length that's a multiple of `multiple`."""
    remainder = len(arr) % multiple
    if remainder == 0:
        return arr
    padding = multiple - remainder
    return np.concatenate([arr, np.zeros(padding, dtype=np.uint8)])


def _pack_generic(flat: np.ndarray, bits: int) -> np.ndarray:
    """Generic bitstream packer for arbitrary bit widths."""
    n = len(flat)
    total_bits = n * bits
    out_bytes = (total_bits + 7) // 8
    out = np.zeros(out_bytes, dtype=np.uint8)

    bit_pos = 0
    for i in range(n):
        val = int(flat[i])
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        # Value may span 2 bytes
        out[byte_idx] |= (val << bit_offset) & 0xFF
        if bit_offset + bits > 8 and byte_idx + 1 < out_bytes:
            out[byte_idx + 1] |= val >> (8 - bit_offset)
        bit_pos += bits

    return out


def _unpack_generic(packed: np.ndarray, bits: int, n_values: int) -> np.ndarray:
    """Generic bitstream unpacker for arbitrary bit widths."""
    mask = (1 << bits) - 1
    out = np.empty(n_values, dtype=np.uint8)

    bit_pos = 0
    for i in range(n_values):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        val = (int(packed[byte_idx]) >> bit_offset)
        if bit_offset + bits > 8 and byte_idx + 1 < len(packed):
            val |= int(packed[byte_idx + 1]) << (8 - bit_offset)
        out[i] = val & mask
        bit_pos += bits

    return out
