"""Core remex encoder/decoder with Matryoshka bit precision."""

import numpy as np
from typing import Optional, Tuple, Iterable
from remex.codebook import (
    coordinate_sigma, lloyd_max_codebook, nested_codebooks,
)
from remex.packing import SUPPORTED_BITS, pack, unpack, packed_nbytes
from remex.rotation import (
    LEGACY_ROTATION, RHTOperator, haar_rotation, identity_rotation,
    rht_rotation, validate_rotation,
)


def _norms_nbytes(norms: Optional[np.ndarray]) -> int:
    """Bytes the norms column costs — zero in scalar mode, where it is absent."""
    return 0 if norms is None else norms.nbytes


def _validate_mean(mean, d: int, normalize: bool, renorm: bool):
    """Check a caller-declared corpus mean, or pass ``None`` through.

    remex never measures this off the data — the caller declares it, the way
    it declares ``scale`` in scalar mode. ``remex.corpus_mean`` is the
    blessed way to compute one.
    """
    if mean is None:
        return None
    if not normalize:
        raise ValueError(
            "mean is only meaningful with normalize=True. Scalar mode "
            "quantizes the coordinates themselves, so there is no direction "
            "to take relative to a centre."
        )
    if not renorm:
        raise ValueError(
            "mean requires renorm=True. Centered codes reconstruct as "
            "mu + m * u_hat with u_hat a unit direction, and renorm=False "
            "leaves the decoded direction its raw, non-unit length."
        )
    mean = np.asarray(mean, dtype=np.float32)
    if mean.shape != (d,):
        raise ValueError(f"mean must have shape ({d},), got {mean.shape}")
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean must be finite")
    return mean


def corpus_mean(X: np.ndarray) -> np.ndarray:
    """The mean of a corpus, as ``Quantizer(mean=...)`` wants it.

    Accumulated in float64 and returned as float32, so the value does not
    depend on how the rows were chunked.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D corpus, got shape {X.shape}")
    return X.mean(axis=0, dtype=np.float64).astype(np.float32)


def _apply_norms(
    scores: np.ndarray, norms: Optional[np.ndarray]
) -> np.ndarray:
    """Rescale unit-sphere scores by the stored norms.

    ``norms is None`` is scalar mode: the codes quantize the coordinates
    themselves, so the lookup already is the approximate inner product and
    there is nothing to rescale.
    """
    return scores if norms is None else scores * norms


class CompressedVectors:
    """Container for quantized vector data.

    Indices are stored as uint8 in memory for fast search/decode.
    Bit-packing is used for serialization (save/load) and for
    computing the true compressed size (nbytes).

    ``norms`` is ``None`` for containers produced by a scalar-mode
    quantizer (``Quantizer(normalize=False)``), which quantizes raw
    coordinates and keeps no per-vector norm.
    """

    __slots__ = ("indices", "norms", "d", "bits", "n", "rotation", "mean",
                 "_x_hat_rot", "_dir_len")

    def __init__(self, indices: np.ndarray, norms: Optional[np.ndarray],
                 d: int, bits: int,
                 rotation: str = LEGACY_ROTATION,
                 mean: Optional[np.ndarray] = None):
        self.indices = indices  # (n, d) uint8 — unpacked for fast access
        self.norms = norms
        self.d = d
        self.bits = bits
        self.n = indices.shape[0]
        self.rotation = validate_rotation(rotation)
        #: Corpus mean these codes are relative to, or None for the plain path.
        #: Part of the encoding the way ``rotation`` is: decoding against a
        #: different one is refused rather than silently wrong.
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self._x_hat_rot = None  # cached dequantized rotated vectors (full precision)
        self._dir_len = {}  # precision -> (n,) length of the decoded direction

    @property
    def has_norms(self) -> bool:
        """False for scalar-mode codes, which store no per-vector norm."""
        return self.norms is not None

    def subset(self, idx: np.ndarray) -> "CompressedVectors":
        """Return a CompressedVectors containing only the given row indices."""
        sub = CompressedVectors(
            self.indices[idx],
            None if self.norms is None else self.norms[idx],
            self.d, self.bits, self.rotation, self.mean,
        )
        if self._x_hat_rot is not None:
            sub._x_hat_rot = self._x_hat_rot[idx]
        return sub

    @property
    def nbytes(self) -> int:
        """Packed memory footprint in bytes (honest compression)."""
        return packed_nbytes(self.n, self.d, self.bits) + _norms_nbytes(self.norms)

    @property
    def nbytes_unpacked(self) -> int:
        """Unpacked memory footprint (what's actually in RAM)."""
        return self.indices.nbytes + _norms_nbytes(self.norms)

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage (using packed size)."""
        return (self.n * self.d * 4) / self.nbytes

    @property
    def resident_bytes(self) -> int:
        """Actual RAM footprint including any active caches."""
        total = self.indices.nbytes + _norms_nbytes(self.norms)
        if self._x_hat_rot is not None:
            total += self._x_hat_rot.nbytes
        return total

    def drop_cache(self):
        """Free the dequantized float32 cache to reclaim memory."""
        self._x_hat_rot = None
        self._dir_len = {}

    def save(self, path: str):
        """Save to compressed .npz file with bit-packed indices.

        Scalar-mode containers write no ``norms`` entry at all; its absence
        is what marks the file as scalar mode on load. Every file written
        by the normalizing path has one, so there is no ambiguity with
        older files.
        """
        packed_idx = pack(self.indices.ravel(), self.bits)
        optional = {} if self.norms is None else {"norms": self.norms}
        if self.mean is not None:
            optional["mean"] = self.mean
        np.savez_compressed(
            path,
            packed_indices=packed_idx,
            d=np.int32(self.d),
            bits=np.int32(self.bits),
            n=np.int32(self.n),
            rotation=np.str_(self.rotation),
            **optional,
        )

    @classmethod
    def load(cls, path: str) -> "CompressedVectors":
        """Load from .npz file, unpacking bit-packed indices.

        A file with no ``rotation`` entry predates the field and is Haar —
        see ``remex.rotation.LEGACY_ROTATION`` for why that is a frozen
        constant rather than the current default.

        A file with no ``norms`` entry is scalar mode (``normalize=False``).
        """
        data = np.load(path)
        d = int(data["d"])
        bits = int(data["bits"])
        n = int(data["n"])
        rotation = str(data["rotation"]) if "rotation" in data else LEGACY_ROTATION

        if "packed_indices" in data:
            flat = unpack(data["packed_indices"], bits, n * d)
            indices = flat.reshape(n, d)
        else:
            # Backward compat: old format stored unpacked indices
            indices = data["indices"]

        norms = data["norms"] if "norms" in data else None
        mean = data["mean"] if "mean" in data else None
        return cls(indices, norms, d, bits, rotation, mean)

    def save_arrow(self, path: str, seed: Optional[int] = None, **extra_metadata):
        """Save to Arrow IPC (Feather v2) format, packing indices for storage.

        Requires pyarrow (optional dependency).

        Args:
            path: Output file path.
            seed: Quantizer seed to store in schema metadata.
            **extra_metadata: Additional key-value pairs for schema metadata.
        """
        packed = PackedVectors.from_compressed(self)
        packed.save_arrow(path, seed=seed, **extra_metadata)

    @classmethod
    def load_arrow(cls, path: str) -> "CompressedVectors":
        """Load from Arrow IPC (Feather v2) format, unpacking to uint8 indices.

        Args:
            path: Arrow IPC file path.

        Returns:
            CompressedVectors with unpacked uint8 indices.
        """
        packed = PackedVectors.load_arrow(path)
        return packed.to_compressed()


class PackedVectors:
    """Memory-efficient packed storage for quantized vectors.

    Stores indices bit-packed in memory, unpacking rows on demand.
    Uses 2-4x less RAM than CompressedVectors for sub-byte bit widths.

    Use ``from_compressed()`` to convert from CompressedVectors,
    ``from_rows()`` to reconstruct from database rows, or
    ``load()`` / ``load_arrow()`` to read from disk.

    Search is supported via ``Quantizer.search_adc()`` and
    ``Quantizer.search_twostage()``.  Cached ``search()`` is not
    supported — use ``to_compressed()`` to convert back if needed.
    """

    __slots__ = ("_packed", "norms", "d", "bits", "n", "rotation", "mean",
                 "_row_bytes", "_row_aligned", "_dir_len")

    def __init__(
        self,
        packed: np.ndarray,
        norms: Optional[np.ndarray],
        n: int,
        d: int,
        bits: int,
        rotation: str = LEGACY_ROTATION,
        mean: Optional[np.ndarray] = None,
    ):
        """
        Args:
            packed: (n, row_bytes) uint8 array of bit-packed indices.
            norms: (n,) float32 array of vector norms, or ``None`` for
                scalar-mode codes (``Quantizer(normalize=False)``).
            n: Number of vectors.
            d: Vector dimension.
            bits: Bits per coordinate.
            rotation: Which rotation encoded these codes. Defaults to
                the frozen historical value, not the library default.
        """
        self._packed = packed  # (n, row_bytes) uint8
        self.norms = norms
        self.n = n
        self.d = d
        self.bits = bits
        self.rotation = validate_rotation(rotation)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self._row_bytes = packed_nbytes(1, d, bits)
        self._row_aligned = (d * bits) % 8 == 0
        self._dir_len = {}  # precision -> (n,) length of the decoded direction

    @property
    def has_norms(self) -> bool:
        """False for scalar-mode codes, which store no per-vector norm."""
        return self.norms is not None

    def unpack_rows(self, start: int, end: int) -> np.ndarray:
        """Decompress a contiguous row slice to uint8 indices.

        Args:
            start: First row index (inclusive).
            end: Last row index (exclusive).

        Returns:
            (end - start, d) uint8 array of indices.
        """
        n_rows = end - start
        if self._row_aligned:
            flat = self._packed[start:end].ravel()
            return unpack(flat, self.bits, n_rows * self.d).reshape(n_rows, self.d)
        else:
            result = np.empty((n_rows, self.d), dtype=np.uint8)
            for i in range(n_rows):
                result[i] = unpack(self._packed[start + i], self.bits, self.d)
            return result

    def unpack_at(self, idx: np.ndarray) -> np.ndarray:
        """Decompress arbitrary row indices to uint8 indices.

        Args:
            idx: Array of row indices to unpack.

        Returns:
            (len(idx), d) uint8 array of indices.
        """
        idx = np.asarray(idx)
        if idx.ndim == 0:
            idx = idx.reshape(1)
        rows = self._packed[idx]  # (len(idx), row_bytes)
        if self._row_aligned:
            flat = rows.ravel()
            return unpack(flat, self.bits, len(idx) * self.d).reshape(len(idx), self.d)
        else:
            result = np.empty((len(idx), self.d), dtype=np.uint8)
            for i in range(len(idx)):
                result[i] = unpack(rows[i], self.bits, self.d)
            return result

    @classmethod
    def from_compressed(cls, compressed: CompressedVectors) -> "PackedVectors":
        """Convert a CompressedVectors to packed in-memory format.

        Args:
            compressed: CompressedVectors with unpacked uint8 indices.

        Returns:
            PackedVectors with bit-packed indices.
        """
        n, d, bits = compressed.n, compressed.d, compressed.bits
        row_bytes = packed_nbytes(1, d, bits)
        row_aligned = (d * bits) % 8 == 0
        if row_aligned:
            packed_flat = pack(compressed.indices.ravel(), bits)
            packed = packed_flat.reshape(n, row_bytes)
        else:
            packed = np.empty((n, row_bytes), dtype=np.uint8)
            for i in range(n):
                packed[i] = pack(compressed.indices[i], bits)
        norms = None if compressed.norms is None else compressed.norms.copy()
        mean = None if compressed.mean is None else compressed.mean.copy()
        return cls(packed, norms, n, d, bits, compressed.rotation, mean)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable,
        norms: Optional[np.ndarray],
        d: int,
        bits: int,
        rotation: str = LEGACY_ROTATION,
    ) -> "PackedVectors":
        """Reconstruct from database packed byte rows.

        Args:
            rows: Iterable of bytes/bytearray/ndarray, one per vector.
            norms: (n,) float32 array of vector norms, or ``None`` for
                scalar-mode codes.
            d: Vector dimension.
            bits: Bits per coordinate.
            rotation: Which rotation encoded these codes. The database
                schema has no column for it, so the caller must supply it
                for anything other than Haar — this default is the frozen
                historical value, not the library default.

        Returns:
            PackedVectors instance.
        """
        row_list = []
        for r in rows:
            if isinstance(r, (bytes, bytearray)):
                row_list.append(np.frombuffer(r, dtype=np.uint8))
            else:
                row_list.append(np.asarray(r, dtype=np.uint8))
        packed = np.stack(row_list)
        n = len(row_list)
        if norms is not None:
            norms = np.asarray(norms, dtype=np.float32)
        return cls(packed, norms, n, d, bits, rotation)

    def at_precision(self, target_bits: int) -> "PackedVectors":
        """Derive a lower-bit representation via Matryoshka right-shift.

        Unpacks in chunks, right-shifts, and repacks at target_bits.

        Args:
            target_bits: Target bit precision (1 to self.bits).

        Returns:
            New PackedVectors at the target precision.
        """
        if target_bits < 1 or target_bits > self.bits:
            raise ValueError(
                f"target_bits must be 1-{self.bits}, got {target_bits}"
            )
        if target_bits == self.bits:
            return self

        shift = self.bits - target_bits
        row_bytes_target = packed_nbytes(1, self.d, target_bits)
        target_aligned = (self.d * target_bits) % 8 == 0
        packed_target = np.empty((self.n, row_bytes_target), dtype=np.uint8)

        chunk = 4096
        for start in range(0, self.n, chunk):
            end = min(start + chunk, self.n)
            indices = self.unpack_rows(start, end)
            shifted = (indices >> shift).astype(np.uint8)
            if target_aligned:
                packed_flat = pack(shifted.ravel(), target_bits)
                packed_target[start:end] = packed_flat.reshape(
                    end - start, row_bytes_target
                )
            else:
                for i in range(end - start):
                    packed_target[start + i] = pack(shifted[i], target_bits)

        return PackedVectors(
            packed_target, self._copy_norms(), self.n, self.d, target_bits,
            self.rotation, None if self.mean is None else self.mean.copy(),
        )

    def to_compressed(self) -> CompressedVectors:
        """Convert to CompressedVectors by unpacking all indices."""
        indices = self.unpack_rows(0, self.n)
        return CompressedVectors(
            indices, self._copy_norms(), self.d, self.bits, self.rotation,
            None if self.mean is None else self.mean.copy(),
        )

    def subset(self, idx: np.ndarray) -> "PackedVectors":
        """Return a PackedVectors containing only the given row indices."""
        idx = np.asarray(idx)
        return PackedVectors(
            self._packed[idx].copy(),
            None if self.norms is None else self.norms[idx].copy(),
            len(idx),
            self.d,
            self.bits,
            self.rotation,
            None if self.mean is None else self.mean.copy(),
        )

    def _copy_norms(self) -> Optional[np.ndarray]:
        return None if self.norms is None else self.norms.copy()

    @property
    def nbytes(self) -> int:
        """Packed memory footprint in bytes."""
        return self._packed.nbytes + _norms_nbytes(self.norms)

    @property
    def resident_bytes(self) -> int:
        """Actual RAM footprint (same as nbytes — no caches)."""
        return self._packed.nbytes + _norms_nbytes(self.norms)

    @property
    def compression_ratio(self) -> float:
        """Ratio vs float32 storage."""
        return (self.n * self.d * 4) / self.nbytes

    def save(self, path: str):
        """Save to compressed .npz file.

        As with ``CompressedVectors.save``, a scalar-mode container writes
        no ``norms`` entry.
        """
        optional = {} if self.norms is None else {"norms": self.norms}
        if self.mean is not None:
            optional["mean"] = self.mean
        np.savez_compressed(
            path,
            packed_indices=self._packed.ravel(),
            d=np.int32(self.d),
            bits=np.int32(self.bits),
            n=np.int32(self.n),
            rotation=np.str_(self.rotation),
            **optional,
        )

    @classmethod
    def load(cls, path: str) -> "PackedVectors":
        """Load from .npz file, keeping indices packed.

        A file with no ``rotation`` entry predates the field and is Haar.
        A file with no ``norms`` entry is scalar mode.
        """
        data = np.load(path)
        d = int(data["d"])
        bits = int(data["bits"])
        n = int(data["n"])
        rotation = str(data["rotation"]) if "rotation" in data else LEGACY_ROTATION
        row_bytes = packed_nbytes(1, d, bits)
        packed_flat = data["packed_indices"]
        packed = packed_flat.reshape(n, row_bytes)
        norms = data["norms"] if "norms" in data else None
        mean = data["mean"] if "mean" in data else None
        return cls(packed, norms, n, d, bits, rotation, mean)

    def save_arrow(self, path: str, seed: Optional[int] = None, **extra_metadata):
        """Save to Arrow IPC (Feather v2) format.

        Stores packed indices as FixedSizeBinary and norms as Float32,
        with quantizer parameters in schema-level metadata. A scalar-mode
        container has no norms, so the file has no ``norms`` column.

        Requires pyarrow (optional dependency).

        Args:
            path: Output file path.
            seed: Quantizer seed to store in metadata.
            **extra_metadata: Additional key-value pairs for schema metadata.
        """
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow IPC format: "
                "pip install pyarrow"
            )

        row_bytes = self._row_bytes
        metadata = {
            b"d": str(self.d).encode(),
            b"bits": str(self.bits).encode(),
            b"n": str(self.n).encode(),
            b"rotation": self.rotation.encode(),
        }
        if seed is not None:
            metadata[b"seed"] = str(seed).encode()
        for k, v in extra_metadata.items():
            key = k.encode() if isinstance(k, str) else k
            metadata[key] = str(v).encode()

        fields = [pa.field("packed_indices", pa.binary(row_bytes))]
        packed_buf = pa.py_buffer(self._packed.tobytes())
        columns = {
            "packed_indices": pa.FixedSizeBinaryArray.from_buffers(
                pa.binary(row_bytes), self.n, [None, packed_buf]
            )
        }
        if self.norms is not None:
            fields.insert(0, pa.field("norms", pa.float32()))
            columns["norms"] = pa.array(self.norms.tolist(), type=pa.float32())

        schema = pa.schema(fields, metadata=metadata)
        table = pa.table(
            {f.name: columns[f.name] for f in fields}, schema=schema
        )
        feather.write_feather(table, path)

    @classmethod
    def load_arrow(cls, path: str, memory_map: bool = False) -> "PackedVectors":
        """Load from Arrow IPC (Feather v2) format.

        Args:
            path: Arrow IPC file path.
            memory_map: If True, memory-map the file for zero-copy access.

        Returns:
            PackedVectors with packed indices.
        """
        try:
            import pyarrow.feather as feather
        except ImportError:
            raise ImportError(
                "pyarrow is required for Arrow IPC format: "
                "pip install pyarrow"
            )

        table = feather.read_table(path, memory_map=memory_map)
        metadata = table.schema.metadata
        d = int(metadata[b"d"])
        bits = int(metadata[b"bits"])
        n = int(metadata[b"n"])
        rotation = (
            metadata[b"rotation"].decode()
            if b"rotation" in metadata
            else LEGACY_ROTATION
        )
        row_bytes = packed_nbytes(1, d, bits)

        if "norms" in table.schema.names:
            norms = table.column("norms").to_numpy().astype(np.float32)
        else:
            norms = None  # scalar mode — the column was never written

        # Extract packed data from contiguous Arrow buffer
        packed_col = table.column("packed_indices")
        chunk = packed_col.chunk(0)
        buffers = chunk.buffers()
        data_buf = buffers[1]
        packed_flat = np.frombuffer(data_buf, dtype=np.uint8)[: n * row_bytes]
        packed = packed_flat.reshape(n, row_bytes).copy()

        return cls(packed, norms, n, d, bits, rotation)


class Quantizer:
    """
    Vector quantizer with Matryoshka bit precision.

    Encodes vectors by:
    1. Normalizing to unit sphere (storing norms separately)
    2. Applying a random orthogonal rotation (makes coordinates ~N(0, 1/d))
    3. Scalar-quantizing each coordinate with a Lloyd-Max codebook

    Data-oblivious: Uses a theoretical Lloyd-Max codebook for N(0, 1/d).
    No training data needed. Based on TurboQuant (Zandieh et al., ICLR 2026).

    ``normalize=False`` drops step 1 and quantizes the coordinates
    themselves — see the ``normalize`` argument below.

    Supports **nested bit precision**: encode once at full bit-width,
    then search at any lower precision by right-shifting indices.
    The top k bits of an n-bit code are a valid k-bit code, with
    centroid tables precomputed for each level.

    Args:
        d: Vector dimension.
        bits: Bits per coordinate (1-8). 3-4 is the sweet spot.
        seed: Random seed for rotation matrix.
        rotation: Which orthogonal rotation to use.

            "haar" (default) — Haar-distributed, via explicit Householder QR.
                Bit-reproducible against the Mojo port (#40). O(d^3) to build:
                measured 1.8 s at d=768, 11.4 s at d=1536, 150 s at d=3072.

            "rht" — randomized Hadamard, O(d^2 log d) to build: 0.4 s at
                d=768, 5.7 s at d=3072 (26x faster). Measured
                indistinguishable from Haar on retrieval recall
                (-0.0001 +/- 0.0013 pooled over 3 corpora x 6 bit widths x
                5 seeds, oaustegard/experiments#11). Bit-reproducible against
                the Mojo port via `polarquant --rotation rht`. Requires an
                even d.

            "none" — no rotation (the identity). Only sensible together with
                `normalize=False`, where the caller has already conditioned
                the coordinates; it makes the code for coordinate *j* an
                exact, matmul-free function of input coordinate *j*, which
                is what a hash key wants. See
                `remex.rotation.identity_rotation`.

            All three are seed-deterministic and exactly orthogonal. The
            rotation is part of the encoding: vectors encoded under one
            CANNOT be decoded under another, so it must match across
            encode/decode exactly as `seed` must.

        normalize: Whether to factor each vector into (unit direction, norm)
            before quantizing. Default True — the behavior described above,
            and the only mode that existing `.pq`/`.npz` files use.

            False selects **scalar mode**: quantize the (optionally rotated)
            coordinates directly with the Lloyd-Max codebook, store no norms
            at all, and let `scale` declare the coordinate spread. Use it
            when the codes are the product — exact-match hash keys, joins,
            bucketing — rather than an approximation of a direction:

            - Unit-norm factorization collapses every constant-direction
              family (`c * 1_vec` and near-parallel neighbourhoods) onto a
              single direction code. On isotropic embeddings that is
              harmless; on enumerated algebraic values it builds
              mega-buckets, and a join that should take minutes turns into
              a collision storm.
            - Scalar mode never casts input to float32 (it works in
              float64) and stores no float32 norms, so values that would
              overflow either — `exp(700)` and friends — quantize to the
              extreme cell instead of to `inf`/`nan`.

            Everything else is unchanged: same Lloyd-Max cell shaping, same
            Matryoshka nesting (right-shift a code for a coarser one, which
            is a per-query collision/recall dial), same
            `(d, bits, seed, rotation)` determinism.

        scale: Coordinate standard deviation the Lloyd-Max codebook is built
            for. Scalar mode only; the normalizing path derives it from the
            unit sphere as `1/sqrt(d)` and rejects an explicit value.
            Defaults to 1.0, i.e. inputs assumed roughly N(0, 1).

            remex stays data-oblivious here: it will not measure your data
            to pick a scale. Condition the input yourself — bound it, or
            push heavy tails through `arcsinh`/`log` — and set `scale` to
            the spread you conditioned it to. Coordinates far outside
            `±3*scale` all land in the outermost cell.
    """

    ROTATIONS = {
        "haar": haar_rotation,
        "rht": rht_rotation,
        "none": identity_rotation,
    }

    def __init__(self, d: int, bits: int = 4, seed: int = 42,
                 rotation: str = "haar", normalize: bool = True,
                 scale: Optional[float] = None, renorm: bool = True,
                 mean: Optional[np.ndarray] = None):
        if bits < 1 or bits > 8:
            raise ValueError(f"bits must be 1-4 or 8, got {bits}")
        if bits not in SUPPORTED_BITS:
            raise ValueError(
                f"bits={bits} is not supported. Use one of "
                f"{list(SUPPORTED_BITS)}. 5-7 bit widths offer negligible "
                f"benefit over 4-bit or 8-bit."
            )

        if rotation not in self.ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(self.ROTATIONS)}, "
                f"got {rotation!r}"
            )

        normalize = bool(normalize)
        if normalize and scale is not None:
            raise ValueError(
                "scale is only meaningful with normalize=False. The "
                "normalizing path quantizes unit vectors, whose rotated "
                "coordinates have a known spread of 1/sqrt(d) — passing a "
                "scale there would be silently ignored."
            )
        # None here means "unit sphere, 1/sqrt(d)"; scalar mode has no such
        # default to fall back on, so it names one.
        sigma = None if normalize else (1.0 if scale is None else scale)

        self.d = d
        self.bits = bits
        self.seed = seed
        self.rotation = rotation
        self.normalize = normalize
        self.renorm = bool(renorm)
        self.mean = _validate_mean(mean, d, normalize, self.renorm)
        self.scale = coordinate_sigma(d, sigma)

        # "rht" is applied in operator form (remex.rotation.RHTOperator):
        # faster from d~768 up, a few KB instead of d^2 floats, and
        # machine-independent output. ``R`` stays available as the same dense
        # matrix, built on first access, for the GPU path and save_params.
        self._R = None
        self._op = RHTOperator(d, seed) if rotation == "rht" else None
        if self._op is None:
            self._R = self.ROTATIONS[rotation](d, seed)
        self.boundaries, self.centroids = lloyd_max_codebook(d, bits, sigma=sigma)

        # Precompute nested centroid tables for all bit levels <= bits
        self._nested = nested_codebooks(d, bits, sigma=sigma)

    def encode(self, X: np.ndarray) -> CompressedVectors:
        """
        Quantize a batch of vectors.

        Args:
            X: (n, d) float array. Need not be unit-normalized.

        Returns:
            CompressedVectors container with indices, and norms unless this
            is a scalar-mode (``normalize=False``) quantizer.
        """
        if not self.normalize:
            return self._encode_scalar(X)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")

        # Compute norms in float64, cast to float32. Removes the 1-ULP
        # reduction-order divergence between BLAS (np.linalg.norm) and
        # Mojo's SIMD reduce_add — needed for byte-identical .pq parity
        # with the Mojo encoder.
        norms64 = np.sqrt(np.sum(X.astype(np.float64) ** 2, axis=1))
        norms = norms64.astype(np.float32)

        target = X if self.mean is None else X - self.mean
        if self.mean is None:
            lengths = norms
        else:
            lengths = np.sqrt(
                np.sum(target.astype(np.float64) ** 2, axis=1)
            ).astype(np.float32)

        X_unit = target / np.maximum(lengths, 1e-8)[:, None]
        X_rot = self._rotate_rows(X_unit)

        indices = np.searchsorted(self.boundaries, X_rot).astype(np.uint8)

        if self.mean is None:
            return CompressedVectors(
                indices, norms, self.d, self.bits, self.rotation
            )
        return CompressedVectors(
            indices, self._centred_lengths(indices, norms64, lengths),
            self.d, self.bits, self.rotation, self.mean,
        )

    def _centred_lengths(self, indices, norms64, fallback):
        """How far along the decoded direction to land so the reconstruction
        has the original vector's length.

        A centred code decodes to ``mu + m * u_hat``. Choosing ``m`` to solve
        ``||mu + m*u_hat|| == ||x||`` puts the whole correction in the norms
        column the container already has, so a centred index costs no extra
        bytes per vector. Expanding the norm gives
        ``m^2 + 2m(mu.u_hat) + (||mu||^2 - ||x||^2) = 0``; the positive root
        is the one on the ray the code points along.

        Where no positive root exists — the original vector is shorter than
        the mean's distance to that ray, which needs a corpus whose mean
        dwarfs its spread — the residual's own length is kept instead, which
        is what an uncentred code would have stored.
        """
        u_rot = self.centroids[indices]
        u = self._unrotate_rows(u_rot).astype(np.float64)
        u /= np.maximum(np.linalg.norm(u, axis=1), 1e-12)[:, None]

        mu = self.mean.astype(np.float64)
        b = u @ mu
        c = float(mu @ mu) - norms64 ** 2
        disc = b * b - c
        m = np.where(disc > 0.0, -b + np.sqrt(np.maximum(disc, 0.0)), fallback)
        return m.astype(np.float32)

    def _encode_scalar(self, X: np.ndarray) -> CompressedVectors:
        """Quantize coordinates directly — no normalization, no norms.

        Stays in float64 throughout. The float32 cast on the normalizing
        path is there for Mojo `.pq` parity, and it is exactly what
        overflows on the heavy-tailed inputs this mode exists for; nothing
        downstream of a scalar-mode code needs that parity.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[np.newaxis]
        if X.shape[1] != self.d:
            raise ValueError(f"Expected d={self.d}, got {X.shape[1]}")

        X_rot = self._rotate_rows(X)
        # searchsorted promotes the float32 boundaries to float64, so an
        # out-of-range magnitude saturates at the outermost cell instead of
        # overflowing to inf. NaN sorts above every boundary; it lands in
        # the top cell rather than raising, which keeps a code well-defined
        # for every input.
        indices = np.searchsorted(self.boundaries, X_rot).astype(np.uint8)

        return CompressedVectors(indices, None, self.d, self.bits, self.rotation)

    @property
    def R(self) -> np.ndarray:
        """The (d, d) rotation matrix. For ``rotation="rht"`` it is built on
        first access (``rht_rotation(d, seed)``); encode and search do not
        use it."""
        if self._R is None:
            self._R = self.ROTATIONS[self.rotation](self.d, self.seed)
        return self._R

    @R.setter
    def R(self, value) -> None:
        self._R = value
        self._op = None  # an explicitly assigned matrix is what gets applied

    def _rotate_rows(self, X: np.ndarray) -> np.ndarray:
        """Apply R to a row-major batch: ``X @ R.T``, identity short-circuited.

        Skipping the multiply is not just a speed-up. With `rotation="none"`
        a code becomes a pure `searchsorted` of the input value, with no
        floating-point summation in between, so identical inputs give
        identical codes across BLAS builds and thread counts.
        """
        if self.rotation == "none":
            return X
        if self._op is not None:
            return self._op.rotate_rows(X)
        return X @ self.R.T

    def _rotate_query(self, q: np.ndarray) -> np.ndarray:
        """Apply R to a single query vector: ``R @ q``, identity short-circuited."""
        if self.rotation == "none":
            return q
        if self._op is not None:
            return self._op.rotate_query(q)
        return self.R @ q

    def _unrotate_rows(self, X_rot: np.ndarray) -> np.ndarray:
        """Undo ``_rotate_rows``: ``X_rot @ R`` (R orthogonal, so R.T is R^-1)."""
        if self.rotation == "none":
            return X_rot
        if self._op is not None:
            return self._op.unrotate_rows(X_rot)
        return X_rot @ self.R

    def _check_rotation(self, compressed) -> None:
        """Refuse to interpret codes a different rotation encoded.

        The rotation is part of the encoding exactly as `seed` is, and the
        two constructions disagree on ~50% of stored bits — so a mismatch
        is not a small error, it is noise that still looks like an answer.
        Recording the rotation on disk is what makes this check possible;
        this is what makes it protective. If you are knowingly reading a
        legacy container whose recorded value is wrong, set
        ``compressed.rotation`` before calling.
        """
        stored = getattr(compressed, "rotation", LEGACY_ROTATION)
        if stored != self.rotation:
            raise ValueError(
                f"rotation mismatch: these codes were encoded with "
                f"{stored!r} but this Quantizer uses {self.rotation!r}. "
                f"Rebuild the Quantizer with rotation={stored!r}. "
                f"(Decoding across rotations returns plausible-looking "
                f"vectors that are wrong — the two disagree on about half "
                f"of all stored bits.)"
            )
        self._check_mode(compressed)

    def _check_mean(self, compressed) -> None:
        """Refuse to interpret codes written against a different centre.

        A centred code is meaningless without its mean and an uncentred one
        is wrong with it, so this is the same class of mismatch as
        ``rotation`` and is refused the same way rather than returning
        vectors that look plausible.
        """
        stored = getattr(compressed, "mean", None)
        if stored is None and self.mean is None:
            return
        if stored is None or self.mean is None:
            which = "no mean" if stored is None else "a corpus mean"
            want = "a corpus mean" if stored is None else "no mean"
            raise ValueError(
                f"mean mismatch: these codes were encoded with {which} but "
                f"this Quantizer uses {want}. Rebuild the Quantizer with the "
                f"mean that wrote them."
            )
        if not np.allclose(stored, self.mean, rtol=1e-5, atol=1e-6):
            raise ValueError(
                "mean mismatch: these codes were encoded against a different "
                "corpus mean. Decoding against this one would shift every "
                "reconstructed vector by the difference."
            )

    def _check_mode(self, compressed) -> None:
        """Refuse to interpret codes the other mode wrote.

        Whether a container carries norms says which mode encoded it, and
        the codebooks differ by a factor of ``scale * sqrt(d)`` — so
        crossing the two rescales every reconstructed coordinate. Caught
        here rather than at the first ``None`` dereference, which would
        only fire on the paths that happen to touch norms.
        """
        has_norms = getattr(compressed, "norms", None) is not None
        if has_norms == self.normalize:
            return
        stored, wanted = (
            ("normalize=True", "normalize=False") if has_norms
            else ("normalize=False (scalar)", "normalize=True")
        )
        raise ValueError(
            f"mode mismatch: these codes were encoded with {stored} but "
            f"this Quantizer uses {wanted}. Rebuild the Quantizer with the "
            f"mode that wrote them — the two use codebooks scaled "
            f"differently, so the reconstruction would be off by a constant "
            f"factor across every coordinate."
        )

    def _query_offset(self, query: np.ndarray) -> float:
        """``q . mu``, the part of a centred score that does not vary by row.

        Constant across the corpus for a given query, so it cannot reorder
        anything; it is added anyway so the scores a search returns match
        the inner products against ``decode()``.
        """
        if self.mean is None:
            return 0.0
        return float(np.asarray(query, dtype=np.float32) @ self.mean)

    def _direction_lengths(
        self, compressed, precision: Optional[int] = None
    ) -> np.ndarray:
        """L2 length of each decoded direction, ``||u_hat||``.

        The stored norm is the length of the *original* vector, but the
        direction it multiplies is a quantized one whose own length is not 1
        -- at 2-bit on real embeddings it runs 0.89-0.97 and varies per
        vector. This is that length, read straight off the codes: no bytes
        are stored for it, and none need to be.

        Rotation is orthogonal, so the length is the same in rotated and
        original space and no unrotate is needed. Cached per precision on the
        container, because the Matryoshka tables give a different length at
        every level.
        """
        cached = compressed._dir_len.get(precision)
        if cached is not None:
            return cached

        centroids = self._resolve_centroids(compressed, precision)
        sq = np.square(centroids.astype(np.float64))

        if isinstance(compressed, PackedVectors):
            shift = (
                0 if (precision is None or precision == self.bits)
                else (self.bits - precision)
            )
            n = compressed.n
            acc = np.empty(n, dtype=np.float64)
            for start in range(0, n, 4096):
                end = min(start + 4096, n)
                chunk = compressed.unpack_rows(start, end)
                if shift > 0:
                    chunk = chunk >> shift
                acc[start:end] = sq[chunk].sum(axis=1)
        else:
            indices = self._resolve_indices(compressed, precision)
            acc = sq[indices].sum(axis=1)

        lengths = np.sqrt(acc).astype(np.float32)
        compressed._dir_len[precision] = lengths
        return lengths

    def _effective_norms(
        self, compressed, precision: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """The scale to multiply a decoded direction by.

        ``norms`` on its own reconstructs a vector of the wrong length;
        dividing by the decoded direction's length fixes it. Returns
        ``norms`` unchanged in scalar mode (no norms to correct) and under
        ``renorm=False``.
        """
        if compressed.norms is None or not self.renorm:
            return compressed.norms
        lengths = self._direction_lengths(compressed, precision)
        return compressed.norms / np.maximum(lengths, 1e-12)

    def decode(
        self, compressed: CompressedVectors, precision: Optional[int] = None
    ) -> np.ndarray:
        """
        Reconstruct vectors from compressed representation.

        Args:
            compressed: CompressedVectors from encode().
            precision: Bit precision for reconstruction (1 to self.bits).
                       None = full precision. Only available in data-oblivious mode.

        Returns:
            (n, d) float32 array of approximate vectors.
        """
        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)
        X_hat_rot = centroids[indices]

        X_hat = self._unrotate_rows(X_hat_rot)
        if compressed.norms is None:  # scalar mode — coordinates are the code
            return X_hat
        X_hat = X_hat * self._effective_norms(compressed, precision)[:, None]
        if self.mean is not None:
            X_hat = X_hat + self.mean
        return X_hat

    def search(
        self,
        compressed: CompressedVectors,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors by approximate inner product.

        Operates in rotated space to avoid full dequantization.
        Caches the dequantized rotated representation for subsequent queries.

        Args:
            compressed: Encoded corpus (CompressedVectors only).
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision for search (1 to self.bits).
                       Lower = faster/coarser, higher = more accurate.
                       None = full precision (self.bits).
                       Only available in data-oblivious mode.

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.

        Raises:
            TypeError: If passed a PackedVectors (use search_adc or
                search_twostage instead, or convert with to_compressed()).
        """
        if isinstance(compressed, PackedVectors):
            raise TypeError(
                "PackedVectors does not support cached search(). "
                "Use search_adc() or search_twostage(), or convert "
                "with packed.to_compressed() first."
            )
        self._check_rotation(compressed)
        query = np.asarray(query, dtype=np.float32)
        q_rot = self._rotate_query(query)

        X_hat_rot = self._get_x_hat_rot(compressed, precision)
        scores = _apply_norms(
            X_hat_rot @ q_rot, self._effective_norms(compressed, precision)
        ) + self._query_offset(query)

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def search_adc(
        self,
        compressed,
        query: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
        chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-efficient search via asymmetric distance computation.

        Computes approximate inner products using a lookup table over
        the uint8 indices, without materializing an (n, d) float32 matrix.
        Peak temporary memory is chunk_size * d * 4 bytes (~6 MB default).

        Slower per-query than ``search()`` (no persistent cache), but
        uses dramatically less RAM. Ideal for the coarse stage of
        two-stage retrieval, or when memory is constrained.

        Accepts both CompressedVectors and PackedVectors. When given
        PackedVectors, indices are unpacked on demand per chunk.

        Args:
            compressed: Encoded corpus (CompressedVectors or PackedVectors).
            query: (d,) query vector.
            k: Number of results.
            precision: Bit precision (1 to self.bits). None = full.
            chunk_size: Vectors per scoring chunk. Controls peak memory.

        Returns:
            (indices, scores): top-k corpus indices and approximate scores.
        """
        query = np.asarray(query, dtype=np.float32)
        q_rot = self._rotate_query(query)

        centroids = self._resolve_centroids(compressed, precision)
        table = np.outer(q_rot, centroids).astype(np.float32)

        if isinstance(compressed, PackedVectors):
            shift = 0 if (precision is None or precision == self.bits) else (self.bits - precision)
            scores = self._adc_score_packed(
                table, compressed, shift, chunk_size,
                self._effective_norms(compressed, precision),
            )
        else:
            indices = self._resolve_indices(compressed, precision)
            scores = self._adc_score_chunked(
                table, indices,
                self._effective_norms(compressed, precision), chunk_size
            )

        scores = scores + self._query_offset(query)

        if k >= compressed.n:
            topk_idx = np.argsort(-scores)
        else:
            topk_idx = np.argpartition(-scores, k)[:k]
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return topk_idx, scores[topk_idx]

    def search_twostage(
        self,
        compressed,
        query: np.ndarray,
        k: int = 10,
        candidates: int = 500,
        coarse_precision: Optional[int] = None,
        coarse_chunk_size: int = 4096,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage retrieval: memory-efficient coarse scan + precise rerank.

        Stage 1 (coarse): ADC lookup-table scan over the full corpus at
        reduced precision. No float32 cache — memory cost is only the
        uint8 indices (already stored) plus a tiny lookup table.

        Stage 2 (fine): Dequantize only the candidate vectors at full
        precision, then rerank by exact (quantized) inner product.

        Accepts both CompressedVectors and PackedVectors. When given
        PackedVectors, indices are unpacked on demand per chunk (coarse)
        and only for the candidate rows (fine).

        Memory profile at 100k vectors, d=384:
          - Single-stage search():    154 MB  (cached n*d float32)
          - Two-stage search_twostage: ~39 MB  (uint8 indices + 6 MB temp)

        Args:
            compressed: Encoded corpus (CompressedVectors or PackedVectors).
            query: (d,) query vector.
            k: Final number of results.
            candidates: Number of coarse candidates (stage 1).
            coarse_precision: Bit precision for coarse pass.
                              Default: max(1, self.bits - 2).
            coarse_chunk_size: Chunk size for ADC scoring.

        Returns:
            (indices, scores): top-k corpus indices and full-precision scores.
            Indices are into the original corpus (not the candidate set).
        """
        if coarse_precision is None:
            coarse_precision = max(1, self.bits - 2)

        query = np.asarray(query, dtype=np.float32)
        q_rot = self._rotate_query(query)
        coarse_k = min(candidates, compressed.n)
        is_packed = isinstance(compressed, PackedVectors)

        # Stage 1: ADC coarse scan — no float32 cache needed
        coarse_centroids = self._resolve_centroids(compressed, coarse_precision)
        coarse_table = np.outer(q_rot, coarse_centroids).astype(np.float32)

        if is_packed:
            coarse_shift = self.bits - coarse_precision
            coarse_scores = self._adc_score_packed(
                coarse_table, compressed, coarse_shift, coarse_chunk_size,
                self._effective_norms(compressed, coarse_precision),
            )
        else:
            coarse_indices = self._resolve_indices(compressed, coarse_precision)
            coarse_scores = self._adc_score_chunked(
                coarse_table, coarse_indices,
                self._effective_norms(compressed, coarse_precision),
                coarse_chunk_size
            )

        if coarse_k >= compressed.n:
            coarse_idx = np.argsort(-coarse_scores)
        else:
            coarse_idx = np.argpartition(-coarse_scores, coarse_k)[:coarse_k]

        # Stage 2: full-precision rerank on small candidate set
        fine_centroids = self._resolve_centroids(compressed, None)
        if is_packed:
            fine_indices = compressed.unpack_at(coarse_idx)
        else:
            fine_indices = compressed.indices[coarse_idx]

        X_hat_cand = fine_centroids[fine_indices]  # (candidates, d)

        fine_norms = self._effective_norms(compressed, None)
        cand_norms = None if fine_norms is None else fine_norms[coarse_idx]
        fine_scores = _apply_norms(X_hat_cand @ q_rot, cand_norms) + self._query_offset(query)
        rerank_order = np.argsort(-fine_scores)[:k]

        original_idx = coarse_idx[rerank_order]
        return original_idx, fine_scores[rerank_order]

    def search_batch(
        self,
        compressed: CompressedVectors,
        queries: np.ndarray,
        k: int = 10,
        precision: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a batch of queries.

        Uses matrix multiplication instead of per-query matvec for
        significantly better throughput via BLAS-level parallelism.

        Args:
            compressed: Encoded corpus (CompressedVectors only).
            queries: (n_queries, d) query matrix.
            k: Number of results per query.
            precision: Bit precision for search (1 to self.bits). None = full.

        Returns:
            (indices, scores): both (n_queries, k) arrays.
                indices[i] = top-k corpus indices for query i.
                scores[i] = corresponding approximate scores, descending.

        Raises:
            TypeError: If passed a PackedVectors.
        """
        if isinstance(compressed, PackedVectors):
            raise TypeError(
                "PackedVectors does not support cached search_batch(). "
                "Use search_adc() or search_twostage(), or convert "
                "with packed.to_compressed() first."
            )
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis]

        n_queries = queries.shape[0]
        Q_rot = self._rotate_rows(queries)  # (n_queries, d)

        X_hat_rot = self._get_x_hat_rot(compressed, precision)
        # (n_queries, n) = (n_queries, d) @ (d, n)
        eff = self._effective_norms(compressed, precision)
        batch_norms = None if eff is None else eff[np.newaxis, :]
        all_scores = _apply_norms(Q_rot @ X_hat_rot.T, batch_norms)
        if self.mean is not None:
            all_scores = all_scores + (
                np.asarray(queries, dtype=np.float32) @ self.mean
            )[:, None]

        all_indices = np.empty((n_queries, min(k, compressed.n)), dtype=np.intp)
        all_topk_scores = np.empty((n_queries, min(k, compressed.n)), dtype=np.float32)

        for i in range(n_queries):
            scores_i = all_scores[i]
            if k >= compressed.n:
                topk_idx = np.argsort(-scores_i)
            else:
                topk_idx = np.argpartition(-scores_i, k)[:k]
                topk_idx = topk_idx[np.argsort(-scores_i[topk_idx])]
            all_indices[i] = topk_idx
            all_topk_scores[i] = scores_i[topk_idx]

        return all_indices, all_topk_scores

    def mse(self, X: np.ndarray, precision: Optional[int] = None) -> float:
        """Compute mean per-vector reconstruction MSE (L2 squared)."""
        compressed = self.encode(X)
        X_hat = self.decode(compressed, precision=precision)
        dtype = np.float32 if self.normalize else np.float64
        X_ref = np.asarray(X, dtype)
        if X_ref.ndim == 1:
            X_ref = X_ref[np.newaxis]
        return float(np.mean(np.sum((X_ref - X_hat) ** 2, axis=1)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adc_score_chunked(
        table: np.ndarray,
        indices: np.ndarray,
        norms: Optional[np.ndarray],
        chunk_size: int,
    ) -> np.ndarray:
        """Score vectors via ADC lookup table, processing in chunks.

        Args:
            table: (d, n_levels) float32 lookup table.
            indices: (n, d) uint8 quantization indices.
            norms: (n,) float32 vector norms, or None in scalar mode.
            chunk_size: Rows per chunk (controls peak memory).

        Returns:
            (n,) float32 approximate inner-product scores.

        Memory: peak allocation is chunk_size * d * 4 bytes.
        At chunk_size=4096, d=384: ~6 MB temporary.
        """
        n = indices.shape[0]
        d = table.shape[0]
        dim_idx = np.arange(d)
        scores = np.empty(n, dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = indices[start:end]  # (chunk, d) uint8
            # Gather: table[j, chunk_idx[i, j]] → (chunk, d) float32
            # Then sum over d → (chunk,) inner-product contribution
            chunk_scores = table[dim_idx, chunk_idx].sum(axis=1)
            chunk_norms = None if norms is None else norms[start:end]
            scores[start:end] = _apply_norms(chunk_scores, chunk_norms)

        return scores

    @staticmethod
    def _adc_score_packed(
        table: np.ndarray,
        packed: "PackedVectors",
        shift: int,
        chunk_size: int,
        norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score packed vectors via ADC, unpacking chunks on demand.

        Args:
            table: (d, n_levels) float32 lookup table.
            packed: PackedVectors with bit-packed indices.
            shift: Right-shift to apply for precision reduction (0 = full).
            chunk_size: Rows per chunk (controls peak memory).
            norms: (n,) scale per vector, or None in scalar mode. Pass
                ``Quantizer._effective_norms`` so the reconstruction-length
                correction reaches this path too; defaults to the packed
                container's own norms.

        Returns:
            (n,) float32 approximate inner-product scores.
        """
        n = packed.n
        d = table.shape[0]
        dim_idx = np.arange(d)
        scores = np.empty(n, dtype=np.float32)
        scale = packed.norms if norms is None else norms

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_idx = packed.unpack_rows(start, end)  # (chunk, d) uint8
            if shift > 0:
                chunk_idx = chunk_idx >> shift
            chunk_scores = table[dim_idx, chunk_idx].sum(axis=1)
            chunk_norms = None if scale is None else scale[start:end]
            scores[start:end] = _apply_norms(chunk_scores, chunk_norms)

        return scores

    def _get_x_hat_rot(
        self, compressed: CompressedVectors, precision: Optional[int] = None
    ) -> np.ndarray:
        """Get dequantized vectors in rotated space, with caching.

        Cache is only used for full-precision queries (precision=None).
        """
        if precision is None and compressed._x_hat_rot is not None:
            return compressed._x_hat_rot

        centroids = self._resolve_centroids(compressed, precision)
        indices = self._resolve_indices(compressed, precision)

        X_hat_rot = centroids[indices]

        # Cache only full-precision results
        if precision is None:
            compressed._x_hat_rot = X_hat_rot

        return X_hat_rot

    def _resolve_centroids(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Get centroid table for the requested precision level.

        Every decode/search path passes through here, which makes it the
        one place the rotation invariant has to hold.
        """
        self._check_rotation(compressed)
        self._check_mean(compressed)
        if precision is None:
            return self.centroids
        if precision < 1 or precision > self.bits:
            raise ValueError(
                f"precision must be 1-{self.bits}, got {precision}"
            )
        return self._nested[precision]

    def _resolve_indices(
        self, compressed: CompressedVectors, precision: Optional[int]
    ) -> np.ndarray:
        """Right-shift indices to the requested precision level."""
        if precision is None or precision == self.bits:
            return compressed.indices
        shift = self.bits - precision
        return compressed.indices >> shift
