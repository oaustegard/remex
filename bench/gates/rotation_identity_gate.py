#!/usr/bin/env python3
"""Gate: a stored index decodes the same way after the library default flips.

The hazard this guards
----------------------
``Quantizer`` can rotate with either the Haar QR construction or a randomized
Hadamard transform. The two give *different codes* from the same
``(d, bits, seed)``, and until the rotation was recorded nothing on disk said
which one wrote an index. The reader therefore fell back on the library's
current default — and the day somebody flips that default, every already-stored
index starts decoding queries in a rotation frame its codes were never written
in. Nothing raises. ``search()`` still returns k neighbours; they are the wrong
k. ``tests/test_rotation_rht.py::test_rotation_is_part_of_the_encoding``
measures the damage: mean cosine falls from >0.99 to <0.5.

The hazard runs in the direction of the *default*, not of the writer. This gate
is what makes the default safe to change: it asserts that changing it is a
no-op for decode of an already-written index.

What is checked
---------------
An index is written with the real library, then its rotation byte is zeroed to
manufacture a pre-field (legacy) file — the exact artifact whose
reinterpretation is the hazard. The library default is then flipped to ``rht``
and the legacy index reopened. Indices, norms and decoded vectors must be
**byte-identical** across that flip, and identical to a Haar reference computed
straight from ``remex.rotation.haar_rotation`` + ``remex.codebook`` without
going through ``core.py`` or ``pq_format.py`` at all.

Running it red
--------------
``--simulate-prefix`` patches the readers to resolve the rotation from the live
module default instead of from the file, which is precisely the pre-fix reader.
The gate must FAIL under that flag. A gate that has only ever been seen green
has not been shown to work.

    python3 bench/gates/rotation_identity_gate.py                    # expect 0
    python3 bench/gates/rotation_identity_gate.py --simulate-prefix  # expect 1

The harness lives in the ``gating`` skill; point ``GATING_SKILL_DIR`` at its
``scripts/`` directory if it is not staged at ``/tmp/gating-skill/scripts``.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_CANDIDATES = [
    os.environ.get("GATING_SKILL_DIR"),
    "/tmp/gating-skill/scripts",
    "/mnt/skills/user/gating/scripts",
    # Vendored fallback (bench/gates/gate.py) so the gate runs in CI, where
    # no skill directory is mounted. A real skill checkout still wins.
    str(Path(__file__).resolve().parent),
]
for _c in _CANDIDATES:
    if _c and (Path(_c) / "gate.py").is_file():
        sys.path.insert(0, _c)
        break
else:  # pragma: no cover - environment problem, not a gate failure
    sys.exit(
        "gate.py harness not found. Set GATING_SKILL_DIR to the gating "
        f"skill's scripts/ directory. Tried: {_CANDIDATES}"
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gate import Gate  # noqa: E402

from remex import CompressedVectors, PackedVectors, Quantizer  # noqa: E402
from remex.codebook import lloyd_max_codebook  # noqa: E402
from remex.packing import pack  # noqa: E402
from remex.pq_format import load_pq, save_pq  # noqa: E402
from remex.rotation import LEGACY_ROTATION, haar_rotation, rht_rotation  # noqa: E402

N, D, BITS, SEED = 96, 64, 4, 11


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _reference_codes(X: np.ndarray, kind: str) -> np.ndarray:
    """Encode without touching core.py or pq_format.py.

    This is the anchor: rotation and codebook straight from their own
    modules, quantization spelled out here. If this agreed with the library
    only because both call the same persistence helper, it would not be an
    anchor at all.
    """
    R = (haar_rotation if kind == "haar" else rht_rotation)(D, SEED)
    boundaries, _ = lloyd_max_codebook(D, BITS)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    unit = X / np.where(norms == 0, 1.0, norms)
    rotated = unit @ R.T
    return np.searchsorted(boundaries, rotated).astype(np.uint8)


def _rotation_default_slot() -> int:
    """Index of `rotation` within Quantizer.__init__.__defaults__.

    Looked up by name rather than assumed to be last: `normalize` and
    `scale` were added after it (#77), and a positional guess here would
    have silently rebound one of those instead — leaving this gate green
    while testing nothing.
    """
    spec = inspect.getfullargspec(Quantizer.__init__)
    n_defaults = len(spec.defaults)
    with_defaults = spec.args[-n_defaults:]
    return with_defaults.index("rotation")


def _library_default_rotation() -> str:
    return Quantizer.__init__.__defaults__[_rotation_default_slot()]


def _flip_library_default(kind: str) -> str:
    """Rebind Quantizer.__init__'s `rotation` default and return the old one."""
    slot = _rotation_default_slot()
    defaults = list(Quantizer.__init__.__defaults__)
    old = defaults[slot]
    defaults[slot] = kind
    Quantizer.__init__.__defaults__ = tuple(defaults)
    return old


def _zero_rotation_byte(path: Path) -> None:
    """Rewrite a .pq so byte 17 is zero — a file from before the field."""
    raw = bytearray(path.read_bytes())
    raw[17] = 0
    path.write_bytes(bytes(raw))


def _bit_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of agreeing bits in the *packed* codes — the bytes on disk.

    Compare the uint8 index arrays directly and you measure the padding: at
    4 bits the high nibble of every byte is always zero and always agrees,
    which floors the statistic near 0.75 and hides what the real codes do.
    The first version of this gate did exactly that and the bracket caught
    it, which is the bracket earning its place.
    """
    ab = np.unpackbits(pack(np.ascontiguousarray(a).ravel(), BITS))
    bb = np.unpackbits(pack(np.ascontiguousarray(b).ravel(), BITS))
    return float((ab == bb).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--simulate-prefix", action="store_true",
        help="patch the readers back to the pre-fix behaviour; gate must go red",
    )
    args = ap.parse_args()

    g = Gate(name="rotation-identity")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, D)).astype(np.float32)

    if args.simulate_prefix:
        # The pre-fix reader: resolve the rotation from whatever the library
        # default happens to be at read time, ignoring what the file says.
        import remex.pq_format as pqf

        _real_load = pqf.load_pq

        def _prefix_load(path):
            cv = _real_load(path)
            # Resolved the way Quantizer resolves it: since 1.0 the raw
            # default is None, meaning "rht" for even d and "haar" for odd.
            live_default = _library_default_rotation() or (
                "rht" if cv.d % 2 == 0 else "haar")
            return CompressedVectors(
                cv.indices, cv.norms, cv.d, cv.bits, live_default
            )

        pqf.load_pq = _prefix_load
        globals()["load_pq"] = _prefix_load
        g.note("--simulate-prefix active: readers trust the module default")

    tmp = Path(tempfile.mkdtemp())

    # ---- 1. anchors: the library agrees with an independent encode -------
    for kind in ("haar", "rht"):
        q = Quantizer(d=D, bits=BITS, seed=SEED, rotation=kind)
        lib = q.encode(X).indices
        ref = _reference_codes(X, kind)
        g.check(
            np.array_equal(lib, ref),
            f"{kind}: library codes match an independent encode",
            f"digest lib={_digest(lib)} ref={_digest(ref)}",
            kind="anchor",
            covers=(),
        )

    # ---- 2. the two rotations are far apart, so identity is non-trivial --
    haar_codes = _reference_codes(X, "haar")
    rht_codes = _reference_codes(X, "rht")
    agreement = _bit_agreement(haar_codes, rht_codes)
    g.bracket(
        "haar and rht codes are near-independent",
        value=agreement, lo=0.40, hi=0.60,
        why=(
            "a misread is a coin flip, not a rounding difference — this is "
            "what makes the byte-identity assertions below non-tautological"
        ),
    )

    # ---- 3. a legacy file survives a default flip ------------------------
    q_haar = Quantizer(d=D, bits=BITS, seed=SEED, rotation="haar")
    cv = q_haar.encode(X)
    pq_path = tmp / "legacy.pq"
    save_pq(pq_path, cv)
    _zero_rotation_byte(pq_path)

    before = load_pq(pq_path)
    g.check(
        before.rotation == LEGACY_ROTATION,
        "legacy .pq (rotation byte zeroed) resolves to haar",
        f"got {before.rotation!r}",
        covers=(),
    )

    old_default = _flip_library_default("rht")
    try:
        after = load_pq(pq_path)
        g.check(
            _library_default_rotation() == "rht",
            "library default really was flipped for this run",
            "guards against the flip silently not taking",
        )
        g.check(
            after.rotation == LEGACY_ROTATION,
            "legacy .pq still resolves to haar after the default flips",
            f"got {after.rotation!r}",
            covers=(),
        )
        g.check(
            np.array_equal(before.indices, after.indices)
            and np.array_equal(before.norms, after.norms),
            "legacy .pq bytes decode identically across the flip",
            f"digest before={_digest(before.indices, before.norms)} "
            f"after={_digest(after.indices, after.norms)}",
            covers=(),
        )
        g.check(
            np.array_equal(after.indices, haar_codes),
            "legacy .pq codes still match the independent haar reference",
            f"digest after={_digest(after.indices)} ref={_digest(haar_codes)}",
            kind="anchor",
            covers=(),
        )

        # ---- 4. an rht index round-trips under a flipped default too -----
        q_rht = Quantizer(d=D, bits=BITS, seed=SEED, rotation="rht")
        rht_path = tmp / "rht.pq"
        save_pq(rht_path, q_rht.encode(X))
        rt = load_pq(rht_path)
        g.check(
            rt.rotation == "rht" and np.array_equal(rt.indices, rht_codes),
            "an rht-written .pq reads back as rht",
            f"rotation={rt.rotation!r} digest={_digest(rt.indices)}",
            covers=(),
        )

        # ---- the recorded value has to be load-bearing, not advisory -----
        # This is the pair of checks the pre-fix reader actually breaks:
        # under it the legacy index claims 'rht', so the correct Haar
        # quantizer is refused and the wrong one is accepted. Recording the
        # rotation without consuming it would leave both of these green.
        decoded = None
        try:
            decoded = q_haar.decode(after)
            refused = False
        except ValueError:
            refused = True
        g.check(
            not refused and decoded is not None,
            "the correct haar Quantizer still decodes the legacy index",
            "a reader that mislabels the index refuses the right quantizer",
            covers=(),
        )

        try:
            Quantizer(d=D, bits=BITS, seed=SEED, rotation="rht").decode(after)
            wrong_refused = False
        except ValueError:
            wrong_refused = True
        g.check(
            wrong_refused,
            "the wrong (rht) Quantizer is refused on the legacy index",
            "enforcement, not just metadata",
            covers=(),
        )

        # ---- 5. the other two containers agree ---------------------------
        npz_path = tmp / "legacy.npz"
        cv.save(npz_path)
        g.check(
            CompressedVectors.load(npz_path).rotation == "haar",
            ".npz round-trips the rotation",
            covers=(),
        )
        pv = PackedVectors.from_compressed(q_rht.encode(X))
        pv_path = tmp / "rht_packed.npz"
        pv.save(pv_path)
        g.check(
            PackedVectors.load(pv_path).rotation == "rht",
            "PackedVectors .npz round-trips the rotation",
            covers=(),
        )
    finally:
        _flip_library_default(old_default)

    # ---- known-bad -------------------------------------------------------
    # Decode the haar index against the rht quantizer: the misread this whole
    # mechanism exists to prevent. It must land far from the truth.
    q_rht = Quantizer(d=D, bits=BITS, seed=SEED, rotation="rht")
    good = q_haar.decode(q_haar.encode(X))
    # Override the recorded value to get past the enforcement — this is
    # exactly the state a mislabelling reader puts the object in, so the
    # damage measured here is the damage the enforcement prevents.
    mislabelled = q_haar.encode(X)
    mislabelled.rotation = "rht"
    crossed = q_rht.decode(mislabelled)

    def _cos(A):
        num = np.sum(X * A, axis=1)
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(A, axis=1)
        return float(np.mean(num / den))

    cos_good, cos_crossed = _cos(good), _cos(crossed)
    g.known_bad(
        "decoding a haar index under an rht quantizer is caught as wrong",
        rejected=(cos_good > 0.9 and cos_crossed < 0.5),
        detail=f"cos(correct)={cos_good:.4f} cos(crossed)={cos_crossed:.4f}",
        covers=(
            "legacy .pq still resolves to haar after the default flips",
            "the correct haar Quantizer still decodes the legacy index",
            "the wrong (rht) Quantizer is refused on the legacy index",
        ),
    )
    g.note(f"haar/rht bit agreement = {agreement:.4f}")

    # ---- coverage limits -------------------------------------------------
    g.coverage(
        "Does NOT cover the Mojo readers. src/pq_format.mojo parses the same "
        "byte 17 and polarquant refuses a mismatch, but neither is exercised "
        "here — that needs a built binary, and the CLI does not build on a "
        "host without a supported GPU."
    )
    g.coverage(
        "Does NOT cover the Arrow container. save_arrow/load_arrow carry a "
        "b'rotation' metadata key with the same absent-means-haar rule, but "
        "pyarrow is an optional dependency. The pytest-arrow CI job installs "
        "it and fails if those tests skip, so the path is covered there — not "
        "here, and not in a local run without the extra."
    )
    g.coverage(
        "Does NOT protect a pre-field READER from a future rht file. An old "
        "remex ignores byte 17 and will decode an rht index as haar, silently. "
        "This protects existing indexes from a future default flip, not the "
        "reverse — that direction is unfixable from this side of the wire."
    )
    g.coverage(
        "Does NOT cover .params. That file embeds R explicitly, so a reader "
        "cannot misread it; its byte 9 is recorded for the --seed path's "
        "benefit only and is not asserted here."
    )
    g.coverage(
        f"One shape only: n={N}, d={D}, bits={BITS}, seed={SEED}. Bit widths "
        "other than 4, and d where the RHT block size is small, are untested "
        "by this gate (tests/ and the Mojo test_rht cover those separately)."
    )
    g.coverage(
        "Does NOT cover LAPACK/BLAS drift across machines. haar_rotation uses "
        "an explicit Householder QR precisely to avoid that, but this gate "
        "runs on one host and cannot see cross-host divergence."
    )
    return g.report()


if __name__ == "__main__":
    sys.exit(main())
