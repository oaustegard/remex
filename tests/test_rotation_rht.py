"""Tests for the randomized-Hadamard rotation option.

The claim this rests on is that an RHT is a legitimate substitute for the Haar
rotation *for what the codec actually needs* — an orthogonal map that leaves
coordinates incoherent, so the Lloyd-Max codebook (fitted to N(0, 1/d)) is
quantizing the distribution it was built for. These check that property
directly rather than trusting the recall result from a different codebase.
"""

import math

import numpy as np
import pytest

from remex.core import Quantizer
from remex.pq_format import save_params
from remex.rotation import haar_rotation, rht_rotation

DIMS = [64, 128, 384, 768, 1024]


@pytest.mark.parametrize("d", DIMS)
def test_orthogonal(d):
    Q = rht_rotation(d, seed=7)
    assert Q.dtype == np.float32 and Q.shape == (d, d)
    err = np.max(np.abs(Q.astype(np.float64) @ Q.astype(np.float64).T - np.eye(d)))
    assert err < 1e-5, f"max|QQ^T - I| = {err:.2e}"


@pytest.mark.parametrize("d", DIMS)
def test_deterministic_from_seed(d):
    assert np.array_equal(rht_rotation(d, 42), rht_rotation(d, 42))
    assert not np.array_equal(rht_rotation(d, 42), rht_rotation(d, 43))


@pytest.mark.parametrize("d", [128, 384, 768])
def test_incoherence_matches_haar(d):
    """The property the codebook depends on: a coordinate spike must come out
    spread. Compared against an INDEPENDENT reference — the max|coord| of a
    uniformly random unit vector, drawn directly — not against Haar alone,
    because two equally-broken rotations would agree with each other.
    """
    rng = np.random.default_rng(3)
    E = np.eye(d, dtype=np.float32)[:64]           # 64 coordinate spikes
    U = rng.standard_normal((4096, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ideal = float(np.mean(np.max(np.abs(U), axis=1)))

    # The lower edge is ATTAINABLE and inclusive, with an ulp of slack: a
    # spike spread to exactly +-1/sqrt(d) in every coordinate is the
    # information-theoretic floor, since the coordinates square to 1. Before
    # 1.0 a power-of-two d took one round and landed exactly there, for every
    # seed (#89). Reaching it is optimal, not a failure, and float32 rounding
    # lands a hair below the float64 constant.
    floor = (1.0 - 1e-6) / math.sqrt(d)
    for name, R in (("haar", haar_rotation(d, 5)), ("rht", rht_rotation(d, 5))):
        mu = float(np.mean(np.max(np.abs(E @ R.T), axis=1)))
        assert floor <= mu < 2.0 * ideal, (
            f"{name} d={d}: spike incoherence {mu:.6f} outside "
            f"[{floor:.6f}, {2 * ideal:.4f})"
        )


def test_identity_would_fail_the_incoherence_check():
    """The check above must be able to reject something. An identity 'rotation'
    is the worst case and has to land outside the bracket."""
    d = 384
    rng = np.random.default_rng(3)
    U = rng.standard_normal((4096, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ideal = float(np.mean(np.max(np.abs(U), axis=1)))
    mu_identity = 1.0                               # a spike stays a spike
    floor = (1.0 - 1e-6) / math.sqrt(d)
    assert not (floor <= mu_identity < 2.0 * ideal)


@pytest.mark.parametrize("d", [128, 768])
def test_coordinates_are_gaussian_after_rotation(d):
    """The codebook is fitted to N(0, 1/d); check the rotation delivers it."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((2000, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    rot = (X @ rht_rotation(d, 5).T).ravel()
    assert abs(float(np.mean(rot))) < 5e-3
    assert abs(float(np.std(rot)) - 1.0 / math.sqrt(d)) / (1.0 / math.sqrt(d)) < 0.05


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_round_trip_fidelity_matches_haar(bits):
    """End to end: RHT must not cost reconstruction quality."""
    d, rng = 384, np.random.default_rng(0)
    X = rng.standard_normal((300, d)).astype(np.float32)
    cos = {}
    for rot in ("haar", "rht"):
        q = Quantizer(d=d, bits=bits, seed=42, rotation=rot)
        Xh = q.decode(q.encode(X))
        num = np.sum(X * Xh, axis=1)
        den = np.linalg.norm(X, axis=1) * np.linalg.norm(Xh, axis=1)
        cos[rot] = float(np.mean(num / den))
    assert abs(cos["haar"] - cos["rht"]) < 2e-3, cos


def _mean_cos(X, A):
    return float(np.mean(np.sum(X * A, 1) / (
        np.linalg.norm(X, axis=1) * np.linalg.norm(A, axis=1))))


def test_rotation_is_part_of_the_encoding():
    """Crossing rotations is refused, not merely bad.

    The codes carry the rotation that wrote them, and every decode/search
    path checks it. This used to return wrong-but-plausible vectors.
    """
    d, rng = 128, np.random.default_rng(1)
    X = rng.standard_normal((64, d)).astype(np.float32)
    qh = Quantizer(d=d, bits=8, seed=42, rotation="haar")
    qr = Quantizer(d=d, bits=8, seed=42, rotation="rht")
    cv = qh.encode(X)

    assert _mean_cos(X, qh.decode(cv)) > 0.99

    for call in (lambda: qr.decode(cv),
                 lambda: qr.search(cv, X[0], k=5),
                 lambda: qr.search_adc(cv, X[0], k=5)):
        with pytest.raises(ValueError, match="rotation mismatch"):
            call()


def test_crossing_rotations_is_worth_refusing():
    """The refusal above has to be load-bearing, not fussiness.

    Override the recorded rotation to bypass the check — the escape hatch a
    caller with genuinely mislabelled legacy data would use — and measure
    what the check is preventing.
    """
    d, rng = 128, np.random.default_rng(1)
    X = rng.standard_normal((64, d)).astype(np.float32)
    qh = Quantizer(d=d, bits=8, seed=42, rotation="haar")
    qr = Quantizer(d=d, bits=8, seed=42, rotation="rht")

    cv = qh.encode(X)
    cv.rotation = "rht"          # lie about it, the way a bad default would
    crossed = qr.decode(cv)
    assert _mean_cos(X, crossed) < 0.5, _mean_cos(X, crossed)


@pytest.mark.parametrize("rotation", ["haar", "rht"])
def test_save_params_accepts_every_mojo_rotation(tmp_path, rotation):
    """Mojo rebuilds both constructions from the seed, so params dump for
    both. The R written is the Quantizer's own, whichever rotation built it."""
    q = Quantizer(d=64, bits=4, seed=1, rotation=rotation)
    path = tmp_path / f"{rotation}.pr"
    save_params(path, q)

    raw = path.read_bytes()
    d, n_levels = 64, 1 << 4
    assert len(raw) == 16 + d * d * 4 + (n_levels - 1) * 4 + n_levels * 4
    R = np.frombuffer(raw, np.float32, count=d * d, offset=16).reshape(d, d)
    assert np.array_equal(R, q.R)


def test_save_params_refuses_a_rotation_mojo_lacks(tmp_path):
    """The guard still has to bite for anything the Mojo port has no
    construction for — it just no longer bites on 'rht'."""
    q = Quantizer(d=64, bits=4, seed=1)
    q.rotation = "hadamard-but-fancier"
    with pytest.raises(ValueError, match="save_params supports rotation in"):
        save_params(tmp_path / "bad.pr", q)


def test_odd_dimension_rejected():
    with pytest.raises(ValueError, match="even dimension"):
        rht_rotation(255, seed=1)


def test_unknown_rotation_rejected():
    with pytest.raises(ValueError, match="rotation must be one of"):
        Quantizer(d=64, bits=4, rotation="hadamard")


# --- rotation is recorded wherever an index is persisted -------------------
# The gate at bench/gates/rotation_identity_gate.py covers .pq and .npz under
# a flipped library default. These cover the surfaces it states it does not:
# Arrow metadata, the .params byte, and the from_rows entry point.

@pytest.mark.parametrize("rotation", ["haar", "rht"])
def test_pq_and_npz_round_trip_rotation(tmp_path, rotation):
    from remex import load_pq, save_pq, CompressedVectors, PackedVectors

    X = np.random.default_rng(0).standard_normal((24, 64)).astype(np.float32)
    cv = Quantizer(d=64, bits=4, seed=1, rotation=rotation).encode(X)
    assert cv.rotation == rotation

    pq = tmp_path / "i.pq"
    save_pq(pq, cv)
    assert load_pq(pq).rotation == rotation

    npz = tmp_path / "i.npz"
    cv.save(npz)
    assert CompressedVectors.load(npz).rotation == rotation

    pv_path = tmp_path / "p.npz"
    PackedVectors.from_compressed(cv).save(pv_path)
    assert PackedVectors.load(pv_path).rotation == rotation


def test_pq_rotation_byte_is_where_the_spec_says(tmp_path):
    """Byte 17, and 0 must mean haar so pre-field files read correctly."""
    from remex import save_pq

    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    for rotation, expected in (("haar", 0), ("rht", 1)):
        p = tmp_path / f"{rotation}.pq"
        save_pq(p, Quantizer(d=64, bits=4, seed=1, rotation=rotation).encode(X))
        assert p.read_bytes()[17] == expected


def test_params_records_the_rotation_byte(tmp_path):
    """.params byte 9 mirrors .pq byte 17, same 0-means-haar convention."""
    for rotation, expected in (("haar", 0), ("rht", 1)):
        p = tmp_path / f"{rotation}.pr"
        save_params(p, Quantizer(d=32, bits=4, seed=1, rotation=rotation))
        assert p.read_bytes()[9] == expected


def test_from_rows_defaults_to_legacy_not_library_default():
    """The DB schema has no rotation column, so the default must be the
    frozen historical value — never whatever the library default is now."""
    from remex import PackedVectors
    from remex.rotation import LEGACY_ROTATION

    X = np.random.default_rng(0).standard_normal((6, 64)).astype(np.float32)
    pv = PackedVectors.from_compressed(
        Quantizer(d=64, bits=4, seed=1, rotation="rht").encode(X)
    )
    rows = [bytes(r) for r in pv._packed]
    rebuilt = PackedVectors.from_rows(rows, pv.norms, 64, 4)
    assert rebuilt.rotation == LEGACY_ROTATION == "haar"
    assert PackedVectors.from_rows(
        rows, pv.norms, 64, 4, rotation="rht"
    ).rotation == "rht"


@pytest.mark.parametrize("rotation", ["haar", "rht"])
def test_arrow_round_trips_rotation(tmp_path, rotation):
    pytest.importorskip("pyarrow")
    from remex import CompressedVectors, PackedVectors

    X = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    cv = Quantizer(d=64, bits=4, seed=1, rotation=rotation).encode(X)
    p = tmp_path / "i.arrow"
    cv.save_arrow(p, seed=1)
    assert CompressedVectors.load_arrow(p).rotation == rotation
    assert PackedVectors.load_arrow(p).rotation == rotation


def test_unknown_rotation_code_in_pq_is_rejected(tmp_path):
    from remex import load_pq, save_pq

    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    p = tmp_path / "i.pq"
    save_pq(p, Quantizer(d=64, bits=4, seed=1).encode(X))
    raw = bytearray(p.read_bytes())
    raw[17] = 7                      # a rotation a future remex might add
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="unknown rotation code"):
        load_pq(p)
