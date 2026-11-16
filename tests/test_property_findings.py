"""Defects found by model-based property tests of 2.2.0.

The harnesses that found them live in oaustegard/experiments,
remex-proptest/ (search, containers, stateful, encode); each test here is
the shrunk, deterministic witness of one finding.

- P-01: the two ``.npz`` writers used different packed layouts when
  ``d * bits % 8 != 0``, so reading one with the other class returned
  wrong codes without an error
- P-02: ``Quantizer.search`` skipped the bits and mean guards on a warm cache
- P-03: a pickled or deep-copied rht ``Quantizer`` segfaulted
- P-04: a rejected first batch left a model-less ``RemexIndex`` unusable
- P-05: ``RemexIndex.load`` bypassed ``open_npz``
- P-06: ``IVFCoarseIndex`` built an index from codes its quantizer refuses
- P-07: ``save_arrow`` extra metadata could relabel the encoding
- P-08: a damaged ``.npz`` member leaked zip/zlib errors
- P-09: a seed of 2**63 or more could not be saved to ``.npz``
- P-10: ``PackedVectors.load`` of an indices-only ``.npz`` raised KeyError
- P-11: vectors shorter than 1e-8 were coded off their direction
- P-12: degenerate codebooks, ``assign_codes`` dtypes, a NaN-ratio warning,
  ``d <= 0`` and ``corpus_mean`` of nothing
"""
import copy
import os
import pickle
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pytest

import remex
from remex import CompressedVectors, IVFCoarseIndex, PackedVectors, Quantizer
from remex.codebook import assign_codes, coordinate_sigma, lloyd_max_codebook
from remex.st import RemexIndex


def data(n=20, d=16, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d)) + shift).astype(np.float32)


# P-01 ---------------------------------------------------------------------

@pytest.mark.parametrize("d,bits,n", [(7, 3, 6), (3, 2, 5), (5, 1, 9), (13, 4, 4)])
def test_npz_written_by_either_class_reads_back_in_both(tmp_path, d, bits, n):
    c = Quantizer(d, bits, seed=1, rotation="haar").encode(data(n, d))
    from_cv = c.save(str(tmp_path / "cv"))
    from_pv = PackedVectors.from_compressed(c).save(str(tmp_path / "pv"))
    for path in (from_cv, from_pv):
        assert np.array_equal(CompressedVectors.load(path).indices, c.indices)
        assert np.array_equal(PackedVectors.load(path).unpack_rows(0, n), c.indices)


# P-02 ---------------------------------------------------------------------

def test_warm_search_refuses_an_uncentred_reader_of_centred_codes():
    X = data(40, 16, shift=3.0)
    owner = Quantizer(16, 2, mean=remex.corpus_mean(X))
    c = owner.encode(X)
    owner.search(c, X[0], k=3)  # warms the cache
    with pytest.raises(ValueError, match="mean"):
        Quantizer(16, 2, renorm=False).search(c, X[0], k=3)


def test_warm_scalar_search_refuses_another_bit_width():
    X = data(30, 16)
    owner = Quantizer(16, 4, normalize=False, scale=1.0, rotation="none")
    c = owner.encode(X)
    owner.search(c, X[0], k=3)
    with pytest.raises(ValueError, match="bits"):
        Quantizer(16, 3, normalize=False, scale=1.0, rotation="none").search(c, X[0], k=3)


# P-03 ---------------------------------------------------------------------

def test_pickled_rht_quantizer_encodes_in_a_fresh_process():
    blob = pickle.dumps(Quantizer(64, 4))
    code = ("import pickle, sys, numpy as np\n"
            "q = pickle.loads(sys.stdin.buffer.read())\n"
            "q.encode(np.ones((3, 64), np.float32))\n")
    r = subprocess.run([sys.executable, "-c", code], input=blob, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()[-500:]


def test_used_rht_quantizer_deepcopies_and_encodes_the_same():
    q = Quantizer(64, 4)
    X = data(5, 64)
    c = q.encode(X)
    q2 = copy.deepcopy(q)
    del q
    assert np.array_equal(q2.encode(X).indices, c.indices)


# P-04 ---------------------------------------------------------------------

def test_rejected_empty_first_add_leaves_the_index_usable():
    idx = RemexIndex(bits=4)
    with pytest.raises(ValueError):
        idx.add([])
    assert idx.dim is None
    assert list(idx.add(np.ones((5, 16), np.float32))) == [0, 1, 2, 3, 4]


def test_rejected_mixed_width_token_batch_leaves_dim_unset():
    idx = RemexIndex(bits=4, similarity="maxsim")
    with pytest.raises(ValueError):
        idx.add([data(3, 8), data(2, 5)])
    assert idx.dim is None


# P-05 ---------------------------------------------------------------------

def test_index_load_finds_the_file_save_wrote_and_names_damage(tmp_path):
    idx = RemexIndex(bits=4)
    idx.add(data(10, 16))
    idx.save(str(tmp_path / "x"))
    assert len(RemexIndex.load(str(tmp_path / "x"))) == 10
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a readable .npz"):
        RemexIndex.load(str(bad))


# P-06 ---------------------------------------------------------------------

def test_ivf_constructor_refuses_codes_of_another_bit_width():
    c = Quantizer(16, 2, seed=1).encode(data(30, 16))
    with pytest.raises(ValueError, match="bits"):
        IVFCoarseIndex(Quantizer(16, 4, seed=1), c, n_bits=2)


# P-07 ---------------------------------------------------------------------

@pytest.mark.parametrize("key", ["rotation", "d", "n", "bits", "mean", "scale"])
def test_arrow_extra_metadata_cannot_relabel_the_encoding(tmp_path, key):
    pytest.importorskip("pyarrow")
    c = Quantizer(16, 2, seed=1).encode(data(10, 16))
    with pytest.raises(ValueError, match="part of the encoding"):
        c.save_arrow(str(tmp_path / "a.arrow"), **{key: "x"})


# P-08 ---------------------------------------------------------------------

def test_damaged_npz_member_raises_the_named_error(tmp_path):
    c = Quantizer(16, 2, seed=1).encode(data(50, 16))
    path = c.save(str(tmp_path / "c"))
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("packed_indices.npy")
    raw = bytearray(open(path, "rb").read())
    start = info.header_offset + 30 + len(info.filename) + len(info.extra)
    for i in range(start + 2, start + info.compress_size - 2):
        raw[i] ^= 0xFF
    open(path, "wb").write(bytes(raw))
    for loader in (CompressedVectors.load, PackedVectors.load):
        with pytest.raises(ValueError, match="damaged .npz"):
            loader(path)


# P-09 ---------------------------------------------------------------------

def test_npz_records_a_seed_beyond_int64(tmp_path):
    seed = 2**64 + 5
    q = Quantizer(8, 3, seed=seed, rotation="rht")
    c = q.encode(data(4, 8))
    back = CompressedVectors.load(c.save(str(tmp_path / "s")))
    assert back.seed == seed
    q.decode(back)


# P-10 ---------------------------------------------------------------------

def test_packedvectors_load_reads_an_indices_only_npz(tmp_path):
    c = Quantizer(7, 3, seed=1, rotation="haar").encode(data(5, 7))
    path = str(tmp_path / "legacy.npz")
    np.savez(path, indices=c.indices, norms=c.norms, d=np.int32(7),
             bits=np.int32(3), n=np.int32(5), rotation=np.str_("haar"))
    assert np.array_equal(PackedVectors.load(path).unpack_rows(0, 5), c.indices)


# P-11 ---------------------------------------------------------------------

def test_codes_of_a_tiny_vector_are_those_of_its_direction():
    q = Quantizer(64, 4, seed=3, rotation="rht")
    x = data(50, 64, seed=3)
    assert np.array_equal(q.encode(x * np.float32(2**-30)).indices, q.encode(x).indices)


# P-12 ---------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1e-300, 1e-46, 1e39])
def test_a_degenerate_scale_is_refused(scale):
    with pytest.raises(ValueError):
        lloyd_max_codebook(8, 4, sigma=scale)


@pytest.mark.parametrize("d", [0, -4])
def test_coordinate_sigma_refuses_non_positive_d(d):
    with pytest.raises(ValueError):
        coordinate_sigma(d)


def test_assign_codes_matches_searchsorted_for_non_native_inputs():
    b32, _ = lloyd_max_codebook(1, 4)
    b64 = b32.astype(np.float64) + 1e-9
    x32 = np.linspace(-3, 3, 1001, dtype=np.float32)
    assert np.array_equal(assign_codes(x32, b64, 4), np.searchsorted(b64, x32))
    xi = np.array([2**24 + 1, -(2**25) - 3, 0], dtype=np.int64)
    assert np.array_equal(assign_codes(xi, b32, 4), np.searchsorted(b32, xi))
    xbe = x32.astype(">f8")
    assert np.array_equal(assign_codes(xbe, b32, 4), np.searchsorted(b32, xbe))
    with pytest.raises(ValueError, match="boundaries"):
        assign_codes(x32, b32[:5], 4)


def test_inf_row_does_not_raise_an_anisotropy_warning():
    X = data(120, 16)
    X[0, 0] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", remex.AnisotropyWarning)
        Quantizer(16, 2).encode(X)


def test_corpus_mean_of_an_empty_corpus_is_refused():
    with pytest.raises(ValueError, match="empty"):
        remex.corpus_mean(np.zeros((0, 4), np.float32))
