"""What a caller is left with when an encoding parameter or a file is wrong.

Each test here was a defect found by a behavior-spec pass over 2.0.0
(oaustegard/experiments, remex-product-description/bug-triage.md):

- B-01: a quantizer with another seed decoded any container without error
- B-02: an interrupted save destroyed the file it was overwriting
- B-03: scalar mode's ``scale`` was not recorded
- B-04: ``save("x")`` wrote ``x.npz`` that ``load("x")`` could not find
- B-05: damaged or foreign files raised errors that did not say so
- B-06: ``d`` was recorded but never compared
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import remex.core
from remex import (
    CompressedVectors, IVFCoarseIndex, PackedVectors, Quantizer, load_pq, save_pq,
)
from remex.gpu import GPUSearcher


def data(n=200, d=64, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def mean_cos(a, b):
    return float(np.mean(np.sum(a * b, 1)
                         / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)))


# --- B-01: the seed is recorded and checked -----------------------------------

def test_a_different_seed_is_refused_on_every_path():
    X = data()
    c = Quantizer(64, 4, seed=1).encode(X)
    q = Quantizer(64, 4, seed=2)
    for call in (lambda: q.decode(c),
                 lambda: q.search(c, X[0], k=3),
                 lambda: q.search_adc(c, X[0], k=3),
                 lambda: q.search_twostage(c, X[0], k=3, candidates=20,
                                           coarse_precision=2),
                 lambda: q.search_batch(c, X[:2], k=3),
                 lambda: q.decode(PackedVectors.from_compressed(c)),
                 lambda: IVFCoarseIndex(q, c, n_bits=4).search_coarse(X[0], k=3, nprobe=16),
                 lambda: GPUSearcher(q, c, backend="numpy").search(X[0], k=3)):
        with pytest.raises(ValueError, match="seed mismatch: these codes were encoded with seed=1"):
            call()


def test_the_right_seed_still_decodes():
    X = data()
    c = Quantizer(64, 4, seed=1).encode(X)
    assert mean_cos(Quantizer(64, 4, seed=1).decode(c), X) > 0.99


@pytest.mark.parametrize("fmt", ["npz", "packed_npz", "arrow"])
def test_the_seed_survives_every_format_that_can_hold_it(tmp_path, fmt):
    c = Quantizer(64, 4, seed=7).encode(data())
    path = tmp_path / "c"
    if fmt == "npz":
        loaded = CompressedVectors.load(c.save(path))
    elif fmt == "packed_npz":
        loaded = PackedVectors.load(PackedVectors.from_compressed(c).save(path))
    else:
        pytest.importorskip("pyarrow")
        c.save_arrow(str(path) + ".arrow")
        loaded = PackedVectors.load_arrow(str(path) + ".arrow")
    assert loaded.seed == 7
    with pytest.raises(ValueError, match="seed mismatch"):
        Quantizer(64, 4, seed=8).decode(loaded)


def test_the_seed_survives_subset_precision_and_conversion():
    c = Quantizer(64, 8, seed=7).encode(data())
    p = PackedVectors.from_compressed(c)
    for derived in (c.subset([0, 1]), p.subset([0, 1]), p.at_precision(2),
                    p.to_compressed()):
        assert derived.seed == 7


def test_an_unrecorded_seed_is_not_checked(tmp_path):
    """Files from remex < 2.1 have no seed entry; they must keep loading."""
    c = Quantizer(64, 4, seed=1).encode(data())
    fields = c._npz_fields()
    del fields["seed"]
    np.savez_compressed(tmp_path / "old.npz", **fields)
    old = CompressedVectors.load(tmp_path / "old.npz")
    assert old.seed is None
    Quantizer(64, 4, seed=1).decode(old)
    Quantizer(64, 4, seed=2).decode(old)  # unknown, so not refused


def test_rotation_none_ignores_the_seed():
    c = Quantizer(64, 4, seed=1, rotation="none", normalize=False).encode(data())
    out = Quantizer(64, 4, seed=2, rotation="none", normalize=False).decode(c)
    assert out.shape == (200, 64)


def test_save_arrow_refuses_a_contradicting_seed(tmp_path):
    pytest.importorskip("pyarrow")
    c = Quantizer(64, 4, seed=7).encode(data())
    with pytest.raises(ValueError, match="contradicts the seed"):
        c.save_arrow(str(tmp_path / "x.arrow"), seed=8)
    assert not (tmp_path / "x.arrow").exists()


def test_remex_index_keeps_the_seed_across_adds(tmp_path):
    from remex import RemexIndex
    index = RemexIndex(None, bits=4, seed=5, dim=64)
    index.add(data(10))
    index.add(data(10, seed=1))
    assert index.compressed.seed == 5
    loaded = RemexIndex.load(index.save(tmp_path / "idx"))
    assert loaded.compressed.seed == 5


# --- B-03: scalar mode's scale ------------------------------------------------

def test_a_different_scale_is_refused_in_scalar_mode(tmp_path):
    q = Quantizer(64, 4, normalize=False, scale=2.0, rotation="none")
    c = CompressedVectors.load(q.encode(data()).save(tmp_path / "s"))
    assert c.scale == 2.0
    with pytest.raises(ValueError, match="scale mismatch"):
        Quantizer(64, 4, normalize=False, scale=1.0, rotation="none").decode(c)
    q.decode(c)


# --- B-06: dimension ----------------------------------------------------------

@pytest.mark.parametrize("rotation,normalize", [("rht", True), ("haar", True),
                                                ("none", False)])
def test_a_different_dimension_is_refused_the_same_way(rotation, normalize):
    X = data()
    c = Quantizer(64, 4, rotation=rotation, normalize=normalize).encode(X)
    q = Quantizer(32, 4, rotation=rotation, normalize=normalize)
    for call in (lambda: q.decode(c), lambda: q.search_adc(c, X[0, :32], k=3)):
        with pytest.raises(ValueError, match="dimension mismatch: these codes are d=64"):
            call()


# --- B-02: an interrupted write keeps the old file -----------------------------

def _writers(c, tmp_path):
    pytest.importorskip("pyarrow")
    q = Quantizer(64, 4)
    return {
        "save": (lambda p: c.save(p), tmp_path / "a.npz",
                 lambda p: CompressedVectors.load(p).n),
        "packed": (lambda p: PackedVectors.from_compressed(c).save(p),
                   tmp_path / "b.npz", lambda p: PackedVectors.load(p).n),
        "arrow": (lambda p: c.save_arrow(str(p)), tmp_path / "c.arrow",
                  lambda p: PackedVectors.load_arrow(str(p)).n),
        "pq": (lambda p: save_pq(p, c), tmp_path / "d.pq",
               lambda p: load_pq(p).n),
        "params": (lambda p: remex.save_params(p, q), tmp_path / "e.params",
                   lambda p: os.path.getsize(p)),
    }


@pytest.mark.parametrize("name", ["save", "packed", "arrow", "pq", "params"])
def test_an_exception_during_a_write_leaves_the_old_file(tmp_path, monkeypatch, name):
    good = Quantizer(64, 4).encode(data())
    write, path, read = _writers(good, tmp_path)[name]
    write(path)
    before = read(path)

    real_replace = os.replace

    def interrupted(src, dst):
        raise KeyboardInterrupt  # after the temp file is complete, before the rename

    monkeypatch.setattr(remex._io.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write(path)
    monkeypatch.setattr(remex._io.os, "replace", real_replace)

    assert read(path) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_a_killed_save_leaves_the_old_file(tmp_path):
    """The 2.0.0 failure: SIGKILL mid-overwrite left 0 bytes and no old copy."""
    path = tmp_path / "index.npz"
    Quantizer(64, 4).encode(data()).save(path)
    before = path.read_bytes()
    child = textwrap.dedent(f"""
        import os
        import numpy as np
        from remex import Quantizer
        def dies_partway(file, **kw):
            with open(file, "wb") as f:
                f.write(b"PK partial")
            os._exit(9)
        np.savez_compressed = dies_partway
        Quantizer(64, 4).encode(np.ones((5, 64), np.float32)).save({str(path)!r})
    """)
    assert subprocess.run([sys.executable, "-c", child]).returncode == 9
    assert path.read_bytes() == before
    CompressedVectors.load(path)


def test_save_still_writes_to_a_file_object():
    import io
    c = Quantizer(64, 4).encode(data())
    buf = io.BytesIO()
    c.save(buf)
    buf.seek(0)
    assert CompressedVectors.load(buf).n == 200


# --- B-04: the .npz suffix ----------------------------------------------------

def test_load_finds_the_name_save_wrote(tmp_path):
    c = Quantizer(64, 4).encode(data())
    written = c.save(str(tmp_path / "index"))
    assert written == str(tmp_path / "index.npz")
    assert CompressedVectors.load(str(tmp_path / "index")).n == 200
    assert PackedVectors.load(str(tmp_path / "index")).n == 200
    with pytest.raises(FileNotFoundError):
        CompressedVectors.load(str(tmp_path / "missing"))


# --- B-05: files that are not containers --------------------------------------

def test_unreadable_files_say_what_they_are(tmp_path):
    good = tmp_path / "good.npz"
    Quantizer(64, 4).encode(data()).save(good)
    raw = good.read_bytes()
    (tmp_path / "empty.npz").write_bytes(b"")
    (tmp_path / "half.npz").write_bytes(raw[: len(raw) // 2])
    (tmp_path / "junk.npz").write_bytes(b"not an archive at all")
    for name in ("empty.npz", "half.npz", "junk.npz"):
        with pytest.raises(ValueError, match="is not a readable .npz archive") as e:
            CompressedVectors.load(tmp_path / name)
        assert "allow_pickle" not in str(e.value).split("(")[0]


def test_a_foreign_archive_is_named_as_such(tmp_path):
    np.savez(tmp_path / "other.npz", a=np.arange(3))
    with pytest.raises(ValueError, match="not a remex container"):
        CompressedVectors.load(tmp_path / "other.npz")
    np.save(tmp_path / "arr.npy", np.arange(3))
    with pytest.raises(ValueError, match="not a remex .npz container"):
        CompressedVectors.load(tmp_path / "arr.npy")


def test_an_unknown_rotation_is_refused_at_save_not_at_load(tmp_path):
    path = tmp_path / "x.npz"
    c = Quantizer(64, 4).encode(data())
    c.save(path)
    before = path.read_bytes()
    c.rotation = "bogus"
    with pytest.raises(ValueError, match="unknown rotation"):
        c.save(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bits", [0, 5, 9])
def test_unsupported_widths_share_one_message(bits):
    with pytest.raises(ValueError, match=rf"bits={bits} is not supported. Use one of \[1, 2, 3, 4, 8\]"):
        Quantizer(64, bits)
