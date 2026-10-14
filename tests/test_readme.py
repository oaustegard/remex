"""Execute every ```python block in README.md against the real API.

The bug class this catches is API drift in the docs: a method that moved to
module level, a constructor that no longer exists, arguments in the wrong
order. Those read fine and are wrong, and nothing else in the suite looks at
them. (remax found three such defects in its own README this way —
oaustegard/remax#63.)

remex's README blocks are illustrative rather than self-contained: they use
`embeddings`, `corpus`, `query` and friends without defining them. Rather
than rewrite the README into something less readable, the fixtures are
supplied here and the blocks run in document order in one shared namespace,
so names a block defines (`pq`, `compressed`) are visible to later blocks —
which is how a reader would build them up.

Fixtures are re-seeded at whatever `d` a block's own `Quantizer(d=...)`
declares, because the README legitimately switches dimension between
sections.
"""
from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
_BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _blocks() -> list[str]:
    return _BLOCK.findall(README.read_text(encoding="utf-8"))


def test_readme_has_python_blocks():
    """Self-guard: if the extractor silently matches nothing, every other
    assertion in this file passes vacuously."""
    assert len(_blocks()) >= 5, (
        "extracted too few ```python blocks from README.md — the fence "
        "convention probably changed and this file is now testing nothing"
    )


def _seed_fixtures(ns: dict, d: int) -> None:
    rng = np.random.default_rng(0)
    ns.update(
        embeddings=rng.standard_normal((50, d)).astype(np.float32),
        corpus=rng.standard_normal((50, d)).astype(np.float32),
        query=rng.standard_normal(d).astype(np.float32),
        d=d,
    )


def test_readme_examples_execute(tmp_path, monkeypatch):
    """Every block runs. Blocks that write files do so in a tmp dir."""
    from remex import PackedVectors, Quantizer

    monkeypatch.chdir(tmp_path)
    # The sentence-transformers block runs against the stand-in model: CI has
    # no torch, and a real run would download weights. What it checks is the
    # remex side of the block, which is what drifts.
    from _fake_st import FakeMultiVectorModel, SentenceTransformer
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = SentenceTransformer
    stub.MultiVectorEncoder = lambda name, **kw: FakeMultiVectorModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    ns: dict = {}
    _seed_fixtures(ns, 384)

    for i, block in enumerate(_blocks()):
        m = re.search(r"Quantizer\(\s*d=(\d+)", block)
        if m:
            _seed_fixtures(ns, int(m.group(1)))
        if "from_rows" in block:
            # from_rows models reading packed bytes back out of a database.
            pv = PackedVectors.from_compressed(
                ns["pq"].encode(ns["corpus"])
            )
            ns["rows"] = [bytes(r) for r in pv._packed]
            ns["norms"] = pv.norms
        try:
            exec(compile(block, f"README.md[block {i}]", "exec"), ns)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            pytest.fail(
                f"README.md python block {i} failed to execute: "
                f"{type(exc).__name__}: {exc}\n\n{block}"
            )


def test_the_executor_would_notice_a_broken_block(tmp_path, monkeypatch):
    """Negative control. The check above must be able to reject something —
    otherwise it is decoration that happens to be green."""
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _seed_fixtures(ns, 64)
    bad = "from remex import Quantizer\nQuantizer(d=64).no_such_method()\n"
    with pytest.raises(AttributeError):
        exec(compile(bad, "<synthetic>", "exec"), ns)


def test_readme_quantizer_signature_claims_hold():
    """The README's headline claim names the tuple that determines a
    quantizer. If a determinant is added and the sentence is not updated,
    that sentence becomes the kind of confident-and-wrong prose this file
    exists to catch."""
    text = README.read_text(encoding="utf-8")
    assert "(d, bits, seed, rotation, normalize, scale)" in text, (
        "README no longer states the full determining tuple; if a determinant "
        "was added or removed, update the sentence and this assertion together"
    )
