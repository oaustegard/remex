"""Relative links in the docs must point at files that exist.

remax#63 found two markdown files cited that had never existed. remex is
currently clean, which is exactly when this check is cheap to add and
worth having: the failure mode is a link that rots silently, long after
whoever wrote it has stopped looking.

No allowlist. remax's version needed one and reported that the allowlist
was itself the fragile part — three of its six known-bads were about the
allowlist going stale. With nothing to exempt here, the simplest thing
that can still fail is the right one; add exemptions only when a real
case demands it, and give each one a test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Markdown inline links: ](target). Skip anchors, URLs and mailto.
_LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)\s]+)\)")

DOC_FILES = sorted(
    p for p in [
        *ROOT.glob("*.md"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("bench/**/*.md"),
    ]
    if p.is_file()
)


def _links(path: Path) -> list[str]:
    return _LINK.findall(path.read_text(encoding="utf-8"))


def test_there_are_docs_to_check():
    """Self-guard: a glob that matches nothing makes every check below
    pass while checking nothing."""
    assert len(DOC_FILES) >= 3, f"found only {len(DOC_FILES)} markdown files"
    assert any(_links(p) for p in DOC_FILES), "no relative links extracted"


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    missing = []
    for target in _links(doc):
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        if not (doc.parent / clean).resolve().exists():
            missing.append(target)
    assert not missing, (
        f"{doc.relative_to(ROOT)} links to files that do not exist: {missing}"
    )


def test_the_scanner_would_notice_a_dead_link(tmp_path):
    """Negative control — the regex has to actually match a link, and a
    dead target has to fail. A scanner whose pattern stops matching reports
    zero dead links forever, which is indistinguishable from success."""
    doc = tmp_path / "x.md"
    doc.write_text("see [gone](does/not/exist.md) and [ok](x.md)\n")
    found = _LINK.findall(doc.read_text())
    assert found == ["does/not/exist.md", "x.md"], found
    dead = [t for t in found if not (doc.parent / t).exists()]
    assert dead == ["does/not/exist.md"]


def test_the_scanner_ignores_urls_and_anchors(tmp_path):
    doc = tmp_path / "y.md"
    doc.write_text("[a](https://example.com/x.md) [b](#section) [c](y.md)\n")
    assert _LINK.findall(doc.read_text()) == ["y.md"]
