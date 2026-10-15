"""File plumbing shared by every remex writer and the ``.npz`` readers.

Two promises, both about what a caller is left with when something goes wrong:

- **An overwrite is all-or-nothing.** Every writer used to open its destination
  for writing directly, which truncates the old file at the first byte; a
  process killed partway left a 0-byte file and no copy of the index it was
  replacing (a 109 MB file, SIGKILL 150 ms in). ``replacing`` writes to a
  sibling temporary name and ``os.replace``s it over the destination only once
  the write has finished, so the destination holds either the old file or the
  new one.
- **A file that is not a remex container says so.** ``np.load`` on a damaged
  file raises ``EOFError``, ``BadZipFile``, or a ``ValueError`` suggesting
  ``allow_pickle=True`` -- which invites the caller to enable pickle loading on
  a file that is merely truncated. ``open_npz`` turns all of those into one
  message naming the file.
"""

from __future__ import annotations

import contextlib
import os
import uuid
import zipfile
from typing import Iterator, Union

import numpy as np

PathOrFile = Union[str, "os.PathLike[str]", object]

#: Entries every remex ``.npz`` container has, whatever version wrote it.
NPZ_REQUIRED = ("d", "bits", "n")


def npz_path(path: PathOrFile) -> PathOrFile:
    """The name ``np.savez`` actually writes: a string path gains ``.npz``.

    NumPy appends the suffix to any path that does not end in it, so
    ``save("index")`` writes ``index.npz``. Returning the real name lets
    callers report it and lets ``replacing`` rename onto it.
    """
    if isinstance(path, (str, os.PathLike)):
        path = os.fspath(path)
        if not path.endswith(".npz"):
            path += ".npz"
    return path


@contextlib.contextmanager
def replacing(path: PathOrFile, suffix: str = "") -> Iterator[PathOrFile]:
    """Yield a temporary sibling of ``path`` to write; rename it over ``path`` on success.

    A file object is yielded unchanged: it is the caller's handle, and there is
    nothing to rename. ``suffix`` is appended to the temporary name, so a writer
    that insists on an extension (``np.savez`` and ``.npz``) writes exactly the
    name it was given. On any exception, including KeyboardInterrupt, the
    temporary file is removed and the destination is untouched.

    The temporary file is created by the writer, not here, so it gets the same
    permissions a direct write would have had.
    """
    if not isinstance(path, (str, os.PathLike)):
        yield path
        return
    path = os.fspath(path)
    head, tail = os.path.split(path)
    tmp = os.path.join(head, f".{tail}.{uuid.uuid4().hex[:12]}.tmp{suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def open_npz(path: PathOrFile):
    """``np.load`` a remex ``.npz`` container, refusing anything else by name.

    A string path that does not exist but whose ``.npz`` sibling does is read
    from the sibling, because that is the name ``save`` wrote it under. A
    missing file still raises ``FileNotFoundError``.
    """
    if isinstance(path, (str, os.PathLike)):
        path = os.fspath(path)
        if not os.path.exists(path) and os.path.exists(npz_path(path)):
            path = npz_path(path)
    try:
        data = np.load(path, allow_pickle=False)
    except (ValueError, EOFError, zipfile.BadZipFile) as e:
        raise ValueError(
            f"{path!r} is not a readable .npz archive: it is empty, truncated, "
            f"or another kind of file ({type(e).__name__}: {e})."
        ) from e
    if not hasattr(data, "files"):  # a bare .npy array, not an archive
        raise ValueError(f"{path!r} is a .npy array, not a remex .npz container")
    missing = [k for k in NPZ_REQUIRED if k not in data.files]
    if missing or not ({"packed_indices", "indices"} & set(data.files)):
        data.close()
        raise ValueError(
            f"{path!r} is an .npz archive but not a remex container: it has no "
            f"{', '.join(missing) or 'packed_indices'} entry "
            f"(entries: {sorted(data.files)})."
        )
    return data
