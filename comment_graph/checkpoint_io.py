"""Portable checkpoint filenames and interruption-safe best-weight replacement."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch


def portable_checkpoint_path(directory, filename, *, max_path_chars=None):
    directory = Path(directory).resolve()
    limit = max_path_chars if max_path_chars is not None else (240 if os.name == "nt" else None)
    path = directory / filename
    if limit is not None and len(str(path)) > limit:
        digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
        path = directory / f"ckpt_{digest}.pt"
    if limit is not None and (len(str(path)) > limit or len(str(directory / ('.ckpt-' + 'x' * 16 + '.tmp'))) > limit):
        raise ValueError("Checkpoint directory is too long; choose a shorter --checkpoint-dir")
    return path


def save_checkpoint_atomic(state, path):
    """Keep the previous best intact until a complete new checkpoint is ready."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=".ckpt-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
