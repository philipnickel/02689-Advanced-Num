"""Utility package containing shared helpers for assignments."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import sys

PathLike = Union[str, Path]


def ensure_repo_root_on_path(caller: PathLike, *, levels_up: int = 2) -> Path:
    """Add the repository root to ``sys.path`` when running scripts directly."""

    root = Path(caller).resolve().parents[levels_up]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root



__all__ = ["ensure_repo_root_on_path"]
