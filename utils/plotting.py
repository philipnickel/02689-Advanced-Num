"""Shared helpers for plotting style and output management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # registers the "science" style family
from matplotlib.figure import Figure


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_DIR = _REPO_ROOT / "assignment_1" / "Plots"
_DEFAULT_STYLES = ("science", "ieee", "bright")

_active_save_dir: Optional[Path] = None


def setup_plotting(
    subdir: Optional[str] = None,
    *,
    base_dir: Optional[Union[Path, str]] = None,
    styles: Optional[Iterable[str]] = None,
) -> Path:
    """Configure matplotlib styles and the default directory for figure output."""

    style_list = list(styles if styles is not None else _DEFAULT_STYLES)
    if style_list:
        plt.style.use(style_list)

    root = _REPO_ROOT
    base_path = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR
    if not base_path.is_absolute():
        base_path = root / base_path

    save_dir = base_path / subdir if subdir else base_path
    save_dir.mkdir(parents=True, exist_ok=True)

    global _active_save_dir
    _active_save_dir = save_dir

    return save_dir


def save_figure(filename: str, fig: Optional[Figure] = None, **kwargs) -> Path:
    """Save a figure to the configured directory and return the path."""

    if _active_save_dir is None:
        raise RuntimeError("setup_plotting must be called before save_figure.")

    target = _active_save_dir / filename
    figure = fig if fig is not None else plt.gcf()
    figure.savefig(target, **kwargs)
    return target


__all__ = ["setup_plotting", "save_figure"]
