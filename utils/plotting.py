"""Shared helpers for plotting style and output management."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import scienceplots  # noqa: F401  # registers the "science" style family

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STYLES = ("science", "ieee", "bright")

_active_save_dir: Optional[Path] = None


def _resolve_dir(base_dir: Union[Path, str], subdir: Union[Path, str]) -> Path:
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = _REPO_ROOT / base_path

    sub_path = Path(subdir)
    if not sub_path.is_absolute():
        sub_path = base_path / sub_path

    return sub_path


def setup_plotting(
    base_dir: Union[Path, str],
    subdir: Union[Path, str],
    *,
    styles: Optional[Iterable[str]] = None,
) -> Path:
    """Configure matplotlib styles and the default directory for figure output."""

    style_list = list(styles if styles is not None else _DEFAULT_STYLES)
    if style_list:
        plt.style.use(style_list)

    save_dir = _resolve_dir(base_dir, subdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global _active_save_dir
    _active_save_dir = save_dir

    return save_dir


def setup_assignment_plotting(
    path: Union[Path, str],
    *,
    styles: Optional[Iterable[str]] = None,
) -> Path:
    """Resolve ``path`` relative to the repo root and prepare the directory."""

    return setup_plotting(_REPO_ROOT, path, styles=styles)


def style_axes(
    axes: Union[plt.Axes, Iterable[plt.Axes]],
    *,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: bool | dict = False,
    grid: bool | dict = True,
) -> None:
    """Apply common styling options to one or many matplotlib ``Axes`` objects."""

    def _style_single(ax: plt.Axes) -> None:
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if legend:
            if isinstance(legend, dict):
                ax.legend(**legend)
            else:
                ax.legend()
        if grid:
            if isinstance(grid, dict):
                ax.grid(**grid)
            else:
                ax.grid()

    if isinstance(axes, plt.Axes):
        _style_single(axes)
        return

    for ax in axes:
        _style_single(ax)


def save_figure(
    filename: Union[str, Path],
    *,
    fig: Optional[Figure] = None,
    tight: bool = True,
    tight_layout_kwargs: Optional[dict] = None,
    **kwargs,
) -> Path:
    """Save a figure to the configured directory and return the path."""

    if _active_save_dir is None:
        raise RuntimeError("setup_plotting must be called before save_figure.")

    name = Path(filename)
    if name.suffix.lower() != ".pdf":
        name = name.with_suffix(".pdf")

    target = _active_save_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)

    figure = fig if fig is not None else plt.gcf()
    if tight:
        kwargs_to_use = tight_layout_kwargs or {}
        figure.tight_layout(**kwargs_to_use)

    figure.savefig(target, **kwargs)
    return target


__all__ = ["setup_plotting", "setup_assignment_plotting", "style_axes", "save_figure"]
