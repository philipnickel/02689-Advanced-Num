"""Plotting utilities for spectral method visualizations."""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """Get repository root directory, handling both local and sphinx-gallery execution.

    Returns the repository root by detecting the presence of pyproject.toml.
    Works in both local execution (via main.py) and sphinx-gallery contexts.

    Returns
    -------
    Path
        Absolute path to the repository root

    """
    try:
        # Try to get caller's __file__ if available (local execution)
        import inspect

        frame = inspect.currentframe().f_back
        caller_file = frame.f_globals.get("__file__")
        if caller_file:
            current = Path(caller_file).resolve().parent
        else:
            # __file__ not available (sphinx-gallery)
            current = Path.cwd()
    except (AttributeError, KeyError):
        # Fallback to cwd
        current = Path.cwd()

    # Walk up until we find pyproject.toml (marks repo root)
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback: assume 2 levels up from script directory
    # Works for Exercises/exercise_X/script.py structure
    return current.parent.parent if caller_file else current.parent.parent


def add_parameter_footer(fig, text: str) -> None:
    r"""
    Add a parameter footer to the bottom-left of a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to annotate
    text : str
        The parameter text to display (use LaTeX formatting with rf'...')

    """
    # Add annotation to figure with consistent styling
    fig.axes[0].annotate(
        text,
        xy=(0, 0),
        xycoords="figure fraction",
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        fontsize=7,
        color="gray",
    )
