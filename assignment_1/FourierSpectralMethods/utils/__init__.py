"""Bridge module so scripts can import the repo-level ``utils`` package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_DIR = _ROOT / "utils"
_INIT_FILE = _PACKAGE_DIR / "__init__.py"

if not _INIT_FILE.exists():  # pragma: no cover - guard for missing package
    raise ImportError("Real 'utils' package not found at expected location")

_SPEC = importlib.util.spec_from_file_location(
    "utils",
    str(_INIT_FILE),
    submodule_search_locations=[str(_PACKAGE_DIR)],
)

if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - failure case
    raise ImportError("Unable to create import spec for project 'utils' package")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
