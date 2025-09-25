"""Bridge package enabling absolute imports when scripts run directly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REAL_PKG_DIR = Path(__file__).resolve().parents[2]
_INIT_FILE = _REAL_PKG_DIR / "__init__.py"

if not _INIT_FILE.exists():  # pragma: no cover - ensure parent package exists
    raise ImportError("Parent assignment_1 package not found")

_SPEC = importlib.util.spec_from_file_location(
    "assignment_1",
    str(_INIT_FILE),
    submodule_search_locations=[str(_REAL_PKG_DIR)],
)

if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - failure case
    raise ImportError("Unable to load parent assignment_1 package")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[__name__] = _MODULE
_SPEC.loader.exec_module(_MODULE)
