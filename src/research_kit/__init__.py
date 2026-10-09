"""research-kit: foundation for a personal research monorepo.

This top-level package performs **zero heavyweight imports**. Importing
``research_kit`` (or any submodule) must never pull in numpy, torch, qiskit,
tenseal, or similar. Backends import those lazily where needed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
