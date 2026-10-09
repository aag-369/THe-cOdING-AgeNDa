"""Exception taxonomy for :mod:`research_kit.labkit`."""

from __future__ import annotations

from research_kit.errors import ResearchError


class LabkitError(ResearchError):
    """Base for every labkit failure."""


class RunExistsError(LabkitError):
    """The target run directory already exists and ``on_exist='error'``."""


class RunStateError(LabkitError):
    """A :class:`~research_kit.labkit.run.Run` was used outside its lifecycle.

    Raised for: using run methods before ``__enter__``, entering twice, or
    using the run after ``__exit__``.
    """


class RunNotFoundError(LabkitError):
    """A run cannot be located from the given hash or path."""


class AmbiguousRunError(RunNotFoundError):
    """A short hash prefix matches more than one run directory."""


class GitError(LabkitError):
    """A git precondition is unmet: ``require_git`` with no repo / no ``git``
    binary, or ``require_clean`` with a dirty tree."""


class InputError(LabkitError):
    """A declared input file or directory is missing or unreadable."""


class MetricsError(LabkitError):
    """A metric value is non-scalar / not JSON-serializable, or a metric key is
    reserved (``wall_time``, ``_step``, or any name starting with ``_``)."""


class ManifestError(LabkitError):
    """``manifest.json`` is missing, invalid, or has an unknown schema version."""


class ReplayError(LabkitError):
    """A run cannot be replayed (corrupt stored config, unusable recorded
    invocation, missing interpreter)."""


class ReplayMismatchError(ReplayError):
    """``strict`` replay found a git SHA or input checksum mismatch."""
