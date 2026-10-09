"""Kit-wide exception base."""

from __future__ import annotations


class ResearchError(Exception):
    """Root of every exception raised deliberately by research-kit.

    Catching this type catches all typed failures from every module while
    letting genuinely unexpected exceptions (``KeyError``, ``TypeError`` from
    bugs, ``KeyboardInterrupt``) propagate untouched.
    """
