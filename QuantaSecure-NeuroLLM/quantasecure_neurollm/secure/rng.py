"""Cryptographic randomness sources.

Production code uses :class:`SystemRandom` (``os.urandom``, i.e. the OS CSPRNG).
:class:`DeterministicDRBG` exists only for reproducible tests and known-answer
generation; it is a SHAKE-256 based generator ``out_i = SHAKE256(seed || ctr_i)``
with a monotonically increasing counter, so two instances with the same seed
produce identical streams.  **Never** use it for real keys.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

__all__ = ["RandomSource", "SystemRandom", "DeterministicDRBG", "default_rng"]


class RandomSource(Protocol):
    def random_bytes(self, n: int) -> bytes: ...


class SystemRandom:
    """OS CSPRNG."""

    def random_bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def __repr__(self) -> str:
        return "SystemRandom()"


class DeterministicDRBG:
    """SHAKE-256 counter-mode DRBG.  TEST ONLY."""

    def __init__(self, seed: bytes | int) -> None:
        if isinstance(seed, int):
            seed = seed.to_bytes(32, "big", signed=False)
        self._seed = hashlib.sha3_256(b"QSNL-test-drbg" + seed).digest()
        self._ctr = 0

    def random_bytes(self, n: int) -> bytes:
        out = hashlib.shake_256(self._seed + self._ctr.to_bytes(8, "big")).digest(n)
        self._ctr += 1
        return out

    def __repr__(self) -> str:
        return "DeterministicDRBG(<test only>)"


_SYSTEM = SystemRandom()


def default_rng() -> RandomSource:
    return _SYSTEM
