"""Seed derivation for *non-cryptographic* randomness.

Scientific randomness (shot sampling, weight initialisation, data augmentation)
uses NumPy's PCG64 generator for speed and reproducibility.  It is **never**
used for key material: cryptographic randomness comes exclusively from
:mod:`quantasecure_neurollm.secure.rng` (``os.urandom`` by default).

Seeds are derived by hashing a label together with integers/bytes so that
independent consumers never share a stream by accident.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["derive_seed", "numpy_rng"]


def derive_seed(label: str, *parts: int | str | bytes) -> int:
    """64-bit seed = SHA3-256(label || parts...)[:8], domain-separated by ``label``."""
    h = hashlib.sha3_256(b"QSNL-seed/" + label.encode())
    for p in parts:
        if isinstance(p, bytes):
            raw = p
        elif isinstance(p, int):
            raw = p.to_bytes(16, "big", signed=True)
        else:
            raw = str(p).encode()
        h.update(len(raw).to_bytes(4, "big") + raw)
    return int.from_bytes(h.digest()[:8], "big")


def numpy_rng(label: str, *parts: int | str | bytes) -> np.random.Generator:
    """A fresh PCG64 generator seeded by :func:`derive_seed`."""
    return np.random.default_rng(derive_seed(label, *parts))
