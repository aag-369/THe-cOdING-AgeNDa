"""Commitment schemes: salted hash commitments and Pedersen commitments.

Hash commitments (used on every intermediate tensor)
----------------------------------------------------
``C = SHA3-256("QSNL-commit-v1" || label || salt || canonical(obj))`` with a
fresh 256-bit salt.  Binding follows from collision resistance of SHA3-256;
hiding holds in the random-oracle model because the 256-bit salt makes ``C``
independent of ``obj`` to anyone without the salt.  The label domain-separates
commitments to different kinds of objects.

Pedersen commitments (used on quantised Born probabilities)
-----------------------------------------------------------
``C(m, r) = m G + r H`` in the prime-order subgroup of edwards25519, with ``G``
the standard base point and ``H`` a hash-to-curve point whose discrete log
w.r.t. ``G`` nobody knows.  Pedersen commitments are *perfectly hiding*
(for uniform ``r``, ``C`` is a uniform group element) and *computationally
binding* (opening one ``C`` to two values yields ``log_G H``).  They are
additively homomorphic -- ``C(m1, r1) + C(m2, r2) = C(m1 + m2, r1 + r2)`` --
which is what lets :mod:`.zkp` prove linear facts about committed
probabilities without revealing them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from ..utils.serialization import encode
from .constant_time import ct_equal
from .ed25519 import BASE_TABLE, IDENTITY, L, FixedBaseTable, Point, hash_to_point, point_add, random_scalar
from .rng import RandomSource, default_rng

__all__ = [
    "HashCommitment",
    "commit_object",
    "verify_object",
    "pedersen_tables",
    "pedersen_commit",
    "pedersen_commit_vector",
]

_DOMAIN = b"QSNL-commit-v1"


@dataclass(frozen=True)
class HashCommitment:
    """A salted hash commitment and the label it was made under."""

    label: str
    digest: bytes

    def hex(self) -> str:
        return self.digest.hex()


def _digest(label: str, salt: bytes, payload: bytes) -> bytes:
    lb = label.encode()
    return hashlib.sha3_256(_DOMAIN + len(lb).to_bytes(2, "big") + lb + salt + payload).digest()


def commit_object(label: str, obj: Any, rng: RandomSource | None = None) -> tuple[HashCommitment, bytes]:
    """Commit to any canonically-encodable object.  Returns ``(commitment, salt)``."""
    salt = (rng or default_rng()).random_bytes(32)
    return HashCommitment(label, _digest(label, salt, encode(obj))), salt


def verify_object(commitment: HashCommitment, obj: Any, salt: bytes) -> bool:
    return ct_equal(_digest(commitment.label, salt, encode(obj)), commitment.digest)


@lru_cache(maxsize=1)
def pedersen_tables() -> tuple[FixedBaseTable, FixedBaseTable]:
    """``(G, H)`` fixed-base tables; ``H`` is derived by hashing, so ``log_G H`` is unknown."""
    return BASE_TABLE, FixedBaseTable(hash_to_point(b"QSNL-pedersen-H-generator-v1"))


def _small_mult(table: FixedBaseTable, m: int) -> Point:
    """Fixed-base multiplication that skips the zero high windows of a small scalar."""
    q = IDENTITY
    m %= L
    for i in range((m.bit_length() + 3) // 4):
        q = point_add(q, table.table[i][(m >> (4 * i)) & 15])
    return q


def pedersen_commit(m: int, r: int) -> Point:
    g, h = pedersen_tables()
    return point_add(_small_mult(g, m), h.mult(r))


def pedersen_commit_vector(values: Sequence[int], rng: RandomSource | None = None) -> tuple[list[Point], list[int]]:
    """Commit to each value with an independent uniform blinding factor."""
    rng = rng or default_rng()
    blinds = [random_scalar(rng) for _ in values]
    return [pedersen_commit(int(v), r) for v, r in zip(values, blinds)], blinds
