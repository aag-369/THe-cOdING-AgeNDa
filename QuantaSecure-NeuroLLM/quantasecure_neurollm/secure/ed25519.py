"""The edwards25519 group and Ed25519 signatures (RFC 8032), pure Python.

Besides signatures, the prime-order subgroup of edwards25519 is the group in
which the Pedersen commitments and Schnorr-style zero-knowledge proofs of
:mod:`.commitments` / :mod:`.zkp` live, so this module also exposes:

* group arithmetic in extended twisted-Edwards coordinates ``(X : Y : Z : T)``
  with the *complete* addition law for ``a = -1`` (no exceptional cases);
* :class:`FixedBaseTable` -- 4-bit windowed precomputation that turns a
  fixed-base scalar multiplication into 64 additions;
* :func:`hash_to_point` -- a nothing-up-my-sleeve generator (try-and-increment
  on SHA-512 output, then cofactor clearing), whose discrete logarithm with
  respect to the base point is unknown.  This is what makes Pedersen
  commitments binding.

Group order: ``8 * L`` with ``L = 2^252 + 27742317777372353535851937790883648493``.
Verification equations are *cofactored* (multiplied by 8), which makes them
insensitive to small-order components an adversary might add to points.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from .rng import RandomSource, default_rng

__all__ = [
    "P",
    "L",
    "Point",
    "IDENTITY",
    "BASE",
    "point_add",
    "point_neg",
    "point_equal",
    "scalar_mult",
    "encode_point",
    "decode_point",
    "FixedBaseTable",
    "BASE_TABLE",
    "hash_to_point",
    "random_scalar",
    "hash_to_scalar",
    "Ed25519PrivateKey",
    "ed25519_verify",
    "Ed25519Error",
]

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, P - 2, P) % P
_D2 = 2 * _D % P
_SQRT_M1 = pow(2, (P - 1) // 4, P)

Point = tuple[int, int, int, int]  # extended coordinates (X, Y, Z, T), x = X/Z, y = Y/Z, xy = T/Z


class Ed25519Error(ValueError):
    pass


def point_add(p1: Point, p2: Point) -> Point:
    """Complete unified addition (RFC 8032 sec. 5.1.4)."""
    a = (p1[1] - p1[0]) * (p2[1] - p2[0]) % P
    b = (p1[1] + p1[0]) * (p2[1] + p2[0]) % P
    c = p1[3] * _D2 * p2[3] % P
    d = 2 * p1[2] * p2[2] % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def point_neg(p: Point) -> Point:
    return ((-p[0]) % P, p[1], p[2], (-p[3]) % P)


def point_equal(p1: Point, p2: Point) -> bool:
    return (p1[0] * p2[2] - p2[0] * p1[2]) % P == 0 and (p1[1] * p2[2] - p2[1] * p1[2]) % P == 0


IDENTITY: Point = (0, 1, 1, 0)


def scalar_mult(s: int, p: Point) -> Point:
    """``s * p`` by double-and-always-add (fixed 256 iterations)."""
    q = IDENTITY
    for i in range(255, -1, -1):
        q = point_add(q, q)
        added = point_add(q, p)
        q = (q, added)[(s >> i) & 1]
    return q


def _recover_x(y: int, sign: int) -> Optional[int]:
    if y >= P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, P - 2, P) % P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * _SQRT_M1 % P
    if (x * x - x2) % P != 0:
        return None
    if (x & 1) != sign:
        x = P - x
    return x


def encode_point(p: Point) -> bytes:
    zinv = pow(p[2], P - 2, P)
    x, y = p[0] * zinv % P, p[1] * zinv % P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def decode_point(s: bytes) -> Point:
    if len(s) != 32:
        raise Ed25519Error("point encoding must be 32 bytes")
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        raise Ed25519Error("invalid point encoding")
    return (x, y, 1, x * y % P)


_BY = 4 * pow(5, P - 2, P) % P
_BX = _recover_x(_BY, 0)
assert _BX is not None
BASE: Point = (_BX, _BY, 1, _BX * _BY % P)


class FixedBaseTable:
    """``table[i][j] = j * 16^i * P``; a scalar multiplication costs 64 additions."""

    def __init__(self, p: Point) -> None:
        self.point = p
        self.table: list[list[Point]] = []
        base = p
        for _ in range(64):
            row = [IDENTITY]
            for _ in range(15):
                row.append(point_add(row[-1], base))
            self.table.append(row)
            base = point_add(row[15], base)  # 16 * base
        self.encoded = encode_point(p)

    def mult(self, s: int) -> Point:
        s %= L
        q = IDENTITY
        for i in range(64):
            q = point_add(q, self.table[i][(s >> (4 * i)) & 15])
        return q


BASE_TABLE = FixedBaseTable(BASE)


def hash_to_point(label: bytes) -> Point:
    """Prime-order-subgroup point with unknown discrete log (try-and-increment + cofactor clearing)."""
    ctr = 0
    while True:
        h = hashlib.sha512(b"QSNL-hash-to-edwards25519/" + label + ctr.to_bytes(4, "big")).digest()
        y = int.from_bytes(h[:32], "little") & ((1 << 255) - 1)
        x = _recover_x(y, h[32] & 1)
        if x is not None:
            pt = scalar_mult(8, (x, y, 1, x * y % P))
            if not point_equal(pt, IDENTITY):
                return pt
        ctr += 1


def random_scalar(rng: RandomSource | None = None) -> int:
    """Uniform scalar mod L (512-bit sample reduced: statistical distance < 2^-250)."""
    return int.from_bytes((rng or default_rng()).random_bytes(64), "little") % L


def hash_to_scalar(*parts: bytes) -> int:
    h = hashlib.sha512()
    for part in parts:
        h.update(len(part).to_bytes(8, "big") + part)
    return int.from_bytes(h.digest(), "little") % L


# ---------------------------------------------------------------------- Ed25519


def _sha512_scalar(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % L


@dataclass(frozen=True)
class Ed25519PrivateKey:
    """RFC 8032 Ed25519 key (pure EdDSA, SHA-512)."""

    seed: bytes

    @classmethod
    def generate(cls, rng: RandomSource | None = None) -> "Ed25519PrivateKey":
        return cls((rng or default_rng()).random_bytes(32))

    def _expand(self) -> tuple[int, bytes]:
        h = hashlib.sha512(self.seed).digest()
        a = int.from_bytes(h[:32], "little")
        a &= (1 << 254) - 8
        a |= 1 << 254
        return a, h[32:]

    @property
    def public_key(self) -> bytes:
        a, _ = self._expand()
        return encode_point(BASE_TABLE.mult(a))  # B has order L, so a*B = (a mod L)*B

    def sign(self, msg: bytes) -> bytes:
        a, prefix = self._expand()
        pk = self.public_key
        r = _sha512_scalar(prefix + msg)
        r_enc = encode_point(BASE_TABLE.mult(r))
        k = _sha512_scalar(r_enc + pk + msg)
        s = (r + k * a) % L
        return r_enc + s.to_bytes(32, "little")

    def __repr__(self) -> str:
        return f"Ed25519PrivateKey(public={self.public_key.hex()[:16]}..., seed=<redacted>)"


def ed25519_verify(public: bytes, msg: bytes, sig: bytes) -> bool:
    """Cofactored verification ``[8][S]B == [8]R + [8][k]A``; rejects non-canonical S."""
    if len(public) != 32 or len(sig) != 64:
        return False
    try:
        a_pt = decode_point(public)
        r_pt = decode_point(sig[:32])
    except Ed25519Error:
        return False
    s = int.from_bytes(sig[32:], "little")
    if s >= L:
        return False
    k = _sha512_scalar(sig[:32] + public + msg)
    lhs = BASE_TABLE.mult(s)
    rhs = point_add(r_pt, scalar_mult(k, a_pt))
    return point_equal(scalar_mult(8, lhs), scalar_mult(8, rhs))
