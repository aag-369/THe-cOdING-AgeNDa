"""X25519 Diffie-Hellman (RFC 7748), pure Python Montgomery ladder.

The ladder performs exactly 255 iterations with an arithmetic conditional swap
(no secret-dependent branches or memory indices); the remaining timing leak is
the variable-time CPython big-integer arithmetic (see :mod:`.constant_time`).
The all-zero shared secret (produced by small-order public keys) is rejected,
as recommended by RFC 7748 sec. 6.1, which gives contributory behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rng import RandomSource, default_rng

__all__ = ["x25519", "x25519_base", "X25519KeyPair", "X25519Error"]

_P = 2**255 - 19
_A24 = 121665
_BASE_U = (9).to_bytes(32, "little")


class X25519Error(ValueError):
    pass


def _decode_scalar(k: bytes) -> int:
    b = bytearray(k)
    b[0] &= 248
    b[31] &= 127
    b[31] |= 64
    return int.from_bytes(b, "little")


def _cswap(swap: int, a: int, b: int) -> tuple[int, int]:
    mask = -swap  # 0 or ...1111
    dummy = mask & (a ^ b)
    return a ^ dummy, b ^ dummy


def x25519(k: bytes, u: bytes) -> bytes:
    """Scalar multiplication on the Montgomery curve (RFC 7748 sec. 5)."""
    if len(k) != 32 or len(u) != 32:
        raise X25519Error("X25519 inputs must be 32 bytes")
    scalar = _decode_scalar(k)
    x1 = int.from_bytes(u, "little") & ((1 << 255) - 1)
    x2, z2, x3, z3 = 1, 0, x1, 1
    swap = 0
    for t in range(254, -1, -1):
        kt = (scalar >> t) & 1
        swap ^= kt
        x2, x3 = _cswap(swap, x2, x3)
        z2, z3 = _cswap(swap, z2, z3)
        swap = kt
        a = (x2 + z2) % _P
        aa = a * a % _P
        b = (x2 - z2) % _P
        bb = b * b % _P
        e = (aa - bb) % _P
        c = (x3 + z3) % _P
        d = (x3 - z3) % _P
        da = d * a % _P
        cb = c * b % _P
        x3 = (da + cb) ** 2 % _P
        z3 = x1 * (da - cb) ** 2 % _P
        x2 = aa * bb % _P
        z2 = e * (aa + _A24 * e) % _P
    x2, x3 = _cswap(swap, x2, x3)
    z2, z3 = _cswap(swap, z2, z3)
    return (x2 * pow(z2, _P - 2, _P) % _P).to_bytes(32, "little")


def x25519_base(k: bytes) -> bytes:
    return x25519(k, _BASE_U)


@dataclass(frozen=True)
class X25519KeyPair:
    private: bytes
    public: bytes

    @classmethod
    def generate(cls, rng: RandomSource | None = None) -> "X25519KeyPair":
        sk = (rng or default_rng()).random_bytes(32)
        return cls(sk, x25519_base(sk))

    def exchange(self, peer_public: bytes) -> bytes:
        shared = x25519(self.private, peer_public)
        if shared == bytes(32):
            raise X25519Error("low-order peer public key (all-zero shared secret)")
        return shared

    def __repr__(self) -> str:
        return f"X25519KeyPair(public={self.public.hex()[:16]}..., private=<redacted>)"
