"""ChaCha20-Poly1305 AEAD (RFC 8439), pure Python.

ChaCha20 is an add-rotate-xor (ARX) cipher with no secret-indexed table
lookups, so -- unlike table-based AES -- it has no cache-timing channel by
construction; it is the recommended AEAD for software-only deployments.
Poly1305 arithmetic is done with Python integers modulo ``2^130 - 5``
(variable-time big integers; see :mod:`.constant_time`).
"""

from __future__ import annotations

import struct

from .aes_gcm import AEADAuthenticationError
from .constant_time import ct_equal

__all__ = ["chacha20_block", "chacha20_xor", "poly1305_mac", "ChaCha20Poly1305"]

_M32 = 0xFFFFFFFF
_SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)  # "expand 32-byte k"


def _quarter(s: list[int], a: int, b: int, c: int, d: int) -> None:
    s[a] = (s[a] + s[b]) & _M32
    s[d] ^= s[a]
    s[d] = ((s[d] << 16) | (s[d] >> 16)) & _M32
    s[c] = (s[c] + s[d]) & _M32
    s[b] ^= s[c]
    s[b] = ((s[b] << 12) | (s[b] >> 20)) & _M32
    s[a] = (s[a] + s[b]) & _M32
    s[d] ^= s[a]
    s[d] = ((s[d] << 8) | (s[d] >> 24)) & _M32
    s[c] = (s[c] + s[d]) & _M32
    s[b] ^= s[c]
    s[b] = ((s[b] << 7) | (s[b] >> 25)) & _M32


def chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """One 64-byte keystream block."""
    state = list(_SIGMA) + list(struct.unpack("<8I", key)) + [counter & _M32] + list(struct.unpack("<3I", nonce))
    w = list(state)
    for _ in range(10):
        _quarter(w, 0, 4, 8, 12)
        _quarter(w, 1, 5, 9, 13)
        _quarter(w, 2, 6, 10, 14)
        _quarter(w, 3, 7, 11, 15)
        _quarter(w, 0, 5, 10, 15)
        _quarter(w, 1, 6, 11, 12)
        _quarter(w, 2, 7, 8, 13)
        _quarter(w, 3, 4, 9, 14)
    return struct.pack("<16I", *((a + b) & _M32 for a, b in zip(w, state)))


def chacha20_xor(key: bytes, counter: int, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    for i in range(0, len(data), 64):
        ks = chacha20_block(key, counter + i // 64, nonce)
        chunk = data[i : i + 64]
        out[i : i + len(chunk)] = bytes(a ^ b for a, b in zip(chunk, ks))
    return bytes(out)


_P1305 = (1 << 130) - 5


def poly1305_mac(key: bytes, msg: bytes) -> bytes:
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:32], "little")
    acc = 0
    for i in range(0, len(msg), 16):
        n = int.from_bytes(msg[i : i + 16] + b"\x01", "little")
        acc = ((acc + n) * r) % _P1305
    return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def _pad16(x: bytes) -> bytes:
    return bytes((16 - len(x) % 16) % 16)


class ChaCha20Poly1305:
    """RFC 8439 AEAD with 96-bit nonces."""

    NONCE_SIZE = 12
    TAG_SIZE = 16
    KEY_SIZE = 32
    name = "ChaCha20-Poly1305"

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("ChaCha20 needs a 32-byte key")
        self._key = key

    def _mac(self, nonce: bytes, aad: bytes, ct: bytes) -> bytes:
        otk = chacha20_block(self._key, 0, nonce)[:32]
        data = aad + _pad16(aad) + ct + _pad16(ct) + struct.pack("<QQ", len(aad), len(ct))
        return poly1305_mac(otk, data)

    def encrypt(self, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("nonce must be 12 bytes")
        ct = chacha20_xor(self._key, 1, nonce, plaintext)
        return ct + self._mac(nonce, aad, ct)

    def decrypt(self, nonce: bytes, data: bytes, aad: bytes = b"") -> bytes:
        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("nonce must be 12 bytes")
        if len(data) < self.TAG_SIZE:
            raise AEADAuthenticationError("ciphertext too short")
        ct, tag = data[: -self.TAG_SIZE], data[-self.TAG_SIZE :]
        if not ct_equal(self._mac(nonce, aad, ct), tag):
            raise AEADAuthenticationError("authentication tag mismatch")
        return chacha20_xor(self._key, 1, nonce, ct)
