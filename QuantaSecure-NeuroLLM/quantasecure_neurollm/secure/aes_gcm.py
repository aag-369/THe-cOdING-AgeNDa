"""AES-256 (FIPS 197) and AES-GCM (NIST SP 800-38D), pure Python.

Implementation notes
--------------------
* The S-box is *computed* at import time (multiplicative inverse in GF(2^8)
  followed by the affine map) instead of being typed in, which removes a
  classic source of transcription bugs; the tests check FIPS-197 vectors.
* Encryption uses the standard four 32-bit T-tables (SubBytes + ShiftRows +
  MixColumns fused).  **Side channel:** T-table lookups indexed by secret
  bytes leak through the data cache; see :mod:`.constant_time`.
* GHASH multiplies in GF(2^128) with the reflected bit order of SP 800-38D.
  The loop is written branch-free (masks instead of ``if``) but Python
  integers are still variable-time.
* Only 96-bit IVs are supported (the recommended size; avoids the GHASH-based
  IV derivation).  Every key in this system is used for exactly one message
  (the ratchet derives a fresh key per message), so the 2^32-invocation limit
  per key is never approached.
"""

from __future__ import annotations

import struct

from .constant_time import ct_equal

__all__ = ["AES256", "AESGCM", "AEADAuthenticationError"]


class AEADAuthenticationError(Exception):
    """Raised when a ciphertext fails authentication (tag mismatch)."""


def _xtime(a: int) -> int:
    a <<= 1
    return (a ^ 0x11B) & 0xFF if a & 0x100 else a


def _gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xtime(a)
        b >>= 1
    return r


def _build_sbox() -> list[int]:
    inv = [0] * 256
    for x in range(1, 256):
        # x^254 = x^{-1} in GF(2^8)
        r, base, e = 1, x, 254
        while e:
            if e & 1:
                r = _gmul(r, base)
            base = _gmul(base, base)
            e >>= 1
        inv[x] = r
    sbox = []
    for x in range(256):
        b = inv[x]
        s = b
        for shift in range(1, 5):
            s ^= ((b << shift) | (b >> (8 - shift))) & 0xFF
        sbox.append(s ^ 0x63)
    return sbox


_SBOX = _build_sbox()


def _build_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    t0 = []
    for x in range(256):
        s = _SBOX[x]
        t0.append((_gmul(s, 2) << 24) | (s << 16) | (s << 8) | _gmul(s, 3))
    ror = lambda v, r: ((v >> r) | (v << (32 - r))) & 0xFFFFFFFF  # noqa: E731
    return t0, [ror(v, 8) for v in t0], [ror(v, 16) for v in t0], [ror(v, 24) for v in t0]


_T0, _T1, _T2, _T3 = _build_tables()
_RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]


class AES256:
    """AES with a 256-bit key (Nk = 8, Nr = 14); encryption direction only (CTR/GCM need no decryption)."""

    ROUNDS = 14

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("AES-256 needs a 32-byte key")
        w = list(struct.unpack(">8I", key))
        for i in range(8, 4 * (self.ROUNDS + 1)):
            t = w[i - 1]
            if i % 8 == 0:
                t = ((t << 8) | (t >> 24)) & 0xFFFFFFFF  # RotWord
                t = (_SBOX[t >> 24] << 24) | (_SBOX[(t >> 16) & 255] << 16) | (_SBOX[(t >> 8) & 255] << 8) | _SBOX[t & 255]
                t ^= _RCON[i // 8 - 1] << 24
            elif i % 8 == 4:
                t = (_SBOX[t >> 24] << 24) | (_SBOX[(t >> 16) & 255] << 16) | (_SBOX[(t >> 8) & 255] << 8) | _SBOX[t & 255]
            w.append(w[i - 8] ^ t)
        self._rk = w

    def encrypt_block(self, block: bytes) -> bytes:
        rk = self._rk
        s0, s1, s2, s3 = struct.unpack(">4I", block)
        s0 ^= rk[0]
        s1 ^= rk[1]
        s2 ^= rk[2]
        s3 ^= rk[3]
        k = 4
        for _ in range(self.ROUNDS - 1):
            t0 = _T0[s0 >> 24] ^ _T1[(s1 >> 16) & 255] ^ _T2[(s2 >> 8) & 255] ^ _T3[s3 & 255] ^ rk[k]
            t1 = _T0[s1 >> 24] ^ _T1[(s2 >> 16) & 255] ^ _T2[(s3 >> 8) & 255] ^ _T3[s0 & 255] ^ rk[k + 1]
            t2 = _T0[s2 >> 24] ^ _T1[(s3 >> 16) & 255] ^ _T2[(s0 >> 8) & 255] ^ _T3[s1 & 255] ^ rk[k + 2]
            t3 = _T0[s3 >> 24] ^ _T1[(s0 >> 16) & 255] ^ _T2[(s1 >> 8) & 255] ^ _T3[s2 & 255] ^ rk[k + 3]
            s0, s1, s2, s3 = t0, t1, t2, t3
            k += 4
        sb = _SBOX
        o0 = (sb[s0 >> 24] << 24) | (sb[(s1 >> 16) & 255] << 16) | (sb[(s2 >> 8) & 255] << 8) | sb[s3 & 255]
        o1 = (sb[s1 >> 24] << 24) | (sb[(s2 >> 16) & 255] << 16) | (sb[(s3 >> 8) & 255] << 8) | sb[s0 & 255]
        o2 = (sb[s2 >> 24] << 24) | (sb[(s3 >> 16) & 255] << 16) | (sb[(s0 >> 8) & 255] << 8) | sb[s1 & 255]
        o3 = (sb[s3 >> 24] << 24) | (sb[(s0 >> 16) & 255] << 16) | (sb[(s1 >> 8) & 255] << 8) | sb[s2 & 255]
        return struct.pack(">4I", o0 ^ rk[k], o1 ^ rk[k + 1], o2 ^ rk[k + 2], o3 ^ rk[k + 3])


_R = 0xE1 << 120
_MASK128 = (1 << 128) - 1


def _gf128_mul(x: int, y: int) -> int:
    """Multiply in GF(2^128) per SP 800-38D (bit 0 = MSB), branch-free inner loop."""
    z, v = 0, y
    for i in range(127, -1, -1):
        z ^= v & -((x >> i) & 1)
        v = (v >> 1) ^ (_R & -(v & 1))
    return z


def _ghash(h: int, aad: bytes, ct: bytes) -> int:
    y = 0
    for data in (aad, ct):
        for i in range(0, len(data), 16):
            block = data[i : i + 16]
            if len(block) < 16:
                block = block + bytes(16 - len(block))
            y = _gf128_mul(y ^ int.from_bytes(block, "big"), h)
    lengths = struct.pack(">QQ", len(aad) * 8, len(ct) * 8)
    return _gf128_mul(y ^ int.from_bytes(lengths, "big"), h)


class AESGCM:
    """AES-256-GCM with 96-bit nonces and 128-bit tags."""

    NONCE_SIZE = 12
    TAG_SIZE = 16
    KEY_SIZE = 32
    name = "AES-256-GCM"

    def __init__(self, key: bytes) -> None:
        self._aes = AES256(key)
        self._h = int.from_bytes(self._aes.encrypt_block(bytes(16)), "big")

    def _ctr(self, j0: bytes, data: bytes) -> bytes:
        prefix, ctr = j0[:12], struct.unpack(">I", j0[12:])[0]
        out = bytearray(len(data))
        for i in range(0, len(data), 16):
            ctr = (ctr + 1) & 0xFFFFFFFF  # inc32
            ks = self._aes.encrypt_block(prefix + struct.pack(">I", ctr))
            chunk = data[i : i + 16]
            out[i : i + len(chunk)] = bytes(a ^ b for a, b in zip(chunk, ks))
        return bytes(out)

    def _tag(self, j0: bytes, aad: bytes, ct: bytes) -> bytes:
        s = _ghash(self._h, aad, ct)
        ek = int.from_bytes(self._aes.encrypt_block(j0), "big")
        return ((s ^ ek) & _MASK128).to_bytes(16, "big")

    def encrypt(self, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
        """Return ``ciphertext || tag``."""
        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("GCM nonce must be 12 bytes")
        j0 = nonce + b"\x00\x00\x00\x01"
        ct = self._ctr(j0, plaintext)
        return ct + self._tag(j0, aad, ct)

    def decrypt(self, nonce: bytes, data: bytes, aad: bytes = b"") -> bytes:
        if len(nonce) != self.NONCE_SIZE:
            raise ValueError("GCM nonce must be 12 bytes")
        if len(data) < self.TAG_SIZE:
            raise AEADAuthenticationError("ciphertext too short")
        ct, tag = data[: -self.TAG_SIZE], data[-self.TAG_SIZE :]
        j0 = nonce + b"\x00\x00\x00\x01"
        if not ct_equal(self._tag(j0, aad, ct), tag):  # verify before decrypting (no unauthenticated release)
            raise AEADAuthenticationError("authentication tag mismatch")
        return self._ctr(j0, ct)
