"""AEAD interface and factory."""

from __future__ import annotations

from typing import Protocol

from .aes_gcm import AEADAuthenticationError, AESGCM
from .chacha20_poly1305 import ChaCha20Poly1305

__all__ = ["AEAD", "make_aead", "AEADAuthenticationError", "SUITES"]


class AEAD(Protocol):
    NONCE_SIZE: int
    TAG_SIZE: int
    KEY_SIZE: int
    name: str

    def encrypt(self, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes: ...

    def decrypt(self, nonce: bytes, data: bytes, aad: bytes = b"") -> bytes: ...


SUITES = {"AES-256-GCM": AESGCM, "ChaCha20-Poly1305": ChaCha20Poly1305}


def make_aead(name: str, key: bytes) -> AEAD:
    try:
        return SUITES[name](key)
    except KeyError as exc:
        raise ValueError(f"unknown AEAD {name!r}; choose from {sorted(SUITES)}") from exc
