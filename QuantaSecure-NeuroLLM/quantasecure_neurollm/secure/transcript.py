"""Fiat-Shamir transcript (Merlin-style) over SHAKE-256.

Every appended message is framed as ``len(label) || label || len(data) || data``
so that no two distinct transcripts hash to the same byte stream.  A
challenge is squeezed from a *copy* of the running sponge and then fed back
into the transcript, so successive challenges are chained.  Making every
prover message part of the hash before the challenge is derived is what turns
an honest-verifier Sigma protocol into a non-interactive proof in the random
oracle model (Fiat & Shamir 1986; Pointcheval & Stern 2000).
"""

from __future__ import annotations

import hashlib

from .ed25519 import L

__all__ = ["Transcript"]


class Transcript:
    def __init__(self, protocol_label: str) -> None:
        self._h = hashlib.shake_256()
        self.append("protocol", protocol_label.encode())

    def append(self, label: str, data: bytes) -> None:
        lb = label.encode()
        self._h.update(len(lb).to_bytes(4, "big") + lb + len(data).to_bytes(8, "big") + data)

    def append_int(self, label: str, value: int) -> None:
        self.append(label, value.to_bytes(32, "big", signed=True))

    def challenge_bytes(self, label: str, n: int) -> bytes:
        self.append("challenge-label", label.encode())
        out = self._h.copy().digest(n)
        self.append("challenge-output", out)
        return out

    def challenge_scalar(self, label: str) -> int:
        """Uniform scalar mod L (64 bytes reduced, bias < 2^-250)."""
        return int.from_bytes(self.challenge_bytes(label, 64), "little") % L
