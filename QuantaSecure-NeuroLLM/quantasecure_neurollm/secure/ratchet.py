"""Hybrid post-quantum Double Ratchet.

This follows the Signal Double Ratchet specification (Perrin & Marlinspike
2016) -- a Diffie-Hellman ratchet driving per-direction symmetric-key chains --
with one addition: every DH-ratchet step also mixes in a fresh **ML-KEM**
shared secret encapsulated to the peer's ratchet KEM key.  The root-key update
is therefore

    RK', CK = KDF_RK(RK, DH(own_ratchet, peer_ratchet) || ML-KEM shared secret)

so a future quantum adversary who can compute discrete logs still cannot
recover chain keys created after a step it did not observe (post-compromise
security against quantum adversaries; cf. the "SPQR"/PQ3 designs).

Properties provided
-------------------
* **Fresh key per message.**  ``KDF_CK`` derives a one-time message key and the
  next chain key with HMAC; the message key is used for exactly one AEAD
  encryption and then deleted, giving forward secrecy at message granularity.
* **Break-in recovery** via the DH (+ KEM) ratchet on every change of direction.
* **Out-of-order / dropped messages** are handled with a bounded store of
  skipped message keys (``max_skip``); a replayed frame fails because its key
  has been consumed.
* **Transactional decryption**: state is rolled back if authentication fails,
  so forged frames cannot desynchronise the ratchet.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from ..utils.serialization import encode
from .aead import AEADAuthenticationError, make_aead
from .kdf import hkdf, hmac_sha256, key_fingerprint
from .mlkem import MLKEM
from .rng import RandomSource, default_rng
from .x25519 import X25519KeyPair

__all__ = ["RatchetHeader", "DoubleRatchet", "RatchetError"]


class RatchetError(Exception):
    pass


@dataclass(frozen=True)
class RatchetHeader:
    dh: bytes  # sender's current ratchet public key
    pn: int  # length of the sender's previous sending chain
    n: int  # message number in the current sending chain
    kem_ct: bytes  # ML-KEM ciphertext for this sending chain (b"" if KEM ratchet disabled)

    def to_dict(self) -> dict[str, Any]:
        return {"dh": self.dh, "pn": self.pn, "n": self.n, "kem_ct": self.kem_ct}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RatchetHeader":
        return cls(bytes(d["dh"]), int(d["pn"]), int(d["n"]), bytes(d["kem_ct"]))

    def encode(self) -> bytes:
        return encode(self.to_dict())


def _kdf_rk(rk: bytes, dh_out: bytes, kem_ss: bytes) -> tuple[bytes, bytes]:
    out = hkdf(dh_out + kem_ss, rk, b"QSNL-ratchet-root-v1", 64)
    return out[:32], out[32:]


def _kdf_ck(ck: bytes) -> tuple[bytes, bytes]:
    """(next chain key, message key)."""
    return hmac_sha256(ck, b"\x02"), hmac_sha256(ck, b"\x01")


def _message_key_material(mk: bytes) -> tuple[bytes, bytes]:
    out = hkdf(mk, None, b"QSNL-message-key-v1", 44)
    return out[:32], out[32:]


@dataclass
class _State:
    dhs: X25519KeyPair
    dhr: Optional[bytes]
    rk: bytes
    cks: Optional[bytes]
    ckr: Optional[bytes]
    ns: int = 0
    nr: int = 0
    pn: int = 0
    skipped: dict[tuple[bytes, int], bytes] = field(default_factory=dict)
    pending_kem_ct: bytes = b""


class DoubleRatchet:
    """One party's ratchet state.  Create with :meth:`initiator` or :meth:`responder`."""

    def __init__(
        self,
        state: _State,
        aead_name: str,
        kem: Optional[MLKEM],
        own_kem_dk: Optional[bytes],
        peer_kem_ek: Optional[bytes],
        max_skip: int,
        rng: RandomSource,
    ) -> None:
        self._s = state
        self.aead_name = aead_name
        self._kem = kem
        self._own_kem_dk = own_kem_dk
        self._peer_kem_ek = peer_kem_ek
        self.max_skip = max_skip
        self._rng = rng
        self.events: list[dict[str, Any]] = []  # audit trail: fingerprints only, never keys
        self.used_key_fingerprints: list[str] = []

    # ------------------------------------------------------------- construction
    @classmethod
    def initiator(
        cls,
        shared_key: bytes,
        peer_dh_public: bytes,
        aead_name: str = "AES-256-GCM",
        kem: Optional[MLKEM] = None,
        own_kem_dk: Optional[bytes] = None,
        peer_kem_ek: Optional[bytes] = None,
        max_skip: int = 64,
        rng: RandomSource | None = None,
    ) -> "DoubleRatchet":
        """RatchetInitAlice: performs the first DH(+KEM) step immediately."""
        rng = rng or default_rng()
        dhs = X25519KeyPair.generate(rng)
        kem_ct, kem_ss = b"", b""
        if kem is not None:
            if peer_kem_ek is None:
                raise RatchetError("KEM ratchet needs the peer's encapsulation key")
            kem_ss, kem_ct = kem.encaps(peer_kem_ek, rng)
        rk, cks = _kdf_rk(shared_key, dhs.exchange(peer_dh_public), kem_ss)
        st = _State(dhs=dhs, dhr=peer_dh_public, rk=rk, cks=cks, ckr=None, pending_kem_ct=kem_ct)
        r = cls(st, aead_name, kem, own_kem_dk, peer_kem_ek, max_skip, rng)
        r._log("dh_ratchet_init", role="initiator", own_dh=key_fingerprint(dhs.public), kem=kem is not None)
        return r

    @classmethod
    def responder(
        cls,
        shared_key: bytes,
        own_dh_pair: X25519KeyPair,
        aead_name: str = "AES-256-GCM",
        kem: Optional[MLKEM] = None,
        own_kem_dk: Optional[bytes] = None,
        peer_kem_ek: Optional[bytes] = None,
        max_skip: int = 64,
        rng: RandomSource | None = None,
    ) -> "DoubleRatchet":
        """RatchetInitBob: waits for the initiator's first message."""
        st = _State(dhs=own_dh_pair, dhr=None, rk=shared_key, cks=None, ckr=None)
        r = cls(st, aead_name, kem, own_kem_dk, peer_kem_ek, max_skip, rng or default_rng())
        r._log("dh_ratchet_init", role="responder", own_dh=key_fingerprint(own_dh_pair.public), kem=kem is not None)
        return r

    # ------------------------------------------------------------------ helpers
    def _log(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})

    @property
    def dh_ratchet_steps(self) -> int:
        return sum(1 for e in self.events if e["event"] == "dh_ratchet_step")

    def _dh_ratchet(self, header: RatchetHeader) -> None:
        s = self._s
        s.pn = s.ns
        s.ns = 0
        s.nr = 0
        s.dhr = header.dh
        ss_in = b""
        if self._kem is not None:
            if self._own_kem_dk is None:
                raise RatchetError("missing own KEM decapsulation key")
            ss_in = self._kem.decaps(self._own_kem_dk, header.kem_ct)
        s.rk, s.ckr = _kdf_rk(s.rk, s.dhs.exchange(s.dhr), ss_in)
        s.dhs = X25519KeyPair.generate(self._rng)
        ss_out = b""
        s.pending_kem_ct = b""
        if self._kem is not None:
            assert self._peer_kem_ek is not None
            ss_out, s.pending_kem_ct = self._kem.encaps(self._peer_kem_ek, self._rng)
        s.rk, s.cks = _kdf_rk(s.rk, s.dhs.exchange(s.dhr), ss_out)
        self._log(
            "dh_ratchet_step",
            peer_dh=key_fingerprint(header.dh),
            own_dh=key_fingerprint(s.dhs.public),
            root=key_fingerprint(s.rk),
            kem_mixed=self._kem is not None,
        )

    def _skip(self, until: int) -> None:
        s = self._s
        if s.ckr is None:
            return
        if until - s.nr > self.max_skip:
            raise RatchetError("too many skipped messages")
        while s.nr < until:
            s.ckr, mk = _kdf_ck(s.ckr)
            assert s.dhr is not None
            s.skipped[(s.dhr, s.nr)] = mk
            s.nr += 1
        while len(s.skipped) > 4 * self.max_skip:  # bound memory: evict oldest
            s.skipped.pop(next(iter(s.skipped)))

    def _seal(self, mk: bytes, plaintext: bytes, ad: bytes) -> bytes:
        key, nonce = _message_key_material(mk)
        self.used_key_fingerprints.append(key_fingerprint(key))
        return make_aead(self.aead_name, key).encrypt(nonce, plaintext, ad)

    def _open(self, mk: bytes, ciphertext: bytes, ad: bytes) -> bytes:
        key, nonce = _message_key_material(mk)
        return make_aead(self.aead_name, key).decrypt(nonce, ciphertext, ad)

    # --------------------------------------------------------------------- API
    def encrypt(self, plaintext: bytes, ad: bytes) -> tuple[RatchetHeader, bytes]:
        s = self._s
        if s.cks is None:
            raise RatchetError("no sending chain yet (responder must receive first)")
        s.cks, mk = _kdf_ck(s.cks)
        header = RatchetHeader(s.dhs.public, s.pn, s.ns, s.pending_kem_ct)
        s.ns += 1
        ct = self._seal(mk, plaintext, ad + header.encode())
        self._log("symmetric_step", direction="send", n=header.n, key=self.used_key_fingerprints[-1])
        return header, ct

    def decrypt(self, header: RatchetHeader, ciphertext: bytes, ad: bytes) -> bytes:
        snapshot = copy.deepcopy(self._s)
        n_events = len(self.events)
        try:
            return self._decrypt(header, ciphertext, ad)
        except Exception:
            self._s = snapshot  # roll back: forged / replayed frames must not change state
            del self.events[n_events:]
            raise

    def _decrypt(self, header: RatchetHeader, ciphertext: bytes, ad: bytes) -> bytes:
        s = self._s
        full_ad = ad + header.encode()
        mk = s.skipped.pop((header.dh, header.n), None)
        if mk is not None:
            pt = self._open(mk, ciphertext, full_ad)
            self._log("symmetric_step", direction="recv-skipped", n=header.n)
            return pt
        if header.dh != s.dhr:
            self._skip(header.pn)
            self._dh_ratchet(header)
        if header.n < s.nr:
            raise RatchetError("message key already used (replay)")
        self._skip(header.n)
        assert s.ckr is not None
        s.ckr, mk = _kdf_ck(s.ckr)
        s.nr += 1
        try:
            pt = self._open(mk, ciphertext, full_ad)
        except AEADAuthenticationError as exc:
            raise RatchetError("authentication failed") from exc
        self._log("symmetric_step", direction="recv", n=header.n)
        return pt
