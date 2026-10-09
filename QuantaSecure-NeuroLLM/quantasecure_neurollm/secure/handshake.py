"""Mutually-authenticated hybrid post-quantum handshake ("QSNL-HS").

Three messages, in the style of TLS 1.3 / PQXDH with signature authentication::

    I -> R  ClientHello    {I, R, eph_x25519_I, ek_mlkem_I, nonce_I}          + HybridSig_I(CH)
    R -> I  ServerHello    {eph_x25519_R, ct = Encaps(ek_I), ek_ratchet_R, nonce_R}
                           + HybridSig_R(th1)  + Finished_R = HMAC(fin_R, th2)
    I -> R  ClientFinished {Finished_I = HMAC(fin_I, th3)}

Key schedule (``th*`` are SHA3-256 transcript hashes)::

    th1  = H(CH || SH_body)
    PRK  = HKDF-Extract(salt = th1, IKM = ss_mlkem || ss_x25519)
    root = ExpandLabel(PRK, "root", th1)          -> seeds the Double Ratchet
    fin_R, fin_I, session_id = ExpandLabel(PRK, ...)

Security properties (informal):

* **Hybrid key secrecy** -- the session key is secret if *either* X25519 (CDH) or
  ML-KEM (IND-CCA, Module-LWE) is secure.
* **Mutual authentication** -- every hello is signed with the sender's pinned
  hybrid (Ed25519 AND ML-DSA) identity key; a man-in-the-middle substituting
  ephemeral keys cannot produce valid signatures.
* **Key confirmation** -- the Finished MACs prove both sides derived the same
  keys and saw the same transcript (downgrade / reflection protection).
* **Forward secrecy** -- only ephemeral keys enter the key exchange.

The responder's ephemeral X25519 key doubles as its first ratchet key and the
initiator's ephemeral ML-KEM key doubles as its ratchet KEM key (as Signal's
signed pre-key does for the DH ratchet).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..utils.serialization import decode, encode
from .constant_time import ct_equal
from .hybrid import HybridPublicKey, HybridSigningKey, combine_shared_secrets, hybrid_verify
from .kdf import expand_label, hmac_sha256, key_fingerprint
from .mlkem import MLKEM
from .ratchet import DoubleRatchet
from .rng import RandomSource, default_rng
from .x25519 import X25519KeyPair

__all__ = ["Suite", "HandshakeError", "HandshakeInitiator", "HandshakeResponder", "HandshakeResult"]

VERSION = 1


class HandshakeError(Exception):
    pass


@dataclass(frozen=True)
class Suite:
    kem: str = "ML-KEM-768"
    signature: str = "ML-DSA-44"
    aead: str = "AES-256-GCM"
    kem_ratchet: bool = True
    max_skip: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {"kem": self.kem, "signature": self.signature, "aead": self.aead, "kem_ratchet": self.kem_ratchet}


@dataclass
class HandshakeResult:
    session_id: bytes
    ratchet: DoubleRatchet
    transcript_hash: bytes
    peer: str
    log: dict[str, Any]


def _h(*parts: bytes) -> bytes:
    h = hashlib.sha3_256()
    for p in parts:
        h.update(len(p).to_bytes(8, "big") + p)
    return h.digest()


TrustLookup = Callable[[str], Optional[HybridPublicKey]]


class HandshakeInitiator:
    def __init__(
        self,
        identity: HybridSigningKey,
        peer_name: str,
        trust: TrustLookup,
        suite: Suite = Suite(),
        rng: RandomSource | None = None,
    ) -> None:
        self.identity = identity
        self.peer_name = peer_name
        self.trust = trust
        self.suite = suite
        self.rng = rng or default_rng()
        self._kem = MLKEM(suite.kem)
        self._eph = X25519KeyPair.generate(self.rng)
        self._ek, self._dk = self._kem.keygen(self.rng)
        self._ch: Optional[bytes] = None

    def client_hello(self) -> bytes:
        body = {
            "type": "client_hello",
            "version": VERSION,
            "suite": self.suite.to_dict(),
            "initiator": self.identity.public.owner,
            "responder": self.peer_name,
            "eph_x25519": self._eph.public,
            "kem_ek": self._ek,
            "nonce": self.rng.random_bytes(32),
        }
        raw = encode(body)
        self._ch = encode({"body": raw, "sig": self.identity.sign("QSNL-HS-CH", raw)})
        return self._ch

    def process_server_hello(self, msg: bytes) -> tuple[bytes, HandshakeResult]:
        if self._ch is None:
            raise HandshakeError("client_hello not sent")
        outer = decode(msg)
        body = decode(outer["body"])
        if body.get("type") != "server_hello" or body.get("version") != VERSION:
            raise HandshakeError("unexpected message")
        peer_pk = self.trust(self.peer_name)
        if peer_pk is None:
            raise HandshakeError(f"no pinned key for {self.peer_name}")
        th1 = _h(self._ch, outer["body"])
        if not hybrid_verify(peer_pk, "QSNL-HS-SH", th1, outer["sig"]):
            raise HandshakeError("server signature invalid")
        ss_kem = self._kem.decaps(self._dk, body["kem_ct"])
        ss_dh = self._eph.exchange(body["eph_x25519"])
        prk = combine_shared_secrets(ss_kem, ss_dh, th1)
        th2 = _h(th1, outer["sig"])
        if not ct_equal(hmac_sha256(expand_label(prk, "finished responder", th1, 32), th2), outer["finished"]):
            raise HandshakeError("server Finished MAC invalid")
        th3 = _h(th2, outer["finished"])
        fin_i = hmac_sha256(expand_label(prk, "finished initiator", th1, 32), th3)
        root = expand_label(prk, "root", th1, 32)
        sid = expand_label(prk, "session id", th1, 16)
        ratchet = DoubleRatchet.initiator(
            root,
            body["eph_x25519"],
            self.suite.aead,
            kem=self._kem if self.suite.kem_ratchet else None,
            own_kem_dk=self._dk,
            peer_kem_ek=body["ratchet_kem_ek"],
            max_skip=self.suite.max_skip,
            rng=self.rng,
        )
        log = {
            "session_id": sid.hex(),
            "suite": self.suite.to_dict(),
            "transcript_hash": th3.hex(),
            "peer_identity": peer_pk.fingerprint(),
            "root_fp": key_fingerprint(root),
            "x25519_fp": key_fingerprint(ss_dh),
            "mlkem_fp": key_fingerprint(ss_kem),
        }
        return encode({"type": "client_finished", "finished": fin_i}), HandshakeResult(sid, ratchet, th3, self.peer_name, log)


class HandshakeResponder:
    def __init__(
        self,
        identity: HybridSigningKey,
        trust: TrustLookup,
        suite: Suite = Suite(),
        rng: RandomSource | None = None,
    ) -> None:
        self.identity = identity
        self.trust = trust
        self.suite = suite
        self.rng = rng or default_rng()
        self._kem = MLKEM(suite.kem)
        self._pending: Optional[dict[str, Any]] = None

    def process_client_hello(self, msg: bytes) -> bytes:
        outer = decode(msg)
        body = decode(outer["body"])
        if body.get("type") != "client_hello" or body.get("version") != VERSION:
            raise HandshakeError("unexpected message")
        if body["responder"] != self.identity.public.owner:
            raise HandshakeError("hello addressed to someone else")
        if body["suite"] != self.suite.to_dict():
            raise HandshakeError("cipher-suite mismatch (downgrade attempt?)")
        peer_pk = self.trust(body["initiator"])
        if peer_pk is None:
            raise HandshakeError(f"unknown initiator {body['initiator']!r}")
        if not hybrid_verify(peer_pk, "QSNL-HS-CH", outer["body"], outer["sig"]):
            raise HandshakeError("client signature invalid")
        eph = X25519KeyPair.generate(self.rng)
        ss_kem, ct = self._kem.encaps(body["kem_ek"], self.rng)
        ss_dh = eph.exchange(body["eph_x25519"])
        ratchet_ek, ratchet_dk = self._kem.keygen(self.rng)
        sh_body = encode(
            {
                "type": "server_hello",
                "version": VERSION,
                "eph_x25519": eph.public,
                "kem_ct": ct,
                "ratchet_kem_ek": ratchet_ek,
                "nonce": self.rng.random_bytes(32),
            }
        )
        th1 = _h(msg, sh_body)
        prk = combine_shared_secrets(ss_kem, ss_dh, th1)
        sig = self.identity.sign("QSNL-HS-SH", th1)
        th2 = _h(th1, sig)
        fin_r = hmac_sha256(expand_label(prk, "finished responder", th1, 32), th2)
        th3 = _h(th2, fin_r)
        self._pending = {
            "peer": body["initiator"],
            "peer_identity": peer_pk.fingerprint(),
            "prk": prk,
            "th1": th1,
            "th3": th3,
            "eph": eph,
            "ratchet_dk": ratchet_dk,
            "peer_kem_ek": body["kem_ek"],
            "ss_dh": ss_dh,
            "ss_kem": ss_kem,
        }
        return encode({"body": sh_body, "sig": sig, "finished": fin_r})

    def process_client_finished(self, msg: bytes) -> HandshakeResult:
        if self._pending is None:
            raise HandshakeError("no handshake in progress")
        p = self._pending
        body = decode(msg)
        expected = hmac_sha256(expand_label(p["prk"], "finished initiator", p["th1"], 32), p["th3"])
        if body.get("type") != "client_finished" or not ct_equal(expected, body["finished"]):
            raise HandshakeError("client Finished MAC invalid")
        root = expand_label(p["prk"], "root", p["th1"], 32)
        sid = expand_label(p["prk"], "session id", p["th1"], 16)
        ratchet = DoubleRatchet.responder(
            root,
            p["eph"],
            self.suite.aead,
            kem=self._kem if self.suite.kem_ratchet else None,
            own_kem_dk=p["ratchet_dk"],
            peer_kem_ek=p["peer_kem_ek"],
            max_skip=self.suite.max_skip,
            rng=self.rng,
        )
        self._pending = None
        log = {
            "session_id": sid.hex(),
            "suite": self.suite.to_dict(),
            "transcript_hash": p["th3"].hex(),
            "peer_identity": p["peer_identity"],
            "root_fp": key_fingerprint(root),
            "x25519_fp": key_fingerprint(p["ss_dh"]),
            "mlkem_fp": key_fingerprint(p["ss_kem"]),
        }
        return HandshakeResult(sid, ratchet, p["th3"], p["peer"], log)
