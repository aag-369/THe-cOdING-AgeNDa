"""Protocol-level tests: handshake, hybrid Double Ratchet, channel, commitments and the ZK proof."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from quantasecure_neurollm.quanta import StateVector
from quantasecure_neurollm.secure import (
    ChannelError,
    DeterministicDRBG,
    DoubleRatchet,
    EgressGuard,
    HandshakeError,
    HandshakeInitiator,
    HandshakeResponder,
    HybridPublicKey,
    HybridSigningKey,
    MLKEM,
    RatchetError,
    SecureSession,
    Suite,
    X25519KeyPair,
    commit_object,
    pedersen_commit,
    prove_measurement,
    verify_measurement,
    verify_object,
)
from quantasecure_neurollm.secure.ed25519 import L, point_add, point_equal
from quantasecure_neurollm.secure.zkp import MeasurementProof, quantize_probabilities
from quantasecure_neurollm.utils import AuditLog
from quantasecure_neurollm.utils.serialization import decode, encode

SUITE = Suite(kem="ML-KEM-512", signature="ML-DSA-44", aead="AES-256-GCM")


@pytest.fixture(scope="module")
def identities() -> dict[str, HybridSigningKey]:
    rng = DeterministicDRBG(10)
    return {name: HybridSigningKey(name, rng=rng) for name in ("alice", "bob", "mallory")}


def trust_of(ids: dict[str, HybridSigningKey], exclude: tuple[str, ...] = ()) -> "callable":  # type: ignore[valid-type]
    table = {n: k.public for n, k in ids.items() if n not in exclude}
    return lambda name: table.get(name)


def do_handshake(ids: dict[str, HybridSigningKey], suite: Suite = SUITE, rng_seed: int = 11):  # type: ignore[no-untyped-def]
    rng = DeterministicDRBG(rng_seed)
    init = HandshakeInitiator(ids["alice"], "bob", trust_of(ids), suite, rng)
    resp = HandshakeResponder(ids["bob"], trust_of(ids), suite, rng)
    sh = resp.process_client_hello(init.client_hello())
    cf, res_i = init.process_server_hello(sh)
    res_r = resp.process_client_finished(cf)
    return res_i, res_r


# ------------------------------------------------------------------ handshake


def test_handshake_agrees_on_session(identities) -> None:  # type: ignore[no-untyped-def]
    res_i, res_r = do_handshake(identities)
    assert res_i.session_id == res_r.session_id
    assert res_i.transcript_hash == res_r.transcript_hash
    assert res_i.log["root_fp"] == res_r.log["root_fp"]
    assert res_i.log["mlkem_fp"] == res_r.log["mlkem_fp"] and res_i.log["x25519_fp"] == res_r.log["x25519_fp"]


def test_handshake_rejects_man_in_the_middle(identities) -> None:  # type: ignore[no-untyped-def]
    rng = DeterministicDRBG(12)
    init = HandshakeInitiator(identities["alice"], "bob", trust_of(identities), SUITE, rng)
    ch = init.client_hello()
    # Mallory cannot accept a hello addressed to bob under her own identity ...
    with pytest.raises(HandshakeError):
        HandshakeResponder(identities["mallory"], trust_of(identities), SUITE, rng).process_client_hello(ch)
    # ... nor re-address alice's signed hello to herself ...
    outer = decode(ch)
    body = decode(outer["body"])
    body["responder"] = "mallory"
    with pytest.raises(HandshakeError):
        HandshakeResponder(identities["mallory"], trust_of(identities), SUITE, rng).process_client_hello(
            encode({"body": encode(body), "sig": outer["sig"]})
        )
    # ... nor impersonate bob with a key pair she generated herself: alice checks bob's *pinned* key.
    fake_bob = HybridSigningKey("bob", rng=rng)
    sh = HandshakeResponder(fake_bob, trust_of(identities), SUITE, rng).process_client_hello(ch)
    with pytest.raises(HandshakeError):
        init.process_server_hello(sh)


def test_handshake_detects_tampered_server_hello(identities) -> None:  # type: ignore[no-untyped-def]
    rng = DeterministicDRBG(13)
    init = HandshakeInitiator(identities["alice"], "bob", trust_of(identities), SUITE, rng)
    resp = HandshakeResponder(identities["bob"], trust_of(identities), SUITE, rng)
    sh = decode(resp.process_client_hello(init.client_hello()))
    body = decode(sh["body"])
    body["eph_x25519"] = X25519KeyPair.generate(rng).public  # MITM swaps the ephemeral key
    with pytest.raises(HandshakeError):
        init.process_server_hello(encode({"body": encode(body), "sig": sh["sig"], "finished": sh["finished"]}))


def test_handshake_rejects_unknown_peer_and_downgrade(identities) -> None:  # type: ignore[no-untyped-def]
    rng = DeterministicDRBG(14)
    init = HandshakeInitiator(identities["alice"], "bob", trust_of(identities), SUITE, rng)
    resp = HandshakeResponder(identities["bob"], trust_of(identities, exclude=("alice",)), SUITE, rng)
    with pytest.raises(HandshakeError):
        resp.process_client_hello(init.client_hello())
    weaker = Suite(kem="ML-KEM-512", signature="ML-DSA-44", aead="ChaCha20-Poly1305")
    init2 = HandshakeInitiator(identities["alice"], "bob", trust_of(identities), weaker, rng)
    resp2 = HandshakeResponder(identities["bob"], trust_of(identities), SUITE, rng)
    with pytest.raises(HandshakeError):
        resp2.process_client_hello(init2.client_hello())


def test_bad_client_finished_rejected(identities) -> None:  # type: ignore[no-untyped-def]
    rng = DeterministicDRBG(15)
    init = HandshakeInitiator(identities["alice"], "bob", trust_of(identities), SUITE, rng)
    resp = HandshakeResponder(identities["bob"], trust_of(identities), SUITE, rng)
    init.process_server_hello(resp.process_client_hello(init.client_hello()))
    with pytest.raises(HandshakeError):
        resp.process_client_finished(encode({"type": "client_finished", "finished": bytes(32)}))


# ------------------------------------------------------------------ ratchet


def ratchet_pair(kem: Optional[MLKEM]) -> tuple[DoubleRatchet, DoubleRatchet]:
    rng = DeterministicDRBG(20)
    sk = rng.random_bytes(32)
    bob_dh = X25519KeyPair.generate(rng)
    a_ek, a_dk = kem.keygen(rng) if kem else (None, None)
    b_ek, b_dk = kem.keygen(rng) if kem else (None, None)
    alice = DoubleRatchet.initiator(sk, bob_dh.public, kem=kem, own_kem_dk=a_dk, peer_kem_ek=b_ek, rng=rng)
    bob = DoubleRatchet.responder(sk, bob_dh, kem=kem, own_kem_dk=b_dk, peer_kem_ek=a_ek, rng=rng)
    return alice, bob


@pytest.mark.parametrize("use_kem", [False, True])
def test_ratchet_conversation_fresh_keys_and_dh_steps(use_kem: bool) -> None:
    alice, bob = ratchet_pair(MLKEM("ML-KEM-512") if use_kem else None)
    for turn in range(3):
        for i in range(2):
            h, c = alice.encrypt(f"a{turn}{i}".encode(), b"ad")
            assert bob.decrypt(h, c, b"ad") == f"a{turn}{i}".encode()
        h, c = bob.encrypt(f"b{turn}".encode(), b"ad")
        assert alice.decrypt(h, c, b"ad") == f"b{turn}".encode()
        if use_kem:
            assert len(h.kem_ct) == MLKEM("ML-KEM-512").params.ct_bytes
    keys = alice.used_key_fingerprints + bob.used_key_fingerprints
    assert len(keys) == len(set(keys)) == 9
    assert alice.dh_ratchet_steps >= 2 and bob.dh_ratchet_steps >= 3


def test_ratchet_out_of_order_and_replay() -> None:
    alice, bob = ratchet_pair(None)
    msgs = [alice.encrypt(f"m{i}".encode(), b"") for i in range(4)]
    assert bob.decrypt(*msgs[2], b"") == b"m2"
    assert bob.decrypt(*msgs[0], b"") == b"m0"
    assert bob.decrypt(*msgs[3], b"") == b"m3"
    assert bob.decrypt(*msgs[1], b"") == b"m1"
    with pytest.raises(Exception):
        bob.decrypt(*msgs[1], b"")  # replay of a consumed key


def test_ratchet_rolls_back_on_forgery() -> None:
    alice, bob = ratchet_pair(MLKEM("ML-KEM-512"))
    h, c = alice.encrypt(b"genuine", b"")
    forged = bytearray(c)
    forged[0] ^= 1
    with pytest.raises(RatchetError):
        bob.decrypt(h, bytes(forged), b"")
    assert bob.decrypt(h, c, b"") == b"genuine"  # state was not corrupted by the forgery


# ------------------------------------------------------------------ channel


def test_session_seal_open_and_tamper(identities) -> None:  # type: ignore[no-untyped-def]
    res_i, res_r = do_handshake(identities, rng_seed=30)
    audit = AuditLog()
    a = SecureSession("alice", "bob", res_i.session_id, res_i.ratchet, audit)
    b = SecureSession("bob", "alice", res_r.session_id, res_r.ratchet, audit)
    secret = {"tokens": np.arange(500, dtype=np.int32), "note": "confidential " * 20}
    wire = a.seal(secret)
    assert b"confidential" not in wire and np.arange(500, dtype=np.int32).tobytes()[:64] not in wire
    kind, payload = b.open(wire)
    assert kind == "data" and payload["note"] == secret["note"]
    np.testing.assert_array_equal(payload["tokens"], secret["tokens"])
    ack = b.seal({"ack": True}, kind="ack")
    assert a.open(ack)[1] == {"ack": True}
    wire2 = bytearray(a.seal({"x": 1}))
    wire2[-5] ^= 0xFF
    with pytest.raises(ChannelError):
        b.open(bytes(wire2))
    with pytest.raises(ChannelError):
        b.open(wire)  # replay
    assert audit.verify_chain()


def test_egress_guard_detects_plaintext() -> None:
    g = EgressGuard(window=8)
    assert g.check(b"secret material here", b"\x00" * 40)
    assert not g.check(b"secret material here", b"header|secret material|trailer")
    # Text that also appears in the public routing header is metadata, not a leak ...
    assert g.check(b"sent by data-owner node", b"src=data-owner|<ciphertext>", public=b"src=data-owner|")
    # ... but the same text inside the encrypted region still is.
    assert not g.check(b"sent by data-owner node", b"src=xx|sent by data-owner", public=b"src=xx|")


# ------------------------------------------------------------------ commitments


def test_hash_commitment_binding_and_hiding() -> None:
    rng = DeterministicDRBG(40)
    c, salt = commit_object("features", {"x": np.ones(3)}, rng)
    assert verify_object(c, {"x": np.ones(3)}, salt)
    assert not verify_object(c, {"x": np.zeros(3)}, salt)
    c2, _ = commit_object("features", {"x": np.ones(3)}, rng)
    assert c.digest != c2.digest  # fresh salt -> unlinkable commitments to equal values


def test_pedersen_homomorphism() -> None:
    a = pedersen_commit(5, 123)
    b = pedersen_commit(7, 456)
    assert point_equal(point_add(a, b), pedersen_commit(12, 579))
    assert not point_equal(pedersen_commit(5, 1), pedersen_commit(6, 1))
    assert point_equal(pedersen_commit(5, L + 1), pedersen_commit(5, 1))


# ------------------------------------------------------------------ ZK proof


@pytest.fixture(scope="module")
def ghz_like_probs() -> np.ndarray:
    sv = StateVector(4).h(0).cnot(0, 1).ry(2, 0.7).cnot(2, 3).rx(3, 0.3)
    return sv.probabilities()


def test_quantization_sums_exactly(ghz_like_probs: np.ndarray) -> None:
    q = quantize_probabilities(ghz_like_probs, 20)
    assert q.sum() == 2**20 and q.min() >= 0
    assert np.max(np.abs(q / 2**20 - ghz_like_probs)) < 2**-20


def test_zk_proof_accepts_honest_prover(ghz_like_probs: np.ndarray) -> None:
    proof = prove_measurement(ghz_like_probs, 4, 2000, b"ctx", rng=DeterministicDRBG(50))
    proof = MeasurementProof.from_dict(decode(encode(proof.to_dict())))  # survives the wire
    report = verify_measurement(proof, expected_context=b"ctx")
    assert report.ok, report.checks
    vals = proof.proven_values()
    assert vals["total"] == 1.0
    np.testing.assert_allclose(vals["marginal_q0"], 0.5, atol=1e-5)


def test_zk_proof_rejects_wrong_claims_shots_and_context(ghz_like_probs: np.ndarray) -> None:
    proof = prove_measurement(ghz_like_probs, 4, 2000, b"ctx", rng=DeterministicDRBG(51))
    assert not verify_measurement(proof, expected_context=b"other").ok
    d = proof.to_dict()
    d["claims"] = list(d["claims"])
    d["claims"][1] += 1000  # lie about the marginal of qubit 0
    assert not verify_measurement(MeasurementProof.from_dict(d)).checks["linear_relations"]
    d = proof.to_dict()
    d["shots"] = np.zeros_like(d["shots"])  # report fabricated all-zero shots
    rep = verify_measurement(MeasurementProof.from_dict(d))
    assert not rep.ok and not rep.checks["shots_consistent"]
    d = proof.to_dict()
    d["responses"] = [(r + 1) % L for r in d["responses"]]
    assert not verify_measurement(MeasurementProof.from_dict(d)).ok
    d = proof.to_dict()
    d["pok_s_m"] = (d["pok_s_m"] + 1) % L
    assert not verify_measurement(MeasurementProof.from_dict(d)).checks["opening_knowledge"]


def test_hybrid_public_key_roundtrip(identities) -> None:  # type: ignore[no-untyped-def]
    pk = identities["alice"].public
    assert HybridPublicKey.from_dict(decode(encode(pk.to_dict()))) == pk
