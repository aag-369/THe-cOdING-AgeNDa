"""End-to-end orchestration of the secure quantum-enhanced reasoning pipeline.

Four mutually distrustful *enclave nodes* run in one process but share no
objects: their only connection is the :class:`~quantasecure_neurollm.secure.channel.SecureBus`,
which carries opaque byte frames and records every one of them::

    data-owner --(tokens)--> quantum-processor --(features + ZK proof)--> neural-encoder
         ^                                                                      |
         +--------------(answer + provenance)-- language-model <--(latent)------+

Phases
------
1. **Provisioning.**  Every node generates a hybrid identity key (Ed25519 + ML-DSA);
   public keys are pinned in a :class:`TrustStore`.
2. **Key establishment.**  One authenticated hybrid handshake (X25519 + ML-KEM,
   signed, key-confirmed) per link, each seeding a hybrid PQ Double Ratchet.
3. **Secure inference.**  Each node opens the incoming frame, checks the
   signed provenance chain and the hash-commitment opening of its input, does
   its work, commits to its output, signs a *stage record* and seals the result
   for the next node; it also returns an encrypted receipt (ack) upstream,
   which advances the DH ratchet in both directions.
4. **Verification.**  The data owner verifies every signature, the commitment
   linkage ``record[i].input == record[i-1].output``, the opening of the
   answer and -- independently of the neural node -- the zero-knowledge proof
   that the measurement shots are consistent with the committed quantum state.
   :mod:`.audit` then inspects the recorded wire traffic.

The confidential document exists in plaintext only inside the data-owner node
(the quantum node sees token ids, decrypted inside its own boundary).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .audit import AuditReport, WireAuditor
from .config import PipelineConfig
from .llm.generation import SamplingConfig, generate
from .neuro.autograd import no_grad
from .secure.channel import SecureBus, SecureSession
from .secure.commitments import HashCommitment, commit_object, verify_object
from .secure.handshake import HandshakeInitiator, HandshakeResponder, Suite
from .secure.hybrid import HybridPublicKey, HybridSigningKey, hybrid_verify
from .secure.rng import DeterministicDRBG, RandomSource, default_rng
from .secure.zkp import MeasurementProof, prove_measurement, verify_measurement
from .training import TrainedArtifacts, summary_prompt
from .utils.logging import AuditLog, get_logger
from .utils.rng import derive_seed
from .utils.serialization import decode, encode
from .utils.timing import Stopwatch

__all__ = [
    "SecurityError",
    "TrustStore",
    "PipelineResult",
    "SecureQuantumReasoningPipeline",
    "SUCCESS_MESSAGE",
    "OWNER",
    "QUANTUM",
    "NEURAL",
    "LLM",
]

OWNER, QUANTUM, NEURAL, LLM = "data-owner", "quantum-processor", "neural-encoder", "language-model"
STAGES = ("submit", "quantum-features", "neural-latent", "llm-answer")
STAGE_LABEL = "QSNL-stage-record-v1"
SUCCESS_MESSAGE = (
    "Secure quantum-enhanced reasoning completed. Original document never exposed in plaintext. "
    "All channel keys rotated. Proof of correct measurement verified."
)


class SecurityError(Exception):
    """A verification step failed; the pipeline aborts rather than continue on untrusted data."""


class TrustStore:
    """Pinned identity keys, provisioned out of band (here: by the orchestrator)."""

    def __init__(self) -> None:
        self._keys: dict[str, HybridPublicKey] = {}

    def register(self, pk: HybridPublicKey) -> None:
        if pk.owner in self._keys and self._keys[pk.owner] != pk:
            raise SecurityError(f"conflicting key for {pk.owner}")
        self._keys[pk.owner] = pk

    def lookup(self, name: str) -> Optional[HybridPublicKey]:
        return self._keys.get(name)


def _sha3(data: bytes) -> str:
    return hashlib.sha3_256(data).hexdigest()


# ============================================================================ nodes
class EnclaveNode:
    """A pipeline module with its own identity and sessions; talks only through the bus."""

    def __init__(self, name: str, bus: SecureBus, trust: TrustStore, audit: AuditLog, suite: Suite, rng: RandomSource) -> None:
        self.name = name
        self.bus = bus
        self.trust = trust
        self.audit = audit
        self.suite = suite
        self.rng = rng
        self.identity = HybridSigningKey(name, suite.signature, rng)
        self.sessions: dict[str, SecureSession] = {}
        self._initiators: dict[str, HandshakeInitiator] = {}
        self._responders: dict[str, HandshakeResponder] = {}
        self.log = get_logger(f"node.{name}")

    # ------------------------------------------------------------ handshakes
    def connect(self, peer: str) -> None:
        init = HandshakeInitiator(self.identity, peer, self.trust.lookup, self.suite, self.rng)
        self._initiators[peer] = init
        self.bus.send(self.name, peer, encode({"hs": "client_hello", "m": init.client_hello()}), "handshake")

    def _on_handshake(self, src: str, data: bytes) -> None:
        env = decode(data)
        kind = env["hs"]
        if kind == "client_hello":
            resp = HandshakeResponder(self.identity, self.trust.lookup, self.suite, self.rng)
            self._responders[src] = resp
            self.bus.send(self.name, src, encode({"hs": "server_hello", "m": resp.process_client_hello(env["m"])}), "handshake")
        elif kind == "server_hello":
            cf, result = self._initiators.pop(src).process_server_hello(env["m"])
            self._install(src, result, "initiator")
            self.bus.send(self.name, src, encode({"hs": "client_finished", "m": cf}), "handshake")
        elif kind == "client_finished":
            self._install(src, self._responders.pop(src).process_client_finished(env["m"]), "responder")
        else:
            raise SecurityError(f"unknown handshake message {kind!r}")

    def _install(self, peer: str, result: Any, role: str) -> None:
        self.sessions[peer] = SecureSession(self.name, peer, result.session_id, result.ratchet, self.audit)
        self.audit.append("handshake", "established", local=self.name, peer=peer, role=role, **result.log)
        self.log.info(
            "handshake %-9s with %-17s sid=%s  ML-KEM fp=%s  X25519 fp=%s  root fp=%s",
            role, peer, result.log["session_id"][:16], result.log["mlkem_fp"], result.log["x25519_fp"], result.log["root_fp"],
        )

    # ---------------------------------------------------------------- traffic
    def send(self, peer: str, payload: dict[str, Any], kind: str = "data") -> None:
        wire = self.sessions[peer].seal(payload, kind)
        rec = self.audit.records[-1]
        self.log.info("sealed %-4s -> %-17s %6d B on wire  (message key fp=%s, chain n=%d)", kind, peer, len(wire), rec.fields["key_fp"], rec.fields["n"])
        self.bus.send(self.name, peer, wire, kind)

    def pump(self) -> int:
        """Process every queued inbound message; returns how many were handled."""
        handled = 0
        while (msg := self.bus.receive(self.name)) is not None:
            src, channel, data = msg
            handled += 1
            if channel == "handshake":
                self._on_handshake(src, data)
                continue
            session = self.sessions.get(src)
            if session is None:
                raise SecurityError(f"{self.name}: data from {src} without a session")
            steps = session.ratchet.dh_ratchet_steps
            kind, payload = session.open(data)
            if session.ratchet.dh_ratchet_steps > steps:
                ev = next(e for e in reversed(session.ratchet.events) if e["event"] == "dh_ratchet_step")
                self.log.info("DH+KEM ratchet step with %-17s new root fp=%s", src, ev["root"])
                self.audit.append("ratchet", "dh_step", node=self.name, peer=src, root_fp=ev["root"], kem_mixed=ev["kem_mixed"])
            if kind == "ack":
                self.audit.append("channel", "receipt", node=self.name, peer=src, receipt=payload.get("receipt"))
            else:
                self.send(src, {"receipt": _sha3(data), "node": self.name}, kind="ack")
                self.on_data(src, payload)
        return handled

    def on_data(self, src: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------- provenance
    def sign_record(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {**record, "node": self.name, "node_identity": self.identity.public.fingerprint(), "time": time.time()}
        raw = encode(record)
        return {"record": raw, "sig": self.identity.sign(STAGE_LABEL, raw)}

    def verify_chain(self, chain: Sequence[dict[str, Any]], upto: int) -> list[dict[str, Any]]:
        """Check signatures, stage order and commitment linkage of the first ``upto`` records."""
        if len(chain) != upto:
            raise SecurityError(f"{self.name}: expected {upto} provenance records, got {len(chain)}")
        records: list[dict[str, Any]] = []
        prev_out: Optional[str] = None
        for entry, stage in zip(chain, STAGES):
            rec = decode(entry["record"])
            pk = self.trust.lookup(rec.get("node", ""))
            if pk is None or pk.fingerprint() != rec.get("node_identity"):
                raise SecurityError(f"{self.name}: unknown signer in stage {stage}")
            if not hybrid_verify(pk, STAGE_LABEL, entry["record"], entry["sig"]):
                raise SecurityError(f"{self.name}: bad signature on stage {stage}")
            if rec["stage"] != stage:
                raise SecurityError(f"{self.name}: unexpected stage {rec['stage']} (wanted {stage})")
            if prev_out is not None and rec["input_commitment"] != prev_out:
                raise SecurityError(f"{self.name}: commitment chain broken at stage {stage}")
            prev_out = rec["output_commitment"]
            records.append(rec)
        return records

    @staticmethod
    def check_opening(label: str, commitment_hex: str, obj: Any, salt: bytes) -> None:
        if not verify_object(HashCommitment(label, bytes.fromhex(commitment_hex)), obj, salt):
            raise SecurityError(f"commitment opening for {label!r} does not match")


def _zk_context(record: dict[str, Any]) -> bytes:
    return encode({"state_commitment": record["state_commitment"], "input_commitment": record["input_commitment"], "node": record["node"]})


class DataOwnerNode(EnclaveNode):
    """Holds the confidential document; submits it and verifies the returned answer."""

    def __init__(self, *args: Any, tokenizer: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.answer_payload: Optional[dict[str, Any]] = None
        self._submitted: Optional[dict[str, Any]] = None
        self.secret_tokens: Optional[np.ndarray] = None

    def submit(self, document: str, task: str = "summarize") -> None:
        tokens = np.asarray(self.tokenizer.encode(document), dtype=np.int32)
        doc_c, _doc_salt = commit_object("document", document, self.rng)
        tok_c, tok_salt = commit_object("tokens", tokens, self.rng)
        rec = self.sign_record({"stage": "submit", "input_commitment": doc_c.hex(), "output_commitment": tok_c.hex(), "n_tokens": int(tokens.size), "task": task})
        self._submitted = {"record": rec, "doc_commitment": doc_c.hex(), "tok_commitment": tok_c.hex()}
        self.secret_tokens = tokens
        self.log.info("document committed: %d bytes -> %d tokens, C_doc=%s.. (plaintext stays inside this node)", len(document.encode()), tokens.size, doc_c.hex()[:16])
        self.send(QUANTUM, {"tokens": tokens, "opening": tok_salt, "task": task, "provenance": [rec]})

    def on_data(self, src: str, payload: dict[str, Any]) -> None:
        if src != LLM:
            raise SecurityError(f"owner received data from unexpected node {src}")
        self.answer_payload = payload

    def verify(self) -> dict[str, Any]:
        """Owner-side end-to-end verification of the returned answer."""
        if self.answer_payload is None or self._submitted is None:
            raise SecurityError("no answer received")
        p = self.answer_payload
        checks: dict[str, bool] = {}
        records = self.verify_chain(p["provenance"], 4)
        checks["provenance_signatures"] = True
        checks["commitment_chain_linked"] = True
        checks["own_submission_intact"] = (
            records[0]["output_commitment"] == self._submitted["tok_commitment"]
            and records[0]["input_commitment"] == self._submitted["doc_commitment"]
        )
        self.check_opening("llm-answer", records[3]["output_commitment"], p["answer"], p["opening"])
        checks["answer_commitment_opened"] = True
        checks["zk_proof_bound_to_record"] = _sha3(encode(p["zk_proof"])) == records[1]["zk_proof_sha3"]
        zk = verify_measurement(MeasurementProof.from_dict(p["zk_proof"]), expected_context=_zk_context(records[1]))
        checks["zk_measurement_proof"] = zk.ok
        self.audit.append("verify", "owner", **{k: bool(v) for k, v in checks.items()})
        return {"checks": checks, "records": records, "zk": zk}


class QuantumProcessorNode(EnclaveNode):
    def __init__(self, *args: Any, artifacts: TrainedArtifacts, config: PipelineConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.art = artifacts
        self.config = config
        self.last_stats: dict[str, Any] = {}

    def on_data(self, src: str, payload: dict[str, Any]) -> None:
        records = self.verify_chain(payload["provenance"], 1)
        tokens = np.asarray(payload["tokens"])
        self.check_opening("tokens", records[0]["output_commitment"], payload["tokens"], payload["opening"])
        cfg = self.config
        feats = self.art.extractor.extract(tokens.astype(np.int64), seed=derive_seed("inference-shots", cfg.generation_seed))
        state_c, _state_salt = commit_object("quantum-state", feats.doc_state, self.rng)
        record_stub = {"state_commitment": state_c.hex(), "input_commitment": records[0]["output_commitment"], "node": self.name}
        t0 = time.perf_counter()
        probs = np.abs(feats.doc_state) ** 2
        proof = prove_measurement(
            probs, cfg.quantum.n_qubits, cfg.quantum.shots, _zk_context(record_stub),
            scale_bits=cfg.crypto.pedersen_scale_bits, delta=cfg.crypto.zk_delta, rng=self.rng,
        )  # fmt: skip
        proof_dict = proof.to_dict()
        out = {"chunk_features": feats.chunk_features, "global_features": feats.global_features}
        feat_c, feat_salt = commit_object("quantum-features", out, self.rng)
        proven = proof.proven_values()
        stats = {
            **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in feats.stats.items()},
            "shots": cfg.quantum.shots,
            "zk_commitments": len(proof.commitments),
            "zk_prove_seconds": time.perf_counter() - t0,
            "proven_marginals": [round(proven[f"marginal_q{q}"], 4) for q in range(cfg.quantum.n_qubits)],
        }
        self.last_stats = stats
        rec = self.sign_record(
            {
                "stage": "quantum-features",
                "input_commitment": records[0]["output_commitment"],
                "output_commitment": feat_c.hex(),
                "state_commitment": state_c.hex(),
                "zk_proof_sha3": _sha3(encode(proof_dict)),
                "model_digest": _sha3(encode([self.art.feature_map.projection, self.art.feature_map.reservoir]))[:16],
                "circuit": {"qubits": cfg.quantum.n_qubits, "gates": len(self.art.feature_map.circuit), "depth": self.art.feature_map.circuit.depth()},
                "measurement_stats": stats,
            }
        )
        self.log.info(
            "simulated %d windows on %d qubits (%d gates, depth %d): half-chain S=%.3f bits, noisy purity=%.3f, fidelity=%.3f, |norm drift|=%.1e (bound %.1e)",
            stats["n_windows"], stats["n_qubits"], stats["gates"], stats["depth"], stats["half_chain_entropy"], stats["purity"],
            stats["fidelity"], stats["max_norm_drift"], stats["norm_error_bound"],
        )  # fmt: skip
        self.log.info(
            "ZK: committed %d Born probabilities (Pedersen, 2^%d scale), %d shots, %d linear statements proven in %.2fs",
            len(proof.commitments), cfg.crypto.pedersen_scale_bits, cfg.quantum.shots, len(proof.claims), stats["zk_prove_seconds"],
        )  # fmt: skip
        self.send(NEURAL, {"features": out, "opening": feat_salt, "zk_proof": proof_dict, "provenance": list(payload["provenance"]) + [rec]})


class NeuralEncoderNode(EnclaveNode):
    def __init__(self, *args: Any, artifacts: TrainedArtifacts, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.art = artifacts
        self.zk_report: Optional[Any] = None

    def on_data(self, src: str, payload: dict[str, Any]) -> None:
        records = self.verify_chain(payload["provenance"], 2)
        self.check_opening("quantum-features", records[1]["output_commitment"], payload["features"], payload["opening"])
        if _sha3(encode(payload["zk_proof"])) != records[1]["zk_proof_sha3"]:
            raise SecurityError("ZK proof does not match the signed quantum record")
        zk = verify_measurement(MeasurementProof.from_dict(payload["zk_proof"]), expected_context=_zk_context(records[1]))
        self.zk_report = zk
        if not zk.ok:
            raise SecurityError(f"measurement-consistency proof rejected: {zk.checks}")
        self.log.info("ZK proof verified: %s; max shot deviation %.4f <= eps %.4f", ", ".join(k for k, v in zk.checks.items() if v), zk.max_deviation, zk.epsilon)
        f = payload["features"]
        with no_grad():
            out = self.art.encoder.encode([np.asarray(f["chunk_features"])], np.asarray(f["global_features"]))
        latent = out.latent.data[0]
        probs = self.art.encoder.topic_probabilities(out)[0]
        result = {"latent": latent, "topic_probs": probs, "quantum_layer": out.quantum_expectations[0]}
        lat_c, lat_salt = commit_object("neural-latent", result, self.rng)
        rec = self.sign_record(
            {
                "stage": "neural-latent",
                "input_commitment": records[1]["output_commitment"],
                "output_commitment": lat_c.hex(),
                "model_digest": self.art.encoder.digest(),
                "zk_verified": True,
                "latent_dim": int(latent.size),
            }
        )
        self.log.info("encoded %d windows -> latent[%d]; PQC layer <Z> = %s", len(f["chunk_features"]), latent.size, np.round(out.quantum_expectations[0][: self.art.encoder.cfg.quantum_qubits], 3).tolist())
        self.send(LLM, {"encoding": result, "opening": lat_salt, "zk_proof": payload["zk_proof"], "provenance": list(payload["provenance"]) + [rec]})


class LanguageModelNode(EnclaveNode):
    def __init__(self, *args: Any, artifacts: TrainedArtifacts, config: PipelineConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.art = artifacts
        self.config = config

    def on_data(self, src: str, payload: dict[str, Any]) -> None:
        records = self.verify_chain(payload["provenance"], 3)
        self.check_opening("neural-latent", records[2]["output_commitment"], payload["encoding"], payload["opening"])
        enc = payload["encoding"]
        lc = self.config.llm
        scfg = SamplingConfig(lc.temperature, lc.top_k, lc.top_p, lc.max_new_tokens, lc.repetition_penalty, seed=self.config.generation_seed)
        tok = self.art.tokenizer
        res = generate(self.art.lm, tok, np.asarray(enc["latent"])[None], summary_prompt(tok), scfg)
        probs = np.asarray(enc["topic_probs"])
        answer = {
            "text": res.text,
            "topic": self.art.topics[int(np.argmax(probs))],
            "topic_confidence": float(np.max(probs)),
            "tokens_generated": len(res.token_ids),
            "mean_logprob": res.mean_logprob,
            "stopped_on_eos": res.stopped_on_eos,
            "sampling": {"temperature": lc.temperature, "top_k": lc.top_k, "top_p": lc.top_p},
        }
        ans_c, ans_salt = commit_object("llm-answer", answer, self.rng)
        rec = self.sign_record(
            {
                "stage": "llm-answer",
                "input_commitment": records[2]["output_commitment"],
                "output_commitment": ans_c.hex(),
                "model_digest": self.art.lm.digest(),
                "tokens_generated": len(res.token_ids),
            }
        )
        self.log.info("generated %d tokens (mean log-prob %.3f, eos=%s), topic=%r (p=%.2f)", len(res.token_ids), res.mean_logprob, res.stopped_on_eos, answer["topic"], answer["topic_confidence"])
        self.send(OWNER, {"answer": answer, "opening": ans_salt, "zk_proof": payload["zk_proof"], "provenance": list(payload["provenance"]) + [rec]})


# ============================================================================ result
@dataclass
class PipelineResult:
    success: bool
    answer: dict[str, Any]
    owner_checks: dict[str, bool]
    zk_checks: dict[str, bool]
    audit: AuditReport
    quantum_stats: dict[str, Any]
    provenance: list[dict[str, Any]]
    timings: dict[str, float]
    wire: dict[str, Any]
    message: str = ""
    errors: list[str] = field(default_factory=list)


class SecureQuantumReasoningPipeline:
    """Wires the four enclave nodes together and runs one confidential document through them."""

    LINKS = ((OWNER, QUANTUM), (QUANTUM, NEURAL), (NEURAL, LLM), (LLM, OWNER))

    def __init__(self, config: PipelineConfig, artifacts: TrainedArtifacts, rng: Optional[RandomSource] = None) -> None:
        self.config = config
        self.art = artifacts
        seed = config.crypto.deterministic_seed
        self.rng = rng or (DeterministicDRBG(seed) if seed is not None else default_rng())
        self.suite = Suite(config.crypto.kem, config.crypto.signature, config.crypto.aead, config.crypto.kem_ratchet, config.crypto.max_skip)
        self.log = get_logger("pipeline")
        self.bus: Optional[SecureBus] = None
        self.audit_log: Optional[AuditLog] = None
        self.nodes: dict[str, EnclaveNode] = {}

    def _pump_all(self) -> None:
        for _ in range(1000):
            if sum(node.pump() for node in self.nodes.values()) == 0:
                return
        raise RuntimeError("message pump did not quiesce")

    def run(self, document: str) -> PipelineResult:
        sw = Stopwatch()
        bus, audit, trust = SecureBus(), AuditLog(), TrustStore()
        self.bus, self.audit_log = bus, audit
        common = dict(bus=bus, trust=trust, audit=audit, suite=self.suite, rng=self.rng)
        with sw.section("provisioning"):
            owner = DataOwnerNode(OWNER, tokenizer=self.art.tokenizer, **common)
            quantum = QuantumProcessorNode(QUANTUM, artifacts=self.art, config=self.config, **common)
            neural = NeuralEncoderNode(NEURAL, artifacts=self.art, **common)
            llm = LanguageModelNode(LLM, artifacts=self.art, config=self.config, **common)
            self.nodes = {n.name: n for n in (owner, quantum, neural, llm)}
            for n in self.nodes.values():
                trust.register(n.identity.public)
                audit.append("provision", "identity", node=n.name, fingerprint=n.identity.public.fingerprint(), signature=self.suite.signature)
                self.log.info("provisioned %-17s hybrid identity Ed25519+%s fp=%s", n.name, self.suite.signature, n.identity.public.fingerprint())
        with sw.section("handshakes"):
            for a, b in self.LINKS:
                self.nodes[a].connect(b)
                self._pump_all()
        errors: list[str] = []
        with sw.section("secure_inference"):
            try:
                owner.submit(document)
                self._pump_all()
            except Exception as exc:  # abort on any verification failure, but still audit what happened
                errors.append(f"{type(exc).__name__}: {exc}")
                self.log.error("pipeline aborted: %s", exc)
        verification: dict[str, Any] = {"checks": {}, "records": [], "zk": None}
        with sw.section("owner_verification"):
            if not errors:
                try:
                    verification = owner.verify()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        with sw.section("wire_audit"):
            auditor = WireAuditor(self.config.crypto.leak_window)
            secrets = {"document": document.encode("utf-8")}
            if owner.secret_tokens is not None:
                t = owner.secret_tokens
                secrets.update({"tokens_int32": t.astype("<i4").tobytes(), "tokens_int64": t.astype("<i8").tobytes(), "tokens_uint16": t.astype("<u2").tobytes()})
            report = auditor.run(bus, audit, self.nodes, secrets, probe_link=(OWNER, QUANTUM))
        zk = verification.get("zk")
        owner_checks = dict(verification.get("checks", {}))
        zk_checks = dict(zk.checks) if zk is not None else {}
        success = not errors and bool(owner_checks) and all(owner_checks.values()) and report.ok
        answer = owner.answer_payload["answer"] if owner.answer_payload else {}
        result = PipelineResult(
            success=success,
            answer=answer,
            owner_checks=owner_checks,
            zk_checks=zk_checks,
            audit=report,
            quantum_stats=quantum.last_stats,
            provenance=[{k: v for k, v in r.items() if k != "measurement_stats"} for r in verification.get("records", [])],
            timings=sw.as_dict(),
            wire={"frames": len(bus.wire), "bytes": bus.total_bytes},
            message=SUCCESS_MESSAGE if success else "Secure pipeline FAILED verification -- see errors and audit report.",
            errors=errors,
        )
        return result
