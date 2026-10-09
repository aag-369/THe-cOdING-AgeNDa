"""End-to-end tests of the secure pipeline (trains the fast configuration once, then uses the cache).

These assert the system-level claims:

* the run succeeds and prints the success criterion;
* the generated text is non-empty and relevant to the confidential document;
* the document / token ids never appear in plaintext in any wire frame
  (checked here independently of the built-in auditor);
* every message used a fresh key and every ratchet advanced;
* active attacks -- a bit flip in flight, a dishonest quantum node lying about
  its measurement statistics -- make the pipeline fail closed.
"""

from __future__ import annotations

import numpy as np
import pytest

import quantasecure_neurollm.pipeline as pipeline_mod
from quantasecure_neurollm.config import fast_config
from quantasecure_neurollm.corpus import CONFIDENTIAL_DOCUMENT, CONFIDENTIAL_TOPIC, EVALUATION_DOCUMENTS, PUBLIC_CORPUS
from quantasecure_neurollm.pipeline import NEURAL, QUANTUM, SUCCESS_MESSAGE, SecureQuantumReasoningPipeline
from quantasecure_neurollm.secure.channel import WireRecord
from quantasecure_neurollm.training import TrainedArtifacts, load_or_train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def artifacts() -> TrainedArtifacts:
    return load_or_train(fast_config())


@pytest.fixture(scope="module")
def run(artifacts: TrainedArtifacts):  # type: ignore[no-untyped-def]
    pipe = SecureQuantumReasoningPipeline(fast_config(), artifacts)
    return pipe, pipe.run(CONFIDENTIAL_DOCUMENT)


def test_pipeline_succeeds_with_success_message(run) -> None:  # type: ignore[no-untyped-def]
    _, result = run
    assert result.errors == []
    assert result.success, (result.owner_checks, result.zk_checks, result.audit.checks)
    assert result.message == SUCCESS_MESSAGE
    assert all(result.owner_checks.values()) and all(result.zk_checks.values()) and result.audit.ok


def test_generated_text_is_nonempty_and_relevant(run) -> None:  # type: ignore[no-untyped-def]
    _, result = run
    text = result.answer["text"].lower()
    assert len(text.split()) >= 10
    assert result.answer["topic"] == CONFIDENTIAL_TOPIC
    qec = next(d for d in PUBLIC_CORPUS if d.topic == CONFIDENTIAL_TOPIC)
    doc_words = set(CONFIDENTIAL_DOCUMENT.lower().replace(",", " ").replace(".", " ").split())
    shared = [k for k in qec.keywords if k in text and any(w.startswith(k) for w in doc_words)]
    assert len(shared) >= 4, shared  # e.g. surface, code, logical, error, threshold, decoder ...


def test_no_plaintext_on_the_wire_independent_check(run) -> None:  # type: ignore[no-untyped-def]
    pipe, _ = run
    frames = [r.data for r in pipe.bus.wire]
    doc = CONFIDENTIAL_DOCUMENT.encode()
    for k in range(0, len(doc) - 16, 4):
        window = doc[k : k + 16]
        assert not any(window in f for f in frames), f"document bytes {window!r} found on the wire"
    tokens = np.asarray(pipe.art.tokenizer.encode(CONFIDENTIAL_DOCUMENT), dtype=np.int32).tobytes()
    for k in range(0, len(tokens) - 16, 8):
        assert not any(tokens[k : k + 16] in f for f in frames)
    # The confidential words do not even appear individually inside the handshake frames.
    for word in (b"microsecond", b"seventy two", b"distance seven"):
        assert not any(word in f for f in frames)


def test_every_message_used_a_fresh_key_and_all_ratchets_advanced(run) -> None:  # type: ignore[no-untyped-def]
    pipe, result = run
    keys = [fp for node in pipe.nodes.values() for s in node.sessions.values() for fp in s.ratchet.used_key_fingerprints]
    assert len(keys) == len(set(keys))
    assert all(v >= 1 for v in result.audit.details["dh_ratchet_steps"].values())
    assert result.audit.details["frames_by_channel"]["data"] == 4 and result.audit.details["frames_by_channel"]["ack"] == 4


def test_provenance_chain_is_complete_and_linked(run) -> None:  # type: ignore[no-untyped-def]
    _, result = run
    stages = [r["stage"] for r in result.provenance]
    assert stages == ["submit", "quantum-features", "neural-latent", "llm-answer"]
    for prev, cur in zip(result.provenance, result.provenance[1:]):
        assert cur["input_commitment"] == prev["output_commitment"]


def test_bit_flip_in_flight_fails_closed(artifacts: TrainedArtifacts) -> None:
    pipe = SecureQuantumReasoningPipeline(fast_config(), artifacts)

    def flip(rec: WireRecord) -> bytes:
        if rec.src == QUANTUM and rec.dst == NEURAL and rec.channel == "data":
            data = bytearray(rec.data)
            data[len(data) // 2] ^= 0x10
            return bytes(data)
        return rec.data

    original_run = pipe.run

    def run_with_interceptor(document: str):  # type: ignore[no-untyped-def]
        orig_bus = pipeline_mod.SecureBus
        pipeline_mod.SecureBus = lambda: orig_bus(interceptor=flip)  # type: ignore[assignment,misc]
        try:
            return original_run(document)
        finally:
            pipeline_mod.SecureBus = orig_bus  # type: ignore[assignment]

    result = run_with_interceptor(CONFIDENTIAL_DOCUMENT)
    assert not result.success
    assert result.answer == {}
    assert any("rejected" in e or "authentication" in e for e in result.errors), result.errors


def test_dishonest_quantum_node_is_caught_by_the_zk_proof(artifacts: TrainedArtifacts, monkeypatch: pytest.MonkeyPatch) -> None:
    honest = pipeline_mod.prove_measurement

    def lying_prover(*args, **kwargs):  # type: ignore[no-untyped-def]
        proof = honest(*args, **kwargs)
        proof.claims[1] += 50_000  # over-report P(qubit 0 = 1) by ~5%
        return proof

    monkeypatch.setattr(pipeline_mod, "prove_measurement", lying_prover)
    result = SecureQuantumReasoningPipeline(fast_config(), artifacts).run(CONFIDENTIAL_DOCUMENT)
    assert not result.success
    assert any("measurement-consistency proof rejected" in e for e in result.errors), result.errors


def test_unseen_evaluation_documents_get_relevant_answers(artifacts: TrainedArtifacts) -> None:
    pipe = SecureQuantumReasoningPipeline(fast_config(), artifacts)
    correct = 0
    for topic, text in EVALUATION_DOCUMENTS:
        result = pipe.run(text)
        assert result.success
        correct += result.answer["topic"] == topic
    assert correct >= 2
