"""Tokenizer, transformer, RoPE, KV cache and sampling tests."""

from __future__ import annotations

import numpy as np
import pytest

from quantasecure_neurollm.config import LLMConfig
from quantasecure_neurollm.corpus import CONFIDENTIAL_DOCUMENT, corpus_digest, tokenizer_training_texts
from quantasecure_neurollm.llm import BPETokenizer, RotaryEmbedding, SamplingConfig, TinyDecoderLM, filter_logits, generate, train_bpe
from quantasecure_neurollm.llm import bpe_merges
from quantasecure_neurollm.neuro import AdamW, Tensor, no_grad

TINY = LLMConfig(d_model=16, n_heads=2, n_layers=2, ffn_hidden=24, max_seq_len=48, n_prefix=2)


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.default()


def test_tokenizer_roundtrip_including_unicode(tok: BPETokenizer) -> None:
    for text in (CONFIDENTIAL_DOCUMENT, "Ψ(x) = ∑ αᵢ|i⟩ — naïve café 🚀", "", "   multiple   spaces\n\nnewlines"):
        assert tok.decode(tok.encode(text)) == text
    ids = tok.encode("surface code", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "surface code"
    assert len(tok.encode(CONFIDENTIAL_DOCUMENT)) < 0.45 * len(CONFIDENTIAL_DOCUMENT.encode())  # BPE compresses


def test_hardcoded_merges_are_reproducible_from_public_corpus() -> None:
    assert bpe_merges.CORPUS_SHA3 == corpus_digest().hex()
    assert tuple(train_bpe(tokenizer_training_texts(), bpe_merges.N_MERGES_REQUESTED)) == bpe_merges.MERGES


def test_rope_scores_depend_only_on_relative_position() -> None:
    rope = RotaryEmbedding(8, 64)
    rng = np.random.default_rng(0)
    q, k = rng.standard_normal(8), rng.standard_normal(8)

    def score(m: int, n: int) -> float:
        qm = rope.apply(Tensor(q.reshape(1, 1, 1, 8)), m).data.ravel()
        kn = rope.apply(Tensor(k.reshape(1, 1, 1, 8)), n).data.ravel()
        return float(qm @ kn)

    assert score(3, 10) == pytest.approx(score(20, 27), abs=1e-12)
    assert score(0, 5) == pytest.approx(score(30, 35), abs=1e-12)
    assert np.linalg.norm(rope.apply(Tensor(q.reshape(1, 1, 1, 8)), 17).data) == pytest.approx(np.linalg.norm(q))


def test_causality_future_tokens_do_not_change_past_logits(tok: BPETokenizer) -> None:
    model = TinyDecoderLM(TINY, tok.vocab_size, d_latent=4, seed=1)
    z = Tensor(np.random.default_rng(1).standard_normal((1, 4)))
    a = np.array([[tok.bos_id, 10, 20, 30, 40, 50]])
    b = a.copy()
    b[0, 4:] = [77, 88]
    with no_grad():
        la, lb = model(a, z).data, model(b, z).data
    p = TINY.n_prefix
    np.testing.assert_allclose(la[:, : p + 4], lb[:, : p + 4], atol=1e-12)
    assert not np.allclose(la[:, p + 4 :], lb[:, p + 4 :])


def test_kv_cache_matches_full_forward(tok: BPETokenizer) -> None:
    model = TinyDecoderLM(TINY, tok.vocab_size, d_latent=4, seed=2)
    z = Tensor(np.random.default_rng(2).standard_normal((1, 4)))
    ids = np.array([[tok.bos_id, tok.sum_id, 5, 9, 13, 2, 101]])
    with no_grad():
        full = model(ids, z).data[0]
        cache = model.new_cache()
        part = model(ids[:, :3], z, start=0, cache=cache).data[0]
        steps = [part]
        pos = cache.length
        for t in range(3, ids.shape[1]):
            steps.append(model(ids[:, t : t + 1], None, start=pos, cache=cache).data[0])
            pos += 1
    np.testing.assert_allclose(np.concatenate(steps, axis=0), full, atol=1e-10)


def test_lm_loss_gradient_check(tok: BPETokenizer) -> None:
    model = TinyDecoderLM(LLMConfig(d_model=8, n_heads=2, n_layers=1, ffn_hidden=8, max_seq_len=16, n_prefix=2), tok.vocab_size, 3, seed=3)
    z = Tensor(np.random.default_rng(3).standard_normal((2, 3)), requires_grad=True)
    ids = np.array([[tok.bos_id, 4, 5, 6], [tok.bos_id, 7, 8, tok.pad_id]])
    mask = np.array([[0, 1, 1, 1], [0, 1, 1, 0]], dtype=float)
    params = [model.tok_emb.weight, model.blocks[0].attn.wq.weight, model.adapter.proj.weight, z]
    model.zero_grad()
    model.sequence_loss(ids, z, mask).backward()
    rng = np.random.default_rng(0)
    for p in params:
        flat = p.data.reshape(-1)
        for i in rng.choice(flat.size, 4, replace=False):
            old = flat[i]
            flat[i] = old + 1e-6
            up = model.sequence_loss(ids, z, mask).item()
            flat[i] = old - 1e-6
            down = model.sequence_loss(ids, z, mask).item()
            flat[i] = old
            assert p.grad.reshape(-1)[i] == pytest.approx((up - down) / 2e-6, abs=1e-6)


def test_filter_logits_topk_topp_greedy() -> None:
    logits = np.log(np.array([0.5, 0.2, 0.15, 0.1, 0.05]))
    p = filter_logits(logits, SamplingConfig(temperature=1.0, top_k=3, top_p=1.0), [], [])
    assert np.count_nonzero(p) == 3 and p.sum() == pytest.approx(1.0)
    p = filter_logits(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=0.6), [], [])
    assert set(np.nonzero(p)[0]) == {0, 1}
    p = filter_logits(logits, SamplingConfig(temperature=0.0), [], [])
    assert p[0] == 1.0
    p = filter_logits(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=1.0), [], [0])
    assert p[0] == 0.0
    p = filter_logits(logits, SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=3.0), [0], [])
    assert p[0] < 0.5


def test_tiny_lm_memorises_conditioned_sentences(tok: BPETokenizer) -> None:
    """Two latents -> two sentences: the soft prompt alone must select the continuation."""
    cfg = LLMConfig(d_model=24, n_heads=2, n_layers=2, ffn_hidden=48, max_seq_len=40, n_prefix=2)
    model = TinyDecoderLM(cfg, tok.vocab_size, d_latent=2, seed=4)
    texts = ["the surface code works", "hydrides superconduct"]
    seqs = [[tok.bos_id, tok.sum_id] + tok.encode(t) + [tok.eos_id] for t in texts]
    width = max(map(len, seqs))
    ids = np.full((2, width), tok.pad_id)
    mask = np.zeros((2, width))
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, 2 : len(s)] = 1
    z = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))
    opt = AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = model.sequence_loss(ids, z, mask)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    for i, t in enumerate(texts):
        res = generate(model, tok, z.data[i : i + 1], [tok.bos_id, tok.sum_id], SamplingConfig(temperature=0.0, max_new_tokens=20))
        assert res.text == t and res.stopped_on_eos
