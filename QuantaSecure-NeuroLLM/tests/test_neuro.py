"""Gradient checks for the autograd engine, layers, quantum layer and hybrid encoder."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pytest

from quantasecure_neurollm.config import NeuroConfig
from quantasecure_neurollm.neuro import (
    AdamW,
    HybridQuantumEncoder,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    Parameter,
    QuantumLayer,
    RMSNorm,
    SwiGLU,
    Tensor,
    UnitaryMeshLayer,
    clip_grad_norm,
    concat,
    cross_entropy,
    log_softmax,
    no_grad,
    softmax,
)
from quantasecure_neurollm.neuro.encoder import pad_chunks


def grad_check(loss_fn: Callable[[], Tensor], params: Sequence[Tensor], h: float = 1e-6, n_probe: int = 12, tol: float = 1e-6) -> None:
    """Compare analytic gradients to central differences on random entries."""
    for p in params:
        p.grad = None
    loss = loss_fn()
    loss.backward()
    rng = np.random.default_rng(0)
    for p in params:
        assert p.grad is not None, "missing gradient"
        flat = p.data.reshape(-1)
        for i in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + h
            up = loss_fn().item()
            flat[i] = old - h
            down = loss_fn().item()
            flat[i] = old
            num = (up - down) / (2 * h)
            ana = p.grad.reshape(-1)[i]
            assert abs(num - ana) <= tol * max(1.0, abs(num)), f"grad mismatch {ana} vs {num}"


def test_elementwise_and_reduction_ops() -> None:
    rng = np.random.default_rng(1)
    a = Parameter(rng.standard_normal((3, 4)))
    b = Parameter(rng.standard_normal((4,)))
    c = Parameter(rng.uniform(0.5, 2.0, (3, 1)))

    def loss() -> Tensor:
        x = (a * b + a / c - b).tanh() + (a * a).sqrt() + a.sigmoid() * a.silu() + (c.log() * 2).exp()
        y = x.sin() * x.cos() + x**3 - x.relu()
        return (y.sum(axis=1, keepdims=True) * c).mean() + y[1:, ::2].sum()

    grad_check(loss, [a, b, c])


def test_matmul_shapes_and_broadcast() -> None:
    rng = np.random.default_rng(2)
    a = Parameter(rng.standard_normal((2, 3, 4)))
    w = Parameter(rng.standard_normal((4, 5)))
    grad_check(lambda: ((a @ w).transpose(0, 2, 1).reshape(2, -1) ** 2).sum(), [a, w])


def test_fused_softmax_logsoftmax_crossentropy() -> None:
    rng = np.random.default_rng(3)
    z = Parameter(rng.standard_normal((4, 6)))
    targets = rng.integers(0, 6, 4)
    weights = np.array([1.0, 0.0, 1.0, 0.5])
    grad_check(lambda: (softmax(z) * np.arange(6)).sum() + log_softmax(z)[0, 2], [z])
    grad_check(lambda: cross_entropy(z, targets, weights), [z])
    probs = np.exp(z.data - z.data.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    expected = -(np.log(probs[np.arange(4), targets]) * weights).sum() / weights.sum()
    assert cross_entropy(z, targets, weights).item() == pytest.approx(expected)


def test_embedding_gather_with_repeats_and_concat() -> None:
    rng = np.random.default_rng(4)
    w = Parameter(rng.standard_normal((5, 3)))
    ids = np.array([[0, 2, 2], [4, 0, 2]])
    grad_check(lambda: (concat([w[ids], w[ids] * 2], axis=-1) ** 2).sum(), [w])


def test_layers_gradients() -> None:
    rng = np.random.default_rng(5)
    x = Parameter(rng.standard_normal((2, 5, 8)))
    lin, rms, ln = Linear(8, 8, rng), RMSNorm(8), LayerNorm(8)
    ffn, mha = SwiGLU(8, 12, rng), MultiHeadAttention(8, 2, rng)
    mask = np.array([[True] * 5, [True, True, True, False, False]])
    params = [x] + lin.parameters() + rms.parameters() + ln.parameters() + ffn.parameters() + mha.parameters()
    grad_check(lambda: (mha(ffn(ln(rms(lin(x)))), mask) ** 2).mean(), params, n_probe=6)


def test_attention_mask_blocks_padded_keys() -> None:
    rng = np.random.default_rng(6)
    mha = MultiHeadAttention(8, 2, rng)
    x = rng.standard_normal((1, 4, 8))
    mask = np.array([[True, True, False, False]])
    y1 = mha(Tensor(x), mask).data
    x2 = x.copy()
    x2[0, 2:] += 100.0  # change only padded positions
    y2 = mha(Tensor(x2), mask).data
    np.testing.assert_allclose(y1[0, :2], y2[0, :2], atol=1e-10)


def test_unitary_mesh_is_norm_preserving_and_unitary() -> None:
    rng = np.random.default_rng(7)
    mesh = UnitaryMeshLayer(10, 6, rng, init_scale=1.0)
    x = rng.standard_normal((7, 10))
    y = mesh(Tensor(x)).data
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), rtol=1e-12)
    u = mesh.unitary()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(5), atol=1e-12)
    z = x[:, :5] + 1j * x[:, 5:]
    np.testing.assert_allclose(y[:, :5] + 1j * y[:, 5:], z @ u.T, atol=1e-12)
    xp = Parameter(x)
    grad_check(lambda: (mesh(xp) * np.arange(10)).sum(), [xp] + mesh.parameters(), n_probe=5)


def test_quantum_layer_parameter_shift_backward() -> None:
    rng = np.random.default_rng(8)
    ql = QuantumLayer(3, 2, rng)
    x = Parameter(rng.uniform(-1, 1, (4, 3)))
    coeffs = rng.standard_normal(ql.n_outputs)
    grad_check(lambda: (ql(x) * coeffs).sum(), [x, ql.weights], h=1e-5, tol=1e-6)
    with no_grad():
        assert not ql(x).requires_grad


def small_encoder(seed: int = 0) -> HybridQuantumEncoder:
    cfg = NeuroConfig(d_model=8, n_heads=2, n_blocks=1, ffn_hidden=12, mesh_depth=2, d_latent=6, quantum_qubits=2, quantum_layers=1)
    return HybridQuantumEncoder(cfg, chunk_dim=5, global_dim=4, n_topics=3, seed=seed)


def test_encoder_end_to_end_gradient_through_pqc() -> None:
    rng = np.random.default_rng(9)
    enc = small_encoder()
    chunks = [rng.standard_normal((3, 5)), rng.standard_normal((2, 5))]
    glob = rng.standard_normal((2, 4))
    padded, mask = pad_chunks(chunks)
    target = rng.standard_normal((2, 6))

    def loss() -> Tensor:
        out = enc(padded, mask, glob)
        return ((out.latent - target) ** 2).mean() + cross_entropy(out.topic_logits, np.array([0, 2]))

    sub = [enc.chunk_proj.weight, enc.quantum.weights, enc.to_quantum.weight, enc.blocks[0].mesh.thetas[0], enc.pool_query]
    grad_check(loss, sub, h=1e-6, n_probe=5, tol=2e-6)


def test_encoder_padding_invariance_and_state_dict_roundtrip() -> None:
    rng = np.random.default_rng(10)
    enc = small_encoder(1)
    enc.fit_normalizer([rng.standard_normal((4, 5))], rng.standard_normal((3, 4)))
    c = rng.standard_normal((2, 5))
    g = rng.standard_normal(4)
    with no_grad():
        a = enc.encode([c], g).latent.data
        b = enc.encode([c, rng.standard_normal((5, 5))], np.stack([g, g])).latent.data[0]  # padded batch
    np.testing.assert_allclose(a[0], b, atol=1e-10)
    enc2 = small_encoder(2)
    enc2.load_state_dict(enc.state_dict())
    with no_grad():
        np.testing.assert_allclose(enc2.encode([c], g).latent.data, a, atol=1e-12)
    assert enc2.digest() == enc.digest()


def test_adamw_and_clipping_fit_toy_regression() -> None:
    rng = np.random.default_rng(11)
    lin = Linear(3, 1, rng)
    x = rng.standard_normal((64, 3))
    y = x @ np.array([[1.5], [-2.0], [0.5]]) + 0.3
    opt = AdamW(lin.parameters(), lr=0.05)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((lin(Tensor(x)) - y) ** 2).mean()
        loss.backward()
        clip_grad_norm(lin.parameters(), 10.0)
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 1e-4 < first
