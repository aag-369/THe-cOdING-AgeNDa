"""Compact decoder-only transformer, implemented from scratch on the autograd engine.

Per block (pre-norm, as in LLaMA-style models)::

    h = x + W_o CausalAttention( RoPE(W_q RMSNorm(x)), RoPE(W_k RMSNorm(x)), W_v RMSNorm(x) )
    y = h + SwiGLU( RMSNorm(h) )

followed by a final RMSNorm and an output projection *tied* to the token
embedding matrix (Press & Wolf 2017), which halves the parameter count of the
vocabulary layers and regularises a small model.

Conditioning ("soft prompt")
----------------------------
:class:`SoftPromptAdapter` maps the encoder latent ``z`` to ``P`` continuous
vectors that are prepended to the token embeddings (Lester et al. 2021).  They
occupy positions ``0..P-1`` of the RoPE table, so every generated token can
attend to the quantum-derived conditioning through ordinary causal attention.

Inference uses a KV cache: keys and values of past positions are stored per
layer and each new token costs ``O(T d)`` attention instead of re-running the
whole prefix; the tests check that cached and uncached logits agree to
``1e-10``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import LLMConfig
from ..neuro.autograd import Tensor, concat, cross_entropy
from ..neuro.layers import Embedding, Linear, Module, RMSNorm, SwiGLU, scaled_dot_product_attention
from .rope import RotaryEmbedding

__all__ = ["CausalSelfAttention", "DecoderBlock", "SoftPromptAdapter", "TinyDecoderLM", "KVCache"]


@dataclass
class KVCache:
    """Per-layer cached keys / values (``(B, H, T, d_head)`` arrays)."""

    keys: list[Optional[np.ndarray]] = field(default_factory=list)
    values: list[Optional[np.ndarray]] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> "KVCache":
        return cls([None] * n_layers, [None] * n_layers)

    @property
    def length(self) -> int:
        return 0 if self.keys[0] is None else int(self.keys[0].shape[2])


class CausalSelfAttention(Module):
    def __init__(self, d: int, n_heads: int, rope: RotaryEmbedding, rng: np.random.Generator, out_gain: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.wq = Linear(d, d, rng, bias=False)
        self.wk = Linear(d, d, rng, bias=False)
        self.wv = Linear(d, d, rng, bias=False)
        self.wo = Linear(d, d, rng, bias=False, gain=out_gain)
        self._rope = rope

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, start: int, cache: Optional[KVCache] = None, layer: int = 0) -> Tensor:
        b, t, d = x.shape
        q = self._rope.apply(self._split(self.wq(x)), start)
        k = self._rope.apply(self._split(self.wk(x)), start)
        v = self._split(self.wv(x))
        if cache is not None:
            if cache.keys[layer] is not None:
                k = concat([Tensor(cache.keys[layer]), k], axis=2)
                v = concat([Tensor(cache.values[layer]), v], axis=2)
            cache.keys[layer], cache.values[layer] = k.data, v.data
        t_k = k.shape[2]
        # Query i sits at absolute position start + i; the t_k keys end at position start + t - 1
        # (they start at `start` without a cache, at 0 with one).  Causal rule: key_pos <= query_pos.
        q_pos = start + np.arange(t)[:, None]
        k_pos = (start + t - t_k) + np.arange(t_k)[None, :]
        bias = np.where(k_pos <= q_pos, 0.0, -1e9)[None, None]
        out, _ = scaled_dot_product_attention(q, k, v, bias)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(b, t, d))


class DecoderBlock(Module):
    def __init__(self, cfg: LLMConfig, rope: RotaryEmbedding, rng: np.random.Generator) -> None:
        super().__init__()
        gain = 1.0 / math.sqrt(2 * cfg.n_layers)
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, rope, rng, gain)
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden, rng, out_gain=gain)

    def forward(self, x: Tensor, start: int, cache: Optional[KVCache], layer: int) -> Tensor:
        x = x + self.attn(self.norm_attn(x), start, cache, layer)
        return x + self.ffn(self.norm_ffn(x))


class SoftPromptAdapter(Module):
    """``z (B, d_latent) -> P conditioning vectors (B, P, d_model)``."""

    def __init__(self, d_latent: int, d_model: int, n_prefix: int, rng: np.random.Generator) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.d_model = d_model
        self.proj = Linear(d_latent, n_prefix * d_model, rng)
        self.norm = RMSNorm(d_model)

    def forward(self, z: Tensor) -> Tensor:
        b = z.shape[0]
        return self.norm(self.proj(z).reshape(b, self.n_prefix, self.d_model))


class TinyDecoderLM(Module):
    def __init__(self, cfg: LLMConfig, vocab_size: int, d_latent: int, seed: int = 0) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.vocab_size = vocab_size
        self._rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_base)
        self.tok_emb = Embedding(vocab_size, cfg.d_model, rng, std=1.0 / math.sqrt(cfg.d_model))
        self.adapter = SoftPromptAdapter(d_latent, cfg.d_model, cfg.n_prefix, rng)
        self.blocks = [DecoderBlock(cfg, self._rope, rng) for _ in range(cfg.n_layers)]
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(
        self,
        ids: np.ndarray,
        latent: Optional[Tensor] = None,
        start: int = 0,
        cache: Optional[KVCache] = None,
    ) -> Tensor:
        """Logits ``(B, P + T, V)`` (or ``(B, T, V)`` without a latent / when continuing a cache)."""
        x = self.tok_emb(ids)
        if latent is not None:
            x = concat([self.adapter(latent), x], axis=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, start, cache, i)
        h = self.norm_f(x)
        return h @ self.tok_emb.weight.transpose(1, 0)

    def sequence_loss(self, ids: np.ndarray, latent: Optional[Tensor], loss_mask: np.ndarray) -> Tensor:
        """Next-token cross-entropy on positions where ``loss_mask`` (over ``ids``) is 1.

        ``loss_mask[b, t] = 1`` means token ``ids[b, t]`` is a prediction target
        (it is predicted from the position just before it).
        """
        logits = self.forward(ids, latent)
        p = self.cfg.n_prefix if latent is not None else 0
        # Logit at sequence position (p + t - 1) predicts ids[:, t] for t >= 1.
        pred = logits[:, p : p + ids.shape[1] - 1, :]
        return cross_entropy(pred, ids[:, 1:], loss_mask[:, 1:])

    def new_cache(self) -> KVCache:
        return KVCache.empty(len(self.blocks))
