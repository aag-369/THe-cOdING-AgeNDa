"""Rotary positional embeddings (RoPE, Su et al. 2021), GPT-NeoX "rotate-half" layout.

For a head dimension ``d`` and position ``m`` the pair ``(x_i, x_{i+d/2})`` is
rotated by the angle ``m * base^(-2i/d)``.  Because rotations compose,
``<R_m q, R_n k> = <q, R_{n-m} k>``: attention scores depend only on the
*relative* offset ``n - m``, which the tests verify numerically.
"""

from __future__ import annotations

import numpy as np

from ..neuro.autograd import Tensor, concat

__all__ = ["RotaryEmbedding"]


class RotaryEmbedding:
    def __init__(self, d_head: int, max_len: int, base: float = 10000.0) -> None:
        if d_head % 2:
            raise ValueError("RoPE needs an even head dimension")
        inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)  # (d/2,)
        angles = np.arange(max_len)[:, None] * inv_freq[None, :]  # (L, d/2)
        angles = np.concatenate([angles, angles], axis=1)  # (L, d)
        self.cos = np.cos(angles)
        self.sin = np.sin(angles)
        self.half = d_head // 2
        self.max_len = max_len

    def apply(self, x: Tensor, start: int) -> Tensor:
        """Rotate ``x`` of shape ``(B, H, T, d)`` for absolute positions ``start .. start + T - 1``."""
        t = x.shape[2]
        if start + t > self.max_len:
            raise ValueError(f"sequence length {start + t} exceeds RoPE table ({self.max_len})")
        cos, sin = self.cos[start : start + t], self.sin[start : start + t]
        rotated = concat([-x[..., self.half :], x[..., : self.half]], axis=-1)
        return x * cos + rotated * sin
