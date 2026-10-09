"""AdamW, global-norm gradient clipping and a warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .layers import Parameter

__all__ = ["AdamW", "clip_grad_norm", "cosine_schedule"]


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter 2019).

    Weight decay is applied only to matrices (``ndim >= 2``); gains, biases and
    quantum rotation angles are not decayed, the usual convention.
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            if self.wd and p.ndim >= 2:
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


def clip_grad_norm(params: Sequence[Parameter], max_norm: float) -> float:
    """Scale all gradients so their joint L2 norm is at most ``max_norm``; returns the pre-clip norm."""
    total = math.sqrt(sum(float(np.sum(p.grad * p.grad)) for p in params if p.grad is not None))
    if total > max_norm > 0:
        scale = max_norm / (total + 1e-12)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale  # not in place: gradient arrays may be shared between tensors
    return total


def cosine_schedule(step: int, total: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
