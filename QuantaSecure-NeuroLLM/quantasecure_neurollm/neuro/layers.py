"""Neural-network building blocks on top of :mod:`.autograd`.

All layers are :class:`Module` subclasses with explicit, typed ``forward``
methods.  Parameters are discovered by walking attributes (``Parameter``,
sub-``Module``, and lists of either), so ``state_dict`` / ``load_state_dict``
and the optimiser need no registration boilerplate.  Non-trainable arrays that
must be checkpointed (e.g. feature-normalisation statistics) are registered
with :meth:`Module.register_buffer`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterator, Optional

import numpy as np

from .autograd import Tensor, concat, softmax

__all__ = [
    "Parameter",
    "Module",
    "Linear",
    "Embedding",
    "RMSNorm",
    "LayerNorm",
    "SwiGLU",
    "MultiHeadAttention",
    "UnitaryMeshLayer",
    "scaled_dot_product_attention",
]


class Parameter(Tensor):
    """A trainable tensor (always ``requires_grad=True``)."""

    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        super().__init__(np.array(data, dtype=np.float64), requires_grad=True, name=name)


class Module:
    """Base class: parameter discovery, (de)serialisation, train/eval flags."""

    def __init__(self) -> None:
        self.training = True
        self._buffers: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- discovery
    def _children(self) -> Iterator[tuple[str, Any]]:
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue
            if isinstance(v, (Parameter, Module)):
                yield k, v
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, (Parameter, Module)):
                        yield f"{k}.{i}", item

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        out: list[tuple[str, Parameter]] = []
        seen: set[int] = set()
        for name, child in self._children():
            full = f"{prefix}{name}"
            if isinstance(child, Parameter):
                if id(child) not in seen:
                    seen.add(id(child))
                    out.append((full, child))
            else:
                for n, p in child.named_parameters(full + "."):
                    if id(p) not in seen:
                        seen.add(id(p))
                        out.append((n, p))
        return out

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def n_params(self) -> int:
        return int(sum(p.size for p in self.parameters()))

    def register_buffer(self, name: str, value: np.ndarray) -> None:
        self._buffers[name] = np.asarray(value, dtype=np.float64)

    def buffer(self, name: str) -> np.ndarray:
        return self._buffers[name]

    def named_buffers(self, prefix: str = "") -> list[tuple[str, np.ndarray]]:
        out = [(f"{prefix}{k}", v) for k, v in self._buffers.items()]
        for name, child in self._children():
            if isinstance(child, Module):
                out += child.named_buffers(f"{prefix}{name}.")
        return out

    # ------------------------------------------------------------ serialisation
    def state_dict(self) -> dict[str, np.ndarray]:
        d = {f"param:{n}": p.data.copy() for n, p in self.named_parameters()}
        d.update({f"buffer:{n}": b.copy() for n, b in self.named_buffers()})
        return d

    def load_state_dict(self, state: dict[str, np.ndarray], strict: bool = True) -> None:
        params = dict(self.named_parameters())
        for n, p in params.items():
            key = f"param:{n}"
            if key not in state:
                if strict:
                    raise KeyError(f"missing parameter {n}")
                continue
            if state[key].shape != p.shape:
                raise ValueError(f"shape mismatch for {n}: {state[key].shape} vs {p.shape}")
            p.data = np.array(state[key], dtype=np.float64)
        bufs = {n for n, _ in self.named_buffers()}
        for key, value in state.items():
            if key.startswith("buffer:") and key[7:] in bufs:
                self._set_buffer(key[7:], value)
        if strict:
            extra = [k for k in state if k.startswith("param:") and k[6:] not in params]
            if extra:
                raise KeyError(f"unexpected parameters {extra[:5]}")

    def _set_buffer(self, dotted: str, value: np.ndarray) -> None:
        head, _, rest = dotted.partition(".")
        if not rest:
            self._buffers[head] = np.asarray(value, dtype=np.float64)
            return
        child: Any = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = child[int(part)] if isinstance(child, (list, tuple)) else getattr(child, part)
        child._buffers[parts[-1]] = np.asarray(value, dtype=np.float64)

    # ------------------------------------------------------------------ modes
    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for _, child in self._children():
            if isinstance(child, Module):
                child.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def digest(self) -> str:
        """Hash of every weight and buffer: identifies the exact model in provenance records."""
        h = hashlib.sha3_256()
        for k, v in sorted(self.state_dict().items()):
            h.update(k.encode() + np.ascontiguousarray(v).tobytes())
        return h.hexdigest()[:16]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)  # type: ignore[attr-defined]


class Linear(Module):
    """``y = x W + b`` with ``W`` initialised ``N(0, gain^2 / fan_in)``."""

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, bias: bool = True, gain: float = 1.0) -> None:
        super().__init__()
        self.weight = Parameter(rng.standard_normal((d_in, d_out)) * gain / math.sqrt(d_in))
        self.bias = Parameter(np.zeros(d_out)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        # One (N, d_in) x (d_in, d_out) GEMM instead of a batched matmul over leading axes.
        y = (x.reshape(-1, x.shape[-1]) @ self.weight).reshape(*lead, self.weight.shape[1])
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, n: int, d: int, rng: np.random.Generator, std: float = 0.1) -> None:
        super().__init__()
        self.weight = Parameter(rng.standard_normal((n, d)) * std)

    def forward(self, ids: np.ndarray) -> Tensor:
        return self.weight[np.asarray(ids, dtype=np.int64)]


class RMSNorm(Module):
    """``x / sqrt(mean(x^2) + eps) * g`` (Zhang & Sennrich 2019)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = Parameter(np.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        return x / rms * self.weight


class LayerNorm(Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = Parameter(np.ones(d))
        self.bias = Parameter(np.zeros(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc / (var + self.eps).sqrt() * self.weight + self.bias


class SwiGLU(Module):
    """``W_down( SiLU(x W_gate) * (x W_up) )`` (Shazeer 2020)."""

    def __init__(self, d: int, hidden: int, rng: np.random.Generator, out_gain: float = 1.0) -> None:
        super().__init__()
        self.gate = Linear(d, hidden, rng, bias=False)
        self.up = Linear(d, hidden, rng, bias=False)
        self.down = Linear(hidden, d, rng, bias=False, gain=out_gain)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.gate(x).silu() * self.up(x))


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, bias: Optional[np.ndarray] = None) -> tuple[Tensor, np.ndarray]:
    """``softmax(q k^T / sqrt(d) + bias) v``; returns the output and the attention weights."""
    d = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / math.sqrt(d))
    if bias is not None:
        scores = scores + bias
    attn = softmax(scores, axis=-1)
    return attn @ v, attn.data


class MultiHeadAttention(Module):
    """Bidirectional multi-head self-attention with an optional key-padding mask."""

    def __init__(self, d: int, n_heads: int, rng: np.random.Generator, out_gain: float = 1.0) -> None:
        super().__init__()
        if d % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.wq = Linear(d, d, rng, bias=False)
        self.wk = Linear(d, d, rng, bias=False)
        self.wv = Linear(d, d, rng, bias=False)
        self.wo = Linear(d, d, rng, bias=False, gain=out_gain)
        self.last_attention: Optional[np.ndarray] = None

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, key_mask: Optional[np.ndarray] = None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self._split(self.wq(x)), self._split(self.wk(x)), self._split(self.wv(x))
        bias = None
        if key_mask is not None:
            bias = np.where(np.asarray(key_mask, dtype=bool), 0.0, -1e9)[:, None, None, :]
        out, attn = scaled_dot_product_attention(q, k, v, bias)
        self.last_attention = attn
        return self.wo(out.transpose(0, 2, 1, 3).reshape(b, t, d))


class UnitaryMeshLayer(Module):
    """Quantum-inspired, exactly norm-preserving layer.

    The ``d`` real features are read as ``d/2`` complex amplitudes
    ``a_j = x_j + i x_{j + d/2}``.  ``depth`` brick-wall layers of
    Mach-Zehnder interferometers (MZIs) act on neighbouring modes ``(j, j+1)``:

        U(theta, phi) = [[e^{i phi} cos theta, -sin theta],
                         [e^{i phi} sin theta,  cos theta]]

    which is unitary for every ``(theta, phi)`` -- the same parameterisation
    used for universal linear-optical processors (Reck et al. 1994; Clements et
    al. 2016).  Hence ``||f(x)||_2 = ||x||_2`` exactly, gradients never explode
    through the layer, and with ``depth >= d/2`` the mesh can represent any
    unitary on ``C^{d/2}``.  Angles are initialised small (near the identity)
    so the residual stream starts undisturbed.
    """

    def __init__(self, d: int, depth: int, rng: np.random.Generator, init_scale: float = 0.1) -> None:
        super().__init__()
        if d % 2:
            raise ValueError("UnitaryMeshLayer needs an even feature dimension")
        self.d = d
        self.m = d // 2
        self.layer_pairs: list[np.ndarray] = []
        self.thetas: list[Parameter] = []
        self.phis: list[Parameter] = []
        for layer in range(depth):
            start = layer % 2
            pairs = np.array([(j, j + 1) for j in range(start, self.m - 1, 2)], dtype=np.int64).reshape(-1, 2)
            if len(pairs) == 0:
                continue
            self.layer_pairs.append(pairs)
            self.thetas.append(Parameter(rng.standard_normal(len(pairs)) * init_scale))
            self.phis.append(Parameter(rng.standard_normal(len(pairs)) * init_scale))

    def forward(self, x: Tensor) -> Tensor:
        re, im = x[..., : self.m], x[..., self.m :]
        for pairs, theta, phi in zip(self.layer_pairs, self.thetas, self.phis):
            a, b = pairs[:, 0], pairs[:, 1]
            untouched = np.setdiff1d(np.arange(self.m), pairs.reshape(-1))
            ar, ai, br, bi = re[..., a], im[..., a], re[..., b], im[..., b]
            cp, sp, ct, st = phi.cos(), phi.sin(), theta.cos(), theta.sin()
            er, ei = cp * ar - sp * ai, sp * ar + cp * ai  # e^{i phi} a
            new_ar, new_ai = ct * er - st * br, ct * ei - st * bi
            new_br, new_bi = st * er + ct * br, st * ei + ct * bi
            order = np.concatenate([a, b, untouched])
            inv = np.argsort(order)
            re = concat([new_ar, new_br, re[..., untouched]], axis=-1)[..., inv]
            im = concat([new_ai, new_bi, im[..., untouched]], axis=-1)[..., inv]
        return concat([re, im], axis=-1)

    def unitary(self) -> np.ndarray:
        """The ``(d/2 x d/2)`` complex unitary currently implemented (for inspection / tests)."""
        u = np.eye(self.m, dtype=np.complex128)
        for pairs, theta, phi in zip(self.layer_pairs, self.thetas, self.phis):
            layer = np.eye(self.m, dtype=np.complex128)
            for (j, k), t, p in zip(pairs, theta.data, phi.data):
                layer[j, j] = np.exp(1j * p) * np.cos(t)
                layer[j, k] = -np.sin(t)
                layer[k, j] = np.exp(1j * p) * np.sin(t)
                layer[k, k] = np.cos(t)
            u = layer @ u
        return u
