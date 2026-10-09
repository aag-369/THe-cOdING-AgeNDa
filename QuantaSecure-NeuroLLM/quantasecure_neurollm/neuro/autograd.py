"""A small reverse-mode automatic-differentiation engine on NumPy arrays.

Every differentiable operation records its parents and a closure that maps the
output gradient to parent gradients (a vector-Jacobian product).
:meth:`Tensor.backward` topologically sorts the graph (iteratively, so deep
graphs do not hit Python's recursion limit) and runs the closures in reverse.

Design notes
------------
* float64 throughout: the models are small, and double precision makes the
  finite-difference gradient checks in the test-suite tight (``~1e-8``).
* Broadcasting is supported for element-wise ops; gradients are summed back to
  the operand shape by :func:`_unbroadcast`.
* Softmax, log-softmax, cross-entropy and SiLU are *fused* ops with analytic
  backward passes -- both faster and more stable than composing them (e.g.
  ``softmax`` backward ``y * (g - <g, y>)`` never forms the Jacobian).
* :func:`custom_op` lets non-NumPy computations join the graph; the quantum
  layer uses it to inject parameter-shift Jacobians.
* :class:`no_grad` disables graph construction for inference.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence, Union

import numpy as np

from ..utils.mathx import logsumexp, stable_softmax

__all__ = [
    "Tensor",
    "no_grad",
    "is_grad_enabled",
    "custom_op",
    "concat",
    "stack",
    "softmax",
    "log_softmax",
    "cross_entropy",
    "where_const",
]

ArrayLike = Union[np.ndarray, float, int, "Tensor"]
_GRAD_ENABLED = True


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction inside the ``with`` block."""
    global _GRAD_ENABLED
    prev = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def _unbroadcast(g: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``g`` over the axes that broadcasting added or stretched to reach ``shape``."""
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    """An ``ndarray`` with an optional gradient and a link to the op that produced it."""

    __slots__ = ("data", "grad", "requires_grad", "_parents", "_backward", "name")
    __array_priority__ = 1000  # make ``ndarray op Tensor`` dispatch to Tensor's reflected ops

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _parents: tuple["Tensor", ...] = (),
        _backward: Optional[Callable[[], None]] = None,
        name: Optional[str] = None,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        self.data = arr if arr.dtype == np.float64 else arr.astype(np.float64)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = _backward
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{', name=' + self.name if self.name else ''})"

    def _accum(self, g: np.ndarray) -> None:
        # Gradients are never mutated in place anywhere in the code base, so aliasing the
        # incoming array (instead of copying it) is safe and saves a full pass over memory.
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Back-propagate from this tensor (a scalar unless ``grad`` is given)."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() on a non-scalar needs an explicit gradient")
            grad = np.ones_like(self.data)
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = grad if self.grad is None else self.grad + grad
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward()

    # --------------------------------------------------------------- arithmetic
    def __add__(self, other: ArrayLike) -> "Tensor":
        o = _t(other)
        out = _make(self.data + o.data, (self, o))

        def bw() -> None:
            self._accum(_unbroadcast(out.grad, self.shape))
            o._accum(_unbroadcast(out.grad, o.shape))

        return _attach(out, bw)

    __radd__ = __add__

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-_t(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return _t(other) + (-self)

    def __neg__(self) -> "Tensor":
        out = _make(-self.data, (self,))
        return _attach(out, lambda: self._accum(-out.grad))

    def __mul__(self, other: ArrayLike) -> "Tensor":
        o = _t(other)
        out = _make(self.data * o.data, (self, o))

        def bw() -> None:
            self._accum(_unbroadcast(out.grad * o.data, self.shape))
            o._accum(_unbroadcast(out.grad * self.data, o.shape))

        return _attach(out, bw)

    __rmul__ = __mul__

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        o = _t(other)
        out = _make(self.data / o.data, (self, o))

        def bw() -> None:
            self._accum(_unbroadcast(out.grad / o.data, self.shape))
            o._accum(_unbroadcast(-out.grad * self.data / (o.data**2), o.shape))

        return _attach(out, bw)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return _t(other) / self

    def __pow__(self, p: float) -> "Tensor":
        out = _make(self.data**p, (self,))
        return _attach(out, lambda: self._accum(out.grad * p * self.data ** (p - 1)))

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        o = _t(other)
        if self.ndim < 2 or o.ndim < 2:
            raise ValueError("matmul operands must be at least 2-D")
        out = _make(self.data @ o.data, (self, o))

        def bw() -> None:
            g = out.grad
            self._accum(_unbroadcast(g @ np.swapaxes(o.data, -1, -2), self.shape))
            o._accum(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, o.shape))

        return _attach(out, bw)

    def __getitem__(self, idx: object) -> "Tensor":
        out = _make(self.data[idx], (self,))
        parts = idx if isinstance(idx, tuple) else (idx,)
        basic = all(p is None or p is Ellipsis or isinstance(p, (int, slice)) for p in parts)

        def bw() -> None:
            g = np.zeros_like(self.data)
            if basic:
                g[idx] = out.grad  # basic indexing selects each element at most once
            else:
                np.add.at(g, idx, out.grad)  # scatter-add handles repeated indices (embeddings)
            self._accum(g)

        return _attach(out, bw)

    # --------------------------------------------------------------- reductions
    def sum(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        out = _make(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.shape).copy())

        return _attach(out, bw)

    def mean(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        n = self.data.size if axis is None else int(np.prod([self.shape[a] for a in np.atleast_1d(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------- shapes
    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = _make(self.data.reshape(shape), (self,))
        return _attach(out, lambda: self._accum(out.grad.reshape(self.shape)))

    def transpose(self, *axes: int) -> "Tensor":
        axes_t = axes if axes else tuple(reversed(range(self.ndim)))
        inv = np.argsort(axes_t)
        out = _make(np.transpose(self.data, axes_t), (self,))
        return _attach(out, lambda: self._accum(np.transpose(out.grad, inv)))

    def swapaxes(self, a: int, b: int) -> "Tensor":
        out = _make(np.swapaxes(self.data, a, b), (self,))
        return _attach(out, lambda: self._accum(np.swapaxes(out.grad, a, b)))

    # ---------------------------------------------------------------- unary ops
    def exp(self) -> "Tensor":
        out = _make(np.exp(self.data), (self,))
        return _attach(out, lambda: self._accum(out.grad * out.data))

    def log(self) -> "Tensor":
        out = _make(np.log(self.data), (self,))
        return _attach(out, lambda: self._accum(out.grad / self.data))

    def sqrt(self) -> "Tensor":
        out = _make(np.sqrt(self.data), (self,))
        return _attach(out, lambda: self._accum(out.grad * 0.5 / out.data))

    def tanh(self) -> "Tensor":
        out = _make(np.tanh(self.data), (self,))
        return _attach(out, lambda: self._accum(out.grad * (1.0 - out.data**2)))

    def sigmoid(self) -> "Tensor":
        out = _make(0.5 * (1.0 + np.tanh(0.5 * self.data)), (self,))  # overflow-free logistic
        return _attach(out, lambda: self._accum(out.grad * out.data * (1.0 - out.data)))

    def relu(self) -> "Tensor":
        out = _make(np.maximum(self.data, 0.0), (self,))
        return _attach(out, lambda: self._accum(out.grad * (self.data > 0)))

    def silu(self) -> "Tensor":
        s = 0.5 * (1.0 + np.tanh(0.5 * self.data))
        out = _make(self.data * s, (self,))
        return _attach(out, lambda: self._accum(out.grad * (s + self.data * s * (1.0 - s))))

    def sin(self) -> "Tensor":
        out = _make(np.sin(self.data), (self,))
        return _attach(out, lambda: self._accum(out.grad * np.cos(self.data)))

    def cos(self) -> "Tensor":
        out = _make(np.cos(self.data), (self,))
        return _attach(out, lambda: self._accum(-out.grad * np.sin(self.data)))


def _t(x: ArrayLike) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data: np.ndarray, parents: tuple[Tensor, ...]) -> Tensor:
    track = _GRAD_ENABLED and any(p.requires_grad for p in parents)
    return Tensor(data, requires_grad=track, _parents=parents if track else ())


def _attach(out: Tensor, bw: Callable[[], None]) -> Tensor:
    if out.requires_grad:
        out._backward = bw
    return out


def custom_op(data: np.ndarray, parents: Sequence[Tensor], vjp: Callable[[np.ndarray], Sequence[Optional[np.ndarray]]]) -> Tensor:
    """Insert an externally-computed op: ``vjp(out_grad)`` returns one gradient per parent."""
    out = _make(np.asarray(data, dtype=np.float64), tuple(parents))

    def bw() -> None:
        for p, g in zip(parents, vjp(out.grad)):
            if g is not None:
                p._accum(g)

    return _attach(out, bw)


def concat(tensors: Sequence[Tensor], axis: int = -1) -> Tensor:
    ts = [_t(t) for t in tensors]
    out = _make(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]

    def bw() -> None:
        for t, g in zip(ts, np.split(out.grad, sizes, axis=axis)):
            t._accum(g)

    return _attach(out, bw)


def stack(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    ts = [_t(t) for t in tensors]
    out = _make(np.stack([t.data for t in ts], axis=axis), tuple(ts))

    def bw() -> None:
        for i, t in enumerate(ts):
            t._accum(np.take(out.grad, i, axis=axis))

    return _attach(out, bw)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    y = stable_softmax(x.data, axis=axis)
    out = _make(y, (x,))

    def bw() -> None:
        g = out.grad
        x._accum(y * (g - np.sum(g * y, axis=axis, keepdims=True)))

    return _attach(out, bw)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    y = x.data - logsumexp(x.data, axis=axis, keepdims=True)
    out = _make(y, (x,))

    def bw() -> None:
        g = out.grad
        x._accum(g - np.exp(y) * np.sum(g, axis=axis, keepdims=True))

    return _attach(out, bw)


def cross_entropy(logits: Tensor, targets: np.ndarray, weights: Optional[np.ndarray] = None) -> Tensor:
    """Weighted mean of ``-log softmax(logits)[target]`` over the leading axes (fused, stable).

    ``logits`` has shape ``(..., V)``; ``targets`` the leading shape; ``weights``
    (same shape as ``targets``) masks padding positions (weight 0).
    """
    v = logits.shape[-1]
    lg = logits.data.reshape(-1, v)
    tg = np.asarray(targets, dtype=np.int64).reshape(-1)
    w = np.ones(tg.shape) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = max(float(w.sum()), 1e-12)
    lse = logsumexp(lg, axis=-1)
    nll = lse - lg[np.arange(len(tg)), tg]
    out = _make(np.array(float((nll * w).sum() / denom)), (logits,))

    def bw() -> None:
        p = np.exp(lg - lse[:, None])
        p[np.arange(len(tg)), tg] -= 1.0
        logits._accum((p * (w / denom)[:, None] * out.grad).reshape(logits.shape))

    return _attach(out, bw)


def where_const(mask: np.ndarray, x: Tensor, value: float) -> Tensor:
    """``x`` where ``mask`` else the constant ``value`` (no gradient into the constant)."""
    out = _make(np.where(mask, x.data, value), (x,))
    return _attach(out, lambda: x._accum(_unbroadcast(np.where(mask, out.grad, 0.0), x.shape)))
