"""A trainable PQC as an autograd layer (parameter-shift backward).

Forward: ``y = <O_k>_{psi(x, w)}`` for a data re-uploading circuit.  Backward:
the exact Jacobians ``dy/dx`` and ``dy/dw`` from the parameter-shift rule
(:meth:`ParameterizedQuantumCircuit.jacobian`) are contracted with the incoming
gradient -- no finite differences, no classical surrogate.  The Jacobian is
computed eagerly in the forward pass only when a gradient will be needed
(training mode with grad enabled); inference runs only the ``B`` unshifted
circuits.

Cost per training step: ``B (1 + 2M)`` state-vector simulations of an
``n``-qubit circuit with ``M`` parameter occurrences, executed as one batch.
"""

from __future__ import annotations

import numpy as np

from ..quanta.pqc import ParameterizedQuantumCircuit, reuploading_ansatz, ring_observables
from .autograd import Tensor, custom_op, is_grad_enabled
from .layers import Module, Parameter

__all__ = ["QuantumLayer"]


class QuantumLayer(Module):
    def __init__(self, n_qubits: int, n_layers: int, rng: np.random.Generator) -> None:
        super().__init__()
        circuit, n_in, n_w = reuploading_ansatz(n_qubits, n_layers)
        self.pqc = ParameterizedQuantumCircuit(circuit, n_in, ring_observables(n_qubits))
        self.n_inputs = n_in
        self.n_outputs = self.pqc.n_outputs
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, n_w) * 0.5)

    def forward(self, x: Tensor) -> Tensor:
        need_grad = is_grad_enabled() and (x.requires_grad or self.weights.requires_grad)
        if not need_grad:
            return Tensor(self.pqc.forward(x.data, self.weights.data))
        f, jx, jw = self.pqc.jacobian(x.data, self.weights.data)

        def vjp(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.einsum("bk,bki->bi", g, jx), np.einsum("bk,bkw->w", g, jw)

        return custom_op(f, (x, self.weights), vjp)
