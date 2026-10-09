"""Correctness tests for the quantum simulator (state vector, density matrix, MPS, PQC)."""

from __future__ import annotations

import numpy as np
import pytest

from quantasecure_neurollm.quanta import (
    GATES,
    MPS,
    DensityMatrix,
    MemoryBudget,
    NoiseModel,
    ParameterizedQuantumCircuit,
    PauliString,
    QuantumCircuit,
    QuantumFeatureExtractor,
    QuantumFeatureMap,
    StateVector,
    norm_error_bound,
    reuploading_ansatz,
    ring_observables,
    run_density,
    run_statevector,
    unitarity_error,
)
from quantasecure_neurollm.quanta import kernels, noise
from quantasecure_neurollm.quanta.gates import TOFFOLI, X, Y, Z
from quantasecure_neurollm.quanta.measurement import expectation_density
from quantasecure_neurollm.quanta.mps import toffoli_decomposition


def random_state(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return v / np.linalg.norm(v)


def random_circuit(n: int, depth: int, rng: np.random.Generator) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for _ in range(depth):
        for q in range(n):
            qc.u3(q, *rng.uniform(-np.pi, np.pi, 3))
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.rzz(q, q + 1, float(rng.uniform(-np.pi, np.pi)))
        a, b = rng.choice(n, 2, replace=False)
        qc.cz(int(a), int(b))
    return qc


def dense_operator(u: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """Reference embedding of a gate into the full Hilbert space (slow, for tests)."""
    dim = 2**n
    out = np.zeros((dim, dim), dtype=complex)
    k = len(qubits)
    for col in range(dim):
        bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
        sub_in = sum(bits[q] << (k - 1 - i) for i, q in enumerate(qubits))
        for sub_out in range(2**k):
            amp = u[sub_out, sub_in]
            if amp == 0:
                continue
            nb = list(bits)
            for i, q in enumerate(qubits):
                nb[q] = (sub_out >> (k - 1 - i)) & 1
            row = sum(b << (n - 1 - q) for q, b in enumerate(nb))
            out[row, col] += amp
    return out


# --------------------------------------------------------------------------- gates


def test_all_gates_are_unitary() -> None:
    rng = np.random.default_rng(0)
    for name, spec in GATES.items():
        args = rng.uniform(-np.pi, np.pi, spec.n_params)
        u = spec.matrix(*args)
        assert unitarity_error(u) < 1e-14, name
        batched = spec.matrix(*[np.full(5, a) for a in args]) if spec.n_params else None
        if batched is not None:
            assert batched.shape == (5,) + u.shape


@pytest.mark.parametrize("qubits", [[0, 2], [2, 0], [1, 3], [3, 1]])
def test_kernel_matches_dense_embedding(qubits: list[int]) -> None:
    rng = np.random.default_rng(1)
    n = 4
    psi = random_state(n, rng)
    m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    u, _ = np.linalg.qr(m)
    out = kernels.apply_gate(psi[None], u, qubits, n)[0]
    np.testing.assert_allclose(out, dense_operator(u, qubits, n) @ psi, atol=1e-13)


@pytest.mark.parametrize("name,qubits", [("rz", [2]), ("cz", [3, 0]), ("rzz", [0, 3]), ("rzz", [3, 1]), ("t", [1])])
def test_diagonal_fast_path_matches_dense(name: str, qubits: list[int]) -> None:
    rng = np.random.default_rng(2)
    n = 4
    psi = random_state(n, rng)[None]
    spec = GATES[name]
    u = spec.matrix(*rng.uniform(-3, 3, spec.n_params))
    fast = kernels.apply_diagonal(psi, np.diagonal(u), qubits, n)
    slow = kernels.apply_gate(psi, u, qubits, n)
    np.testing.assert_allclose(fast, slow, atol=1e-14)


def test_cnot_convention_control_first() -> None:
    sv = StateVector.from_bitstring("10").cnot(0, 1)
    assert sv.probabilities()[0b11] == pytest.approx(1.0)
    sv = StateVector.from_bitstring("01").cnot(0, 1)
    assert sv.probabilities()[0b01] == pytest.approx(1.0)


def test_toffoli_truth_table_and_decomposition() -> None:
    for bits in range(8):
        sv = StateVector.from_bitstring(format(bits, "03b")).toffoli(0, 1, 2)
        expected = bits ^ 1 if (bits >> 1) == 0b11 else bits
        assert sv.probabilities()[expected] == pytest.approx(1.0)
    qc = QuantumCircuit(3)
    for g, qs in toffoli_decomposition(0, 1, 2):
        qc.append(g, qs)
    u = qc.to_unitary()
    np.testing.assert_allclose(u, TOFFOLI, atol=1e-12)


def test_circuit_unitary_and_inverse() -> None:
    rng = np.random.default_rng(3)
    qc = random_circuit(4, 3, rng)
    u = qc.to_unitary()
    assert unitarity_error(u) < 1e-12
    ui = qc.inverse().to_unitary()
    np.testing.assert_allclose(ui @ u, np.eye(16), atol=1e-12)
    # Reference: multiply dense embeddings gate by gate.
    ref = np.eye(16, dtype=complex)
    bound, _ = qc.bind()
    for op in bound:
        ref = dense_operator(op.matrix, list(op.qubits), 4) @ ref
    np.testing.assert_allclose(u, ref, atol=1e-12)


# ----------------------------------------------------------------- state vector


def test_bell_state_properties() -> None:
    sv = StateVector(2).h(0).cnot(0, 1)
    np.testing.assert_allclose(sv.probabilities(), [0.5, 0, 0, 0.5], atol=1e-15)
    assert sv.entanglement_entropy([0]) == pytest.approx(1.0, abs=1e-12)
    assert sv.expectation("ZZ") == pytest.approx(1.0)
    assert sv.expectation("XX") == pytest.approx(1.0)
    assert sv.expectation("YY") == pytest.approx(-1.0)
    assert sv.expectation("ZI") == pytest.approx(0.0, abs=1e-15)


def test_ghz_16_qubits() -> None:
    n = 16
    sv = StateVector(n).h(0)
    for q in range(n - 1):
        sv.cnot(q, q + 1)
    p = sv.probabilities()
    assert p[0] == pytest.approx(0.5) and p[-1] == pytest.approx(0.5)
    for cut in (1, 5, 8, 15):
        assert sv.entanglement_entropy(list(range(cut))) == pytest.approx(1.0, abs=1e-10)
    assert sv.norm_drift() < 1e-13


def test_norm_drift_within_a_priori_bound() -> None:
    rng = np.random.default_rng(4)
    qc = random_circuit(10, 40, rng)
    psi, report = run_statevector(qc, renormalize_tol=None)
    assert report.max_norm_drift <= report.error_bound
    assert report.error_bound == pytest.approx(norm_error_bound(len(qc), 2))


def test_sampling_matches_born_rule() -> None:
    rng = np.random.default_rng(5)
    sv = StateVector(3).ry(0, 1.1).cnot(0, 1).ry(2, 0.4)
    shots = 200_000
    counts = np.bincount(sv.sample(shots, rng), minlength=8) / shots
    np.testing.assert_allclose(counts, sv.probabilities(), atol=4e-3)


def test_projective_measurement_collapses() -> None:
    rng = np.random.default_rng(6)
    sv = StateVector(2).h(0).cnot(0, 1)
    out = sv.measure(0, rng)
    assert sv.probabilities()[0b11 if out else 0b00] == pytest.approx(1.0)


def test_batched_parameters_match_individual_runs() -> None:
    qc = QuantumCircuit(2, n_params=2)
    qc.ry(0, qc.p(0)).cx(0, 1).rz(1, qc.p(1, scale=2.0))
    params = np.array([[0.1, 0.2], [1.0, -0.5], [2.5, 3.0]])
    batch, _ = run_statevector(qc, params)
    for i, p in enumerate(params):
        single = StateVector(2).ry(0, p[0]).cnot(0, 1).rz(1, 2 * p[1]).amplitudes
        np.testing.assert_allclose(batch[i], single, atol=1e-14)


def test_memory_budget_refuses_huge_states() -> None:
    with pytest.raises(MemoryError):
        StateVector(30, budget=MemoryBudget(max_bytes=2**20))


# --------------------------------------------------------------- density matrix


def test_noiseless_density_matches_statevector() -> None:
    rng = np.random.default_rng(7)
    qc = random_circuit(4, 3, rng)
    qc.toffoli(3, 0, 2).swap(1, 3).x(2)  # exercise the permutation fast path too
    psi, _ = run_statevector(qc)
    rho = run_density(qc)[0]
    np.testing.assert_allclose(rho, np.outer(psi[0], psi[0].conj()), atol=1e-12)


def test_density_gate_fast_paths_match_two_sided_reference() -> None:
    rng = np.random.default_rng(70)
    n = 3
    psi = random_state(n, rng)
    rho = np.outer(psi, psi.conj())[None]
    for name, qubits in (("cx", [2, 0]), ("ccx", [1, 2, 0]), ("swap", [0, 2]), ("rzz", [2, 1]), ("u3", [1]), ("ryy", [0, 2])):
        spec = GATES[name]
        u = spec.matrix(*rng.uniform(-3, 3, spec.n_params))
        full = dense_operator(u, qubits, n)
        expected = full @ rho[0] @ full.conj().T
        np.testing.assert_allclose(kernels.apply_gate_dm(rho, u, qubits, n, spec.diagonal)[0], expected, atol=1e-13)
    assert kernels.permutation_indices(GATES["h"].matrix(), [0], n) is None


@pytest.mark.parametrize("q", [0, 2])
def test_closed_form_channels_match_kraus(q: int) -> None:
    rng = np.random.default_rng(8)
    n = 3
    psi = random_state(n, rng)
    rho = np.outer(psi, psi.conj())[None]
    for p in (0.0, 0.1, 0.37):
        a = kernels.depolarize_dm(rho, p, [q], n)
        b = kernels.apply_kraus_dm(rho, noise.depolarizing_kraus(p), [q], n)
        np.testing.assert_allclose(a, b, atol=1e-13)
        a = kernels.amplitude_damp_dm(rho, p, q, n)
        b = kernels.apply_kraus_dm(rho, noise.amplitude_damping_kraus(p), [q], n)
        np.testing.assert_allclose(a, b, atol=1e-13)
        a = kernels.phase_damp_dm(rho, p, q, n)
        b = kernels.apply_kraus_dm(rho, noise.phase_damping_kraus(p), [q], n)
        np.testing.assert_allclose(a, b, atol=1e-13)
        fused = kernels.damp_dm(rho, p, p / 2, q, n)
        seq = kernels.apply_kraus_dm(kernels.apply_kraus_dm(rho, noise.amplitude_damping_kraus(p), [q], n), noise.phase_damping_kraus(p / 2), [q], n)
        np.testing.assert_allclose(fused, seq, atol=1e-13)
    before = rho.copy()
    kernels.depolarize_dm(rho, 0.2, [q], n)
    kernels.damp_dm(rho, 0.2, 0.1, q, n)
    np.testing.assert_array_equal(rho, before)  # out-of-place by default


def test_two_qubit_depolarizing_closed_form() -> None:
    rng = np.random.default_rng(9)
    n = 3
    psi = random_state(n, rng)
    rho = np.outer(psi, psi.conj())[None]
    p = 0.2
    paulis = [np.eye(2), X, Y, Z]
    kraus = [np.sqrt(1 - 15 * p / 16) * np.eye(4)]
    kraus += [np.sqrt(p / 16) * np.kron(a, b) for i, a in enumerate(paulis) for j, b in enumerate(paulis) if i or j]
    assert noise.is_trace_preserving(kraus)
    np.testing.assert_allclose(
        kernels.depolarize_dm(rho, p, [2, 0], n), kernels.apply_kraus_dm(rho, kraus, [2, 0], n), atol=1e-13
    )


def test_kraus_sets_trace_preserving() -> None:
    for fn in (noise.depolarizing_kraus, noise.amplitude_damping_kraus, noise.phase_damping_kraus, noise.bit_flip_kraus):
        assert noise.is_trace_preserving(fn(0.23))


def test_noisy_circuit_gives_valid_state_and_lower_purity() -> None:
    rng = np.random.default_rng(10)
    qc = random_circuit(4, 3, rng)
    nm = NoiseModel(0.01, 0.03, 0.02, 0.01)
    dm = DensityMatrix(4).run(qc, noise=nm)
    assert dm.is_valid()
    assert dm.purity() < 0.999
    ideal = StateVector(4).run(qc)
    assert 0.5 < dm.fidelity_with(ideal) < 1.0


def test_amplitude_damping_relaxes_to_ground_state() -> None:
    dm = DensityMatrix(1).apply("x", [0])
    for _ in range(200):
        dm.amplitude_damp(0.05, 0)
    assert dm.probabilities()[0] == pytest.approx(1.0, abs=1e-4)


def test_density_expectation_non_diagonal() -> None:
    rng = np.random.default_rng(11)
    n = 3
    psi = random_state(n, rng)
    rho = np.outer(psi, psi.conj())
    for label in ("XIY", "ZXZ", "YYI"):
        p = PauliString.from_label(label)
        mats = {"I": np.eye(2), "X": X, "Y": Y, "Z": Z}
        full = mats[label[0]]
        for c in label[1:]:
            full = np.kron(full, mats[c])
        assert expectation_density(rho[None], p, n)[0] == pytest.approx(np.real(np.trace(rho @ full)), abs=1e-13)


def test_mixed_state_entropy_and_mutual_information() -> None:
    dm = DensityMatrix.from_statevector(StateVector(2).h(0).cnot(0, 1))
    assert dm.subsystem_entropy([0]) == pytest.approx(1.0, abs=1e-12)
    from quantasecure_neurollm.quanta import mutual_information

    assert mutual_information(dm.matrix, [0], [1], 2) == pytest.approx(2.0, abs=1e-10)
    assert DensityMatrix.maximally_mixed(3).von_neumann_entropy() == pytest.approx(3.0, abs=1e-12)


# ------------------------------------------------------------------------- MPS


def test_mps_matches_statevector_without_truncation() -> None:
    rng = np.random.default_rng(12)
    n = 8
    qc = random_circuit(n, 4, rng)
    qc.toffoli(0, 3, 6)
    mps = MPS(n, max_bond=256, cutoff=0.0).run(qc)
    sv = StateVector(n).run(qc)
    assert abs(np.vdot(mps.to_statevector(), sv.amplitudes)) ** 2 == pytest.approx(1.0, abs=1e-10)
    for label in ("ZIIIIIIZ", "IXYIIIII", "IIIZZIII"):
        assert mps.expectation(label) == pytest.approx(sv.expectation(label), abs=1e-10)
    for cut in (1, 4, 7):
        assert mps.bond_entropy(cut) == pytest.approx(sv.entanglement_entropy(list(range(cut))), abs=1e-9)
    assert mps.truncation_error == 0.0


def test_mps_sampling_distribution() -> None:
    rng = np.random.default_rng(13)
    n = 5
    qc = random_circuit(n, 2, rng)
    mps = MPS(n).run(qc)
    shots = 100_000
    freq = np.bincount(mps.sample(shots, rng), minlength=2**n) / shots
    np.testing.assert_allclose(freq, StateVector(n).run(qc).probabilities(), atol=6e-3)


def test_mps_scales_to_40_qubit_ghz() -> None:
    n = 40
    mps = MPS(n, max_bond=4)
    mps.apply("h", [0])
    for q in range(n - 1):
        mps.apply("cx", [q, q + 1])
    assert max(mps.bond_dimensions()) == 2
    assert mps.bond_entropy(20) == pytest.approx(1.0, abs=1e-10)
    assert mps.expectation(PauliString.z(0, 39)) == pytest.approx(1.0, abs=1e-12)
    samples = mps.sample(200, np.random.default_rng(0))
    assert set(np.unique(samples)) <= {0, 2**n - 1}


def test_mps_truncation_is_reported() -> None:
    rng = np.random.default_rng(14)
    n = 8
    qc = random_circuit(n, 6, rng)
    mps = MPS(n, max_bond=4).run(qc)
    exact = StateVector(n).run(qc).amplitudes
    fid = abs(np.vdot(mps.to_statevector(), exact)) ** 2
    assert mps.truncation_error > 0
    assert fid >= 1 - 1.5 * mps.truncation_error - 1e-9  # first-order fidelity bound


# ------------------------------------------------------------------------- PQC


def test_parameter_shift_matches_finite_differences() -> None:
    rng = np.random.default_rng(15)
    qc, n_in, n_w = reuploading_ansatz(3, 2)
    pqc = ParameterizedQuantumCircuit(qc, n_in, ring_observables(3))
    x = rng.uniform(-1, 1, (4, n_in))
    w = rng.uniform(-np.pi, np.pi, n_w)
    f, jx, jw = pqc.jacobian(x, w)
    np.testing.assert_allclose(f, pqc.forward(x, w), atol=1e-14)
    fx, fw = pqc.finite_difference_jacobian(x, w)
    np.testing.assert_allclose(jx, fx, atol=1e-8)
    np.testing.assert_allclose(jw, fw, atol=1e-8)


def test_parameter_shift_with_shared_and_scaled_parameters() -> None:
    qc = QuantumCircuit(2, n_params=2)
    qc.rx(0, qc.p(0)).ry(1, qc.p(0, scale=-2.0)).rzz(0, 1, qc.p(1, scale=0.5)).u3(0, qc.p(1), 0.3, qc.p(0))
    qc.rxx(0, 1, qc.p(1)).phase(1, qc.p(0, scale=3.0))
    pqc = ParameterizedQuantumCircuit(qc, 1, [PauliString.from_label("XZ"), PauliString.from_label("YY")])
    x, w = np.array([[0.7], [-1.3]]), np.array([0.4])
    _, jx, jw = pqc.jacobian(x, w)
    fx, fw = pqc.finite_difference_jacobian(x, w)
    np.testing.assert_allclose(jx, fx, atol=1e-8)
    np.testing.assert_allclose(jw, fw, atol=1e-8)


def test_shot_based_parameter_shift_is_unbiased() -> None:
    rng = np.random.default_rng(16)
    qc, n_in, n_w = reuploading_ansatz(2, 1)
    pqc = ParameterizedQuantumCircuit(qc, n_in, ring_observables(2))
    x = np.array([[0.3, -0.8]])
    w = rng.uniform(-1, 1, n_w)
    _, jx_exact, _ = pqc.jacobian(x, w)
    estimates = [pqc.jacobian(x, w, shots=4000, rng=rng)[1] for _ in range(20)]
    np.testing.assert_allclose(np.mean(estimates, axis=0), jx_exact, atol=0.02)


def test_pqc_counts_every_parameter_occurrence() -> None:
    qc = QuantumCircuit(2, n_params=1)
    qc.rx(0, qc.p(0)).rzz(0, 1, qc.p(0, scale=2.0)).h(1)
    pqc = ParameterizedQuantumCircuit(qc, 1, [PauliString.z(0)])
    assert pqc.n_occurrences == 2
    assert pqc.n_weights == 0


# ------------------------------------------------------------------ feature map


def test_feature_extractor_shapes_and_determinism() -> None:
    vocab = 300
    fmap = QuantumFeatureMap(6, 2, vocab)
    ext = QuantumFeatureExtractor(fmap, NoiseModel(0.002, 0.01, 0.003, 0.002, 0.01), chunk_tokens=20, chunk_stride=16)
    rng = np.random.default_rng(17)
    docs = [rng.integers(0, vocab, 90), rng.integers(0, vocab, 15)]
    out = ext.extract_many(docs, seed=1)
    assert out[0].chunk_features.shape[1] == ext.chunk_dim
    assert out[0].global_features.shape == (ext.global_dim,)
    assert out[1].chunk_features.shape[0] == 1  # short doc -> only the global window
    again = ext.extract_many(docs, seed=1)
    np.testing.assert_array_equal(out[0].chunk_features, again[0].chunk_features)
    assert 0.0 < out[0].stats["purity"] < 1.0
    assert out[0].stats["max_norm_drift"] <= out[0].stats["norm_error_bound"]
    names_chunk, names_glob = ext.feature_names()
    assert len(names_chunk) == ext.chunk_dim and len(names_glob) == ext.global_dim


def test_feature_map_preserves_similarity() -> None:
    vocab = 400
    fmap = QuantumFeatureMap(6, 2, vocab)
    ext = QuantumFeatureExtractor(fmap, NoiseModel())
    rng = np.random.default_rng(18)
    base = rng.integers(0, vocab, 60)
    near = base.copy()
    near[:6] = rng.integers(0, vocab, 6)
    far = rng.integers(0, vocab, 60)
    f = [x.chunk_features[0] for x in ext.extract_many([base, near, far])]
    assert np.linalg.norm(f[0] - f[1]) < np.linalg.norm(f[0] - f[2])
