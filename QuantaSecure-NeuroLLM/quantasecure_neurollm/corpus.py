"""Hard-coded text data (all passages are original prose written for this artifact).

* :data:`PUBLIC_CORPUS` -- six short physics abstracts with reference summaries.
  This is the *only* data any model component is trained on (tokenizer merges,
  feature normalisation, encoder, language model).
* :data:`CONFIDENTIAL_DOCUMENT` -- the held-out "confidential preprint" that the
  demo processes through the secure pipeline.  It is never used for training,
  never logged, and never placed on the wire in plaintext.
* :data:`EVALUATION_DOCUMENTS` -- further held-out passages used by the tests to
  check that the pipeline generalises beyond the confidential document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = [
    "TopicDocument",
    "PUBLIC_CORPUS",
    "TOPICS",
    "CONFIDENTIAL_TITLE",
    "CONFIDENTIAL_DOCUMENT",
    "CONFIDENTIAL_TOPIC",
    "EVALUATION_DOCUMENTS",
    "corpus_digest",
    "tokenizer_training_texts",
]


@dataclass(frozen=True)
class TopicDocument:
    topic: str
    title: str
    abstract: str
    summary: str
    keywords: tuple[str, ...]


PUBLIC_CORPUS: tuple[TopicDocument, ...] = (
    TopicDocument(
        topic="quantum error correction",
        title="Logical qubit memory below the surface code threshold",
        abstract=(
            "Quantum error correction protects fragile quantum information by encoding one logical qubit into many "
            "physical qubits. We operate a surface code on a superconducting processor and repeatedly measure "
            "stabilizers to detect bit flip and phase flip errors. A real time decoder converts the stream of "
            "syndrome measurements into corrections within each cycle. Increasing the code distance from three to "
            "five to seven suppresses the logical error rate by roughly a factor of two at each step, which shows "
            "that the physical error rate is below the threshold of the code. Leakage removal and improved readout "
            "reduce correlated errors that would otherwise limit the logical lifetime. The logical memory now "
            "outlives its best physical qubit, a key milestone toward fault tolerant quantum computation."
        ),
        summary=(
            "Finding: the surface code logical error rate falls as the code distance grows, so the qubits operate "
            "below threshold. Method: repeated stabilizer measurements with a real time decoder. Implication: a "
            "path to fault tolerant quantum computing."
        ),
        keywords=("surface", "code", "logical", "error", "qubit", "threshold", "decoder", "fault", "tolerant", "stabilizer"),
    ),
    TopicDocument(
        topic="superconductivity",
        title="Near room temperature superconductivity in compressed hydrides",
        abstract=(
            "Hydrogen rich compounds become superconducting at remarkably high temperatures when squeezed to "
            "megabar pressures in a diamond anvil cell. We synthesize a lanthanum hydride by laser heating and track "
            "its electrical resistance while cooling. Resistance drops sharply to zero near two hundred fifty "
            "kelvin, and the transition shifts to lower temperature under an applied magnetic field, as expected "
            "for a superconductor. Magnetic susceptibility measurements show flux expulsion, the Meissner effect. "
            "Density functional calculations attribute the high critical temperature to strong electron phonon "
            "coupling from light hydrogen vibrations in a clathrate lattice. The challenge now is to stabilize such "
            "phases at lower pressure."
        ),
        summary=(
            "Finding: a compressed lanthanum hydride superconducts near two hundred fifty kelvin. Method: resistance "
            "and magnetic susceptibility in a diamond anvil cell. Implication: electron phonon coupling in hydrides "
            "may enable room temperature superconductors."
        ),
        keywords=("hydride", "superconduct", "pressure", "kelvin", "resistance", "phonon", "diamond", "temperature"),
    ),
    TopicDocument(
        topic="gravitational waves",
        title="Gravitational waves and light from a binary neutron star merger",
        abstract=(
            "Two neutron stars spiraling together emit gravitational waves whose frequency and amplitude rise until "
            "the stars merge. Laser interferometer detectors recorded such a chirp lasting about one minute, and "
            "matched filtering against relativistic waveform templates measured the masses and the distance of the "
            "source. Less than two seconds after the merger a short gamma ray burst was observed, and telescopes "
            "then located a kilonova glowing from the radioactive decay of freshly made heavy elements. Comparing "
            "the arrival times of gravity and light constrains the speed of gravitational waves to equal the speed "
            "of light. The event opened multi messenger astronomy and gave an independent measurement of the Hubble "
            "constant."
        ),
        summary=(
            "Finding: gravitational waves from merging neutron stars arrived with a gamma ray burst and a kilonova. "
            "Method: interferometer detection with matched filtering and telescope follow up. Implication: multi "
            "messenger astronomy and a new Hubble constant estimate."
        ),
        keywords=("gravitational", "neutron", "merger", "kilonova", "interferometer", "gamma", "waves", "light"),
    ),
    TopicDocument(
        topic="topological matter",
        title="Majorana zero modes at the edge of a topological superconductor",
        abstract=(
            "Topological insulators conduct electricity only along their surfaces through spin momentum locked "
            "states protected by time reversal symmetry. When such a surface is placed in contact with a "
            "conventional superconductor, the proximity effect can create a topological superconductor. We "
            "fabricate semiconductor nanowires with strong spin orbit coupling on an aluminum shell and apply a "
            "magnetic field. Tunneling spectroscopy reveals a zero bias conductance peak that persists over a wide "
            "range of field and gate voltage, consistent with Majorana zero modes bound to the wire ends. Because "
            "these modes obey non abelian braiding statistics, they could store quantum information nonlocally and "
            "protect it from local noise."
        ),
        summary=(
            "Finding: a stable zero bias conductance peak signals Majorana zero modes at the nanowire ends. Method: "
            "tunneling spectroscopy of proximitized nanowires in a magnetic field. Implication: topologically "
            "protected qubits based on braiding."
        ),
        keywords=("majorana", "topological", "nanowire", "braiding", "conductance", "tunneling", "zero", "modes"),
    ),
    TopicDocument(
        topic="dark matter",
        title="Searching for weakly interacting dark matter with liquid xenon",
        abstract=(
            "Most of the matter in the universe is invisible dark matter, and weakly interacting massive particles "
            "remain a leading candidate. We operate a dual phase liquid xenon time projection chamber deep "
            "underground to shield it from cosmic rays. A particle scattering off a xenon nucleus produces prompt "
            "scintillation light and ionization electrons that are drifted to a gas layer and amplified. The ratio "
            "of the two signals separates nuclear recoils from electron recoils caused by background radioactivity. "
            "After a year of exposure no excess above the expected background was observed, setting the strongest "
            "limits yet on the spin independent scattering cross section for masses near thirty giga electron volts."
        ),
        summary=(
            "Finding: no dark matter signal above background, giving the tightest cross section limits near thirty "
            "giga electron volts. Method: a deep underground liquid xenon time projection chamber. Implication: "
            "weakly interacting dark matter must couple even more feebly."
        ),
        keywords=("dark", "matter", "xenon", "background", "cross", "section", "underground", "recoil"),
    ),
    TopicDocument(
        topic="quantum cryptography",
        title="Satellite quantum key distribution across intercontinental distances",
        abstract=(
            "Quantum key distribution lets two parties grow a shared secret key whose security follows from the "
            "laws of quantum mechanics rather than from computational assumptions. Any eavesdropper who measures "
            "the transmitted photons disturbs their polarization and raises the quantum bit error rate. We send "
            "decoy state photons from a low earth orbit satellite to ground stations separated by thousands of "
            "kilometers. Error correction and privacy amplification distill a secure key from the raw detections, "
            "and the satellite acts as a trusted relay to join keys between stations. The demonstration shows that "
            "free space links can overcome the exponential photon loss of optical fiber and points toward a global "
            "quantum internet."
        ),
        summary=(
            "Finding: a satellite distributed secure quantum keys between distant ground stations. Method: decoy "
            "state photons with error correction and privacy amplification. Implication: free space links toward a "
            "global quantum internet."
        ),
        keywords=("key", "distribution", "satellite", "photons", "secure", "privacy", "decoy", "internet"),
    ),
)

TOPICS: tuple[str, ...] = tuple(d.topic for d in PUBLIC_CORPUS)

CONFIDENTIAL_TITLE = "CONFIDENTIAL PREPRINT -- Real-time decoding of a distance-7 surface code"
CONFIDENTIAL_TOPIC = "quantum error correction"
CONFIDENTIAL_DOCUMENT = (
    "We report a logical qubit encoded in a distance seven surface code on a superconducting quantum processor with "
    "seventy two physical qubits. Each error correction cycle measures every stabilizer, and a neural network "
    "decoder running on dedicated hardware returns corrections in under one microsecond, keeping pace with the "
    "cycle time. Compared with a distance five code the logical error rate per cycle is suppressed by a factor of "
    "two point one, confirming operation below the surface code threshold. We identify rare bursts of correlated "
    "errors caused by high energy events and mitigate leakage with dedicated reset gates. The logical qubit "
    "lifetime exceeds that of the best physical qubit by more than a factor of two. These results indicate that "
    "scaling the code distance will continue to reduce logical errors, a prerequisite for fault tolerant quantum "
    "algorithms."
)

EVALUATION_DOCUMENTS: tuple[tuple[str, str], ...] = (
    (
        "superconductivity",
        "We compress yttrium hydride above one hundred fifty gigapascal in a diamond anvil cell and observe the "
        "electrical resistance vanish below two hundred twenty kelvin. Applying a magnetic field lowers the "
        "transition temperature, and the critical field follows the expected superconducting dependence. "
        "Calculations show strong electron phonon coupling from hydrogen vibrations, pointing toward light element "
        "superconductors at higher temperature.",
    ),
    (
        "gravitational waves",
        "A gravitational wave chirp from the inspiral of two neutron stars was detected by the interferometer "
        "network and localized on the sky. Matched filtering estimated component masses near one point four solar "
        "masses. Telescopes found the optical kilonova in a nearby galaxy, and the coincident gamma ray burst "
        "confirmed that gravity and light travel at the same speed.",
    ),
    (
        "quantum cryptography",
        "We implement quantum key distribution over a fiber network and a free space link using decoy state "
        "photons. The quantum bit error rate stays low enough for error correction and privacy amplification to "
        "distill secret keys between distant nodes, showing a scalable route to a quantum internet secured by "
        "physics.",
    ),
)


def tokenizer_training_texts() -> list[str]:
    """Text the BPE merges are learned from (public corpus only)."""
    out: list[str] = []
    for d in PUBLIC_CORPUS:
        out += [d.title, d.abstract, d.summary]
    return out


def corpus_digest() -> bytes:
    h = hashlib.sha3_256()
    for d in PUBLIC_CORPUS:
        for part in (d.topic, d.title, d.abstract, d.summary):
            h.update(part.encode() + b"\x00")
    return h.digest()
