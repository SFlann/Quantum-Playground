"""
tsim_surface.py — surface-code helpers for the tsim/Stim/PyMatching walkthrough
================================================================================

This module plays the same role for the ``9.tsim`` notebooks that
``7.Modelling-Noisy-Circuits/noisy_engine.jl`` plays for the ITensor series:
it is the shared engine the notebooks ``include`` so the narrative cells stay
clean.  Where the ITensor engine propagates the *exact physical MPS state*, this
engine builds **Stim circuits**, samples them with **tsim** (QuEra's
``bloqade-tsim`` — a Stim-API simulator that natively handles the non-Clifford
``T`` gate via ZX stabiliser-rank decomposition), and decodes the syndromes with
**PyMatching** (MWPM), mirroring the series' MWPM decoder.

Design choices (see the notebook prose for the physics):

* **Single-patch experiments** (§1, §2, §6 of the walkthrough and the memory
  benchmark) delegate geometry to ``stim.Circuit.generated`` — the reference
  rotated-planar surface code, correct at every distance ``d``.  tsim consumes
  the Stim text verbatim.
* **Multi-patch experiments** (§3, §4 — logical Bell and the magic-Bell
  ``H·T·CNOT``) are built here.  Patches are placed **side-by-side** with
  disjoint qubit indices (never interleaved — the layout convention from the
  MPS notebooks, which keeps the non-local cost of a transversal CNOT visible).
  Detectors across the transversal CNOT are generated from the CSS
  stabiliser-flow rule and *validated* by a zero-noise determinism check
  (``assert_deterministic``): with no noise every detector must stay silent, so
  a wrong detector rule is caught immediately rather than reasoned about.
* **Noise** is the **phenomenological** model of ITensor notebook 7 (per-round
  Pauli error rate ``p`` on data qubits + measurement-flip rate ``q``), so the
  two engines can be compared apples-to-apples.  It is also graphlike, so MWPM
  applies without hook-error decomposition.

The public API deliberately echoes the Julia names (``estimate_pL``,
``run_experiment``) so the comparison notebook reads as a like-for-like port.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import stim
import tsim
import pymatching

# --------------------------------------------------------------------------- #
#  0.  Small utilities
# --------------------------------------------------------------------------- #

def _tsim_circuit(text: str) -> "tsim.Circuit":
    """Parse Stim-format ``text`` into a tsim circuit."""
    return tsim.Circuit(text)


def wilson_interval(k: int, n: int, z: float = 1.0) -> Tuple[float, float]:
    """Wilson score interval half-width proxy for a binomial rate ``k/n``.

    Returns ``(phat, stderr)`` where ``stderr`` is the ordinary binomial
    standard error ``sqrt(phat(1-phat)/n)`` (matched to notebook 7's error
    bars).  ``k=0`` or ``k=n`` return a small floor so log-plots behave.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    se = math.sqrt(max(phat * (1.0 - phat), 1e-12) / n)
    return (phat, se)


# --------------------------------------------------------------------------- #
#  1.  Single-patch rotated surface code (delegated to Stim)
# --------------------------------------------------------------------------- #

def memory_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    p: float = 0.0,
    q: Optional[float] = None,
    phenomenological: bool = False,
) -> "stim.Circuit":
    """A rotated-planar surface-code *memory* experiment as a Stim circuit.

    Parameters
    ----------
    distance : code distance ``d`` (odd).
    rounds   : number of syndrome-extraction rounds.
    basis    : ``"Z"`` (prepare/measure ``|0>_L``) or ``"X"`` (``|+>_L``).
    p        : per-round **data** depolarising rate (phenomenological).  Applied
               through Stim's ``after_clifford_depolarization`` /
               ``after_reset_flip_probability`` knobs so the reference detector
               structure is preserved.
    q        : **measurement** flip rate; defaults to ``p`` when ``None``.

    The returned circuit is standard Stim text; feed ``str(circ)`` to tsim.
    """
    basis = basis.upper()
    if basis not in ("Z", "X"):
        raise ValueError("basis must be 'Z' or 'X'")
    if q is None:
        q = p
    # Phenomenological: only per-round data depolarisation + measurement flips, matching
    # ITensor notebook 7's model (data Pauli at rate p, check-flip at rate q).  Full
    # (default): also attach noise to every Clifford and reset (circuit-level), giving a
    # lower threshold.
    ac = ar = 0.0 if phenomenological else p
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=ac,
        after_reset_flip_probability=ar,
        before_measure_flip_probability=q,
        before_round_data_depolarization=p,
    )


# --------------------------------------------------------------------------- #
#  2.  Sampling + decoding (tsim front-end, PyMatching back-end)
# --------------------------------------------------------------------------- #

@dataclass
class ShotResult:
    """Outcome of a decoded batch of shots."""
    detectors: np.ndarray          # (shots, n_det) uint8
    observables: np.ndarray        # (shots, n_obs) uint8
    predictions: np.ndarray        # (shots, n_obs) uint8  (decoder guess)
    logical_errors: np.ndarray     # (shots,) bool  (any observable mispredicted)

    @property
    def num_logical_errors(self) -> int:
        return int(self.logical_errors.sum())

    @property
    def shots(self) -> int:
        return self.detectors.shape[0]


def _matching_from_circuit(circuit: "stim.Circuit") -> "pymatching.Matching":
    """Build a PyMatching decoder from a circuit's decomposed DEM.

    Works for both plain Stim circuits and tsim circuits — the latter expose a
    Stim-compatible ``detector_error_model()``.  We prefer Stim's own DEM when
    the circuit is a ``stim.Circuit`` (it decomposes hook errors), and fall back
    to tsim's DEM for circuits carrying non-Clifford instructions.
    """
    if isinstance(circuit, stim.Circuit):
        dem = circuit.detector_error_model(decompose_errors=True)
    else:
        dem = circuit.detector_error_model()
    return pymatching.Matching.from_detector_error_model(dem)


def run_experiment(
    circuit,
    shots: int,
    matching: Optional["pymatching.Matching"] = None,
    seed: Optional[int] = None,
) -> ShotResult:
    """Sample ``circuit`` with **tsim** and decode with **PyMatching**.

    ``circuit`` may be a ``stim.Circuit`` or a ``tsim.Circuit``; a
    ``stim.Circuit`` is handed to tsim as text so the *same* sampler runs in
    every experiment (Clifford or not).  Returns a :class:`ShotResult`.
    """
    if isinstance(circuit, stim.Circuit):
        matching = matching or _matching_from_circuit(circuit)
        tcirc = _tsim_circuit(str(circuit))
    else:
        matching = matching or _matching_from_circuit(circuit)
        tcirc = circuit

    sampler = tcirc.compile_detector_sampler()
    det, obs = sampler.sample(shots=shots, separate_observables=True)
    det = np.asarray(det, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)

    pred = matching.decode_batch(det)
    pred = np.asarray(pred, dtype=np.uint8)
    logical_errors = np.any(pred != obs, axis=1)
    return ShotResult(det, obs, pred, logical_errors)


def estimate_pL(
    circuit,
    shots: int,
    matching: Optional["pymatching.Matching"] = None,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Monte-Carlo estimate of the logical error rate ``p_L`` with a 1σ bar.

    Mirrors ``estimate_pL`` in ITensor notebook 7: returns ``(p_L, stderr)``
    from a binomial over ``shots`` decoded shots.
    """
    res = run_experiment(circuit, shots, matching=matching, seed=seed)
    return wilson_interval(res.num_logical_errors, res.shots)


# --------------------------------------------------------------------------- #
#  3.  Rotated-code geometry, parsed once from Stim (for multi-patch builds)
# --------------------------------------------------------------------------- #

@dataclass
class PatchSchedule:
    """The stabiliser-extraction schedule of one rotated-SC patch at distance d.

    Parsed once from ``stim.Circuit.generated`` so the multi-patch builder can
    replicate the *exact* reference schedule at any distance, for a patch whose
    qubit indices are shifted by an integer offset and whose coordinates are
    shifted in ``x`` (side-by-side placement).
    """
    d: int
    data: List[int]                       # data qubit indices
    x_anc: List[int]                       # X-type ancilla indices
    z_anc: List[int]                       # Z-type ancilla indices
    anc_order: List[int]                   # order ancillas appear in MR
    cx_steps: List[List[Tuple[int, int]]]  # 4 CX layers, each a list of (c,t)
    coords: Dict[int, Tuple[float, float]] # qubit -> (x,y)
    obs_data: List[int]                    # data qubits in OBSERVABLE_INCLUDE

    @property
    def n_anc(self) -> int:
        return len(self.anc_order)


def _parse_schedule(d: int) -> PatchSchedule:
    """Extract the reference schedule for a distance-``d`` rotated memory-Z code."""
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=1
    )
    text = str(circ)

    coords: Dict[int, Tuple[float, float]] = {}
    for m in re.finditer(r"QUBIT_COORDS\(([^)]+)\)\s+(\d+)", text):
        x, y = (float(v) for v in m.group(1).split(","))
        coords[int(m.group(2))] = (x, y)

    # First round block only (before REPEAT) gives us H set, CX layers, MR order.
    head = text.split("REPEAT")[0]
    lines = [ln.strip() for ln in head.splitlines()]

    reset = next(ln for ln in lines if ln.startswith("R "))
    all_qubits = [int(t) for t in reset.split()[1:]]

    h_line = next(ln for ln in lines if ln.startswith("H "))
    x_anc = [int(t) for t in h_line.split()[1:]]

    mr_line = next(ln for ln in lines if ln.startswith("MR"))
    anc_order = [int(t) for t in mr_line.split()[1:]]
    z_anc = [a for a in anc_order if a not in x_anc]

    m_line = next(ln for ln in lines if re.match(r"^M ", ln))
    data = [int(t) for t in m_line.split()[1:]]

    cx_steps: List[List[Tuple[int, int]]] = []
    for ln in lines:
        if ln.startswith("CX "):
            toks = [int(t) for t in ln.split()[1:]]
            cx_steps.append(list(zip(toks[0::2], toks[1::2])))
    cx_steps = cx_steps[:4]  # one round = 4 CX layers in the rotated code

    # Logical-Z observable support (bottom row of data).
    obs_line = next(ln for ln in text.splitlines() if ln.startswith("OBSERVABLE_INCLUDE"))
    # Rebuild it from coordinates instead of rec[] (row y == 1, the bottom row).
    ymin = min(coords[q][1] for q in data)
    obs_data = sorted((q for q in data if coords[q][1] == ymin), key=lambda q: coords[q][0])

    return PatchSchedule(d, data, x_anc, z_anc, anc_order, cx_steps, coords, obs_data)


# --------------------------------------------------------------------------- #
#  4.  Measurement-record bookkeeping for hand-built circuits
# --------------------------------------------------------------------------- #

class Builder:
    """Accumulates Stim instructions while tracking measurement-record indices.

    Detectors are emitted by naming the *absolute* measurement index of each bit
    we want to XOR; :meth:`detector` converts those to the ``rec[-k]`` offsets
    Stim expects at the current point in the stream.  This removes all manual
    ``rec[]`` counting from the multi-patch builder.
    """

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.nmeas: int = 0

    def emit(self, line: str) -> None:
        self.lines.append(line)

    def coords(self, q: int, x: float, y: float) -> None:
        self.emit(f"QUBIT_COORDS({x}, {y}) {q}")

    def op(self, name: str, qubits: Sequence[int], arg: Optional[float] = None) -> None:
        if not qubits:
            return
        head = name if arg is None else f"{name}({arg})"
        self.emit(head + " " + " ".join(str(q) for q in qubits))

    def tick(self) -> None:
        self.emit("TICK")

    def measure(self, name: str, qubits: Sequence[int], arg: Optional[float] = None) -> List[int]:
        """Emit a measurement op and return the absolute index of each new bit."""
        self.op(name, qubits, arg)
        idxs = list(range(self.nmeas, self.nmeas + len(qubits)))
        self.nmeas += len(qubits)
        return idxs

    def detector(self, abs_indices: Sequence[int], coord: Tuple[float, float, float]) -> None:
        recs = " ".join(f"rec[{i - self.nmeas}]" for i in sorted(abs_indices, reverse=True))
        cx, cy, cz = coord
        self.emit(f"DETECTOR({cx}, {cy}, {cz}) {recs}")

    def observable(self, abs_indices: Sequence[int], index: int = 0) -> None:
        recs = " ".join(f"rec[{i - self.nmeas}]" for i in sorted(abs_indices, reverse=True))
        self.emit(f"OBSERVABLE_INCLUDE({index}) {recs}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def stim(self) -> "stim.Circuit":
        return stim.Circuit(self.text())


# --------------------------------------------------------------------------- #
#  5.  Two-patch logical circuits (Bell / magic-Bell) with a transversal CNOT
# --------------------------------------------------------------------------- #
#
#  The transversal CNOT is the one construction Stim's generators do not give us,
#  so we build it here.  The subtle part is the detectors *across* the CNOT.
#
#  From round >= 1 both stabiliser types are deterministic (each check compares
#  to its own previous round).  A transversal CX with control patch A and target
#  patch B conjugates the physical checks as
#
#        X_A  ->  X_A X_B        Z_B  ->  Z_A Z_B
#        X_B  ->  X_B            Z_A  ->  Z_A
#
#  so, at the *first* round after the CNOT, the control patch's X-plaquette and
#  the target patch's Z-plaquette each pick up the same-location plaquette on the
#  other patch (previous round).  Every other detector is the ordinary
#  ``now XOR prev``.  This rule is validated by :func:`assert_deterministic`:
#  under zero noise every detector must stay silent, so a wrong rule is caught.

def _support_maps(s: "PatchSchedule"):
    """Data-qubit support of each ancilla, read off the CX schedule."""
    zsup = {a: set() for a in s.z_anc}   # Z-anc is CX *target*: pairs (data, anc)
    xsup = {a: set() for a in s.x_anc}   # X-anc is CX *control*: pairs (anc, data)
    for layer in s.cx_steps:
        for (c, t) in layer:
            if t in zsup:
                zsup[t].add(c)
            if c in xsup:
                xsup[c].add(t)
    return xsup, zsup


class Patch:
    """One rotated-SC patch at a qubit-index offset and side-by-side x-shift."""

    def __init__(self, s: "PatchSchedule", qoffset: int, xshift: float):
        self.s = s
        self.data = [q + qoffset for q in s.data]
        self.x_anc = [q + qoffset for q in s.x_anc]
        self.z_anc = [q + qoffset for q in s.z_anc]
        self.anc_order = [q + qoffset for q in s.anc_order]
        self.coords = {q + qoffset: (x + xshift, y) for q, (x, y) in s.coords.items()}
        self.cx_steps = [[(c + qoffset, t + qoffset) for (c, t) in layer]
                         for layer in s.cx_steps]
        xsup, zsup = _support_maps(s)
        self.xsup = {a + qoffset: {q + qoffset for q in sup} for a, sup in xsup.items()}
        self.zsup = {a + qoffset: {q + qoffset for q in sup} for a, sup in zsup.items()}
        self.loc_of = {q + qoffset: s.coords[q] for q in s.anc_order}   # -> local coord
        self.by_loc = {s.coords[q]: q + qoffset for q in s.anc_order}   # local coord ->
        self.data_by_loc = {s.coords[q]: q + qoffset for q in s.data}
        self.obs_z = [q + qoffset for q in s.obs_data]                  # logical-Z (bottom row)
        xmin = min(s.coords[q][0] for q in s.data)
        self.obs_x = sorted((q + qoffset for q in s.data if s.coords[q][0] == xmin),
                            key=lambda q: self.coords[q][1])             # logical-X (left column)

    @property
    def corner(self) -> int:
        """Data qubit shared by the logical-X and logical-Z strings (injection site)."""
        return next(q for q in self.obs_x if q in self.obs_z)


@dataclass
class TwoPatchInfo:
    A: Patch
    B: Patch
    meas_basis: str
    n_obs: int = 3   # obs 0 = logical(A), 1 = logical(B), 2 = joint correlation


def two_patch_circuit(
    d: int = 3,
    R_pre: int = 2,
    R_post: int = 2,
    prepA: str = "X",
    prepB: str = "Z",
    do_cx: bool = True,
    meas_basis: str = "Z",
    p: float = 0.0,
    q: Optional[float] = None,
    naive_t: bool = False,
) -> Tuple[str, TwoPatchInfo]:
    """Build a two-patch logical circuit as Stim text (fed to tsim).

    Two distance-``d`` patches sit side-by-side (disjoint indices, B shifted in
    ``x``).  With the defaults (``prepA='X'``, ``prepB='Z'``, ``do_cx=True``) this
    is the logical **Bell** circuit |+>_L|0>_L --CX--> Bell.  Set ``prepA='A'`` to
    inject |A>_L = T|+>_L on patch A (a physical ``T`` on its corner qubit), giving
    the **magic-Bell** ``H·T·CNOT`` whose ideal correlators are ``<ZZ>=1`` and
    ``<XX>=cos(pi/4)=0.707``.

    ``naive_t=True`` instead slaps a bare ``T`` on patch A's whole logical-X string
    *after* encoding — the pedagogically instructive **wrong** way (it breaks
    stabilisers and gives the wrong ``<XX>``; used in a Demo).

    Noise is phenomenological: ``DEPOLARIZE1(p)`` on data each round and a
    measurement-flip rate ``q`` (defaults to ``p``).  Observables:
    ``0`` = logical(A), ``1`` = logical(B), ``2`` = joint correlation (the only
    deterministic one for a Bell state, in the measured basis).
    """
    if q is None:
        q = p
    prepA, prepB, meas_basis = prepA.upper(), prepB.upper(), meas_basis.upper()
    s = _parse_schedule(d)
    A = Patch(s, 0, 0.0)
    B = Patch(s, 1000, float(2 * d + 2))
    patches = [A, B]
    b = Builder()
    for P in patches:
        for qb in sorted(P.coords):
            x, y = P.coords[qb]
            b.coords(qb, x, y)

    all_data = A.data + B.data
    all_anc = A.anc_order + B.anc_order
    b.op("R", all_data + all_anc)
    for P, pr in ((A, prepA), (B, prepB)):
        if pr in ("X", "A"):
            b.op("H", P.data)
        if pr == "A":                       # inject |A>_L via physical T on the corner
            b.op("T", [P.corner])
    b.tick()

    prep = {A: prepA, B: prepB}
    anc_hist = {P: {a: [] for a in P.anc_order} for P in patches}
    coordz = 0

    def emit_round(cross: Optional[Tuple[Patch, Patch]] = None) -> None:
        nonlocal coordz
        if p > 0:
            b.op("DEPOLARIZE1", all_data, p)
        for P in patches:
            b.op("H", P.x_anc)
        b.tick()
        for layer_i in range(4):
            pairs: List[int] = []
            for P in patches:
                pairs += [x for pr in P.cx_steps[layer_i] for x in pr]
            b.op("CX", pairs)
            b.tick()
        for P in patches:
            b.op("H", P.x_anc)
        b.tick()
        for P in patches:
            idxs = b.measure("MR", P.anc_order, q if q > 0 else None)
            for a, ix in zip(P.anc_order, idxs):
                anc_hist[P][a].append(ix)
        for P in patches:
            for a in P.anc_order:
                is_x = a in P.x_anc
                loc = P.loc_of[a]
                hist = anc_hist[P][a]
                r = len(hist) - 1
                cx, cy = P.coords[a]
                if r == 0:
                    want = "X" if is_x else "Z"
                    if prep[P] == want:     # only prep-basis checks are det at round 0
                        b.detector([hist[0]], (cx, cy, coordz))
                    continue
                bits = [hist[r], hist[r - 1]]
                if cross is not None:
                    ctrl, tgt = cross
                    if P is ctrl and is_x:      # X_A picks up same-loc X_B (prev)
                        bits.append(anc_hist[tgt][tgt.by_loc[loc]][r - 1])
                    if P is tgt and not is_x:   # Z_B picks up same-loc Z_A (prev)
                        bits.append(anc_hist[ctrl][ctrl.by_loc[loc]][r - 1])
                b.detector(bits, (cx, cy, coordz))
        coordz += 1

    for _ in range(R_pre):
        emit_round()

    if naive_t:
        b.op("T", A.obs_x)
        b.tick()

    if do_cx:
        pairs = []
        for loc, qa in A.data_by_loc.items():
            pairs += [qa, B.data_by_loc[loc]]
        b.op("CX", pairs)
        b.tick()
        emit_round(cross=(A, B))
        for _ in range(R_post - 1):
            emit_round()
    else:
        for _ in range(R_post):
            emit_round()

    if meas_basis == "X":
        b.op("H", all_data)
        b.tick()
    data_idx: Dict[int, int] = {}
    for P in patches:
        idxs = b.measure("M", P.data, q if q > 0 else None)
        for dq, ix in zip(P.data, idxs):
            data_idx[dq] = ix

    for P in patches:
        anc_set = P.z_anc if meas_basis == "Z" else P.x_anc
        sup = P.zsup if meas_basis == "Z" else P.xsup
        for a in anc_set:
            cx, cy = P.coords[a]
            bits = [data_idx[dq] for dq in sup[a]] + [anc_hist[P][a][-1]]
            b.detector(bits, (cx, cy, coordz))

    obsA = A.obs_z if meas_basis == "Z" else A.obs_x
    obsB = B.obs_z if meas_basis == "Z" else B.obs_x
    b.observable([data_idx[q] for q in obsA], 0)
    b.observable([data_idx[q] for q in obsB], 1)
    b.observable([data_idx[q] for q in obsA] + [data_idx[q] for q in obsB], 2)

    return b.text(), TwoPatchInfo(A, B, meas_basis)


# --------------------------------------------------------------------------- #
#  5b.  General Q-patch circuits (deep / many-T scaling benchmark)
# --------------------------------------------------------------------------- #
#
#  ``two_patch_circuit`` is the pedagogical 2-patch special case.  The deep-circuit
#  scaling study (``example_deep_circuit_scaling.ipynb``) needs an *arbitrary* number
#  of side-by-side patches and an *arbitrary* gate list, so it can vary the three
#  cost axes independently at fixed d=3: logical-qubit count ``Q`` (width), non-Clifford
#  ``T`` count (magic), and syndrome-round depth ``R`` (time).  ``multi_patch_circuit``
#  is the straight generalisation of ``two_patch_circuit``:
#
#    * patch ``i`` (1-indexed) sits at qubit offset ``1000*i`` and x-shift ``i*(2d+2)``;
#    * ``preps[i]`` prepares patch ``i`` in ``|0>_L`` ("Z"), ``|+>_L`` ("X") or the magic
#      state ``|A>_L = T|+>_L`` ("A", a physical ``T`` on the corner) — this is how we
#      seed logical ``T`` without a mid-circuit logical ``H`` (which the transversal
#      construction of this series handles elsewhere, arXiv:2412.01391);
#    * each gate ``("CNOT", c, t)`` is a transversal CX between patches ``c`` and ``t``,
#      followed by one syndrome round carrying the *same* CSS cross-detector correction
#      proved for the 2-patch case (one CNOT per round, so the single-CNOT rule composes);
#    * each gate ``("T", p)`` is a physical ``T`` on patch ``p``'s corner qubit.
#
#  The T-free (pure-Clifford) circuits stay deterministic under zero noise — verified by
#  ``assert_deterministic`` in the module self-test for GHZ and centre-crossing chains at
#  Q up to 6 — which is the correctness anchor for the general builder.

def multi_patch_circuit(
    d: int = 3,
    Q: int = 2,
    preps: Optional[Sequence[str]] = None,
    gates: Optional[Sequence[tuple]] = None,
    R_pre: int = 1,
    R_gap: int = 1,
    R_post: int = 1,
    meas_basis: str = "Z",
    p: float = 0.0,
    q: Optional[float] = None,
) -> str:
    """Build a ``Q``-patch distance-``d`` logical circuit as Stim text (fed to tsim).

    Parameters
    ----------
    Q          : number of logical qubits (side-by-side patches).
    preps      : length-``Q`` prep bases, each ``"Z"`` / ``"X"`` / ``"A"`` (default all
                 ``"Z"`` = ``|0>_L``).
    gates      : ordered list of ``("CNOT", ctrl, tgt)`` (1-indexed patches) or
                 ``("T", patch)``.  Each gate is applied, then ``R_gap`` syndrome rounds
                 run; the round immediately after a CNOT carries the cross-detector
                 correction.
    R_pre/R_post : idle syndrome rounds before the first / after the last gate.
    meas_basis : final transversal readout basis (``"Z"`` or ``"X"``).
    p, q       : phenomenological data-depolarise / measurement-flip rates (``q``←``p``).

    Returns Stim text; observable ``i`` is the logical operator of patch ``i+1``.
    Mirrors the ITensor ``run_circuit_mq`` gate tuples so the two engines run the *same*
    logical circuit (an ITensor leading ``(:H, i)`` on a fresh ``|0>`` maps to ``preps[i]="X"``).
    """
    if q is None:
        q = p
    preps = [x.upper() for x in (preps if preps is not None else ["Z"] * Q)]
    gates = list(gates or [])
    meas_basis = meas_basis.upper()
    if len(preps) != Q:
        raise ValueError(f"preps must have length Q={Q}")
    s = _parse_schedule(d)
    patches = [Patch(s, 1000 * (i + 1), i * float(2 * d + 2)) for i in range(Q)]
    b = Builder()
    for P in patches:
        for qb in sorted(P.coords):
            x, y = P.coords[qb]
            b.coords(qb, x, y)

    all_data = [dq for P in patches for dq in P.data]
    all_anc = [a for P in patches for a in P.anc_order]
    b.op("R", all_data + all_anc)
    for P, pr in zip(patches, preps):
        if pr in ("X", "A"):
            b.op("H", P.data)
        if pr == "A":                        # inject |A>_L via physical T on the corner
            b.op("T", [P.corner])
    b.tick()

    anc_hist = {i: {a: [] for a in P.anc_order} for i, P in enumerate(patches)}
    coordz = [0]

    def emit_round(cross: Optional[Sequence[Tuple[int, int]]] = None) -> None:
        """One syndrome-extraction round; ``cross`` = list of (ctrl_idx, tgt_idx) 0-based."""
        cross = list(cross or [])
        if p > 0:
            b.op("DEPOLARIZE1", all_data, p)
        for P in patches:
            b.op("H", P.x_anc)
        b.tick()
        for layer_i in range(4):
            pairs: List[int] = []
            for P in patches:
                pairs += [x for pr in P.cx_steps[layer_i] for x in pr]
            b.op("CX", pairs)
            b.tick()
        for P in patches:
            b.op("H", P.x_anc)
        b.tick()
        for i, P in enumerate(patches):
            idxs = b.measure("MR", P.anc_order, q if q > 0 else None)
            for a, ix in zip(P.anc_order, idxs):
                anc_hist[i][a].append(ix)
        role: Dict[int, Tuple[str, int]] = {}       # patch idx -> ("ctrl"/"tgt", partner idx)
        for (ci, ti) in cross:
            role[ci] = ("ctrl", ti)
            role[ti] = ("tgt", ci)
        for i, P in enumerate(patches):
            for a in P.anc_order:
                is_x = a in P.x_anc
                loc = P.loc_of[a]
                hist = anc_hist[i][a]
                r = len(hist) - 1
                cx, cy = P.coords[a]
                if r == 0:
                    want = "X" if is_x else "Z"
                    if preps[i] == want:            # only prep-basis checks det at round 0
                        b.detector([hist[0]], (cx, cy, coordz[0]))
                    continue
                bits = [hist[r], hist[r - 1]]
                if i in role:
                    kind, j = role[i]
                    partner = patches[j]
                    if kind == "ctrl" and is_x:     # X_ctrl picks up same-loc X_tgt (prev)
                        bits.append(anc_hist[j][partner.by_loc[loc]][r - 1])
                    if kind == "tgt" and not is_x:  # Z_tgt picks up same-loc Z_ctrl (prev)
                        bits.append(anc_hist[j][partner.by_loc[loc]][r - 1])
                b.detector(bits, (cx, cy, coordz[0]))
        coordz[0] += 1

    for _ in range(R_pre):
        emit_round()

    for g in gates:
        if g[0] == "CNOT":
            ci, ti = g[1] - 1, g[2] - 1
            A, B = patches[ci], patches[ti]
            pairs = []
            for loc, qa in A.data_by_loc.items():
                pairs += [qa, B.data_by_loc[loc]]
            b.op("CX", pairs)
            b.tick()
            emit_round(cross=[(ci, ti)])
            for _ in range(R_gap - 1):
                emit_round()
        elif g[0] == "T":
            b.op("T", [patches[g[1] - 1].corner])
            b.tick()
            for _ in range(R_gap):
                emit_round()
        else:
            raise ValueError(f"unknown gate {g!r}")

    for _ in range(R_post):
        emit_round()

    if meas_basis == "X":
        b.op("H", all_data)
        b.tick()
    data_idx: Dict[int, int] = {}
    for P in patches:
        idxs = b.measure("M", P.data, q if q > 0 else None)
        for dq, ix in zip(P.data, idxs):
            data_idx[dq] = ix

    for i, P in enumerate(patches):
        anc_set = P.z_anc if meas_basis == "Z" else P.x_anc
        sup = P.zsup if meas_basis == "Z" else P.xsup
        for a in anc_set:
            cx, cy = P.coords[a]
            bits = [data_idx[dq] for dq in sup[a]] + [anc_hist[i][a][-1]]
            b.detector(bits, (cx, cy, coordz[0]))

    for i, P in enumerate(patches):
        obs = P.obs_z if meas_basis == "Z" else P.obs_x
        b.observable([data_idx[qq] for qq in obs], i)
    return b.text()


def count_T(gates: Sequence[tuple]) -> int:
    """Number of physical ``T`` injections a gate list will emit (magic count)."""
    return sum(1 for g in gates if g and g[0] == "T")


def sample_correlator(
    circuit_text: str,
    shots: int,
    decode: bool = True,
) -> float:
    """Estimate the joint logical correlator ``<PP>`` from a two-patch circuit.

    Samples with tsim; if ``decode`` and the noiseless DEM is non-trivial, the
    per-patch logical bits are corrected with PyMatching before correlating.
    Returns ``<PP> = P(agree) - P(disagree)`` (``+1`` = perfectly correlated).
    """
    t = _tsim_circuit(circuit_text)
    det, obs = t.compile_detector_sampler().sample(shots=shots, separate_observables=True)
    det = np.asarray(det, dtype=np.uint8)
    obs = np.asarray(obs, dtype=np.uint8)
    a = obs[:, 0].astype(int)
    bb = obs[:, 1].astype(int)
    if decode:
        try:
            m = pymatching.Matching.from_detector_error_model(t.detector_error_model())
            pred = np.asarray(m.decode_batch(det), dtype=np.uint8)
            a ^= pred[:, 0]
            bb ^= pred[:, 1]
        except ValueError:
            # p=0 (empty DEM) -> no correction needed; sampler already exact.
            pass
    disagree = (a ^ bb) != 0
    return float(1.0 - 2.0 * np.mean(disagree))


# --------------------------------------------------------------------------- #
#  6.  Validation helpers
# --------------------------------------------------------------------------- #

def assert_deterministic(circuit, shots: int = 256) -> None:
    """Assert every detector stays silent under zero noise (correctness oracle).

    A wrong detector rule (e.g. across a transversal CNOT) makes some detector
    non-deterministic, which this catches immediately.
    """
    c = circuit if isinstance(circuit, tsim.Circuit) else _tsim_circuit(str(circuit))
    det, _ = c.compile_detector_sampler().sample(shots=shots, separate_observables=True)
    det = np.asarray(det)
    fired = det.any(axis=0)
    if fired.any():
        bad = np.where(fired)[0].tolist()
        raise AssertionError(
            f"{len(bad)} detector(s) non-deterministic under zero noise: {bad[:20]}"
        )


if __name__ == "__main__":
    # ---- smoke test: single-patch memory + decoder ----
    print("tsim_surface smoke test")
    for d in (3, 5):
        c = memory_circuit(d, rounds=d, basis="Z", p=0.0)
        assert_deterministic(c)
        print(f"  d={d} memory-Z zero-noise: detectors silent  OK")

    c = memory_circuit(3, rounds=3, basis="Z", p=0.01)
    pL, se = estimate_pL(c, shots=2000)
    print(f"  d=3 p=0.01 memory-Z: p_L = {pL:.4f} +/- {se:.4f}")
    assert 0.0 < pL < 0.2, "p_L out of sane range"

    # sub-threshold ordering: d=5 should beat d=3 at low p
    p = 0.003
    pL3, _ = estimate_pL(memory_circuit(3, 3, "Z", p), shots=4000)
    pL5, _ = estimate_pL(memory_circuit(5, 5, "Z", p), shots=4000)
    print(f"  sub-threshold p={p}: p_L(d=3)={pL3:.4f}  p_L(d=5)={pL5:.4f}")

    sched = _parse_schedule(3)
    print(f"  parsed d=3 schedule: {len(sched.data)} data, "
          f"{len(sched.x_anc)} X-anc, {len(sched.z_anc)} Z-anc, "
          f"{len(sched.cx_steps)} CX layers, obs_data={sched.obs_data}")

    # ---- two-patch: zero-noise determinism + correlators ----
    print("two-patch checks")
    for basis in ("Z", "X"):
        txt, _ = two_patch_circuit(prepA="X", prepB="Z", meas_basis=basis, p=0.0)
        assert_deterministic(txt)
    print("  Bell zero-noise: all detectors silent  OK")
    zz = sample_correlator(two_patch_circuit(prepA="X", prepB="Z", meas_basis="Z", p=0.0)[0],
                           shots=20000, decode=False)
    xx = sample_correlator(two_patch_circuit(prepA="X", prepB="Z", meas_basis="X", p=0.0)[0],
                           shots=20000, decode=False)
    print(f"  Bell <ZZ>={zz:+.3f}  <XX>={xx:+.3f}  (ideal +1 / +1)")
    assert zz > 0.97 and xx > 0.97

    mzz = sample_correlator(two_patch_circuit(prepA="A", prepB="Z", meas_basis="Z", p=0.0)[0],
                            shots=20000, decode=False)
    mxx = sample_correlator(two_patch_circuit(prepA="A", prepB="Z", meas_basis="X", p=0.0)[0],
                            shots=20000, decode=False)
    print(f"  magic-Bell <ZZ>={mzz:+.3f}  <XX>={mxx:+.3f}  (ideal +1 / +0.707)")
    assert mzz > 0.97 and 0.60 < mxx < 0.80

    mxx_n = sample_correlator(two_patch_circuit(prepA="A", prepB="Z", meas_basis="X", p=0.01)[0],
                              shots=20000, decode=True)
    print(f"  magic-Bell <XX> decoded @ p=0.01: {mxx_n:+.3f}")

    # ---- general Q-patch builder: Clifford circuits stay deterministic ----
    print("multi-patch checks (correctness anchor for the deep-circuit study)")
    for Q in (3, 4):
        preps = ["X"] + ["Z"] * (Q - 1)
        ghz = [("CNOT", i, i + 1) for i in range(1, Q)]                 # GHZ chain
        cross = [("CNOT", i, Q + 1 - i) for i in range(1, Q // 2 + 1)]  # centre-crossing
        for label, gs in (("GHZ-chain", ghz), ("centre-cross", cross)):
            for mb in ("Z", "X"):
                assert_deterministic(multi_patch_circuit(3, Q, preps=preps, gates=gs,
                                                         meas_basis=mb, p=0.0))
        print(f"  Q={Q}: GHZ + centre-crossing Clifford circuits deterministic  OK")
    # magic-Bell reproduced through the general builder (sanity of T injection)
    gzz = sample_correlator(multi_patch_circuit(3, 2, preps=["A", "Z"],
                            gates=[("CNOT", 1, 2)], meas_basis="Z", p=0.0), shots=20000, decode=False)
    gxx = sample_correlator(multi_patch_circuit(3, 2, preps=["A", "Z"],
                            gates=[("CNOT", 1, 2)], meas_basis="X", p=0.0), shots=20000, decode=False)
    print(f"  general-builder magic-Bell <ZZ>={gzz:+.3f} <XX>={gxx:+.3f}  (ideal +1 / +0.707)")
    assert gzz > 0.97 and 0.60 < gxx < 0.80
    print("smoke test done")
