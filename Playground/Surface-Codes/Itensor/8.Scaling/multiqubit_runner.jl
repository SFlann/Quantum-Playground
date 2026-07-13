# ============================================================================
#  multiqubit_runner.jl  —  Q-logical-qubit non-Clifford circuits at fixed d
#
#  A thin generalisation layer over `scaling_engine.jl`. That engine already
#  parameterises the patch count (`build_code(d; npatch)`) and its `logical_CNOT!`,
#  `logical_H!`, `joint_expect`/`frame_sign` are patch-agnostic — but `run_circuit`,
#  the S/T teleportation gadget, and the exact reference were hard-wired to two data
#  patches + a patch-3 magic ancilla. This file lifts those to arbitrary `Q`:
#
#    - `teleport_{S,T}_at!`  : the gadget with an explicit ancilla patch index.
#    - `run_circuit_mq`      : prepare Q data patches (1..Q), reuse ONE magic-ancilla
#                              patch (index Q+1) for every S/T, transversal CNOTs only.
#    - `logical_expect`      : frame-corrected ⟨⊗ Pauli⟩ over ANY set of patches.
#    - `ref_state_mq`/`refN` : exact 2^Q state vector + Pauli-string correlators.
#
#  Layout stays side-by-side (per project convention), so a transversal CNOT between
#  non-adjacent logical qubits visibly spans the chain — the cost this study measures.
#  Nothing in the engine is modified; the original notebooks keep working.
# ============================================================================
include("scaling_engine.jl")

"""
    teleport_S_at!(M, p, anc, R; C, B, use_AD) -> M

Logical S on data patch `p` teleported through magic ancilla patch `anc` (any index): prep |Y⟩ on
`anc`, transversal CNOT `p→anc`, commit one R-round sliding-decoded epoch on `anc`, measure it out,
apply the conditional byproduct Z_L on `p`. Generalises the engine's `teleport_S!` (which fixed anc=3).
"""
function teleport_S_at!(M::Machine, p, anc, R; C, B, use_AD=true)
    c = M.code
    prepare_logical!(M, anc, :Y); logical_CNOT!(M, p, anc)
    run_epoch!(M, anc, R; C, B, use_AD)
    m_raw = measure_patch_Z_raw!(M, anc)
    fbit = 0; for q in c.ZL_support; fbit ⊻= M.fx[anc][c.didx[q]]; end
    (m_raw ⊻ fbit) == 1 && apply_ZL_phys!(M, p)
    M
end

"""
    teleport_T_at!(M, p, anc, R; C, B, use_AD) -> M

Logical T on data patch `p` teleported through magic ancilla patch `anc`: prep |A⟩=T|+⟩, transversal
CNOT `p→anc`, commit, measure out, and apply the conditional (itself teleported) S byproduct on `p`.
Generalises the engine's `teleport_T!`.
"""
function teleport_T_at!(M::Machine, p, anc, R; C, B, use_AD=true)
    c = M.code
    prepare_logical!(M, anc, :A); logical_CNOT!(M, p, anc)
    run_epoch!(M, anc, R; C, B, use_AD)
    m_raw = measure_patch_Z_raw!(M, anc)
    fbit = 0; for q in c.ZL_support; fbit ⊻= M.fx[anc][c.didx[q]]; end
    (m_raw ⊻ fbit) == 1 && teleport_S_at!(M, p, anc, R; C, B, use_AD)
    M
end

"""
    apply_logical_mq!(M, gate, anc, R; C, B, use_AD) -> M

Dispatch one logical gate for the multi-qubit runner. `gate` is `(:X|:Z|:H|:S|:T, patch)` or
`(:CNOT, ctrl, tgt)`; S/T teleport through the shared ancilla patch `anc`.
"""
function apply_logical_mq!(M::Machine, gate, anc, R; C, B, use_AD)
    g = gate[1]
    g === :X    ? logical_X!(M, gate[2]) :
    g === :Z    ? logical_Z!(M, gate[2]) :
    g === :H    ? logical_H!(M, gate[2]) :
    g === :S    ? teleport_S_at!(M, gate[2], anc, R; C, B, use_AD) :
    g === :T    ? teleport_T_at!(M, gate[2], anc, R; C, B, use_AD) :
    g === :CNOT ? logical_CNOT!(M, gate[2], gate[3]) : error("unknown gate $g")
end

"""
    run_circuit_mq(c, Q, circuit; ec, R, C, B, use_AD) -> Machine

Run a `Q`-logical-qubit circuit on code `c` (which must have `npatch ≥ Q+1`). Data qubits are patches
`1..Q` prepared in |0⟩_L; patch `Q+1` is a single magic ancilla reused (prepared fresh, measured out)
for every S/T. With `ec`, one sliding-decoded idle epoch runs on every data patch after each gate.
Read out with `logical_expect`.
"""
function run_circuit_mq(c::Code, Q::Int, circuit; ec=true, R=6, C=2, B=2, use_AD=true)
    anc = Q + 1
    c.npatch >= anc || error("need npatch ≥ Q+1 = $anc, got npatch=$(c.npatch)")
    M = Machine(c)
    for p in 1:Q; prepare_logical!(M, p, :zero); end
    for gate in circuit
        apply_logical_mq!(M, gate, anc, R; C, B, use_AD)
        ec && (for p in 1:Q; run_epoch!(M, p, R; C, B, use_AD); end)
    end
    M
end

"""
    logical_expect(M, specs) -> Float64

Frame-corrected logical correlator ⟨∏ L_p⟩ over an arbitrary set of patches. `specs` is a list of
`(patch, L)` with `L ∈ {:X,:Y,:Z}`; patches not listed are identity. Reuses the engine's
`_op_and_sign` (physical operator, phase, and Pauli-frame sign per patch), so it generalises `corr2`
to any number of logical qubits.
"""
function logical_expect(M::Machine, specs)
    c = M.code; Opsi = copy(M.psi); phase = 1.0 + 0im; sgn = 1
    for (p, L) in specs
        ops, ph, s = _op_and_sign(M, p, L)
        for (g,q) in ops; Opsi = apply(op(g, sd(c,p,q)), Opsi; cutoff=CUTOFF[]); end
        phase *= ph; sgn *= s
    end
    real(phase * inner(M.psi, Opsi)) * sgn
end

# --- exact 2^Q state-vector reference (validation ground truth) --------------
"embed a single-qubit gate `U` on qubit `i` of `Q` (qubit 1 = most significant)."
_embed1(U, i, Q) = foldl(kron, [k == i ? U : _I for k in 1:Q])
"the 2^Q CNOT permutation matrix (control `ctrl`, target `tgt`; qubit 1 = most significant)."
function _embed_cnot(ctrl, tgt, Q)
    N = 2^Q; Mt = zeros(ComplexF64, N, N)
    for idx in 0:(N-1)
        cbit = (idx >> (Q - ctrl)) & 1
        out  = cbit == 1 ? idx ⊻ (1 << (Q - tgt)) : idx
        Mt[out+1, idx+1] = 1.0
    end
    Mt
end
"""
    ref_state_mq(Q, circuit) -> Vector{ComplexF64}

Exact 2^Q-amplitude state vector after applying `circuit` (same gate tuples as `run_circuit_mq`) to
|0…0⟩. Qubit ordering matches `logical_expect` (patch i ↔ qubit i, qubit 1 most significant).
"""
function ref_state_mq(Q, circuit)
    ψ = zeros(ComplexF64, 2^Q); ψ[1] = 1.0
    for gate in circuit
        g = gate[1]
        if g === :CNOT
            ψ = _embed_cnot(gate[2], gate[3], Q) * ψ
        else
            U = g===:X ? _X : g===:Z ? _Z : g===:H ? _H : g===:S ? _S : g===:T ? _T : error("gate $g")
            ψ = _embed1(U, gate[2], Q) * ψ
        end
    end
    ψ
end
"""
    refN(ψ, Q, specs) -> Float64

Exact Pauli-string correlator ⟨∏ P_i⟩ on reference state `ψ`. `specs = [(qubit, :X/:Y/:Z)...]`; other
qubits are identity. The exact counterpart of `logical_expect`.
"""
function refN(ψ, Q, specs)
    ops = Any[_I for _ in 1:Q]
    for (i, P) in specs; ops[i] = _Pd[P]; end
    real(ψ' * foldl(kron, ops) * ψ)
end

println("multiqubit_runner.jl loaded — run_circuit_mq(c, Q, circuit); needs npatch ≥ Q+1")
