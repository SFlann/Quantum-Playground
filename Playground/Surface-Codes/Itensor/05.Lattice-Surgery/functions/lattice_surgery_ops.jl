# =============================================================================
#  lattice_surgery_ops.jl
#  The lattice-surgery CNOT: joint logical Pauli-product measurements + the
#  classical byproduct corrections. Depends on surface_code_ls.jl.
#
#  Joint logical measurement.
#  ------------------------------------------------------------------
#  A ZZ- (XX-) merge in lattice surgery measures the JOINT logical operator
#  O = O_a ⊗ O_b (a product of physical Paulis on the two patches' logical
#  supports) WITHOUT revealing either patch's logical value separately. Because
#  a logical operator commutes with every stabilizer, projecting onto its ±1
#  eigenspace keeps both patches in the code space and simply fixes their JOINT
#  parity. We realise that projection directly and exactly on the MPS:
#      outcome ± with prob (1 ± <ψ|O|ψ>)/2 ,   ψ -> (1 ± O)ψ / ‖·‖ .
#  This is the faithful projective measurement the physical merge implements;
#  doing it directly (rather than via a parity ancilla threaded across ~35
#  sites) avoids the truncation error that a long non-local ancilla circuit
#  accumulates. `specs` is a list of (patch, L, hpar) tuples.
# =============================================================================

# apply the product operator O = ∏ logop(L,h) to a fresh copy of psi
function _apply_joint(psi, specs; cutoff = threshold)
    Opsi = copy(psi)
    for (p, L, h) in specs
        basis, supp = logop(L, h)
        for q in supp; Opsi = apply(op(string(basis), site_of[(p, q)]), Opsi; cutoff); end
    end
    Opsi
end

# EXACT projective joint logical-Pauli measurement. Returns (bit, psi); bit 0 = +1.
function measure_joint_exact!(psi, specs; cutoff = threshold)
    Opsi = _apply_joint(psi, specs; cutoff)
    ex = real(inner(psi, Opsi))                     # <ψ|O|ψ> ∈ [-1,1]
    if rand() < 0.5*(1 + ex)
        bit = 0; newpsi = +(psi,  Opsi; cutoff)     # ∝ (1+O)ψ
    else
        bit = 1; newpsi = +(psi, -Opsi; cutoff)     # ∝ (1-O)ψ
    end
    bit, newpsi / sqrt(real(inner(newpsi, newpsi)))
end

# non-collapsing joint expectation <∏ logop(L,h)>  (for verification / teaching).
joint_expect(psi, specs; cutoff = threshold) = real(inner(psi, _apply_joint(psi, specs; cutoff)))

# =============================================================================
#  The lattice-surgery CNOT gadget.
#
#      ancilla (patch 3) prepared in |+>_L
#      m1 = M_ZZ(control, ancilla)      ZZ-merge
#      m2 = M_XX(ancilla, target)       XX-merge
#      m3 = M_Z (ancilla)               read out & discard the ancilla
#
#  Stabilizer flow (ancilla |+> ⇒ +X_a) leaves the control/target pair
#  stabilized by  (-1)^{m1⊕m3} Z_c Z_t  and  (-1)^{m2} X_c X_t  — a CNOT up to
#  the Pauli byproduct
#      X_target^{m1 ⊕ m3}      (fixes the Z_cZ_t sign)
#      Z_control^{m2}          (fixes the X_cX_t sign).
#  hc, ht are the H-parity flags of the control and target patches, so the
#  gadget composes correctly with a mid-circuit logical H.
# =============================================================================
function ls_cnot!(psi, pc, pt; hc = 0, ht = 0, verbose = false, cutoff = threshold)
    psi = inject_det!(psi, 3, :plus; cutoff)                          # ancilla |+>_L
    m1, psi = measure_joint_exact!(psi, [(pc,:Z,hc), (3,:Z,0)]; cutoff)  # ZZ-merge
    m2, psi = measure_joint_exact!(psi, [(3,:X,0),  (pt,:X,ht)]; cutoff) # XX-merge
    m3, psi = measure_joint_exact!(psi, [(3,:Z,0)]; cutoff)             # read ancilla
    (m1 ⊻ m3) == 1 && (psi = applyL!(psi, pt, ht, :X; cutoff))          # byproduct on target
    m2 == 1        && (psi = applyL!(psi, pc, hc, :Z; cutoff))          # byproduct on control
    verbose && println("    LS-CNOT outcomes: m1(ZZ)=$m1  m2(XX)=$m2  m3(anc-Z)=$m3  ",
                       "→ corrections  X_t^$(m1 ⊻ m3)  Z_c^$m2")
    psi, (m1, m2, m3)
end

# Honest destructive JOINT logical readout: measure the listed patches on the
# SAME collapsing state and return the JOINT parity bit (∏ of the logicals).
# Measuring each patch on an independent copy (as `readout` does per-patch)
# would destroy the correlation of an entangled state — the pitfall this avoids.
function joint_readout!(psi, specs; cutoff = threshold)
    S = Set{Tuple{Int,Tuple{Int,Int}}}()
    for (p, L, h) in specs
        basis, supp = logop(L, h)
        basis == :X && (for q in data_coords; psi = apply(op("H", site_of[(p,q)]), psi; cutoff); end)
        for q in supp; push!(S, (p, q)); end
    end
    par = 0
    for (p, L, h) in specs, q in data_coords
        s = site_of[(p, q)]; sz = real(inner(psi', apply(op("Sz", s), psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = apply(b == 0 ? P_up(s) : P_dn(s), psi; cutoff); psi = psi / sqrt(real(inner(psi, psi)))
        (p, q) in S && (par ⊻= b)
    end
    par, psi
end
