# ============================================================================
#  photonic_mps.jl  —  a DYNAMIC-SITE ITensor MPS engine for photonic FBQC
#  (09.Ring-Fusion-Network edition: the 08 engine + a §J 6-RING resource state,
#   a paid-fusion counter FUSES, and a SHOW_FUSIONS beam-splitter Bell readout.
#   The 08 primitives are unchanged, so every 08 verification still holds.)
#
#  Every other ITensor engine in this repo (8.Scaling/scaling_engine.jl, …) fixes
#  its `siteinds` list ONCE and reuses persistent data+ancilla qubits. A photonic
#  fusion machine works the opposite way: photons are created, measured once, and
#  gone — at any instant only the active FRONTIER of photons exists. This engine
#  models exactly that: it is an MPS whose site list GROWS (a fresh photon is
#  kron-appended as a new tensor with a trivial bond) and SHRINKS (a measured
#  photon in a definite Z-eigenstate is contracted into its neighbour and removed).
#  So the active qubit count stays bounded by the code size, not the number of
#  cycles — the capability a foliated memory needs, done on a genuine ITensor MPS.
#
#  Vocabulary (FBQC):  NODES = the photons that carry logical worldlines + the
#  throw-away check photons;  LEAVES = the dangling photons grown on a node and
#  consumed in a FUSION.  A fusion is a destructive Bell (XX&ZZ) measurement of two
#  leaves.  `fused_cz!` realises a CZ edge between two persisting nodes purely by
#  fusing leaves (no direct CZ), with a Z-only Pauli byproduct tracked classically.
#
#  Everything here is verified against the exact dense-vector `fusion_network.jl`
#  and against direct reference gates (see the notebook's verification cells).
#  Convention: qubit sites are "S=1/2"; label -> MPS position via `r.pos`.
# ============================================================================
using ITensors, ITensorMPS, LinearAlgebra, Random

# X, H, Z, Sz, Id are built-in ITensor ops for S=1/2; define CZ and CNOT to be safe.
ITensors.op(::OpName"CZ", ::SiteType"S=1/2")   = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
ITensors.op(::OpName"CNOT", ::SiteType"S=1/2") = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]

const CUTOFF = Ref(1e-12)
const MAXDIM = Ref(1_000_000)

# ---------------------------------------------------------------------------
#  the photon register: an MPS whose site list grows and shrinks
# ---------------------------------------------------------------------------
mutable struct Reg
    psi::MPS
    sites::Vector{Index}
    pos::Dict{Symbol,Int}     # label -> 1-based MPS position
    pendZ::Dict{Symbol,Int}   # per-photon pending classical Z byproduct (from fused_cz!)
end
Reg() = Reg(MPS(), Index[], Dict{Symbol,Int}(), Dict{Symbol,Int}())

Base.length(r::Reg) = length(r.sites)
sidx(r::Reg, l::Symbol) = r.sites[r.pos[l]]

"Append a fresh photon (|+> default, or |0>) at the right end of the MPS; label it."
function addphoton!(r::Reg, label::Symbol; state=:plus)
    s = siteinds("S=1/2", 1)[1]
    a1, a2 = state == :plus ? (1/sqrt(2), 1/sqrt(2)) :
             state == :zero ? (1.0, 0.0) : error("state $state")
    if isempty(r.sites)
        A = ITensor(s); A[s=>1] = a1; A[s=>2] = a2
        r.psi = MPS(ITensor[A])
    else
        l = Index(1, "Link")
        A = ITensor(l, s); A[l=>1, s=>1] = a1; A[l=>1, s=>2] = a2
        last = r.psi[end] * onehot(l => 1)           # trailing dim-1 link on old last tensor
        r.psi = MPS(ITensor[r.psi[1:end-1]..., last, A])
    end
    push!(r.sites, s); r.pos[label] = length(r.sites); r.pendZ[label] = 0
    return label
end

"Rename a photon's label, carrying its pending-Z byproduct."
function relabel!(r::Reg, old::Symbol, new::Symbol)
    r.pos[new] = r.pos[old]; delete!(r.pos, old)
    r.pendZ[new] = get(r.pendZ, old, 0); delete!(r.pendZ, old)
    return new
end

_ap!(r::Reg, g) = (r.psi = apply(g, r.psi; cutoff=CUTOFF[], maxdim=MAXDIM[]); r)
h!(r::Reg, l)      = _ap!(r, op("H", sidx(r,l)))
x!(r::Reg, l)      = _ap!(r, op("X", sidx(r,l)))
z!(r::Reg, l)      = _ap!(r, op("Z", sidx(r,l)))
cz!(r::Reg, a, b)  = _ap!(r, op("CZ", sidx(r,a), sidx(r,b)))

"⟨P⟩ for a Pauli string given as label=>'X'/'Y'/'Z'."
function expect_pauli(r::Reg, terms)
    Opsi = copy(r.psi)
    for (l,c) in terms
        Opsi = apply(op(string(c), sidx(r,l)), Opsi; cutoff=CUTOFF[])
    end
    real(inner(r.psi, Opsi))
end

"Projectively measure a Pauli involution (label=>basis). Outcome 0↔+1, 1↔-1. Collapses in place."
function measure_pauli!(r::Reg, terms; force=nothing)
    Opsi = copy(r.psi)
    for (l,c) in terms
        Opsi = apply(op(string(c), sidx(r,l)), Opsi; cutoff=CUTOFF[])
    end
    ev = real(inner(r.psi, Opsi))
    p_plus = (1 + ev)/2
    outcome = force !== nothing ? force : (rand() < p_plus ? 0 : 1)
    sgn = outcome == 0 ? 1 : -1
    newpsi = +(r.psi, sgn*Opsi; cutoff=CUTOFF[])
    normalize!(newpsi)
    r.psi = newpsi
    return outcome
end

"Remove a photon already collapsed to |outcome>_Z (product); contract it into a neighbour."
function drop!(r::Reg, label::Symbol, outcome::Int)
    j = r.pos[label]; s = r.sites[j]; psi = r.psi; n = length(psi)
    T = psi[j] * onehot(s => outcome+1)
    if n == 1
        r.psi = MPS()
    elseif j == 1
        psi[2] = T * psi[2]; r.psi = MPS(ITensor[psi[k] for k in 2:n])
    elseif j == n
        psi[n-1] = psi[n-1] * T; r.psi = MPS(ITensor[psi[k] for k in 1:n-1])
    else
        psi[j+1] = T * psi[j+1]; r.psi = MPS(ITensor[psi[k] for k in 1:n if k != j])
    end
    deleteat!(r.sites, j); delete!(r.pos, label); delete!(r.pendZ, label)
    for (k,pp) in r.pos; pp > j && (r.pos[k] = pp - 1); end
    length(r.psi) > 0 && normalize!(r.psi)
    return r
end

"Measure photon in basis b ('X'/'Z') and remove it; returns the outcome. (Fast local path.)"
measdrop!(r::Reg, label::Symbol, b::Char) = sample_drop!(r, label, b)

"""
    fused_cz!(r, u, v) -> (Δz_u, Δz_v)

Apply CZ(u,v) up to a Pauli byproduct, realised WITHOUT a direct CZ on u,v: via a
two-photon CZ *bond* resource (b1,b2) and two type-II leaf fusions. u,v persist (we
grow and fuse LEAVES, never u,v). Verified identity:
    fused_cz!(u,v) = CZ(u,v) · Z_u^{mz1⊕mx2} · Z_v^{mx1⊕mz2}
The two Z byproducts are folded into the classical pending-Z frame (`r.pendZ`) — the
state is never corrected. Returns the two byproduct bits added to u and v.
"""
function fused_cz!(r::Reg, u::Symbol, v::Symbol)
    bu = gensym(:bond); bv = gensym(:bond); lu = gensym(:leaf); lv = gensym(:leaf)
    addphoton!(r, bu); addphoton!(r, bv); cz!(r, bu, bv)          # the CZ bond resource
    addphoton!(r, lu); cz!(r, u, lu)                             # leaf on u
    addphoton!(r, lv); cz!(r, v, lv)                             # leaf on v
    (mx1, mz1) = fuse!(r, lu, bu)                                # fuse leaves
    (mx2, mz2) = fuse!(r, lv, bv)
    zu = mz1 ⊻ mx2; zv = mx1 ⊻ mz2
    r.pendZ[u] ⊻= zu; r.pendZ[v] ⊻= zv                          # record, never apply
    return (zu, zv)
end

"Measure photon `label` in X and drop it, correcting the outcome by its pending Z. Returns the ±1 bit (0/1)."
function measX!(r::Reg, label::Symbol)
    pz = get(r.pendZ, label, 0)
    raw = measdrop!(r, label, 'X')
    return raw ⊻ pz            # a pending Z before an X-measurement flips the outcome
end

# ---------------------------------------------------------------------------
#  A type-II fusion IS a beam-splitter Bell-state measurement. `FUSES` counts
#  every one we PAY for (the physically expensive linear-optical operation), and
#  `SHOW_FUSIONS[]=true` makes each fusion print its beam-splitter Bell readout —
#  the two commuting parities (X₁X₂, Z₁Z₂) a real BSM click pattern returns. This
#  is the "surface every fusion explicitly" instrumentation asked for in the plan.
# ---------------------------------------------------------------------------
const FUSES = Ref(0)                       # running tally of PAID (inter-ring) fusions
const SHOW_FUSIONS = Ref(false)            # if true, print each fusion's Bell readout
reset_fuses!() = (FUSES[] = 0)

"""
    sample_drop!(r, label, basis; force=nothing) -> outcome::Int

Projectively measure ONE photon in basis 'X' or 'Z' and remove its site — the fast
local path. The MPS is orthogonalized to the site (incremental, so consecutive
nearby measurements are cheap); the outcome is sampled from the local reduced
density, the site tensor is projected/normalized, and the leftover is contracted
into a neighbour (same splice as `drop!`). Physics identical to measure-then-drop
via the exact `+(psi,±Opsi)` route, but with no global MPS addition.
"""
function sample_drop!(r::Reg, label::Symbol, basis::Char; force=nothing)
    j = r.pos[label]; s = r.sites[j]
    orthogonalize!(r.psi, j)
    A = r.psi[j]
    basis == 'X' && (A = noprime(A * op("H", s)))     # rotate X→Z locally, no global apply
    T0 = A * onehot(s => 1)
    p0 = min(max(real(scalar(T0 * dag(T0))), 0.0), 1.0)   # ⟨Π₀⟩ (env = 𝟙 at the ortho centre)
    o = force !== nothing ? force : (rand() < p0 ? 0 : 1)
    T = A * onehot(s => o + 1)
    nrm = sqrt(o == 0 ? p0 : 1 - p0)
    nrm > 1e-12 && (T = T / nrm)
    # splice T into a neighbour and remove the site (same book-keeping as drop!)
    psi = r.psi; n = length(psi)
    if n == 1
        r.psi = MPS()
    elseif j == 1
        psi[2] = T * psi[2]; r.psi = MPS(ITensor[psi[k] for k in 2:n])
    elseif j == n
        psi[n-1] = psi[n-1] * T; r.psi = MPS(ITensor[psi[k] for k in 1:n-1])
    else
        psi[j+1] = T * psi[j+1]; r.psi = MPS(ITensor[psi[k] for k in 1:n if k != j])
    end
    deleteat!(r.sites, j); delete!(r.pos, label); delete!(r.pendZ, label)
    for (k, pp) in r.pos; pp > j && (r.pos[k] = pp - 1); end
    return o
end

"""
    fuse!(r, a, b; force=nothing) -> (mx, mz)

Type-II fusion of photons a,b: a destructive Bell measurement returning the two
commuting parities (X₁X₂, Z₁Z₂); both photons are consumed. Realised by the exact
Bell-measurement circuit — CNOT(a→b), then measure a in X and b in Z: CNOT maps
X_a → X_aX_b and Z_b → Z_aZ_b, so the two LOCAL readouts ARE the joint parities.
(Same physics as the joint-projector route, verified against it; much faster on
long MPS.) `force=(fx,fz)` conditions the two parities, for deterministic tests.
"""
function fuse!(r::Reg, a::Symbol, b::Symbol; force=nothing)
    fx = force === nothing ? nothing : force[1]
    fz = force === nothing ? nothing : force[2]
    _ap!(r, op("CNOT", sidx(r, a), sidx(r, b)))              # the Bell-analyzer circuit
    mx = sample_drop!(r, a, 'X'; force=fx)                   # ≡ X₁X₂ parity (beam-splitter output 1)
    mz = sample_drop!(r, b, 'Z'; force=fz)                   # ≡ Z₁Z₂ parity (beam-splitter output 2)
    FUSES[] += 1                                             # we paid for one linear-optical fusion
    if SHOW_FUSIONS[]
        bell = mx == 0 ? (mz == 0 ? "Φ⁺" : "Ψ⁺") : (mz == 0 ? "Φ⁻" : "Ψ⁻")
        println("    beam-splitter fusion ", a, "⋈", b,
                "  →  Bell outcome ", bell, "   (X₁X₂=", (-1)^mx, ", Z₁Z₂=", (-1)^mz, ")")
    end
    return (mx, mz)
end

# ===========================================================================
#  §J  the 6-RING resource state, and CZ-through-fusion surfaced explicitly
# ---------------------------------------------------------------------------
#  PsiQuantum's state-of-the-art resource state is the 6-RING: a 6-photon cluster
#  (graph) state whose CZ edges form a cycle. In a real machine a near-deterministic
#  resource-state generator makes these upstream, at the source; the ring CZs are the
#  cheap, deterministic part. So we PREPARE the ring directly (|+>^n + ring CZs) and do
#  NOT simulate its creation — the interesting physics starts when rings meet in fusions.
#  Its graph-state stabilizers are  K_i = X_i Z_{i-1} Z_{i+1}  (indices cyclic).
# ===========================================================================

"""
    ring_resource_state!(r, labels) -> labels

Prepare an n-photon **CZ-ring** resource state directly: append n photons in |+>,
then apply a CZ on each ring edge (i, i+1) cyclically. `labels[i]` names photon i.
These edges are the pre-made, source-side part of the network — free, not fusions.
"""
function ring_resource_state!(r::Reg, labels::Vector{Symbol})
    n = length(labels)
    for l in labels; addphoton!(r, l; state=:plus); end          # |+>^n : the ring photons
    for i in 1:n
        j = i == n ? 1 : i + 1
        cz!(r, labels[i], labels[j])                             # deterministic ring CZ (made at source)
    end
    return labels
end

"Cyclic neighbours on an n-ring (1-based)."
ring_prev(i, n) = i == 1 ? n : i - 1
ring_next(i, n) = i == n ? 1 : i + 1

nothing
