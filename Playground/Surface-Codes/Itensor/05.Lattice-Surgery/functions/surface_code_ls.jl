# =============================================================================
#  surface_code_ls.jl
#  Consolidated surface-code machinery for the lattice-surgery notebooks.
#
#  This is the SAME rotated-planar d=3 encoding used throughout the project
#  (see 1.Single-logical-qubit ... 4.Magic-states). It bundles:
#    * base machinery      (geometry, stabilizer measurement, MWPM decoder)
#    * teleportation magic  (|A>,|Y> injection, S/T gate teleportation)
#    * frame-aware logical ops (logical H as transversal-H + an H-parity flag)
#    * deterministic logical-state preparation (forced +1 stabilizer projection)
#
#  Conventions (unchanged from the earlier notebooks):
#    - 3 patches, laid out SIDE-BY-SIDE on one MPS: all of patch 1, then all of
#      patch 2, then all of patch 3.  Within a patch: data, Z-aux, X-aux.
#      (Side-by-side, NOT interleaved, so a between-patch gate's non-locality
#       cost is visible — see the project note on MPS layouts.)
#    - Patches 1,2 are the two logical data qubits; patch 3 is a recyclable
#      ancilla (magic-state injection AND the lattice-surgery ancilla).
#    - logical Z = Z on row y=0;  logical X = X on column x=0   (H-parity 0).
#    - After a transversal logical H the patch sits in its DUAL code: the Pauli
#      TYPE of each logical is unchanged but its support swaps row0<->col0.
#      We track this with a per-patch H-parity flag `hpar` and `logop(L,hpar)`.
# =============================================================================
using ITensors, ITensorMPS, LinearAlgebra, Random
Random.seed!(0xB011)
const threshold = 1e-6     # MPS truncation cutoff

# Physical single-qubit T and S gates for S=1/2 sites (Up=|0>, Dn=|1>).
ITensors.op(::OpName"T", ::SiteType"S=1/2") = ComplexF64[1 0; 0 exp(im*π/4)]
ITensors.op(::OpName"S", ::SiteType"S=1/2") = ComplexF64[1 0; 0 im]
# Controlled-Z (used by the parity-ancilla joint measurement, kept for reference).
ITensors.op(::OpName"CZ2", ::SiteType"S=1/2") = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]

const d = 3     # code distance; d=3 keeps the ~half-MPS-spanning ops tractable.

# ---- geometry for a single patch ----
data_coords = vec([(x, y) for x in 0:(d-1), y in 0:(d-1)])
z_aux_local = Tuple{Float64,Float64}[]
x_aux_local = Tuple{Float64,Float64}[]
for x in 0:(d-2), y in 0:(d-2)
    push!((x + y) % 2 == 0 ? z_aux_local : x_aux_local, (x + 0.5, y + 0.5))
end
for y in 1:2:(d-2);  push!(z_aux_local, (-0.5,    y + 0.5)); end
for y in 0:2:(d-2);  push!(z_aux_local, (d - 0.5, y + 0.5)); end
for x in 0:2:(d-2);  push!(x_aux_local, (x + 0.5, -0.5));    end
for x in 1:2:(d-2);  push!(x_aux_local, (x + 0.5, d - 0.5)); end
const Nz_per_patch = length(z_aux_local)

# ---- MPS site ordering: side-by-side (patch 1, then 2, then 3) ----
all_keys = Tuple{Int, Tuple{Float64,Float64}}[]
for p in 1:3
    for q in data_coords;  push!(all_keys, (p, q)); end
    for a in z_aux_local;  push!(all_keys, (p, a)); end
    for a in x_aux_local;  push!(all_keys, (p, a)); end
end
sites = siteinds("S=1/2", length(all_keys))
site_of = Dict{Tuple{Int, Tuple{Float64,Float64}}, Index}(
    k => sites[i] for (i, k) in enumerate(all_keys))
mps_index_of = Dict(k => i for (i, k) in enumerate(all_keys))

data_neighbors_of(aux_coord) =
    [q for q in data_coords if abs(q[1]-aux_coord[1])==0.5 && abs(q[2]-aux_coord[2])==0.5]

P_up(s) = 0.5 * op("Id", s) + op("Sz", s)
P_dn(s) = 0.5 * op("Id", s) - op("Sz", s)

# ---- projective single-site Z measurement (bit 0 = Up = |0>) ----
function measure_Z!(psi, site; cutoff = threshold)
    sz = real(inner(psi', apply(op("Sz", site), psi; cutoff)))
    if rand() < 0.5 + sz
        psi = apply(P_up(site), psi; cutoff); bit = 0
    else
        psi = apply(P_dn(site), psi; cutoff); bit = 1
    end
    bit, psi / sqrt(real(inner(psi, psi)))
end

function measure_Z_stab(psi, p::Int, aux_coord; cutoff = threshold)
    aux_site = site_of[(p, aux_coord)]; nbrs = data_neighbors_of(aux_coord)
    order = length(nbrs) == 4 ? [2, 4, 1, 3] : [1, 2]
    for q in nbrs[order]
        psi = apply(op("CNOT", site_of[(p, q)], aux_site), psi; cutoff)
    end
    measure_Z!(psi, aux_site; cutoff)
end

function reset_aux!(psi, aux_site; cutoff = threshold)
    sz = real(inner(psi', apply(op("Sz", aux_site), psi; cutoff)))
    sz < 0 ? apply(op("X", aux_site), psi; cutoff) : psi
end

function measure_X_stab(psi, p::Int, aux_coord; cutoff = threshold)
    aux_site = site_of[(p, aux_coord)]
    psi = apply(op("H", aux_site), psi; cutoff)
    nbrs = data_neighbors_of(aux_coord); order = length(nbrs) == 4 ? [2, 1, 4, 3] : [1, 2]
    for q in nbrs[order]
        psi = apply(op("CNOT", aux_site, site_of[(p, q)]), psi; cutoff)
    end
    psi = apply(op("H", aux_site), psi; cutoff)
    measure_Z!(psi, aux_site; cutoff)
end

function project_to_codespace!(psi, p::Int; cutoff = threshold)
    z_syn = Int[]
    for ac in z_aux_local
        psi = reset_aux!(psi, site_of[(p, ac)]; cutoff)
        bit, psi = measure_Z_stab(psi, p, ac; cutoff); push!(z_syn, bit)
    end
    x_syn = Int[]
    for ac in x_aux_local
        psi = reset_aux!(psi, site_of[(p, ac)]; cutoff)
        bit, psi = measure_X_stab(psi, p, ac; cutoff); push!(x_syn, bit)
    end
    (; z_syn, x_syn), psi
end

# ---- transversal logical gates ----
logical_H!(psi, p::Int; cutoff = threshold) =
    (for q in data_coords; psi = apply(op("H", site_of[(p, q)]), psi; cutoff); end; psi)
tH! = logical_H!   # alias used by the H-parity frame code

function logical_CNOT!(psi, p_ctrl::Int, p_tgt::Int; cutoff = threshold)
    for q in data_coords
        psi = apply(op("CNOT", site_of[(p_ctrl, q)], site_of[(p_tgt, q)]), psi; cutoff)
    end
    psi
end

# =============================================================================
#  Teleportation magic: |A>,|Y> injection and S/T gate teleportation
# =============================================================================
const XL_col0 = [(0, y) for y in 0:(d-1)]   # logical-X support (column x=0), hpar 0
const ZL_support = [(x, 0) for x in 0:(d-1)] # logical-Z support (row  y=0), hpar 0

function reset_patch_data!(psi, p; cutoff = threshold)
    for q in data_coords
        s = site_of[(p, q)]
        sz = real(inner(psi', apply(op("Sz", s), psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = apply(b == 0 ? P_up(s) : P_dn(s), psi; cutoff)
        psi = psi / sqrt(real(inner(psi, psi)))
        b == 1 && (psi = apply(op("X", s), psi; cutoff))
    end
    psi
end

seed_ops(sym) = sym === :zero ? String[] : sym === :one ? ["X"] :
                sym === :plus ? ["H"]   : sym === :minus ? ["H","Z"] :
                sym === :A ? ["H","T"] : sym === :Y ? ["H","S"] : error("seed $sym")

function inject_seed!(psi, p, sym::Symbol; cutoff = threshold)
    psi = reset_patch_data!(psi, p; cutoff)
    c = site_of[(p, (0, 0))]
    for g in seed_ops(sym); psi = apply(op(g, c), psi; cutoff); end
    for q in XL_col0[2:end]; psi = apply(op("H", site_of[(p, q)]), psi; cutoff); end
    _, psi = project_to_codespace!(psi, p; cutoff)
    psi
end

function measure_patch_Z!(psi, p; cutoff = threshold)
    par = 0
    for q in data_coords
        s = site_of[(p, q)]
        sz = real(inner(psi', apply(op("Sz", s), psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = apply(b == 0 ? P_up(s) : P_dn(s), psi; cutoff)
        psi = psi / sqrt(real(inner(psi, psi)))
        q in ZL_support && (par ⊻= b)
    end
    par, psi
end

apply_ZL!(psi, p; cutoff = threshold) =
    (for q in ZL_support; psi = apply(op("Z", site_of[(p, q)]), psi; cutoff); end; psi)
apply_XL!(psi, p; cutoff = threshold) =
    (for q in XL_col0;   psi = apply(op("X", site_of[(p, q)]), psi; cutoff); end; psi)

function teleport_S!(psi, p; cutoff = threshold)
    psi = inject_seed!(psi, 3, :Y; cutoff)
    psi = logical_CNOT!(psi, p, 3; cutoff)
    m, psi = measure_patch_Z!(psi, 3; cutoff)
    m == 1 && (psi = apply_ZL!(psi, p; cutoff))
    psi
end

function teleport_T!(psi, p; cutoff = threshold)
    psi = inject_seed!(psi, 3, :A; cutoff)
    psi = logical_CNOT!(psi, p, 3; cutoff)
    m, psi = measure_patch_Z!(psi, 3; cutoff)
    m == 1 && (psi = teleport_S!(psi, p; cutoff))
    psi
end

# =============================================================================
#  Frame-aware logical operators (logical H = transversal-H + H-parity flag)
# =============================================================================
const row0 = ZL_support          # Z_L support at hpar 0
const col0 = XL_col0             # X_L support at hpar 0

# Physical (Pauli type, support) implementing patch's logical operator L (:Z/:X)
# at H-parity h. Transversal-H keeps the Pauli TYPE and swaps row0<->col0.
function logop(L, h)
    if L === :Z
        return h == 0 ? (:Z, row0) : (:Z, col0)
    else
        return h == 0 ? (:X, col0) : (:X, row0)
    end
end
ev(b) = 1 - 2*b

# destructive frame-aware logical readout of ONE patch (returns parity bit).
# NOTE: copies psi internally, so it does NOT collapse the caller's state — use
# joint_readout! (in lattice_surgery_ops.jl) when a JOINT parity is needed.
function readout(psi, p, h, L; cutoff = threshold)
    basis, supp = logop(L, h); S = Set(supp); psi = copy(psi)
    basis == :X && (for q in data_coords; psi = apply(op("H", site_of[(p,q)]), psi; cutoff); end)
    par = 0
    for q in data_coords
        s = site_of[(p,q)]; sz = real(inner(psi', apply(op("Sz", s), psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = apply(b == 0 ? P_up(s) : P_dn(s), psi; cutoff); psi = psi / sqrt(real(inner(psi, psi)))
        q in S && (par ⊻= b)
    end
    par
end

# frame-aware logical Pauli applied physically (P = :X or :Z)
function applyL!(psi, p, h, P; cutoff = threshold)
    basis, supp = logop(P, h)
    for q in supp; psi = apply(op(string(basis), site_of[(p, q)]), psi; cutoff); end
    psi
end

# =============================================================================
#  Deterministic logical-state preparation: seed + FORCE every stabilizer to +1
#  (project the syndrome ancilla onto |0>). Gives the trivial-syndrome logical
#  representative with NO random Pauli frame — ideal ground truth for the demos.
# =============================================================================
function force_Z_stab!(psi, p, ac; cutoff = threshold)
    aux = site_of[(p, ac)]; nbrs = data_neighbors_of(ac)
    order = length(nbrs) == 4 ? [2,4,1,3] : [1,2]
    for q in nbrs[order]; psi = apply(op("CNOT", site_of[(p,q)], aux), psi; cutoff); end
    psi = apply(P_up(aux), psi; cutoff); psi / sqrt(real(inner(psi,psi)))
end
function force_X_stab!(psi, p, ac; cutoff = threshold)
    aux = site_of[(p, ac)]; psi = apply(op("H", aux), psi; cutoff)
    nbrs = data_neighbors_of(ac); order = length(nbrs) == 4 ? [2,1,4,3] : [1,2]
    for q in nbrs[order]; psi = apply(op("CNOT", aux, site_of[(p,q)]), psi; cutoff); end
    psi = apply(op("H", aux), psi; cutoff)
    psi = apply(P_up(aux), psi; cutoff); psi / sqrt(real(inner(psi,psi)))
end
function inject_det!(psi, p, sym; cutoff = threshold)
    psi = reset_patch_data!(psi, p; cutoff)
    c = site_of[(p,(0,0))]
    for g in seed_ops(sym); psi = apply(op(g, c), psi; cutoff); end
    for q in XL_col0[2:end]; psi = apply(op("H", site_of[(p,q)]), psi; cutoff); end
    for ac in z_aux_local; psi = reset_aux!(psi, site_of[(p,ac)]; cutoff); psi = force_Z_stab!(psi, p, ac; cutoff); end
    for ac in x_aux_local; psi = reset_aux!(psi, site_of[(p,ac)]; cutoff); psi = force_X_stab!(psi, p, ac; cutoff); end
    psi
end
