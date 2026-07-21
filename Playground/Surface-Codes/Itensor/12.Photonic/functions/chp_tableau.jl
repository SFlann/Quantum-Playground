# ============================================================================
#  chp_tableau.jl — a minimal Aaronson–Gottesman stabilizer tableau (CHP).
#
#  Everything the ring fusion network does is CLIFFORD: cluster-state resource
#  prep (H + CZ), beam-splitter fusions (CNOT + two single-photon readouts),
#  and single-photon pattern measurements. So the full network can ALSO be run
#  on a stabilizer tableau — ~10⁴× faster than the exact MPS — which is how the
#  wide sweeps below (hundreds of runs, GF(2) detector analysis, error scans)
#  are done. The exact ITensor MPS remains the ground truth: the notebook
#  verifies every discovered detector composition on it directly.
#  (Base Julia only; no packages. Standard CHP: arXiv:quant-ph/0406196.)
# ============================================================================

mutable struct Tab
    n::Int
    x::BitMatrix   # (2n) × n
    z::BitMatrix
    r::BitVector   # 2n phases
end

function Tab(n::Int)
    x = falses(2n, n); z = falses(2n, n); r = falses(2n)
    for i in 1:n
        x[i, i] = true          # destabilizers X_i
        z[n+i, i] = true        # stabilizers  Z_i  (state |0...0>)
    end
    Tab(n, x, z, r)
end

# rowsum: row h ← row h * row i  (exact AG phase bookkeeping)
function rowsum!(t::Tab, h::Int, i::Int)
    s = 2*(t.r[h] ? 1 : 0) + 2*(t.r[i] ? 1 : 0)
    @inbounds for j in 1:t.n
        x1 = t.x[i,j]; z1 = t.z[i,j]; x2 = t.x[h,j]; z2 = t.z[h,j]
        # g(x1,z1,x2,z2) per the paper
        if x1
            if z1
                s += (z2 ? 1 : 0) - (x2 ? 1 : 0)               # Y: g = z2 - x2
            else
                s += (z2 && x2) ? 1 : (z2 && !x2 ? -1*0 : 0)   # X: g = z2*(2x2-1)
                s += (z2 && !x2) ? -1 : 0
            end
        elseif z1
            s += (x2 && !z2) ? 1 : 0                            # Z: g = x2*(1-2z2)
            s += (x2 && z2) ? -1 : 0
        end
        t.x[h,j] ⊻= x1; t.z[h,j] ⊻= z1
    end
    t.r[h] = mod(s, 4) == 2
    return t
end

h!(t::Tab, q) = (@inbounds for i in 1:2t.n
    t.r[i] ⊻= t.x[i,q] && t.z[i,q]
    t.x[i,q], t.z[i,q] = t.z[i,q], t.x[i,q]
end; t)

s!(t::Tab, q) = (@inbounds for i in 1:2t.n
    t.r[i] ⊻= t.x[i,q] && t.z[i,q]
    t.z[i,q] ⊻= t.x[i,q]
end; t)

cnot!(t::Tab, c, q) = (@inbounds for i in 1:2t.n
    t.r[i] ⊻= t.x[i,c] && t.z[i,q] && (t.x[i,q] == t.z[i,c])
    t.x[i,q] ⊻= t.x[i,c]; t.z[i,c] ⊻= t.z[i,q]
end; t)

cz!(t::Tab, a, b) = (h!(t, b); cnot!(t, a, b); h!(t, b); t)
x!(t::Tab, q) = (@inbounds for i in 1:2t.n; t.r[i] ⊻= t.z[i,q]; end; t)
z!(t::Tab, q) = (@inbounds for i in 1:2t.n; t.r[i] ⊻= t.x[i,q]; end; t)

# --- deterministic-phase accumulation with an explicit scratch row ---
"accumulate row i of t into scratch (xs, zs, phase s∈0:3); returns new s"
function scratch_rowsum(t::Tab, xs::BitVector, zs::BitVector, s::Int, i::Int)
    @inbounds for j in 1:t.n
        x1 = t.x[i,j]; z1 = t.z[i,j]; x2 = xs[j]; z2 = zs[j]
        if x1
            if z1; s += (z2 ? 1 : 0) - (x2 ? 1 : 0)
            else;  s += z2 ? (x2 ? 1 : -1) : 0
            end
        elseif z1
            s += x2 ? (z2 ? -1 : 1) : 0
        end
        xs[j] ⊻= x1; zs[j] ⊻= z1
    end
    s += 2*(t.r[i] ? 1 : 0)
    return mod(s, 4)
end

"measure Z_q with correct deterministic branch"
function measure_z!(t::Tab, q; force=nothing)
    n = t.n
    p = 0
    @inbounds for i in n+1:2n
        if t.x[i,q]; p = i; break; end
    end
    if p > 0
        @inbounds for i in 1:2n
            (i != p && t.x[i,q]) && rowsum!(t, i, p)
        end
        t.x[p-n, :] .= t.x[p, :]; t.z[p-n, :] .= t.z[p, :]; t.r[p-n] = t.r[p]
        t.x[p, :] .= false; t.z[p, :] .= false; t.z[p, q] = true
        o = force !== nothing ? force : rand(0:1)
        t.r[p] = (o == 1)
        return o
    else
        xs = falses(n); zs = falses(n); s = 0
        @inbounds for i in 1:n
            t.x[i,q] && (s = scratch_rowsum(t, xs, zs, s, n + i))
        end
        o = s == 2 ? 1 : 0
        force !== nothing && force != o && error("forced $force but deterministic $o")
        return o
    end
end

measure_x!(t::Tab, q; force=nothing) = (h!(t, q); o = measure_z!(t, q; force=force); h!(t, q); o)

"is the Pauli ∏_{q∈S} P_q (P='X' or 'Z') deterministic? returns (det::Bool, value 0/1)"
function pauli_expectation(t::Tab, S::Vector{Int}, P::Char)
    # map product to Z_{s1} by a temporary Clifford, read, then undo
    ops = Function[]
    if P == 'X'
        for q in S; h!(t, q); end
    end
    s1 = S[1]
    for q in S[2:end]; cnot!(t, q, s1); end
    # deterministic iff no stabilizer has x at s1
    n = t.n
    p = any(t.x[i, s1] for i in n+1:2n)
    det = !p
    val = 0
    if det
        xs = falses(n); zs = falses(n); s = 0
        @inbounds for i in 1:n
            t.x[i,s1] && (s = scratch_rowsum(t, xs, zs, s, n + i))
        end
        val = s == 2 ? 1 : 0
    end
    for q in reverse(S[2:end]); cnot!(t, q, s1); end
    if P == 'X'
        for q in S; h!(t, q); end
    end
    return det, val
end

"forced projection of ∏X_S to +1 (layer-0 prep)"
function project_xprod!(t::Tab, S::Vector{Int})
    for q in S; h!(t, q); end
    s1 = S[1]
    for q in S[2:end]; cnot!(t, q, s1); end
    measure_z!(t, s1; force=0)
    for q in reverse(S[2:end]); cnot!(t, q, s1); end
    for q in S; h!(t, q); end
end

nothing

"determinism/value of a MIXED product ∏ P_i on qubits qs with bases ps ('X','Y','Z')"
function mixed_expectation(t::Tab, qs::Vector{Int}, ps::Vector{Char})
    # rotate each qubit's basis to Z, CNOT-chain to qs[1], read, undo
    for (q,p) in zip(qs,ps)
        p == 'X' && h!(t,q)
        p == 'Y' && (s!(t,q); s!(t,q); s!(t,q); h!(t,q))   # S† then H : Y -> Z
    end
    s1 = qs[1]
    for q in qs[2:end]; cnot!(t, q, s1); end
    n = t.n
    det = !any(t.x[i,s1] for i in n+1:2n)
    val = 0
    if det
        xs = falses(n); zs = falses(n); sacc = 0
        for i in 1:n
            t.x[i,s1] && (sacc = scratch_rowsum(t, xs, zs, sacc, n+i))
        end
        val = sacc == 2 ? 1 : 0
    end
    for q in reverse(qs[2:end]); cnot!(t, q, s1); end
    for (q,p) in zip(qs,ps)
        p == 'X' && h!(t,q)
        p == 'Y' && (h!(t,q); s!(t,q))                     # undo: H then S
    end
    return det, val
end

# ---------------------------------------------------------------------------
#  §  analysis utilities
# ---------------------------------------------------------------------------
"""
    reduced_group(t, targets) -> [(pauli_string, sign), ...]

The stabilizer group of the REDUCED state on `targets`: all independent
elements of the global stabilizer group supported only on those qubits.
(Used to read off, e.g., what operator of the input a given ring photon
carries — the Choi-state probe of §4.)
"""
function reduced_group(t::Tab, targets::Vector{Int})
    n = t.n
    others = setdiff(1:n, targets)
    G = falses(n, 2*length(others))
    for (ri, row) in enumerate(n+1:2n)
        for (j, q) in enumerate(others)
            G[ri, j] = t.x[row, q]; G[ri, length(others)+j] = t.z[row, q]
        end
    end
    A = permutedims(Matrix{Int}(G)) .% 2
    piv = Int[]; row = 1
    for col in 1:size(A,2)
        pr = 0
        for rr in row:size(A,1); if A[rr,col]==1; pr=rr; break; end; end
        pr == 0 && continue
        A[row,:], A[pr,:] = A[pr,:], A[row,:]
        for rr in 1:size(A,1); rr != row && A[rr,col]==1 && (A[rr,:] .⊻= A[row,:]); end
        piv = vcat(piv, col); row += 1; row > size(A,1) && break
    end
    out = Tuple{String,Int}[]
    for f in setdiff(1:size(A,2), piv)
        c = zeros(Int, size(A,2)); c[f] = 1
        for (i,p) in enumerate(piv); i <= size(A,1) || continue; c[p] = A[i,f]; end
        xs = falses(n); zs = falses(n); s = 0
        for ri in 1:size(A,2); c[ri]==1 && (s = scratch_rowsum(t, xs, zs, s, n+ri)); end
        ps = ""
        for q in targets
            ps *= xs[q] ? (zs[q] ? "Y" : "X") : (zs[q] ? "Z" : "I")
        end
        push!(out, (ps, s == 2 ? -1 : 1))
    end
    return out
end

"""
    affine_invariants(R) -> (basis, consts)

All GF(2)-affine invariants of a run matrix R (rows = runs, cols = record
bits): the parity combinations that come out the SAME in every run. These
ARE the network's detectors (plus gauge constraints) — exactly how detector
error models are built for real hardware.
"""
function affine_invariants(R::Matrix{Int})
    A = R .⊻ R[1:1, :]
    m, n = size(A)
    M = copy(A) .% 2
    pivots = Int[]; row = 1
    for col in 1:n
        pr = 0
        for rr in row:m; if M[rr,col] == 1; pr = rr; break; end; end
        pr == 0 && continue
        M[row,:], M[pr,:] = M[pr,:], M[row,:]
        for rr in 1:m
            rr != row && M[rr,col] == 1 && (M[rr,:] .⊻= M[row,:])
        end
        push!(pivots, col); row += 1
        row > m && break
    end
    basis = Vector{Vector{Int}}()
    for f in setdiff(1:n, pivots)
        v = zeros(Int, n); v[f] = 1
        for (i,p) in enumerate(pivots)
            i <= size(M,1) || continue
            v[p] = M[i,f]
        end
        push!(basis, v)
    end
    consts = [reduce(⊻, R[1,:] .* v) % 2 for v in basis]
    return basis, consts
end

nothing
# ---------------------------------------------------------------------------
#  §  the ring-native slab on the tableau — MIRRORS the notebook's MPS `slab!`
#     exactly (same resource states, same fusions, same pattern, same record
#     names), so wide sweeps (hundreds of runs) take seconds. The notebook
#     verifies the same detector compositions on the exact MPS.
# ---------------------------------------------------------------------------
const CHP_XSTAB = [[2,3,5,6],[4,5,7,8],[1,2],[8,9]]
const CHP_SPATIAL_ODD = [
    ("Z1245", 1,:n3, 2,:n3), ("Z1245", 4,:n3, 5,:n3),
    ("Z5689", 5,:n6, 6,:n6), ("Z5689", 8,:n3, 9,:n3),
    ("Z47",   4,:n6, 7,:n6),
    ("Z36",   3,:n3, 6,:n3),
]
const CHP_SPATIAL_EVEN = [
    ("X2356", 2,:n3, 3,:n3), ("X2356", 5,:n6, 6,:n6),
    ("X4578", 4,:n3, 5,:n3), ("X4578", 7,:n6, 8,:n6),
    ("X12",   1,:n6, 2,:n6),
    ("X89",   8,:n3, 9,:n3),
]
chp_spatial(t) = isodd(t) ? CHP_SPATIAL_ODD : CHP_SPATIAL_EVEN
chp_fused(t, i) = [s for (_,a,sa,b,sb) in chp_spatial(t) for (q,s) in ((a,sa),(b,sb)) if q == i]
chp_pad(t, i)   = [s for s in (:n2,:n3) if !(s in chp_fused(t, i))]
chp_prune(t, i) = [s for s in (:n5,:n6) if !(s in chp_fused(t, i))]

mutable struct ChpSim
    t::Tab
    next::Int
    id::Dict{Symbol,Int}
end
ChpSim(n) = ChpSim(Tab(n), 1, Dict{Symbol,Int}())
chp_alloc!(s::ChpSim, l::Symbol; plus=true) = (q = s.next; s.next += 1; s.id[l] = q; plus && h!(s.t, q); q)
chp_q(s::ChpSim, l) = s.id[l]
chp_n(t,i,k) = Symbol("t$(t)_r$(i)_n$(k)")
chp_l(t,i,sl) = Symbol("t$(t)_r$(i)_L$(sl)")

"beam-splitter fusion on the tableau: CNOT then the two local readouts = (X₁X₂, Z₁Z₂)"
chp_fuse!(s::ChpSim, a, b) = (cnot!(s.t, chp_q(s,a), chp_q(s,b));
                              (measure_x!(s.t, chp_q(s,a)), measure_z!(s.t, chp_q(s,b))))

function chp_ring!(s::ChpSim, t, i)
    for k in 1:6; chp_alloc!(s, chp_n(t,i,k)); end
    for k in 1:6; cz!(s.t, chp_q(s,chp_n(t,i,k)), chp_q(s,chp_n(t,i, k==6 ? 1 : k+1))); end
    chp_alloc!(s, chp_l(t,i,:n1)); cz!(s.t, chp_q(s,chp_n(t,i,1)), chp_q(s,chp_l(t,i,:n1)))
    chp_alloc!(s, chp_l(t,i,:n4)); cz!(s.t, chp_q(s,chp_n(t,i,4)), chp_q(s,chp_l(t,i,:n4)))
    for sl in chp_fused(t, i)
        k = parse(Int, string(sl)[2:2])
        chp_alloc!(s, chp_l(t,i,sl)); cz!(s.t, chp_q(s,chp_n(t,i,k)), chp_q(s,chp_l(t,i,sl)))
    end
end

"full memory run on the tableau; returns the record. err ∈ (:none,:X,:Z); errafter=-1 → input error"
function chp_memory(nslabs; err=:none, errslot=5, errafter=99)
    s = ChpSim(18 + 96*nslabs); rec = Dict{String,Int}()
    for i in 1:9; chp_alloc!(s, Symbol("W$i"); plus=false); end
    for st in CHP_XSTAB; project_xprod!(s.t, [chp_q(s,Symbol("W$i")) for i in st]); end
    if errafter == -1 && err != :none
        err == :X ? x!(s.t, chp_q(s,Symbol("W$errslot"))) : z!(s.t, chp_q(s,Symbol("W$errslot")))
    end
    for i in 1:9
        chp_alloc!(s, Symbol("LW$i")); cz!(s.t, chp_q(s,Symbol("W$i")), chp_q(s,Symbol("LW$i")))
    end
    W = [Symbol("W$i") for i in 1:9]; LW = [Symbol("LW$i") for i in 1:9]
    for t in 1:nslabs
        for i in 1:9; chp_ring!(s, t, i); end
        for i in 1:9
            (mx, mz) = chp_fuse!(s, LW[i], chp_l(t,i,:n1))
            rec["t$(t)_tf$(i)_x"] = mx; rec["t$(t)_tf$(i)_z"] = mz
        end
        for (f, (cell, a, sa, b, sb)) in enumerate(chp_spatial(t))
            (mx, mz) = chp_fuse!(s, chp_l(t,a,sa), chp_l(t,b,sb))
            rec["t$(t)_sf$(f)_$(cell)_x"] = mx; rec["t$(t)_sf$(f)_$(cell)_z"] = mz
        end
        for i in 1:9
            for sl in chp_prune(t, i)
                k = parse(Int, string(sl)[2:2])
                rec["t$(t)_pr$(i)_$(sl)"] = measure_z!(s.t, chp_q(s,chp_n(t,i,k)))
            end
            rec["t$(t)_xA$(i)"]   = measure_x!(s.t, chp_q(s,W[i]))
            rec["t$(t)_xn1_$(i)"] = measure_x!(s.t, chp_q(s,chp_n(t,i,1)))
            for sl in vcat(chp_pad(t, i), chp_fused(t, i))
                k = parse(Int, string(sl)[2:2])
                rec["t$(t)_xn$(k)_$(i)"] = measure_x!(s.t, chp_q(s,chp_n(t,i,k)))
            end
        end
        W = [chp_n(t,i,4) for i in 1:9]; LW = [chp_l(t,i,:n4) for i in 1:9]
        if t == errafter && err != :none
            err == :X ? x!(s.t, chp_q(s,W[errslot])) : z!(s.t, chp_q(s,W[errslot]))
        end
    end
    for i in 1:9; rec["fin_lz$(i)"] = measure_z!(s.t, chp_q(s,LW[i])); end
    tz = Tab(s.t.n, copy(s.t.x), copy(s.t.z), copy(s.t.r))
    tx = Tab(s.t.n, copy(s.t.x), copy(s.t.z), copy(s.t.r))
    for i in 1:9
        rec["fin_wZ$(i)"] = measure_z!(tz, chp_q(s,W[i]))
        rec["fin_wX$(i)"] = measure_x!(tx, chp_q(s,W[i]))
    end
    return rec
end

nothing
