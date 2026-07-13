# ============================================================================
#  noisy_engine.jl  —  the sliding-window MPS decoder engine
#
#  This file is a faithful copy of the code of notebook
#  6.Real-Time-Decoder/example_sliding_window_itensor.ipynb (§1–§9), the most
#  sophisticated tool built in this series: the ITensor MPS surface-code engine
#  with a software Pauli frame, an online (2+1)-D sliding-window MWPM decoder
#  with artificial defects, transversal Clifford gates, and magic-state T/S
#  teleportation. Notebook 7 (Modelling-Noisy-Circuits) `include`s this file and
#  builds a stochastic noise layer on top of it, so the decoder logic itself is
#  reused verbatim rather than re-implemented.
#
#  Nothing here is new; see notebook 6 for the derivation and validation. The
#  per-round data-error / measurement-error injection HOOKS that notebook 7's
#  noise sampler drives (`run_epoch!`'s `data_errs` / `meas_errs`, and
#  `run_circuit`'s `errors` / `meas`) already live in this engine.
# ============================================================================

# ---------------------------------------------------------------------------
# [engine cell 2]
# ---------------------------------------------------------------------------
using ITensors, ITensorMPS, LinearAlgebra, Random
Random.seed!(0xB011)
const threshold = 1e-6          # MPS truncation cutoff
const d = 3                     # code distance

ITensors.op(::OpName"T", ::SiteType"S=1/2") = ComplexF64[1 0; 0 exp(im*π/4)]
ITensors.op(::OpName"S", ::SiteType"S=1/2") = ComplexF64[1 0; 0 im]
ITensors.op(::OpName"SWAPg", ::SiteType"S=1/2") = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

data_coords = vec([(x, y) for x in 0:(d-1), y in 0:(d-1)])
z_aux_local = Tuple{Float64,Float64}[]; x_aux_local = Tuple{Float64,Float64}[]
for x in 0:(d-2), y in 0:(d-2)
    push!((x + y) % 2 == 0 ? z_aux_local : x_aux_local, (x + 0.5, y + 0.5))
end
for y in 1:2:(d-2);  push!(z_aux_local, (-0.5,    y + 0.5)); end
for y in 0:2:(d-2);  push!(z_aux_local, (d - 0.5, y + 0.5)); end
for x in 0:2:(d-2);  push!(x_aux_local, (x + 0.5, -0.5));    end
for x in 1:2:(d-2);  push!(x_aux_local, (x + 0.5, d - 0.5)); end
all_keys = Tuple{Int, Tuple{Float64,Float64}}[]
for p in 1:3
    for q in data_coords;  push!(all_keys, (p, q)); end
    for a in z_aux_local;  push!(all_keys, (p, a)); end
    for a in x_aux_local;  push!(all_keys, (p, a)); end
end
sites   = siteinds("S=1/2", length(all_keys))
site_of = Dict(k => sites[i] for (i, k) in enumerate(all_keys))
const didx  = Dict(q => i for (i,q) in enumerate(data_coords))   # data coord -> index 1..9
const Ndata = length(data_coords)
const Nz_per_patch = length(z_aux_local); const Nx_per_patch = length(x_aux_local)
const XL_col0 = [(0, y) for y in 0:(d-1)]        # logical-X support (left column)
const ZL_support = [(x, 0) for x in 0:(d-1)]     # logical-Z support (bottom row)

"""
    data_neighbors_of(ac) -> Vector{Tuple{Int,Int}}

The data qubits a check at half-integer coordinate `ac` acts on (its <=4 diagonal neighbours). Used
both to apply the check's CNOTs during extraction and to build the matching-graph geometry.
"""
data_neighbors_of(ac) = [q for q in data_coords if abs(q[1]-ac[1])==0.5 && abs(q[2]-ac[2])==0.5]

"projector onto |0> for site `s` (0.5*Id + Sz); used to collapse a measured ancilla to bit 0."
P_up(s) = 0.5*op("Id", s) + op("Sz", s)
"projector onto |1> for site `s` (0.5*Id - Sz); used to collapse a measured ancilla to bit 1."
P_dn(s) = 0.5*op("Id", s) - op("Sz", s)

"""
    measure_Z!(psi, site; cutoff) -> (bit, psi)

Projectively measure one physical qubit `site` in the Z basis, sampling the Born-rule outcome.

- `psi`  : the MPS state; `site` : the ITensor index of the qubit to measure.
Returns `bit` (0 = |0>, 1 = |1>) and the collapsed, renormalised state. This is the primitive every
ancilla read-out is built from.
"""
function measure_Z!(psi, site; cutoff = threshold)
    sz = real(inner(psi', apply(op("Sz", site), psi; cutoff)))
    if rand() < 0.5 + sz; psi = apply(P_up(site), psi; cutoff); bit = 0
    else; psi = apply(P_dn(site), psi; cutoff); bit = 1 end
    bit, psi / sqrt(real(inner(psi, psi)))
end

"""
    reset_aux!(psi, aux; cutoff) -> psi

Reset an ancilla qubit `aux` to |0> before it is reused for the next round's stabiliser measurement
(flips it with X if it is currently in |1>). Keeps ancillas clean between rounds.
"""
reset_aux!(psi, aux; cutoff = threshold) =
    (real(inner(psi', apply(op("Sz", aux), psi; cutoff))) < 0 ? apply(op("X", aux), psi; cutoff) : psi)

"""
    measure_Z_stab(psi, p, ac; cutoff) -> (bit, psi)

Measure one Z-plaquette of patch `p` (the check at coordinate `ac`): entangle its data neighbours onto
the check ancilla with CNOTs (in a hook-error-avoiding order) and read the ancilla in Z. `bit` = 0
means the stabiliser is satisfied. Detects X errors on the data qubits it covers.
"""
function measure_Z_stab(psi, p, ac; cutoff = threshold)
    aux = site_of[(p, ac)]; nbrs = data_neighbors_of(ac)
    order = length(nbrs) == 4 ? [2,4,1,3] : [1,2]
    for q in nbrs[order]; psi = apply(op("CNOT", site_of[(p,q)], aux), psi; cutoff); end
    measure_Z!(psi, aux; cutoff)
end

"""
    measure_X_stab(psi, p, ac; cutoff) -> (bit, psi)

Measure one X-plaquette of patch `p` (the check at `ac`): conjugate the ancilla with H so its CNOTs
measure the X-parity of the data neighbours, then read it in Z. Detects Z errors on those data qubits.
"""
function measure_X_stab(psi, p, ac; cutoff = threshold)
    aux = site_of[(p, ac)]; psi = apply(op("H", aux), psi; cutoff)
    nbrs = data_neighbors_of(ac); order = length(nbrs) == 4 ? [2,1,4,3] : [1,2]
    for q in nbrs[order]; psi = apply(op("CNOT", aux, site_of[(p,q)]), psi; cutoff); end
    psi = apply(op("H", aux), psi; cutoff); measure_Z!(psi, aux; cutoff)
end

"""
    measure_raw_syndrome(psi, p; cutoff) -> (z, x, psi)

Measure ONE full round of both check types on patch `p`. Returns the length-`Nz_per_patch` Z-syndrome
`z` (detects X errors) and length-`Nx_per_patch` X-syndrome `x` (detects Z errors), plus the collapsed
state. Outcomes are random per shot; nothing is corrected — the raw bits feed the sliding decoder.
"""
function measure_raw_syndrome(psi, p; cutoff = threshold)
    z = Int[]; for ac in z_aux_local
        psi = reset_aux!(psi, site_of[(p,ac)]; cutoff); b,psi = measure_Z_stab(psi,p,ac;cutoff); push!(z,b); end
    x = Int[]; for ac in x_aux_local
        psi = reset_aux!(psi, site_of[(p,ac)]; cutoff); b,psi = measure_X_stab(psi,p,ac;cutoff); push!(x,b); end
    z, x, psi
end

"""
    seed_ops(sym) -> Vector{String}

The single-qubit gate sequence (applied to a patch's corner qubit during prep) that seeds logical
state `sym` before the logical-X column is spread out. `:zero,:one,:plus,:minus` are the Paulis'
eigenstates; `:A = T|+>` and `:Y = S|+>` are the magic states injected for T and S teleportation.
"""
seed_ops(sym) = sym === :zero ? String[] : sym === :one ? ["X"] :
                sym === :plus ? ["H"]   : sym === :minus ? ["H","Z"] :
                sym === :A ? ["H","T"]  : sym === :Y ? ["H","S"] : error("seed $sym")

"""
    reset_patch_data!(psi, p; cutoff) -> psi

Reset all 9 data qubits of patch `p` to |0> (measure each and X-flip any found in |1>). The clean slate
`prepare_logical!` starts from before seeding a logical state.
"""
function reset_patch_data!(psi, p; cutoff = threshold)
    for q in data_coords
        s = site_of[(p, q)]; sz = real(inner(psi', apply(op("Sz", s), psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = apply(b == 0 ? P_up(s) : P_dn(s), psi; cutoff); psi = psi / sqrt(real(inner(psi, psi)))
        b == 1 && (psi = apply(op("X", s), psi; cutoff))
    end
    psi
end
println("patch: $(length(data_coords)) data, $Nz_per_patch Z-checks, $Nx_per_patch X-checks",
        "   |   MPS sites (3 patches): $(length(sites))")

# ---------------------------------------------------------------------------
# [engine cell 4]
# ---------------------------------------------------------------------------
"""
    Machine

Mutable container for the whole simulation state.
Fields:
- `psi`   : the ITensor MPS holding all three patches.
- `ref_z` : patch -> reference Z-syndrome (the gauge X-errors are measured against).
- `ref_x` : patch -> reference X-syndrome.
- `fx`    : patch -> length-Ndata bit vector, the decoder's estimated X error per data qubit.
- `fz`    : patch -> estimated Z error per data qubit.
Together `(fx, fz)` are the software Pauli frame; readout and the T-gate commitment consume them.
"""
mutable struct Machine
    psi
    ref_z::Dict{Int,Vector{Int}}; ref_x::Dict{Int,Vector{Int}}
    fx::Dict{Int,Vector{Int}};     fz::Dict{Int,Vector{Int}}
end
"Construct an empty Machine wrapping MPS `psi` (per-patch dicts filled by `prepare_logical!`)."
Machine(psi) = Machine(psi, Dict(), Dict(), Dict(), Dict())

const z_support = Dict(ac => Set(data_neighbors_of(ac)) for ac in z_aux_local)
const x_support = Dict(ac => Set(data_neighbors_of(ac)) for ac in x_aux_local)

"""
    flips(cset, auxs, supp) -> Vector{Int}

The sorted indices of checks (from list `auxs`, with supports `supp`) that a candidate error `cset`
(a set of data qubits) violates — i.e. the syndrome a correction/error produces. Helper for the
recovery lookup below.
"""
flips(cset, auxs, supp) = sort([i for (i,ac) in enumerate(auxs) if isodd(length(intersect(cset, supp[ac])))])

"""
    build_lookup(auxs, supp; avoid) -> Dict{Vector{Int}, Vector{Tuple{Int,Int}}}

Precompute a syndrome -> minimum-weight recovery table by brute force over all data-error patterns.

- `auxs, supp` : the checks and their supports (Z-checks for the X-recovery, X-checks for the Z one).
- `avoid`      : a logical support; corrections with ODD overlap on it are rejected, so the recovery is
  a pure *destabiliser* (no logical component) — it fixes the gauge without changing the encoded value.

Returns, for each achievable syndrome (sorted lit-check indices), the lightest data-qubit correction.
"""
function build_lookup(auxs, supp; avoid = nothing)
    coords = collect(data_coords); A = avoid === nothing ? nothing : Set(avoid)
    table = Dict{Vector{Int}, Vector{Tuple{Int,Int}}}()
    for mask in 0:(2^length(coords) - 1)
        e = Set(coords[i] for i in 1:length(coords) if (mask >> (i-1)) & 1 == 1)
        A !== nothing && isodd(length(intersect(e, A))) && continue
        k = flips(e, auxs, supp); w = length(e)
        (!haskey(table,k) || w < length(table[k])) && (table[k] = collect(e))
    end
    table
end
const XREC = build_lookup(z_aux_local, z_support; avoid = ZL_support)   # X-recovery  <- Z-syndrome
const ZREC = build_lookup(x_aux_local, x_support; avoid = XL_col0)      # Z-recovery  <- X-syndrome

"""
    prepare_logical!(M, p, sym; cutoff) -> M

Prepare patch `p` of machine `M` in logical state `sym`, the way a real device would but with NO
physical feed-forward recovery.

Steps: reset data to |0>; seed the logical state (`seed_ops`) and spread logical-X across the left
column; take ONE syndrome round whose random outcome becomes the REFERENCE (`ref_z`,`ref_x`); then seed
the Pauli frame with the destabiliser recovery for that outcome (`XREC`/`ZREC`) instead of applying it
to the atoms. So the random gauge is absorbed into the initial frame and later readout is deterministic.
"""
function prepare_logical!(M, p, sym; cutoff = threshold)
    M.psi = reset_patch_data!(M.psi, p; cutoff)
    c = site_of[(p,(0,0))]
    for g in seed_ops(sym); M.psi = apply(op(g, c), M.psi; cutoff); end
    for q in XL_col0[2:end]; M.psi = apply(op("H", site_of[(p,q)]), M.psi; cutoff); end
    z, x, M.psi = measure_raw_syndrome(M.psi, p; cutoff)
    M.ref_z[p] = z; M.ref_x[p] = x
    M.fx[p] = zeros(Int, Ndata); M.fz[p] = zeros(Int, Ndata)
    zl = sort([i for (i,b) in enumerate(z) if b == 1]); xl = sort([i for (i,b) in enumerate(x) if b == 1])
    for q in get(XREC, zl, Tuple{Int,Int}[]); M.fx[p][didx[q]] ⊻= 1; end
    for q in get(ZREC, xl, Tuple{Int,Int}[]); M.fz[p][didx[q]] ⊻= 1; end
    M
end
println("prep ready: ", length(XREC), " X-recoveries, ", length(ZREC), " Z-recoveries (into the frame)")

# ---------------------------------------------------------------------------
# [engine cell 6]
# ---------------------------------------------------------------------------
"""
    z_aux_containing(q) / x_aux_containing(q) -> Vector{Int}

Indices of the Z- (resp. X-) checks that touch data qubit `q`. Used once to classify each qubit as a
spatial matching edge (touches 2 checks) or a boundary edge (touches 1), building `edge_data`/`bedge_data`.
"""
z_aux_containing(q) = [i for (i,a) in enumerate(z_aux_local) if abs(a[1]-q[1])==0.5 && abs(a[2]-q[2])==0.5]
x_aux_containing(q) = [i for (i,a) in enumerate(x_aux_local) if abs(a[1]-q[1])==0.5 && abs(a[2]-q[2])==0.5]

edge_data_local  = Dict{Tuple{Int,Int}, Tuple{Int,Int}}(); bedge_data_local = Dict{Int, Tuple{Int,Int}}()
for q in data_coords
    zs = z_aux_containing(q)
    if length(zs) == 2; a,b = minmax(zs[1],zs[2]); edge_data_local[(a,b)] = q
    elseif length(zs) == 1; bedge_data_local[zs[1]] = q end
end
xedge_data_local  = Dict{Tuple{Int,Int}, Tuple{Int,Int}}(); xbedge_data_local = Dict{Int, Tuple{Int,Int}}()
for q in data_coords
    xs = x_aux_containing(q)
    if length(xs) == 2; a,b = minmax(xs[1],xs[2]); xedge_data_local[(a,b)] = q
    elseif length(xs) == 1; xbedge_data_local[xs[1]] = q end
end

"""
    build_graph_generic(R, Nstab, edge_data, bedge_data) -> NamedTuple

Build the (2+1)-D matching graph for `R` rounds of a stabiliser type and precompute all-pairs shortest
paths (Floyd-Warshall).

- `R`          : rounds spanned (a whole epoch, or a single sliding window).
- `Nstab`      : number of checks of this type (`Nz_per_patch` or `Nx_per_patch`).
- `edge_data`  : (check i, check j) -> the data qubit they share (a spatial edge).
- `bedge_data` : check i -> its boundary data qubit (a boundary edge).

Nodes are (check, round) + one boundary `BND`. Returns a NamedTuple: `dist` (matching weights), `nxt`
(path reconstruction), `BND`, `node_id(i,r)`, `decode_id(u)`, `edge_kind(u,v)` -> (kind, payload, round),
`path(a,b)`, and `R`. `edge_kind` names the fault an edge represents: :spatial/:boundary -> a data-qubit
X error (kept by `corr_from_pairs`); :time -> a measurement error (no data flip).
"""
function build_graph_generic(R::Int, Nstab::Int, edge_data, bedge_data)
    Ntot = R*Nstab + 1; BND = Ntot
    node_id(i,r) = (r-1)*Nstab + i                     # (check i, round r) -> linear id
    decode_id(u) = (mod1(u,Nstab), div(u-1,Nstab) + 1) # inverse of node_id
    dist = fill(Inf, Ntot, Ntot); nxt = fill(-1, Ntot, Ntot)
    for i in 1:Ntot; dist[i,i] = 0.0; nxt[i,i] = i; end
    add!(u,v) = (dist[u,v]=1.0; dist[v,u]=1.0; nxt[u,v]=v; nxt[v,u]=u)   # symmetric unit edge
    for r in 1:R, ((i,j),_) in edge_data; add!(node_id(i,r), node_id(j,r)); end   # spatial
    for r in 1:(R-1), i in 1:Nstab; add!(node_id(i,r), node_id(i,r+1)); end        # time
    for r in 1:R, (i,_) in bedge_data; add!(node_id(i,r), BND); end                # boundary
    for k in 1:Ntot, i in 1:Ntot, j in 1:Ntot
        if dist[i,k] + dist[k,j] < dist[i,j]; dist[i,j] = dist[i,k] + dist[k,j]; nxt[i,j] = nxt[i,k]; end
    end
    # classify one elementary edge and name its fault; the 3rd entry is the edge's round (for the
    # sliding decoder's core/buffer bookkeeping and surface-crossing detection).
    function edge_kind(u,v)
        u,v = minmax(u,v)
        v == BND && return (:boundary, bedge_data[decode_id(u)[1]], decode_id(u)[2])
        iu,ru = decode_id(u); iv,rv = decode_id(v)
        ru == rv ? (:spatial, edge_data[minmax(iu,iv)], ru) : (:time, (iu, min(ru,rv)), min(ru,rv))
    end
    path(a,b) = (nxt[a,b]==-1 ? Int[] : (p=[a]; while a!=b; a=nxt[a,b]; push!(p,a); end; p))
    (; dist, nxt, BND, node_id, decode_id, edge_kind, path, R)
end

"""
    mwpm(defects, dist, BND) -> Vector{Tuple{Int,Int}}

Exact minimum-weight perfect matching of the lit detector nodes `defects`, via a Held-Karp subset DP.

Each defect pairs with another defect (cost = `dist` between them) or with the boundary sink `BND`
(cost = distance to boundary); total cost is minimised -> most likely fault set. Returns the matching
as pairs `(u,v)` (`v == BND` = matched to boundary). `O(2^n · n)` — needed because sliding windows at
realistic noise light more detectors than a brute-force `n!!` matcher can handle. `dp[mask+1]` = min
cost to resolve the subset `mask`; `choice` records decisions for reconstruction.
"""
function mwpm(defects::Vector{Int}, dist, BND)
    n = length(defects); n == 0 && return Tuple{Int,Int}[]
    bnd = [dist[defects[i], BND] for i in 1:n]
    dd  = [i==j ? 0.0 : dist[defects[i], defects[j]] for i in 1:n, j in 1:n]
    FULL = (1<<n) - 1; dp = fill(Inf, 1<<n); dp[1] = 0.0; choice = fill((-1,-1), 1<<n)
    for mask in 0:FULL
        dp[mask+1] == Inf && continue
        i = 0; while i < n && ((mask>>i)&1 == 1); i += 1; end
        i == n && continue
        nm = mask | (1<<i)
        if isfinite(bnd[i+1]) && dp[mask+1]+bnd[i+1] < dp[nm+1]; dp[nm+1]=dp[mask+1]+bnd[i+1]; choice[nm+1]=(i,-1) end
        for j in (i+1):(n-1)
            ((mask>>j)&1 == 1) && continue; isfinite(dd[i+1,j+1]) || continue
            nm2 = mask | (1<<i) | (1<<j)
            if dp[mask+1]+dd[i+1,j+1] < dp[nm2+1]; dp[nm2+1]=dp[mask+1]+dd[i+1,j+1]; choice[nm2+1]=(i,j) end
        end
    end
    pairs = Tuple{Int,Int}[]; mask = FULL
    while mask != 0
        (i,j) = choice[mask+1]; i == -1 && break
        if j == -1; push!(pairs,(defects[i+1],BND)); mask &= ~(1<<i)
        else; push!(pairs,(defects[i+1],defects[j+1])); mask &= ~(1<<i); mask &= ~(1<<j) end
    end
    pairs
end
println("matching graph + subset-DP MWPM loaded")

# ---------------------------------------------------------------------------
# [engine cell 8]
# ---------------------------------------------------------------------------
"""
    corr_from_pairs(pairs, g; rounds, off) -> Set{Tuple{Int,Int}}

Turn an MWPM matching into the net data-qubit X-flips it prescribes.

Walk each pair's `g.path`; spatial/boundary edges name a data qubit to flip, time edges are measurement
errors (skipped). `rounds` (if given) commits only edges whose global round lies in that range — how the
sliding decoder commits just its core; `off` maps window-local rounds to global. Even multiplicities
cancel, so only odd-count qubits are returned.
"""
function corr_from_pairs(pairs, g; rounds=nothing, off=0)
    cnt = Dict{Tuple{Int,Int},Int}()
    for (a,b) in pairs, k in 1:(length(g.path(a,b))-1)
        pth=g.path(a,b); kind=g.edge_kind(pth[k],pth[k+1])
        kind[1]===:time && continue
        gr = kind[3]+off
        (rounds===nothing || gr in rounds) && (cnt[kind[2]]=get(cnt,kind[2],0)+1)
    end
    Set(q for (q,c) in cnt if isodd(c))
end

"""
    sliding_decode(hist, reference, Nstab, ed, bed; C, B, use_AD) -> Set{Tuple{Int,Int}}

Sliding-window decode of one epoch's raw syndrome history for a single check type.

- `hist`       : `hist[r]` = the length-`Nstab` raw syndrome recorded at round r.
- `reference`  : the gauge the first round's detectors are measured against (chained from the previous
                 epoch); detectors are `det[r] = hist[r] XOR hist[r-1]`, with `hist[0] = reference`.
- `Nstab,ed,bed` : the graph geometry for this check type (`build_graph_generic` args).
- `C`          : commit (core) size; `B` : buffer (look-ahead); window `W = C+B`.
- `use_AD`     : deposit artificial defects at commitment surfaces (true) or not (false, for the demo).

Windows overlap and step by `C`; each commits the data-qubit flips on its core rounds and, when a
matched chain crosses the surface via a time edge, hands the crossing check to the next window as a
pre-lit artificial defect (`AD`). Returns the union of committed flips for the whole epoch. Locals:
`det` global detector stream; `AD` incoming artificial defects; `subdet` the window's (copied) slice;
`core_end` its last committed round; `newAD` the crossings handed forward.
"""
function sliding_decode(hist, reference, Nstab, ed, bed; C, B, use_AD=true)
    R = length(hist)
    det = Vector{Vector{Int}}(undef, R); prev = reference
    for r in 1:R; det[r] = hist[r] .⊻ prev; prev = hist[r]; end
    corr = Set{Tuple{Int,Int}}(); AD = Int[]; W = C+B; start = 1
    while start <= R
        stop = min(start+W-1, R); w = stop-start+1; last_window = stop==R
        g = build_graph_generic(w, Nstab, ed, bed)
        subdet = [copy(det[start+rl-1]) for rl in 1:w]
        for i in AD; subdet[1][i] ⊻= 1; end                 # inject artificial defects on local round 1
        lit = [g.node_id(i,rl) for rl in 1:w for i in 1:Nstab if subdet[rl][i]==1]
        pairs = mwpm(lit, g.dist, g.BND)
        core_end = last_window ? stop : start+C-1
        corr = symdiff(corr, corr_from_pairs(pairs, g; rounds=start:core_end, off=start-1))
        newAD = Int[]
        if use_AD && !last_window
            surf_lo = core_end-(start-1)                     # local round index of the commit surface
            for (a,b) in pairs, k in 1:(length(g.path(a,b))-1)
                pth=g.path(a,b); kind=g.edge_kind(pth[k],pth[k+1])
                (kind[1]===:time && kind[3]==surf_lo) && push!(newAD, kind[2][1])
            end
        end
        AD = newAD; start = core_end+1
    end
    corr
end

"""
    decode_epoch_sliding!(M, p, zhist, xhist; C, B, use_AD) -> M

Decode one epoch of patch `p` on both check types and update the software frame.

`zhist`/`xhist` are the epoch's Z-/X-syndrome histories. The Z-graph decode yields X-corrections
(XORed into `fx`); the X-graph decode yields Z-corrections (into `fz`) — so an X error, a Z error, or a
Y error (both at once) are each tracked on the appropriate graph(s). The reference then chains to the
epoch's last round so a persistent uncorrected error is counted once.
"""
function decode_epoch_sliding!(M, p, zhist, xhist; C, B, use_AD=true)
    for q in sliding_decode(zhist, M.ref_z[p], Nz_per_patch, edge_data_local, bedge_data_local; C,B,use_AD)
        M.fx[p][didx[q]] ⊻= 1; end
    for q in sliding_decode(xhist, M.ref_x[p], Nx_per_patch, xedge_data_local, xbedge_data_local; C,B,use_AD)
        M.fz[p][didx[q]] ⊻= 1; end
    M.ref_z[p] = zhist[end]; M.ref_x[p] = xhist[end]
    M
end

"""
    run_epoch!(M, p, R; data_errs, meas_errs, C, B, use_AD, cutoff) -> M

Run one idle epoch on patch `p`: measure `R` noisy syndrome rounds on the MPS, then sliding-decode them
into the frame.

- `data_errs :: Vector{(round, "X"/"Z", coord)}` : a physical Pauli applied to a data qubit just before
  that round's measurement (persists thereafter). Inject a Y as an X and a Z on the same qubit/round.
- `meas_errs :: Vector{(round, :Z/:X, check_index)}` : flip only the RECORDED bit of a check that round
  (a pure measurement error; the state is untouched).
- `R,C,B,use_AD` : epoch length and sliding-window parameters.
Locals `zhist`/`xhist` accumulate the raw rounds; the state carries the true errors, the frame the
estimate.
"""
function run_epoch!(M, p, R; data_errs=[], meas_errs=[], C=2, B=2, use_AD=true, cutoff=threshold)
    zhist = Vector{Vector{Int}}(undef,R); xhist = Vector{Vector{Int}}(undef,R)
    for r in 1:R
        for (rr,P,q) in data_errs; rr==r && (M.psi = apply(op(P, site_of[(p,q)]), M.psi; cutoff)); end
        z,x,M.psi = measure_raw_syndrome(M.psi,p;cutoff)
        for (rr,typ,si) in meas_errs; rr==r && (typ===:Z ? (z[si]=1-z[si]) : (x[si]=1-x[si])); end
        zhist[r]=z; xhist[r]=x
    end
    decode_epoch_sliding!(M,p,zhist,xhist; C,B,use_AD)
    M
end
println("sliding-window epoch decoder loaded")

# ---------------------------------------------------------------------------
# [engine cell 10]
# ---------------------------------------------------------------------------
"apply the physical logical-Z (Z on the bottom row) to patch p — used as a T/S byproduct."
apply_ZL_phys!(M, p; cutoff = threshold) =
    (for q in ZL_support; M.psi = apply(op("Z", site_of[(p,q)]), M.psi; cutoff); end)
"apply the physical logical-X (X on the left column) to patch p."
apply_XL_phys!(M, p; cutoff = threshold) =
    (for q in XL_col0;    M.psi = apply(op("X", site_of[(p,q)]), M.psi; cutoff); end)
"logical X on patch p; a known Pauli commutes with all stabilisers, so frame and reference are unchanged."
logical_X!(M, p; cutoff = threshold) = (apply_XL_phys!(M, p; cutoff); M)
"logical Z on patch p; frame and reference unchanged (see logical_X!)."
logical_Z!(M, p; cutoff = threshold) = (apply_ZL_phys!(M, p; cutoff); M)

"""
    logical_CNOT!(M, pc, pt; cutoff) -> M

Transversal logical CNOT (control `pc`, target `pt`): bitwise physical CNOT across the patches, then
propagate frame and reference by the CNOT conjugation X_c->X_cX_t, Z_t->Z_cZ_t:
`fx[pt] ⊻= fx[pc]`, `fz[pc] ⊻= fz[pt]`, and the same on `ref_z`/`ref_x`. This is also the entangling
step inside the T/S teleportation gadget, where it folds the data patch's frame onto the ancilla.
"""
function logical_CNOT!(M, pc, pt; cutoff = threshold)
    for q in data_coords; M.psi = apply(op("CNOT", site_of[(pc,q)], site_of[(pt,q)]), M.psi; cutoff); end
    M.fx[pt] .⊻= M.fx[pc]; M.fz[pc] .⊻= M.fz[pt]
    M.ref_z[pt] .⊻= M.ref_z[pc]; M.ref_x[pc] .⊻= M.ref_x[pt]
    M
end

"the 90-degree patch rotation on data coordinates, sigma:(x,y) -> (y, d-1-x)."
sigma_pt(q) = (q[2], (d - 1) - q[1])
"""
    sigma_cycles() -> Vector{Vector{Tuple{Int,Int}}}

Decompose the rotation `sigma_pt` into its qubit cycles, so the Hadamard can realise it physically as a
network of SWAP gates (moving atoms). Computed once at setup.
"""
function sigma_cycles()
    seen = Set{Tuple{Int,Int}}(); cycles = Vector{Vector{Tuple{Int,Int}}}()
    for q0 in data_coords
        q0 in seen && continue
        cyc = Tuple{Int,Int}[]; q = q0
        while !(q in seen); push!(cyc, q); push!(seen, q); q = sigma_pt(q); end
        length(cyc) > 1 && push!(cycles, cyc)
    end
    cycles
end
const SIGMA_CYCLES = sigma_cycles()
const SIGMA_PERM = [didx[sigma_pt(q)] for q in data_coords]      # data index i -> index of sigma(i)
sigma_auxf(a) = (a[2], (d - 1) - a[1])                          # the rotation on check coordinates
const ZAUX_TO_XAUX = Dict{Int,Int}(); const XAUX_TO_ZAUX = Dict{Int,Int}()
for (i,a) in enumerate(z_aux_local)
    j = findfirst(b -> all(isapprox.(b, sigma_auxf(a))), x_aux_local); ZAUX_TO_XAUX[i] = j
end
for (j,a) in enumerate(x_aux_local)
    i = findfirst(b -> all(isapprox.(b, sigma_auxf(a))), z_aux_local); XAUX_TO_ZAUX[j] = i
end

"""
    logical_H!(M, p; cutoff) -> M

Transversal logical Hadamard on patch `p`: physical `H` on every data qubit, then the 90° rotation via
the SWAP network (`SIGMA_CYCLES`) that maps the H-swapped checks back onto the standard layout. Frame
and reference are then transformed to match: `H` swaps X<->Z, and the rotation is a PULL-BACK
(`new[i] = old[sigma(i)]`) — for the frame on data qubits (`SIGMA_PERM`) and for the reference on checks
(`ZAUX_TO_XAUX`/`XAUX_TO_ZAUX`). Getting the pull-back direction wrong silently randomises post-H readout.
"""
function logical_H!(M, p; cutoff = threshold)
    for q in data_coords; M.psi = apply(op("H", site_of[(p, q)]), M.psi; cutoff); end
    for cyc in SIGMA_CYCLES, i in 1:(length(cyc)-1)
        M.psi = apply(op("SWAPg", site_of[(p, cyc[i])], site_of[(p, cyc[i+1])]), M.psi; cutoff); end
    oldfx = copy(M.fx[p]); oldfz = copy(M.fz[p]); nfx = zeros(Int, Ndata); nfz = zeros(Int, Ndata)
    for i in 1:Ndata; nfx[i] = oldfz[SIGMA_PERM[i]]; nfz[i] = oldfx[SIGMA_PERM[i]]; end
    M.fx[p] = nfx; M.fz[p] = nfz
    oldrz = copy(M.ref_z[p]); oldrx = copy(M.ref_x[p])
    nrz = zeros(Int, Nz_per_patch); nrx = zeros(Int, Nx_per_patch)
    for i in 1:Nz_per_patch; nrz[i] = oldrx[ZAUX_TO_XAUX[i]]; end
    for i in 1:Nx_per_patch; nrx[i] = oldrz[XAUX_TO_ZAUX[i]]; end
    M.ref_z[p] = nrz; M.ref_x[p] = nrx
    M
end
println("transversal gates + frame/reference propagation loaded")

# ---------------------------------------------------------------------------
# [engine cell 12]
# ---------------------------------------------------------------------------
"physical (Pauli type, support) representing a patch's logical Z or X in the standard orientation."
logop(L) = L === :Z ? (:Z, ZL_support) : (:X, XL_col0)
"""
    _apply_joint(psi, specs; cutoff) -> psi

Apply the product of logical operators named by `specs` (a list of `(patch, :Z/:X)`) to a COPY of `psi`.
Helper for the non-collapsing expectation in `joint_expect`.
"""
function _apply_joint(psi, specs; cutoff = threshold)
    Opsi = copy(psi)
    for (p, L) in specs
        basis, supp = logop(L)
        for q in supp; Opsi = apply(op(string(basis), site_of[(p,q)]), Opsi; cutoff); end
    end
    Opsi
end
"non-collapsing exact expectation <prod of logical Paulis> on the physical state (no frame correction)."
joint_expect(psi, specs; cutoff = threshold) = real(inner(psi, _apply_joint(psi, specs; cutoff)))

"""
    frame_sign(M, p, L) -> ±1

The sign the Pauli frame imposes on a logical `L` (:Z or :X) readout of patch `p`: a logical-Z readout
flips under X-frame on `ZL_support`; a logical-X readout flips under Z-frame on `XL_col0`.
"""
function frame_sign(M, p, L)
    if L === :Z; s = 0; for q in ZL_support; s ⊻= M.fx[p][didx[q]]; end; return 1 - 2s
    else; s = 0; for q in XL_col0; s ⊻= M.fz[p][didx[q]]; end; return 1 - 2s end
end
"frame-corrected two-patch correlator for L in {:Z,:X} (convenience wrapper over joint_expect)."
corr(M, b1, b2) = joint_expect(M.psi, [(1,b1),(2,b2)]) * frame_sign(M,1,b1) * frame_sign(M,2,b2)

"""
    _op_and_sign(M, p, L) -> (ops, phase, sign)

Build the physical operator list, phase, and frame sign for a single-patch logical `L` in `{:X,:Y,:Z}`.
`:Y = i X_L Z_L`; because `apply` composes right-to-left the built product needs phase `-i`, and its
frame sign is the product of the X_L and Z_L signs. Backs the general correlator `corr2`.
"""
function _op_and_sign(M, p, L)
    ops = Tuple{String,Tuple{Int,Int}}[]; ph = 1.0 + 0im
    if L === :Z
        for q in ZL_support; push!(ops,("Z",q)); end
        s=0; for q in ZL_support; s ⊻= M.fx[p][didx[q]]; end; sgn = 1-2s
    elseif L === :X
        for q in XL_col0; push!(ops,("X",q)); end
        s=0; for q in XL_col0; s ⊻= M.fz[p][didx[q]]; end; sgn = 1-2s
    else
        for q in XL_col0; push!(ops,("X",q)); end
        for q in ZL_support; push!(ops,("Z",q)); end
        ph = -1.0im
        s1=0; for q in XL_col0; s1 ⊻= M.fz[p][didx[q]]; end
        s2=0; for q in ZL_support; s2 ⊻= M.fx[p][didx[q]]; end; sgn = (1-2s1)*(1-2s2)
    end
    ops, ph, sgn
end
"""
    corr2(M, L1, L2; cutoff) -> Float64

Frame-corrected two-patch logical correlator <L1 (x) L2> with each factor in `{:X,:Y,:Z}` — the
verification readout (e.g. checking a Y error or a wrong-S error via `<YX>` as well as `<XX>`).
"""
function corr2(M, L1, L2; cutoff = threshold)
    o1,ph1,s1 = _op_and_sign(M,1,L1); o2,ph2,s2 = _op_and_sign(M,2,L2)
    Opsi = copy(M.psi)
    for (g,q) in o1; Opsi = apply(op(g,site_of[(1,q)]),Opsi;cutoff); end
    for (g,q) in o2; Opsi = apply(op(g,site_of[(2,q)]),Opsi;cutoff); end
    real(ph1*ph2*inner(M.psi,Opsi))*s1*s2
end
println("frame-corrected readout (incl. Y) loaded")

# ---------------------------------------------------------------------------
# [engine cell 14]
# ---------------------------------------------------------------------------
"""
    measure_patch_Z_raw!(M, p; cutoff) -> Int

Destructive logical-Z of patch `p`: measure every data qubit in Z, returning the raw parity over the
logical row `ZL_support`. Used to read out (and discard) the magic ancilla in the teleportation gadget;
the frame correction is applied separately by the caller.
"""
function measure_patch_Z_raw!(M, p; cutoff = threshold)
    par = 0
    for q in data_coords
        s = site_of[(p,q)]; sz = real(inner(M.psi', apply(op("Sz", s), M.psi; cutoff)))
        b = rand() < 0.5 + sz ? 0 : 1
        M.psi = apply(b == 0 ? P_up(s) : P_dn(s), M.psi; cutoff); M.psi = M.psi / sqrt(real(inner(M.psi,M.psi)))
        q in ZL_support && (par ⊻= b)
    end
    par
end

"""
    teleport_S!(M, p, R; C, B, cutoff, force_wrong, use_AD) -> M

Teleport a logical S onto patch `p` from a |Y> = S|+> ancilla: prep ancilla (patch 3), CNOT data->anc,
COMMIT (sliding-decode one R-round epoch on the ancilla), measure the ancilla, and apply the conditional
byproduct Z_L on `p` iff the frame-corrected outcome `m` is 1. `force_wrong` flips the committed bit.
"""
function teleport_S!(M, p, R; C, B, cutoff = threshold, force_wrong = false, use_AD = true)
    prepare_logical!(M, 3, :Y; cutoff); logical_CNOT!(M, p, 3; cutoff)
    run_epoch!(M, 3, R; C, B, use_AD, cutoff)
    m_raw = measure_patch_Z_raw!(M, 3; cutoff)
    fbit = 0; for q in ZL_support; fbit ⊻= M.fx[3][didx[q]]; end
    m = m_raw ⊻ fbit ⊻ (force_wrong ? 1 : 0)
    m == 1 && apply_ZL_phys!(M, p; cutoff); M
end

"""
    teleport_T!(M, p, R; C, B, cutoff, force_wrong, use_AD) -> M

Teleport a logical T onto patch `p` from a |A> = T|+> ancilla, same structure as `teleport_S!`; the
byproduct here is a conditional S (itself teleported). The commit epoch protects the outcome against
seam/measurement errors before the frame bit is consumed.
"""
function teleport_T!(M, p, R; C, B, cutoff = threshold, force_wrong = false, use_AD = true)
    prepare_logical!(M, 3, :A; cutoff); logical_CNOT!(M, p, 3; cutoff)
    run_epoch!(M, 3, R; C, B, use_AD, cutoff)
    m_raw = measure_patch_Z_raw!(M, 3; cutoff)
    fbit = 0; for q in ZL_support; fbit ⊻= M.fx[3][didx[q]]; end
    m = m_raw ⊻ fbit ⊻ (force_wrong ? 1 : 0)
    m == 1 && teleport_S!(M, p, R; C, B, cutoff); M
end
println("T-gate commitment loaded")

# ---------------------------------------------------------------------------
# [engine cell 16]
# ---------------------------------------------------------------------------
"""
    apply_logical!(M, gate, R; C, B, use_AD, cutoff) -> M

Dispatch one logical gate onto the machine. `gate` is `(:X|:Z|:H|:S|:T, patch)` or `(:CNOT, ctrl, tgt)`;
`R,C,B,use_AD` are forwarded to the T/S commitment epochs.
"""
function apply_logical!(M, gate, R; C, B, use_AD, cutoff = threshold)
    g = gate[1]
    if     g === :X;    return logical_X!(M, gate[2]; cutoff)
    elseif g === :Z;    return logical_Z!(M, gate[2]; cutoff)
    elseif g === :H;    return logical_H!(M, gate[2]; cutoff)
    elseif g === :S;    return teleport_S!(M, gate[2], R; C, B, use_AD, cutoff)
    elseif g === :T;    return teleport_T!(M, gate[2], R; C, B, use_AD, cutoff)
    elseif g === :CNOT; return logical_CNOT!(M, gate[2], gate[3]; cutoff)
    else; error("unknown gate $g"); end
end

"""
    run_circuit(circuit; ec, R, C, B, use_AD, errors, meas, cutoff) -> Machine

Run a full 2-qubit circuit on freshly prepared |0>_L (x) |0>_L. After each gate (when `ec`), run one
sliding-decoded idle epoch on both data patches.

- `circuit` : list of gate tuples (see `apply_logical!`).
- `R,C,B`   : epoch length and sliding-window commit/buffer sizes; `use_AD` toggles artificial defects.
- `errors :: Vector{(gate_k, round, patch, "X"/"Z", coord)}` : inject a data Pauli into that gate's epoch.
- `meas   :: Vector{(gate_k, round, patch, :Z/:X, check_index)}` : inject a measurement error.
Returns the final `Machine`; read out with `corr`/`corr2`.
"""
function run_circuit(circuit; ec=true, R=6, C=2, B=2, use_AD=true, errors=[], meas=[], cutoff=threshold)
    M = Machine(MPS(sites, "Up"))
    prepare_logical!(M, 1, :zero; cutoff); prepare_logical!(M, 2, :zero; cutoff)
    for (k, gate) in enumerate(circuit)
        apply_logical!(M, gate, R; C, B, use_AD, cutoff)
        if ec
            de1 = [(rr,P,q) for (kk,rr,p,P,q) in errors if kk==k && p==1]
            de2 = [(rr,P,q) for (kk,rr,p,P,q) in errors if kk==k && p==2]
            me1 = [(rr,t,si) for (kk,rr,p,t,si) in meas if kk==k && p==1]
            me2 = [(rr,t,si) for (kk,rr,p,t,si) in meas if kk==k && p==2]
            run_epoch!(M, 1, R; data_errs=de1, meas_errs=me1, C, B, use_AD, cutoff)
            run_epoch!(M, 2, R; data_errs=de2, meas_errs=me2, C, B, use_AD, cutoff)
        end
    end
    M
end
println("runner ready")

# ---------------------------------------------------------------------------
# [engine cell 18]
# ---------------------------------------------------------------------------
const _I = ComplexF64[1 0; 0 1];  const _X = ComplexF64[0 1; 1 0]
const _Z = ComplexF64[1 0; 0 -1]; const _H = ComplexF64[1 1; 1 -1]/sqrt(2)
const _S = ComplexF64[1 0; 0 im]; const _T = ComplexF64[1 0; 0 exp(im*π/4)]; const _Y = ComplexF64[0 -im; im 0]
"embed a single-qubit gate `g` on qubit `p` (1 or 2) of the 2-qubit reference."
_1q(g, p) = p == 1 ? kron(g, _I) : kron(_I, g)
const _CN12 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
"""
    ref_state(circuit) -> Vector{ComplexF64}

Exact 4-amplitude statevector after applying `circuit` (same gate tuples as `run_circuit`) to |00>.
"""
function ref_state(circuit)
    ψ = ComplexF64[1, 0, 0, 0]
    for gate in circuit
        g = gate[1]
        M = g===:X ? _1q(_X,gate[2]) : g===:Z ? _1q(_Z,gate[2]) : g===:H ? _1q(_H,gate[2]) :
            g===:S ? _1q(_S,gate[2]) : g===:T ? _1q(_T,gate[2]) : g===:CNOT ? _CN12 : error()
        ψ = M * ψ
    end
    ψ
end
_Pd = Dict(:X=>_X, :Y=>_Y, :Z=>_Z)
"exact two-qubit correlator <a (x) b>, a,b in {:X,:Y,:Z}, on reference state ψ."
ref2(ψ, a, b) = real(ψ' * kron(_Pd[a], _Pd[b]) * ψ)
println("exact reference loaded")

