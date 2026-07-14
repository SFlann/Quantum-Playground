# ============================================================================
#  scaling_engine.jl  —  d-parameterised MPS surface-code engine (union-find)
#
#  This is a refactor of 7.Modelling-Noisy-Circuits/noisy_engine.jl built for a
#  SCALING study: the code distance `d` is a runtime parameter (a `Code` object)
#  instead of a module-level `const`, so patches of several distances can coexist
#  in one session and be swept. Three things change relative to notebook 7:
#
#   1. DECODER = UNION-FIND.  The exact Held-Karp subset-DP MWPM was O(2^n) in the
#      defect count and the gauge-seeding lookup table was O(2^(d^2)) — the latter
#      alone hangs before d=5 even loads. Both are replaced by a Delfosse-Nickerson
#      union-find decoder (weighted-growth clusters + peeling), which is near-linear
#      and decodes each connected defect cluster independently (the "component
#      decomposition" that keeps cost bounded at low p).
#
#   2. GRAPH CACHE.  The (2+1)-D matching-graph geometry is static per window width,
#      so it is built once and cached per `(check-type, w)` instead of rebuilt inside
#      the sliding loop every window/epoch/shot.
#
#   3. MPS TRUNCATION CONTROL.  `cutoff` still sets numerical precision; a `maxdim`
#      cap now bounds bond growth, and every gate application is routed through `_ap`
#      which records the largest bond dimension seen and FLAGS when the cap binds
#      (so an under-resolved d=7 run is never silently trusted).
#
#  Scope: transversal Clifford gates + magic-state S/T teleportation on THREE
#  patches (2 data + 1 magic ancilla), i.e. non-Clifford two-qubit circuits — the
#  regime that genuinely needs the MPS. No lattice surgery. See notebooks 6 and 7
#  for the derivation of the frame/reference bookkeeping reused verbatim here.
# ============================================================================

using ITensors, ITensorMPS, LinearAlgebra, Random, Printf
Random.seed!(0xB011)

ITensors.op(::OpName"T", ::SiteType"S=1/2") = ComplexF64[1 0; 0 exp(im*π/4)]
ITensors.op(::OpName"S", ::SiteType"S=1/2") = ComplexF64[1 0; 0 im]
ITensors.op(::OpName"SWAPg", ::SiteType"S=1/2") = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

# ---------------------------------------------------------------------------
#  Global MPS-truncation context (single-threaded learning tool → globals are fine)
# ---------------------------------------------------------------------------
const CUTOFF     = Ref(1e-6)          # SVD truncation cutoff (numerical precision)
const MAXDIM     = Ref(1_000_000)     # bond-dimension cap (huge = effectively uncapped)
const CHIMAX     = Ref(0)             # largest bond dimension seen since last reset
const CAP_HITS   = Ref(0)             # # gate-applies where the maxdim cap was binding

"Reset the bond-dimension diagnostics (`CHIMAX`, `CAP_HITS`) before a fresh run."
reset_trunc!() = (CHIMAX[] = 0; CAP_HITS[] = 0)

"""
    set_precision!(; cutoff, maxdim) -> nothing

Set the global MPS truncation controls used by every gate application.
`cutoff` is the SVD discard threshold (numerical precision); `maxdim` caps the bond
dimension (pass a large number to leave it effectively uncapped). Also resets the
`CHIMAX`/`CAP_HITS` diagnostics.
"""
function set_precision!(; cutoff = 1e-6, maxdim = 1_000_000)
    CUTOFF[] = cutoff; MAXDIM[] = maxdim; reset_trunc!(); nothing
end

"""
    _ap(psi, g) -> psi

Apply gate ITensor `g` to MPS `psi` with the global `cutoff`/`maxdim`, then update the
truncation diagnostics: record the largest bond dimension in `CHIMAX` and, if the new
bond dimension reaches the `maxdim` cap, increment `CAP_HITS` (the cap is binding, so the
state may be under-resolved). This is the single choke-point through which all state
evolution passes.
"""
function _ap(psi, g)
    phi = apply(g, psi; cutoff = CUTOFF[], maxdim = MAXDIM[])
    χ = maxlinkdim(phi)
    χ > CHIMAX[] && (CHIMAX[] = χ)
    χ >= MAXDIM[] && (CAP_HITS[] += 1)
    phi
end
"Non-collapsing expectation ⟨psi|g|psi⟩ (a measurement probe; does not touch diagnostics)."
_ex(psi, g) = real(inner(psi', apply(g, psi; cutoff = CUTOFF[])))

# ===========================================================================
#  §1  The code geometry — a `Code` object built for a chosen distance `d`
# ===========================================================================
"""
    Code

All distance-`d` geometry for a rotated planar surface code laid out on `npatch` patches
(here 3: two data patches + one magic ancilla). Replaces notebook 7's module-level `const`s
so several distances can coexist. Fields:
- `d, npatch`                : distance and patch count.
- `data_coords`              : the `d^2` data-qubit integer coordinates.
- `z_aux, x_aux`             : Z- and X-check half-integer coordinates.
- `sites, site_of, didx`     : ITensor indices; `(patch,coord)->index`; `datacoord->1..d^2`.
- `Ndata, Nz, Nx`            : counts per patch.
- `XL_col0, ZL_support`      : logical-X (left column) and logical-Z (bottom row) supports.
- `z_support, x_support`     : check coord -> set of data qubits it acts on.
- `edge_data,bedge_data`     : Z matching-graph geometry (spatial / boundary edges).
- `xedge_data,xbedge_data`   : X matching-graph geometry.
- `sigma_perm, sigma_cycles` : the 90° rotation on data qubits (for transversal H).
- `zaux2xaux, xaux2zaux`     : the rotation on checks (for transversal H).
- `zcache, xcache`           : window-width -> cached matching graph (built lazily).
"""
struct Code
    d::Int; npatch::Int
    data_coords::Vector{Tuple{Int,Int}}
    z_aux::Vector{Tuple{Float64,Float64}}; x_aux::Vector{Tuple{Float64,Float64}}
    sites::Vector{<:Index}
    site_of::Dict{Tuple{Int,Tuple{Int,Int}},Index}
    site_of_aux::Dict{Tuple{Int,Tuple{Float64,Float64}},Index}
    didx::Dict{Tuple{Int,Int},Int}
    Ndata::Int; Nz::Int; Nx::Int
    XL_col0::Vector{Tuple{Int,Int}}; ZL_support::Vector{Tuple{Int,Int}}
    z_support::Dict{Tuple{Float64,Float64},Set{Tuple{Int,Int}}}
    x_support::Dict{Tuple{Float64,Float64},Set{Tuple{Int,Int}}}
    edge_data::Dict{Tuple{Int,Int},Tuple{Int,Int}};  bedge_data::Dict{Int,Tuple{Int,Int}}
    xedge_data::Dict{Tuple{Int,Int},Tuple{Int,Int}}; xbedge_data::Dict{Int,Tuple{Int,Int}}
    sigma_perm::Vector{Int}; sigma_cycles::Vector{Vector{Tuple{Int,Int}}}
    zaux2xaux::Dict{Int,Int}; xaux2zaux::Dict{Int,Int}
    zcache::Dict{Int,Any}; xcache::Dict{Int,Any}
end

"data qubits a check at half-integer coord `ac` touches (its ≤4 diagonal neighbours)."
_nbrs(data_coords, ac) = [q for q in data_coords if abs(q[1]-ac[1])==0.5 && abs(q[2]-ac[2])==0.5]

"""
    build_code(d; npatch=3) -> Code

Construct the full distance-`d` geometry (data/check layout, MPS site indices, logical
supports, and both matching-graph geometries) for `npatch` side-by-side patches. `d` must be
odd (the 90° rotation used by the transversal Hadamard maps the code onto itself only for
odd `d`). This is the single place distance enters; everything downstream reads the `Code`.
"""
function build_code(d::Int; npatch::Int = 3)
    isodd(d) || error("d must be odd (got $d)")
    data_coords = vec([(x, y) for x in 0:(d-1), y in 0:(d-1)])
    z_aux = Tuple{Float64,Float64}[]; x_aux = Tuple{Float64,Float64}[]
    for x in 0:(d-2), y in 0:(d-2)
        push!((x + y) % 2 == 0 ? z_aux : x_aux, (x + 0.5, y + 0.5))
    end
    for y in 1:2:(d-2); push!(z_aux, (-0.5,   y + 0.5)); end
    for y in 0:2:(d-2); push!(z_aux, (d - 0.5, y + 0.5)); end
    for x in 0:2:(d-2); push!(x_aux, (x + 0.5, -0.5));    end
    for x in 1:2:(d-2); push!(x_aux, (x + 0.5, d - 0.5)); end

    # MPS site ordering: per patch, data then that patch's Z- then X-ancillas (patches
    # side-by-side, so a non-local cross-patch CNOT visibly costs bond dimension — see the
    # scaling notebook's discussion). Ancillas kept next to their patch's data.
    dkeys = Tuple{Int,Tuple{Int,Int}}[]; akeys = Tuple{Int,Tuple{Float64,Float64}}[]
    order = Tuple{Any,Any}[]   # (patch, coord) in MPS order, coord Int or Float tuple
    for p in 1:npatch
        for q in data_coords; push!(order, (p, q)); end
        for a in z_aux;       push!(order, (p, a)); end
        for a in x_aux;       push!(order, (p, a)); end
    end
    sites = siteinds("S=1/2", length(order))
    site_of = Dict{Tuple{Int,Tuple{Int,Int}},Index}()
    site_of_aux = Dict{Tuple{Int,Tuple{Float64,Float64}},Index}()
    for (i,(p,c)) in enumerate(order)
        if c isa Tuple{Int,Int}; site_of[(p,c)] = sites[i]
        else; site_of_aux[(p,c)] = sites[i]; end
    end

    didx = Dict(q => i for (i,q) in enumerate(data_coords))
    Ndata = length(data_coords); Nz = length(z_aux); Nx = length(x_aux)
    XL_col0    = [(0, y) for y in 0:(d-1)]
    ZL_support = [(x, 0) for x in 0:(d-1)]
    z_support = Dict(ac => Set(_nbrs(data_coords, ac)) for ac in z_aux)
    x_support = Dict(ac => Set(_nbrs(data_coords, ac)) for ac in x_aux)

    # matching-graph geometry: each data qubit touching two checks of a type is a spatial edge
    # between them; touching one is a boundary edge.
    zc(q) = [i for (i,a) in enumerate(z_aux) if abs(a[1]-q[1])==0.5 && abs(a[2]-q[2])==0.5]
    xc(q) = [i for (i,a) in enumerate(x_aux) if abs(a[1]-q[1])==0.5 && abs(a[2]-q[2])==0.5]
    edge_data = Dict{Tuple{Int,Int},Tuple{Int,Int}}(); bedge_data = Dict{Int,Tuple{Int,Int}}()
    for q in data_coords
        zs = zc(q)
        length(zs)==2 ? (edge_data[minmax(zs[1],zs[2])] = q) :
        length(zs)==1 ? (bedge_data[zs[1]] = q) : nothing
    end
    xedge_data = Dict{Tuple{Int,Int},Tuple{Int,Int}}(); xbedge_data = Dict{Int,Tuple{Int,Int}}()
    for q in data_coords
        xs = xc(q)
        length(xs)==2 ? (xedge_data[minmax(xs[1],xs[2])] = q) :
        length(xs)==1 ? (xbedge_data[xs[1]] = q) : nothing
    end

    # 90° rotation for the transversal Hadamard
    sigma_pt(q)   = (q[2], (d - 1) - q[1])
    sigma_auxf(a) = (a[2], (d - 1) - a[1])
    sigma_perm = [didx[sigma_pt(q)] for q in data_coords]
    seen = Set{Tuple{Int,Int}}(); sigma_cycles = Vector{Vector{Tuple{Int,Int}}}()
    for q0 in data_coords
        q0 in seen && continue
        cyc = Tuple{Int,Int}[]; q = q0
        while !(q in seen); push!(cyc, q); push!(seen, q); q = sigma_pt(q); end
        length(cyc) > 1 && push!(sigma_cycles, cyc)
    end
    zaux2xaux = Dict{Int,Int}(); xaux2zaux = Dict{Int,Int}()
    for (i,a) in enumerate(z_aux)
        zaux2xaux[i] = findfirst(b -> all(isapprox.(b, sigma_auxf(a))), x_aux)
    end
    for (j,a) in enumerate(x_aux)
        xaux2zaux[j] = findfirst(b -> all(isapprox.(b, sigma_auxf(a))), z_aux)
    end

    Code(d, npatch, data_coords, z_aux, x_aux, sites, site_of, site_of_aux, didx,
         Ndata, Nz, Nx, XL_col0, ZL_support, z_support, x_support,
         edge_data, bedge_data, xedge_data, xbedge_data,
         sigma_perm, sigma_cycles, zaux2xaux, xaux2zaux,
         Dict{Int,Any}(), Dict{Int,Any}())
end

"convenience accessors for a data / ancilla site index."
sd(c::Code, p, q::Tuple{Int,Int})       = c.site_of[(p,q)]
sa(c::Code, p, a::Tuple{Float64,Float64}) = c.site_of_aux[(p,a)]

# ===========================================================================
#  §2  MPS stabiliser-extraction primitives (threaded with a `Code`)
# ===========================================================================
"projector onto |0> / |1> for site `s`."
P_up(s) = 0.5*op("Id", s) + op("Sz", s)
P_dn(s) = 0.5*op("Id", s) - op("Sz", s)

"""
    measure_Z!(psi, site) -> (bit, psi)

Projectively measure one physical qubit in Z, sampling the Born-rule outcome and returning the
collapsed, renormalised state. The primitive every ancilla read-out is built from.
"""
function measure_Z!(psi, site)
    sz = _ex(psi, op("Sz", site))
    if rand() < 0.5 + sz; psi = _ap(psi, P_up(site)); bit = 0
    else; psi = _ap(psi, P_dn(site)); bit = 1 end
    bit, psi / sqrt(real(inner(psi, psi)))
end

"reset ancilla `aux` to |0> (X-flip it if found in |1>) before its next round."
reset_aux!(psi, aux) = (_ex(psi, op("Sz", aux)) < 0 ? _ap(psi, op("X", aux)) : psi)

"""
    measure_Z_stab(psi, c, p, ac) -> (bit, psi)

Measure one Z-plaquette of patch `p` (check at `ac`): CNOT its data neighbours onto the ancilla
in a hook-error-avoiding order, then read the ancilla in Z. `bit=0` ⇒ satisfied. Detects X errors.
"""
function measure_Z_stab(psi, c::Code, p, ac)
    aux = sa(c, p, ac); nbrs = _nbrs(c.data_coords, ac)
    ord = length(nbrs) == 4 ? [2,4,1,3] : [1,2]
    for q in nbrs[ord]; psi = _ap(psi, op("CNOT", sd(c,p,q), aux)); end
    measure_Z!(psi, aux)
end

"""
    measure_X_stab(psi, c, p, ac) -> (bit, psi)

Measure one X-plaquette of patch `p`: conjugate the ancilla with H so its CNOTs read X-parity,
then measure in Z. Detects Z errors.
"""
function measure_X_stab(psi, c::Code, p, ac)
    aux = sa(c, p, ac); psi = _ap(psi, op("H", aux))
    nbrs = _nbrs(c.data_coords, ac); ord = length(nbrs) == 4 ? [2,1,4,3] : [1,2]
    for q in nbrs[ord]; psi = _ap(psi, op("CNOT", aux, sd(c,p,q))); end
    psi = _ap(psi, op("H", aux)); measure_Z!(psi, aux)
end

"""
    measure_raw_syndrome(psi, c, p) -> (z, x, psi)

One full round of both check types on patch `p`: the length-`Nz` Z-syndrome (detects X errors) and
length-`Nx` X-syndrome (detects Z errors), plus the collapsed state. Nothing is corrected.
"""
function measure_raw_syndrome(psi, c::Code, p)
    z = Int[]; for ac in c.z_aux
        psi = reset_aux!(psi, sa(c,p,ac)); b,psi = measure_Z_stab(psi,c,p,ac); push!(z,b); end
    x = Int[]; for ac in c.x_aux
        psi = reset_aux!(psi, sa(c,p,ac)); b,psi = measure_X_stab(psi,c,p,ac); push!(x,b); end
    z, x, psi
end

"single-qubit prep sequence that seeds logical `sym` on a patch corner before spreading logical-X."
seed_ops(sym) = sym === :zero ? String[] : sym === :one ? ["X"] :
                sym === :plus ? ["H"]   : sym === :minus ? ["H","Z"] :
                sym === :A ? ["H","T"]  : sym === :Y ? ["H","S"] : error("seed $sym")

"reset all data qubits of patch `p` to |0>."
function reset_patch_data!(psi, c::Code, p)
    for q in c.data_coords
        s = sd(c,p,q); sz = _ex(psi, op("Sz", s))
        b = rand() < 0.5 + sz ? 0 : 1
        psi = _ap(psi, b == 0 ? P_up(s) : P_dn(s)); psi = psi / sqrt(real(inner(psi, psi)))
        b == 1 && (psi = _ap(psi, op("X", s)))
    end
    psi
end

# ===========================================================================
#  §3  The union-find decoder
# ===========================================================================
# The matching graph for `w` rounds of one check type. Nodes are (check,round) detectors
# 1..w*Nstab plus a single conceptual boundary `BND=w*Nstab+1` (never unioned through — see
# `uf_decode`). Edges are elementary faults, each tagged by `edge_kind`:
#   :spatial / :boundary -> a data-qubit error (committed as a flip),
#   :time                -> a measurement error (drives artificial defects, no data flip).
# Geometry is static per `w`, so `get_graph` caches it per check type.
"""
    build_window_graph(w, Nstab, edge_data, bedge_data) -> NamedTuple

Build (and return) the elementary (2+1)-D matching graph for `w` rounds: internal edges
(`int_edges`, spatial+time), boundary edges (`bnd_edges`, a node per boundary fault), a
per-edge fault tag (`ekind`), and the `node_id(i,r)` / `BND` helpers. No all-pairs distances
are computed (union-find does not need them). Cached by `get_graph`.
"""
function build_window_graph(w::Int, Nstab::Int, edge_data, bedge_data)
    node_id(i,r) = (r-1)*Nstab + i
    BND = w*Nstab + 1
    int_edges = Tuple{Int,Int}[]; int_kind = Tuple{Symbol,Any,Int}[]
    bnd_edges = Int[];            bnd_kind = Tuple{Symbol,Any,Int}[]
    for r in 1:w, ((i,j),q) in edge_data                       # spatial
        push!(int_edges, (node_id(i,r), node_id(j,r))); push!(int_kind, (:spatial, q, r))
    end
    for r in 1:(w-1), i in 1:Nstab                             # time
        push!(int_edges, (node_id(i,r), node_id(i,r+1))); push!(int_kind, (:time, (i,r), r))
    end
    for r in 1:w, (i,q) in bedge_data                          # boundary
        push!(bnd_edges, node_id(i,r)); push!(bnd_kind, (:boundary, q, r))
    end
    # adjacency of internal edges per node (for union-find growth)
    nnodes = w*Nstab
    node_int = [Int[] for _ in 1:nnodes]
    for (ei,(u,v)) in enumerate(int_edges); push!(node_int[u], ei); push!(node_int[v], ei); end
    ekind = Dict{Tuple{Int,Int},Tuple{Symbol,Any,Int}}()
    for (ei,(u,v)) in enumerate(int_edges); ekind[minmax(u,v)] = int_kind[ei]; end
    for (bi,u) in enumerate(bnd_edges);     ekind[minmax(u,BND)] = bnd_kind[bi]; end
    (; w, Nstab, BND, nnodes, node_id, int_edges, bnd_edges, node_int, ekind)
end

"fetch the cached matching graph for width `w` and check type (`:z`/`:x`), building it on first use."
function get_graph(c::Code, checktype::Symbol, w::Int)
    cache, Nstab, ed, bed = checktype === :z ?
        (c.zcache, c.Nz, c.edge_data, c.bedge_data) :
        (c.xcache, c.Nx, c.xedge_data, c.xbedge_data)
    get!(cache, w) do; build_window_graph(w, Nstab, ed, bed); end
end

# simple union-find with path compression + per-root defect count and boundary flag
mutable struct UF
    parent::Vector{Int}; rank::Vector{Int}
    dcount::Vector{Int}; bnd::BitVector
end
UF(n) = UF(collect(1:n), zeros(Int,n), zeros(Int,n), falses(n))
function _find(uf::UF, x)
    while uf.parent[x] != x; uf.parent[x] = uf.parent[uf.parent[x]]; x = uf.parent[x]; end
    x
end
function _union!(uf::UF, a, b)
    ra = _find(uf,a); rb = _find(uf,b); ra == rb && return ra
    uf.rank[ra] < uf.rank[rb] && ((ra,rb) = (rb,ra))
    uf.parent[rb] = ra
    uf.dcount[ra] += uf.dcount[rb]; uf.bnd[ra] |= uf.bnd[rb]
    uf.rank[ra] == uf.rank[rb] && (uf.rank[ra] += 1)
    ra
end
_neutral(uf::UF, r) = iseven(uf.dcount[r]) || uf.bnd[r]

"""
    uf_decode(g, defects) -> Vector{Tuple{Int,Int}}

Delfosse-Nickerson union-find decode of one window. `defects` are lit detector node ids on graph
`g`. Grow every odd (non-neutral) cluster by half-edges until all clusters are neutral (even defect
count, or a boundary edge filled), then peel each cluster's spanning forest to a minimum edge set
that explains its defects. Returns the chosen elementary edges as `(u,v)` node pairs (`v==g.BND` is
a boundary edge). Only defect-containing clusters ever grow, so disconnected clusters are decoded
independently — the "component decomposition" that bounds cost at low `p`.
"""
function uf_decode(g, defects)
    isempty(defects) && return Tuple{Int,Int}[]
    uf = UF(g.nnodes)
    isdef = falses(g.nnodes); for u in defects; isdef[u] = true; uf.dcount[u] = 1; end
    support  = zeros(Int, length(g.int_edges))    # 0,1,2 half-grown per internal edge
    bsupport = zeros(Int, length(g.bnd_edges))    # 0,1,2 per boundary edge

    while true
        growing = Set{Int}()
        for u in defects
            r = _find(uf, u); _neutral(uf, r) || push!(growing, r)
        end
        isempty(growing) && break
        newint = Int[]; newbnd = Int[]
        for (ei,(u,v)) in enumerate(g.int_edges)
            support[ei] == 2 && continue
            ru = _find(uf,u); rv = _find(uf,v); ru == rv && continue
            inc = (ru in growing) + (rv in growing); inc == 0 && continue
            support[ei] = min(2, support[ei] + inc)
            support[ei] == 2 && push!(newint, ei)
        end
        for (bi,u) in enumerate(g.bnd_edges)
            bsupport[bi] == 2 && continue
            (_find(uf,u) in growing) || continue
            bsupport[bi] += 1
            bsupport[bi] == 2 && push!(newbnd, bi)
        end
        for ei in newint; (u,v) = g.int_edges[ei]; _union!(uf, u, v); end
        for bi in newbnd; uf.bnd[_find(uf, g.bnd_edges[bi])] = true; end
    end

    # peel each neutral cluster's spanning forest
    grown_int = [g.int_edges[ei] for ei in eachindex(g.int_edges) if support[ei] == 2]
    grown_bnd = Set(g.bnd_edges[bi] for bi in eachindex(g.bnd_edges) if bsupport[bi] == 2)
    adj = Dict{Int,Vector{Int}}()
    push_adj!(a,b) = (push!(get!(adj,a,Int[]), b); push!(get!(adj,b,Int[]), a))
    for (u,v) in grown_int; push_adj!(u,v); end
    for u in grown_bnd;     push_adj!(u, g.BND); end
    nodes = Set{Int}(); for (u,v) in grown_int; push!(nodes,u); push!(nodes,v); end
    for u in grown_bnd; push!(nodes,u); end; for u in defects; push!(nodes,u); end

    groups = Dict{Int,Vector{Int}}()   # cluster root -> its nodes (BND excluded)
    for n in nodes; push!(get!(groups, _find(uf,n), Int[]), n); end
    correction = Tuple{Int,Int}[]
    for (_, gnodes) in groups
        gset = Set(gnodes)                      # this cluster's nodes (BND is shared, excluded)
        hasb = any(n in grown_bnd for n in gnodes)
        start = hasb ? g.BND : gnodes[1]
        # BFS spanning tree from `start`, staying inside this cluster (BND's adjacency lists
        # boundary nodes of OTHER clusters too, so restrict every hop to `gset` ∪ {BND}).
        parent = Dict{Int,Int}(start => start); order = [start]
        queue = [start]; head = 1
        while head <= length(queue)
            u = queue[head]; head += 1
            for v in get(adj, u, Int[])
                (v == g.BND || v in gset) || continue
                haskey(parent, v) && continue
                parent[v] = u; push!(order, v); push!(queue, v)
            end
        end
        synd = Dict(n => (n <= g.nnodes && isdef[n]) for n in order)
        for u in Iterators.reverse(order)
            u == start && continue
            p = parent[u]
            if u != g.BND && synd[u]
                push!(correction, (u, p))
                p != g.BND && (synd[p] = !synd[p])
            end
        end
    end
    correction
end

"the fault tag of the elementary edge `(u,v)` on graph `g`."
edge_kind(g, u, v) = g.ekind[minmax(u,v)]

"""
    commit_edges(g, edges; rounds, off) -> Set{Tuple{Int,Int}}

Turn a decoder's chosen edges into the net data-qubit flips they prescribe. Spatial/boundary edges
name a data qubit; time edges (measurement errors) are skipped. `rounds` (global) selects which
window rounds to commit (the sliding core); `off` maps window-local rounds to global. Even
multiplicities cancel.
"""
function commit_edges(g, edges; rounds=nothing, off=0)
    cnt = Dict{Tuple{Int,Int},Int}()
    for (u,v) in edges
        k = edge_kind(g,u,v); k[1] === :time && continue
        gr = k[3] + off
        (rounds === nothing || gr in rounds) && (cnt[k[2]] = get(cnt,k[2],0) + 1)
    end
    Set(q for (q,cn) in cnt if isodd(cn))
end

"the check indices whose time edge crosses the commit surface `surf_lo` (handed to the next window as artificial defects)."
function ad_from_edges(g, edges, surf_lo)
    ad = Int[]
    for (u,v) in edges
        k = edge_kind(g,u,v)
        (k[1] === :time && k[3] == surf_lo) && push!(ad, k[2][1])
    end
    ad
end

"""
    sliding_decode(c, hist, reference, checktype; C, B, use_AD) -> Set{Tuple{Int,Int}}

Sliding-window union-find decode of one epoch's raw syndrome history for one check type.
`hist[r]` is the round-r raw syndrome; detectors are `det[r]=hist[r] XOR hist[r-1]` with
`hist[0]=reference`. Windows of width `W=C+B` step by the commit core `C`; each commits its core
rounds' data flips and, when `use_AD`, hands surface-crossing time edges to the next window as
pre-lit artificial defects. Returns the union of committed data-qubit flips.
"""
function sliding_decode(c::Code, hist, reference, checktype::Symbol; C, B, use_AD=true)
    Nstab = checktype === :z ? c.Nz : c.Nx
    R = length(hist)
    det = Vector{Vector{Int}}(undef, R); prev = reference
    for r in 1:R; det[r] = hist[r] .⊻ prev; prev = hist[r]; end
    corr = Set{Tuple{Int,Int}}(); AD = Int[]; W = C + B; start = 1
    while start <= R
        stop = min(start+W-1, R); w = stop-start+1; last_window = stop == R
        g = get_graph(c, checktype, w)
        subdet = [copy(det[start+rl-1]) for rl in 1:w]
        for i in AD; subdet[1][i] ⊻= 1; end
        defects = [g.node_id(i,rl) for rl in 1:w for i in 1:Nstab if subdet[rl][i]==1]
        edges = uf_decode(g, defects)
        core_end = last_window ? stop : start+C-1
        corr = symdiff(corr, commit_edges(g, edges; rounds=start:core_end, off=start-1))
        AD = (use_AD && !last_window) ? ad_from_edges(g, edges, core_end-(start-1)) : Int[]
        start = core_end + 1
    end
    corr
end

# ===========================================================================
#  §4  Machine, gauge seeding (union-find, replacing the 2^(d^2) table), epochs
# ===========================================================================
"""
    Machine

Simulation state for a `Code`: the MPS `psi`; per-patch reference syndromes `ref_z`/`ref_x` (the
gauge errors are measured against); and the software Pauli frame `fx`/`fz` (per-patch, per-data-qubit
estimated X/Z error) consumed by readout and the T/S commitment.
"""
mutable struct Machine
    code::Code
    psi
    ref_z::Dict{Int,Vector{Int}}; ref_x::Dict{Int,Vector{Int}}
    fx::Dict{Int,Vector{Int}};     fz::Dict{Int,Vector{Int}}
end
Machine(c::Code) = Machine(c, MPS(c.sites, "Up"), Dict(), Dict(), Dict(), Dict())

"""
    seed_frame_uf(c, checktype, lit) -> Vector{Tuple{Int,Int}}

Gauge-seed helper: single-round union-find decode of a reference syndrome (`lit` = lit check
indices). Replaces notebook 7's `build_lookup`, whose 2^(d^2) brute force hangs before d=5.
Returns the data qubits whose frame bit the reference gauge flips.
"""
function seed_frame_uf(c::Code, checktype::Symbol, lit::Vector{Int})
    isempty(lit) && return Tuple{Int,Int}[]
    g = get_graph(c, checktype, 1)                 # one round → spatial+boundary edges only
    defects = [g.node_id(i,1) for i in lit]
    collect(commit_edges(g, uf_decode(g, defects)))
end

"""
    prepare_logical!(M, p, sym) -> M

Prepare patch `p` in logical state `sym` with NO physical feed-forward: reset data, seed the state,
spread logical-X across the left column, take one syndrome round whose random outcome becomes the
REFERENCE, and absorb that random gauge into the initial Pauli frame via `seed_frame_uf`.
"""
function prepare_logical!(M::Machine, p, sym)
    c = M.code
    M.psi = reset_patch_data!(M.psi, c, p)
    for g in seed_ops(sym); M.psi = _ap(M.psi, op(g, sd(c,p,(0,0)))); end
    for q in c.XL_col0[2:end]; M.psi = _ap(M.psi, op("H", sd(c,p,q))); end
    z, x, M.psi = measure_raw_syndrome(M.psi, c, p)
    M.ref_z[p] = z; M.ref_x[p] = x
    M.fx[p] = zeros(Int, c.Ndata); M.fz[p] = zeros(Int, c.Ndata)
    zl = sort([i for (i,b) in enumerate(z) if b == 1])
    xl = sort([i for (i,b) in enumerate(x) if b == 1])
    for q in seed_frame_uf(c, :z, zl); M.fx[p][c.didx[q]] ⊻= 1; end
    for q in seed_frame_uf(c, :x, xl); M.fz[p][c.didx[q]] ⊻= 1; end
    # The min-weight syndrome match lands in a RANDOM logical class; the freshly prepared state is
    # logically trivial (physical <Z_L>=<X_L>=±1 as intended), so force the frame to be a pure
    # destabiliser — even overlap with each logical support. A full logical operator has zero
    # syndrome, so XOR-ing it in fixes the parity without changing what the frame corrects.
    isodd(sum(M.fx[p][c.didx[q]] for q in c.ZL_support)) &&
        (for q in c.XL_col0;    M.fx[p][c.didx[q]] ⊻= 1; end)
    isodd(sum(M.fz[p][c.didx[q]] for q in c.XL_col0)) &&
        (for q in c.ZL_support; M.fz[p][c.didx[q]] ⊻= 1; end)
    M
end

"""
    decode_epoch_sliding!(M, p, zhist, xhist; C, B, use_AD) -> M

Decode one epoch of patch `p` on both check types and update the software frame: the Z-graph decode
yields X-corrections (into `fx`), the X-graph decode yields Z-corrections (into `fz`); a Y error hits
both. The reference then chains to the epoch's last round.
"""
function decode_epoch_sliding!(M::Machine, p, zhist, xhist; C, B, use_AD=true)
    c = M.code
    for q in sliding_decode(c, zhist, M.ref_z[p], :z; C, B, use_AD); M.fx[p][c.didx[q]] ⊻= 1; end
    for q in sliding_decode(c, xhist, M.ref_x[p], :x; C, B, use_AD); M.fz[p][c.didx[q]] ⊻= 1; end
    M.ref_z[p] = zhist[end]; M.ref_x[p] = xhist[end]
    M
end

"""
    run_epoch!(M, p, R; data_errs, meas_errs, C, B, use_AD) -> M

Run one idle epoch on patch `p`: measure `R` noisy syndrome rounds on the MPS, then sliding-decode
them into the frame. `data_errs :: (round,"X"/"Z",coord)` apply a physical Pauli to a data qubit just
before that round (a Y is an X and a Z); `meas_errs :: (round,:Z/:X,check_index)` flip only a recorded
bit that round.
"""
function run_epoch!(M::Machine, p, R; data_errs=[], meas_errs=[], C=2, B=2, use_AD=true)
    c = M.code
    zhist = Vector{Vector{Int}}(undef,R); xhist = Vector{Vector{Int}}(undef,R)
    for r in 1:R
        for (rr,P,q) in data_errs; rr==r && (M.psi = _ap(M.psi, op(P, sd(c,p,q)))); end
        z,x,M.psi = measure_raw_syndrome(M.psi, c, p)
        for (rr,typ,si) in meas_errs; rr==r && (typ===:Z ? (z[si]=1-z[si]) : (x[si]=1-x[si])); end
        zhist[r]=z; xhist[r]=x
    end
    decode_epoch_sliding!(M, p, zhist, xhist; C, B, use_AD)
    M
end

# ===========================================================================
#  §5  Transversal Clifford gates + frame/reference propagation
# ===========================================================================
apply_ZL_phys!(M, p) = (for q in M.code.ZL_support; M.psi = _ap(M.psi, op("Z", sd(M.code,p,q))); end)
apply_XL_phys!(M, p) = (for q in M.code.XL_col0;    M.psi = _ap(M.psi, op("X", sd(M.code,p,q))); end)
logical_X!(M, p) = (apply_XL_phys!(M, p); M)
logical_Z!(M, p) = (apply_ZL_phys!(M, p); M)

"""
    logical_CNOT!(M, pc, pt) -> M

Transversal logical CNOT (control `pc`, target `pt`): bitwise physical CNOT across the patches, then
propagate frame and reference by the conjugation X_c→X_cX_t, Z_t→Z_cZ_t. Also the entangling step in
the T/S gadget. NOTE: for side-by-side patches this couples non-adjacent MPS regions, the dominant
bond-dimension cost that this scaling study measures.
"""
function logical_CNOT!(M::Machine, pc, pt)
    c = M.code
    for q in c.data_coords; M.psi = _ap(M.psi, op("CNOT", sd(c,pc,q), sd(c,pt,q))); end
    M.fx[pt] .⊻= M.fx[pc]; M.fz[pc] .⊻= M.fz[pt]
    M.ref_z[pt] .⊻= M.ref_z[pc]; M.ref_x[pc] .⊻= M.ref_x[pt]
    M
end

"""
    logical_H!(M, p) -> M

Transversal logical Hadamard: physical H on every data qubit, then the 90° rotation via the SWAP
network (`sigma_cycles`). Frame and reference transform to match — H swaps X↔Z, and the rotation is
a PULL-BACK (`new[i]=old[sigma(i)]`) on the frame (`sigma_perm`) and on the reference checks
(`zaux2xaux`/`xaux2zaux`).
"""
function logical_H!(M::Machine, p)
    c = M.code
    for q in c.data_coords; M.psi = _ap(M.psi, op("H", sd(c,p,q))); end
    for cyc in c.sigma_cycles, i in 1:(length(cyc)-1)
        M.psi = _ap(M.psi, op("SWAPg", sd(c,p,cyc[i]), sd(c,p,cyc[i+1]))); end
    oldfx = copy(M.fx[p]); oldfz = copy(M.fz[p])
    nfx = zeros(Int, c.Ndata); nfz = zeros(Int, c.Ndata)
    for i in 1:c.Ndata; nfx[i] = oldfz[c.sigma_perm[i]]; nfz[i] = oldfx[c.sigma_perm[i]]; end
    M.fx[p] = nfx; M.fz[p] = nfz
    oldrz = copy(M.ref_z[p]); oldrx = copy(M.ref_x[p])
    nrz = zeros(Int, c.Nz); nrx = zeros(Int, c.Nx)
    for i in 1:c.Nz; nrz[i] = oldrx[c.zaux2xaux[i]]; end
    for i in 1:c.Nx; nrx[i] = oldrz[c.xaux2zaux[i]]; end
    M.ref_z[p] = nrz; M.ref_x[p] = nrx
    M
end

# ===========================================================================
#  §6  Frame-corrected logical read-out
# ===========================================================================
logop(L) = L === :Z ? ("Z", :Z) : ("X", :X)
"apply the product of logical Paulis named by `specs` (list of (patch,:Z/:X)) to a COPY of `psi`."
function _apply_joint(c::Code, psi, specs)
    Opsi = copy(psi)
    for (p, L) in specs
        (basis, _) = logop(L); supp = L === :Z ? c.ZL_support : c.XL_col0
        for q in supp; Opsi = apply(op(basis, sd(c,p,q)), Opsi; cutoff=CUTOFF[]); end
    end
    Opsi
end
"non-collapsing exact expectation ⟨∏ logical Paulis⟩ on the physical state (no frame correction)."
joint_expect(c::Code, psi, specs) = real(inner(psi, _apply_joint(c, psi, specs)))

"""
    frame_sign(M, p, L) -> ±1

The sign the Pauli frame imposes on a logical `L` (:Z/:X) readout of patch `p`: a Z-readout flips
under X-frame on `ZL_support`; an X-readout flips under Z-frame on `XL_col0`.
"""
function frame_sign(M::Machine, p, L)
    c = M.code
    if L === :Z; s = 0; for q in c.ZL_support; s ⊻= M.fx[p][c.didx[q]]; end; return 1 - 2s
    else;        s = 0; for q in c.XL_col0;    s ⊻= M.fz[p][c.didx[q]]; end; return 1 - 2s end
end

"frame-corrected single-patch logical expectation ⟨L⟩ — an exact ±1 pass/fail oracle for a basis state."
logical_readout(M::Machine, p, L) = joint_expect(M.code, M.psi, [(p, L)]) * frame_sign(M, p, L)
"frame-corrected two-patch correlator ⟨L₁⊗L₂⟩, L∈{:Z,:X}."
corr(M::Machine, b1, b2) = joint_expect(M.code, M.psi, [(1,b1),(2,b2)]) * frame_sign(M,1,b1) * frame_sign(M,2,b2)

"physical operator list, phase, and frame sign for a single-patch logical `L`∈{:X,:Y,:Z}."
function _op_and_sign(M::Machine, p, L)
    c = M.code; ops = Tuple{String,Tuple{Int,Int}}[]; ph = 1.0 + 0im
    if L === :Z
        for q in c.ZL_support; push!(ops,("Z",q)); end
        s=0; for q in c.ZL_support; s ⊻= M.fx[p][c.didx[q]]; end; sgn = 1-2s
    elseif L === :X
        for q in c.XL_col0; push!(ops,("X",q)); end
        s=0; for q in c.XL_col0; s ⊻= M.fz[p][c.didx[q]]; end; sgn = 1-2s
    else
        for q in c.XL_col0; push!(ops,("X",q)); end
        for q in c.ZL_support; push!(ops,("Z",q)); end
        ph = -1.0im
        s1=0; for q in c.XL_col0; s1 ⊻= M.fz[p][c.didx[q]]; end
        s2=0; for q in c.ZL_support; s2 ⊻= M.fx[p][c.didx[q]]; end; sgn = (1-2s1)*(1-2s2)
    end
    ops, ph, sgn
end
"""
    corr2(M, L1, L2) -> Float64

Frame-corrected two-patch logical correlator ⟨L₁⊗L₂⟩ with each factor in {:X,:Y,:Z} — the general
verification readout (e.g. ⟨YX⟩ as well as ⟨XX⟩).
"""
function corr2(M::Machine, L1, L2)
    c = M.code
    o1,ph1,s1 = _op_and_sign(M,1,L1); o2,ph2,s2 = _op_and_sign(M,2,L2)
    Opsi = copy(M.psi)
    for (g,q) in o1; Opsi = apply(op(g, sd(c,1,q)), Opsi; cutoff=CUTOFF[]); end
    for (g,q) in o2; Opsi = apply(op(g, sd(c,2,q)), Opsi; cutoff=CUTOFF[]); end
    real(ph1*ph2*inner(M.psi,Opsi))*s1*s2
end

# ===========================================================================
#  §7  Magic-state S/T teleportation
# ===========================================================================
"""
    measure_patch_Z_raw!(M, p) -> Int

Destructive logical-Z of patch `p`: measure every data qubit in Z and return the raw parity over
`ZL_support`. Reads out (and discards) the magic ancilla; the caller applies the frame correction.
"""
function measure_patch_Z_raw!(M::Machine, p)
    c = M.code; par = 0
    for q in c.data_coords
        s = sd(c,p,q); sz = _ex(M.psi, op("Sz", s))
        b = rand() < 0.5 + sz ? 0 : 1
        M.psi = _ap(M.psi, b == 0 ? P_up(s) : P_dn(s)); M.psi = M.psi / sqrt(real(inner(M.psi,M.psi)))
        q in c.ZL_support && (par ⊻= b)
    end
    par
end

"""
    teleport_S!(M, p, R; C, B, force_wrong, use_AD) -> M

Teleport a logical S onto patch `p` from a |Y>=S|+> ancilla (patch 3): prep ancilla, CNOT data→anc,
commit (one sliding-decoded R-round epoch on the ancilla), measure it, and apply the conditional
byproduct Z_L iff the frame-corrected outcome is 1. `force_wrong` flips the committed bit.
"""
function teleport_S!(M::Machine, p, R; C, B, force_wrong=false, use_AD=true)
    c = M.code
    prepare_logical!(M, 3, :Y); logical_CNOT!(M, p, 3)
    run_epoch!(M, 3, R; C, B, use_AD)
    m_raw = measure_patch_Z_raw!(M, 3)
    fbit = 0; for q in c.ZL_support; fbit ⊻= M.fx[3][c.didx[q]]; end
    m = m_raw ⊻ fbit ⊻ (force_wrong ? 1 : 0)
    m == 1 && apply_ZL_phys!(M, p); M
end

"""
    teleport_T!(M, p, R; C, B, force_wrong, use_AD) -> M

Teleport a logical T onto patch `p` from a |A>=T|+> ancilla, same structure as `teleport_S!`; the
byproduct here is a conditional (itself teleported) S. The commit epoch protects the outcome before
the frame bit is consumed.
"""
function teleport_T!(M::Machine, p, R; C, B, force_wrong=false, use_AD=true)
    c = M.code
    prepare_logical!(M, 3, :A); logical_CNOT!(M, p, 3)
    run_epoch!(M, 3, R; C, B, use_AD)
    m_raw = measure_patch_Z_raw!(M, 3)
    fbit = 0; for q in c.ZL_support; fbit ⊻= M.fx[3][c.didx[q]]; end
    m = m_raw ⊻ fbit ⊻ (force_wrong ? 1 : 0)
    m == 1 && teleport_S!(M, p, R; C, B); M
end

"""
    apply_logical!(M, gate, R; C, B, use_AD) -> M

Dispatch one logical gate. `gate` is `(:X|:Z|:H|:S|:T, patch)` or `(:CNOT, ctrl, tgt)`;
`R,C,B,use_AD` forward to the T/S commitment epochs.
"""
function apply_logical!(M::Machine, gate, R; C, B, use_AD)
    g = gate[1]
    g === :X    ? logical_X!(M, gate[2]) :
    g === :Z    ? logical_Z!(M, gate[2]) :
    g === :H    ? logical_H!(M, gate[2]) :
    g === :S    ? teleport_S!(M, gate[2], R; C, B, use_AD) :
    g === :T    ? teleport_T!(M, gate[2], R; C, B, use_AD) :
    g === :CNOT ? logical_CNOT!(M, gate[2], gate[3]) : error("unknown gate $g")
end

"""
    run_circuit(c, circuit; ec, R, C, B, use_AD, errors, meas) -> Machine

Run a full 2-qubit circuit on freshly prepared |0>_L⊗|0>_L for code `c`. After each gate (when `ec`)
run one sliding-decoded idle epoch on both data patches. `errors :: (gate_k,round,patch,"X"/"Z",coord)`
and `meas :: (gate_k,round,patch,:Z/:X,check_index)` inject faults into a gate's epoch. Read out with
`corr`/`corr2`/`logical_readout`.
"""
function run_circuit(c::Code, circuit; ec=true, R=6, C=2, B=2, use_AD=true, errors=[], meas=[])
    M = Machine(c)
    prepare_logical!(M, 1, :zero); prepare_logical!(M, 2, :zero)
    for (k, gate) in enumerate(circuit)
        apply_logical!(M, gate, R; C, B, use_AD)
        if ec
            de1 = [(rr,P,q) for (kk,rr,p,P,q) in errors if kk==k && p==1]
            de2 = [(rr,P,q) for (kk,rr,p,P,q) in errors if kk==k && p==2]
            me1 = [(rr,t,si) for (kk,rr,p,t,si) in meas if kk==k && p==1]
            me2 = [(rr,t,si) for (kk,rr,p,t,si) in meas if kk==k && p==2]
            run_epoch!(M, 1, R; data_errs=de1, meas_errs=me1, C, B, use_AD)
            run_epoch!(M, 2, R; data_errs=de2, meas_errs=me2, C, B, use_AD)
        end
    end
    M
end

# ===========================================================================
#  §8  Stochastic noise model + Monte-Carlo estimator
# ===========================================================================
"""
    NoiseModel

Phenomenological per-round stochastic Pauli noise: `p` = per-data-qubit Pauli-error prob per round
(type drawn from {X,Y,Z} with weights `wx,wy,wz`; a Y is coincident X and Z); `q` = per-measurement
wrong-bit prob; `p_gate` = extra depolarising prob per data qubit after a transversal CNOT.
"""
struct NoiseModel
    p::Float64; q::Float64; wx::Float64; wy::Float64; wz::Float64; p_gate::Float64
end
NoiseModel(; p=0.0, q=0.0, wx=1.0, wy=1.0, wz=1.0, p_gate=0.0) = NoiseModel(p,q,wx,wy,wz,p_gate)

"sample which Pauli fires given an error occurred (∝ wx,wy,wz)."
function draw_pauli(nm::NoiseModel)
    u = rand() * (nm.wx + nm.wy + nm.wz)
    u < nm.wx ? :X : u < nm.wx + nm.wy ? :Y : :Z
end

"""
    sample_epoch_noise(c, R, nm; perfect_last_meas) -> (data_errs, meas_errs)

Draw one epoch's faults in the format `run_epoch!` consumes: each round every data qubit fires a
Pauli with prob `nm.p` and every check's recorded bit flips with prob `nm.q` (a Y is two data
entries). `perfect_last_meas` suppresses measurement noise on the final round (clean time boundary).
"""
function sample_epoch_noise(c::Code, R::Int, nm::NoiseModel; perfect_last_meas::Bool=false)
    data_errs = Tuple{Int,String,Tuple{Int,Int}}[]; meas_errs = Tuple{Int,Symbol,Int}[]
    for r in 1:R
        for q in c.data_coords
            if rand() < nm.p
                P = draw_pauli(nm)
                P === :X ? push!(data_errs,(r,"X",q)) :
                P === :Z ? push!(data_errs,(r,"Z",q)) :
                          (push!(data_errs,(r,"X",q)); push!(data_errs,(r,"Z",q)))
            end
        end
        if !(perfect_last_meas && r == R)
            for i in 1:c.Nz; (rand() < nm.q) && push!(meas_errs,(r,:Z,i)); end
            for i in 1:c.Nx; (rand() < nm.q) && push!(meas_errs,(r,:X,i)); end
        end
    end
    data_errs, meas_errs
end

"""
    run_memory_experiment(c, sym, R, nm; C, B, use_AD) -> Machine

Prepare patch 1 of code `c` in logical `sym`, expose it to `R` noisy rounds (final round clean),
sliding-decode into the frame, and return the machine.
"""
function run_memory_experiment(c::Code, sym::Symbol, R::Int, nm::NoiseModel; C=2, B=2, use_AD=true)
    M = Machine(c); prepare_logical!(M, 1, sym)
    de, me = sample_epoch_noise(c, R, nm; perfect_last_meas=true)
    run_epoch!(M, 1, R; data_errs=de, meas_errs=me, C, B, use_AD)
    M
end

"""
    estimate_pL(c, sym, L, R, nm, N; C, B, use_AD, seed) -> (pL, se)

Monte-Carlo logical error probability of the `R`-round memory of `sym` (code `c`), read in basis `L`,
under `nm`. A shot fails iff `logical_readout < 0`. Returns the failure fraction and its binomial std
error. Pair `sym`↔`L`: `:zero`↔`:Z`, `:plus`↔`:X`.
"""
function estimate_pL(c::Code, sym, L, R, nm, N; C=2, B=2, use_AD=true, seed=nothing)
    seed !== nothing && Random.seed!(seed)
    fails = 0
    for _ in 1:N
        M = run_memory_experiment(c, sym, R, nm; C, B, use_AD)
        (logical_readout(M, 1, L) < 0) && (fails += 1)
    end
    pL = fails / N; (pL, sqrt(pL*(1-pL)/N))
end

# ===========================================================================
#  §9  Exact 2-qubit reference (validation ground truth)
# ===========================================================================
const _I = ComplexF64[1 0; 0 1];  const _X = ComplexF64[0 1; 1 0]
const _Z = ComplexF64[1 0; 0 -1]; const _H = ComplexF64[1 1; 1 -1]/sqrt(2)
const _S = ComplexF64[1 0; 0 im]; const _T = ComplexF64[1 0; 0 exp(im*π/4)]; const _Y = ComplexF64[0 -im; im 0]
_1q(g, p) = p == 1 ? kron(g, _I) : kron(_I, g)
const _CN12 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
"exact 4-amplitude statevector after `circuit` on |00>."
function ref_state(circuit)
    ψ = ComplexF64[1, 0, 0, 0]
    for gate in circuit
        g = gate[1]
        Mm = g===:X ? _1q(_X,gate[2]) : g===:Z ? _1q(_Z,gate[2]) : g===:H ? _1q(_H,gate[2]) :
             g===:S ? _1q(_S,gate[2]) : g===:T ? _1q(_T,gate[2]) : g===:CNOT ? _CN12 : error()
        ψ = Mm * ψ
    end
    ψ
end
const _Pd = Dict(:X=>_X, :Y=>_Y, :Z=>_Z)
"exact two-qubit correlator ⟨a⊗b⟩, a,b∈{:X,:Y,:Z}, on reference state ψ."
ref2(ψ, a, b) = real(ψ' * kron(_Pd[a], _Pd[b]) * ψ)

println("scaling_engine.jl loaded — build_code(d) to construct a distance-d code")
