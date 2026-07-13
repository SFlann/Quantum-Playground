# ============================================================================
#  bench_deep_itensor.jl  —  ITensor MPS timing for the deep-circuit / many-T
#  scaling study (9.tsim/example_deep_circuit_scaling.ipynb).
#
#  Times the exact-MPS engine `run_circuit_mq` (8.Scaling) on the SAME three
#  logical-circuit families the tsim side sweeps, all at fixed d = 3:
#
#    * WIDTH  — centre-crossing entangler `xent(Q)`: Q/2 Bell pairs that cross
#               the middle of the chain + a T on each end.  χ (bond dimension)
#               grows ~2^{Q/2}: the MPS entanglement wall.
#    * MAGIC  — `magic(L)` at Q = 3: L layers of [T on all 3 patches + CNOT
#               chain], so the physical-T count is 3L.  The MPS is FREE in T
#               (χ stays flat — reported), but each T is a teleportation gadget,
#               so wall-time grows ~linearly in the T count (not exponentially).
#    * DEPTH  — fixed magic-Bell, varying QEC round count R (temporal depth).
#
#  Emits data/deep_itensor_bench.json for the notebook to read (the established
#  bench_itensor.jl -> JSON handoff of this directory).  No JSON package is in
#  the project, so JSON is hand-written (all values are numbers/strings).
#
#  Run (real julia binary avoids the juliaup lock; see reference-julia-execution):
#    <julia> --project=.. bench_deep_itensor.jl            # full
#    <julia> --project=.. bench_deep_itensor.jl quick      # fast sanity subset
# ============================================================================

include("../8.Scaling/multiqubit_runner.jl")   # -> run_circuit_mq, build_code, CHIMAX, ...

const QUICK = "quick" in ARGS

# ---- the three logical-circuit families (mirror functions/tsim_surface builds) ----

# WIDTH: H the left half (=prep |+>), CNOT i -> (Q+1-i) crossing the centre, T on the ends.
xent(Q) = vcat(Any[(:H, i) for i in 1:(Q ÷ 2)],
               Any[(:CNOT, i, Q + 1 - i) for i in 1:(Q ÷ 2)],
               Any[(:T, 1), (:T, Q)])

# MAGIC: Q=3, L layers of [T on all 3 + CNOT chain]; leading H on patch 1 (=prep |+>).
function magic(L)
    g = Any[(:H, 1)]
    for _ in 1:L
        append!(g, Any[(:T, 1), (:T, 2), (:T, 3), (:CNOT, 1, 2), (:CNOT, 2, 3)])
    end
    g
end

# DEPTH: the single-T magic Bell, held over a varying number of QEC rounds.
bell1() = Any[(:H, 1), (:T, 1), (:CNOT, 1, 2)]

# ---- validation specs: exhaustive single- + a few joint-Pauli correlators ----
function specs_for(Q)
    s = Vector{Any}[]
    for i in 1:Q
        push!(s, [(i, :Z)]); push!(s, [(i, :X)])
    end
    push!(s, [(i, :Z) for i in 1:Q]); push!(s, [(i, :X) for i in 1:Q])
    s
end

function run_point(Q, circ; ec, R)
    c = build_code(3; npatch = Q + 1)
    Random.seed!(21)
    set_precision!(cutoff = 1e-6, maxdim = 4096)   # also resets CHIMAX / CAP_HITS
    t = @elapsed M = run_circuit_mq(c, Q, circ; ec = ec, R = R, C = 2, B = 2, use_AD = true)
    ψ = ref_state_mq(Q, circ)
    err = maximum(abs(logical_expect(M, s) - refN(ψ, Q, s)) for s in specs_for(Q))
    (secs = t, chi = CHIMAX[], cap = CAP_HITS[], err = err,
     nT = count(g -> g[1] === :T, circ), sites = length(c.sites))
end

# ---- tiny hand-rolled JSON writer (flat dicts of numbers) ----
jval(x::Bool) = x ? "true" : "false"
jval(x::Integer) = string(x)
jval(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
jval(x::AbstractString) = "\"$x\""
jrow(d) = "{" * join(["\"$k\": $(jval(v))" for (k, v) in d], ", ") * "}"
jarr(rows) = "[\n    " * join([jrow(r) for r in rows], ",\n    ") * "\n  ]"

function main()
    println("bench_deep_itensor.jl  (quick=$QUICK)  d=3 fixed")
    run_point(2, xent(2); ec = false, R = 1)   # warm up the JIT so point 1 isn't penalised
    t_all = @elapsed begin

    # WIDTH sweep --------------------------------------------------------------
    Qs = QUICK ? (2, 4) : (2, 4, 6)
    width = Dict{String,Any}[]
    println("\n[WIDTH] centre-crossing entangler, ec=false, R=2")
    println("Q  sites  nT  chi   cap   maxErr    secs")
    for Q in Qs
        r = run_point(Q, xent(Q); ec = false, R = 2)
        push!(width, Dict("Q" => Q, "sites" => r.sites, "nT" => r.nT,
                          "chi" => r.chi, "cap" => r.cap, "err" => r.err, "secs" => r.secs))
        @printf("%d  %-5d  %-2d  %-4d  %-4d  %.1e  %7.2f\n",
                Q, r.sites, r.nT, r.chi, r.cap, r.err, r.secs)
    end

    # MAGIC sweep --------------------------------------------------------------
    Ls = QUICK ? (1, 2, 3) : (1, 2, 3, 4, 5, 6)
    magicr = Dict{String,Any}[]
    println("\n[MAGIC] Q=3, L layers of (T x3 + CNOT chain), ec=false, R=2")
    println("L  nT  chi   cap   maxErr    secs   (chi flat = free in T)")
    for L in Ls
        r = run_point(3, magic(L); ec = false, R = 2)
        push!(magicr, Dict("L" => L, "nT" => r.nT, "chi" => r.chi,
                           "cap" => r.cap, "err" => r.err, "secs" => r.secs))
        @printf("%d  %-2d  %-4d  %-4d  %.1e  %7.2f\n", L, r.nT, r.chi, r.cap, r.err, r.secs)
    end

    # DEPTH sweep --------------------------------------------------------------
    Rs = QUICK ? (1, 2, 4) : (1, 2, 4, 6, 8)
    depth = Dict{String,Any}[]
    println("\n[DEPTH] magic-Bell (Q=2, 1 T), ec=true, varying QEC rounds R")
    println("R  chi   maxErr    secs")
    for R in Rs
        r = run_point(2, bell1(); ec = true, R = R)
        push!(depth, Dict("R" => R, "chi" => r.chi, "err" => r.err, "secs" => r.secs))
        @printf("%d  %-4d  %.1e  %7.2f\n", R, r.chi, r.err, r.secs)
    end

    end  # t_all

    open(joinpath(@__DIR__, "data", "deep_itensor_bench.json"), "w") do io
        write(io, "{\n")
        write(io, "  \"engine\": \"itensor_mps\",\n")
        write(io, "  \"quick\": $(jval(QUICK)),\n")
        write(io, "  \"d\": 3,\n")
        write(io, "  \"total_secs\": $(jval(t_all)),\n")
        write(io, "  \"width\": $(jarr(width)),\n")
        write(io, "  \"magic\": $(jarr(magicr)),\n")
        write(io, "  \"depth\": $(jarr(depth))\n")
        write(io, "}\n")
    end
    @printf("\nwrote data/deep_itensor_bench.json  (total %.1f s)\n", t_all)
end

main()
