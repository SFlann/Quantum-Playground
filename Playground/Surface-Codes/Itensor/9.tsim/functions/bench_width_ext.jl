# ============================================================================
#  bench_width_ext.jl — extend the WIDTH sweep of the deep-circuit study to
#  larger Q, to test whether the ITensor and tsim time curves ever cross.
#
#  The first large MPS contraction pays a big one-time Julia/ITensor method-
#  compilation cost (empirically ~900 s the first time χ reaches a few hundred),
#  which otherwise corrupts the first timed large-Q point.  We therefore run one
#  UNTIMED warm-up at the largest Q before timing anything, so every reported
#  time is "warm" and the curve is monotone.
#
#  Writes data/deep_itensor_width.json = { "width": [ {Q,sites,chi,cap,err,secs} ] }
#  (the notebook prefers this over the width block of deep_itensor_bench.json).
#
#  Run alone (memory!):  <julia> --project=.. bench_width_ext.jl
# ============================================================================
include("../8.Scaling/multiqubit_runner.jl")

xent(Q) = vcat(Any[(:H, i) for i in 1:(Q ÷ 2)],
               Any[(:CNOT, i, Q + 1 - i) for i in 1:(Q ÷ 2)],
               Any[(:T, 1), (:T, Q)])

function specs_for(Q)
    s = Vector{Any}[]
    for i in 1:Q; push!(s, [(i, :Z)]); push!(s, [(i, :X)]); end
    push!(s, [(i, :Z) for i in 1:Q]); push!(s, [(i, :X) for i in 1:Q]); s
end

const QS = (2, 4, 6, 8, 10)

jval(x::Integer) = string(x)
jval(x::AbstractFloat) = isfinite(x) ? string(x) : "null"
jrow(d) = "{" * join(["\"$k\": $(jval(v))" for (k, v) in d], ", ") * "}"

println("bench_width_ext.jl  d=3  Q ∈ $QS")
# --- untimed warm-up at the largest Q: pays the one-time compilation cost ---
println("warming up at Q=$(maximum(QS)) (untimed, pays JIT/contraction compilation)…"); flush(stdout)
let Qw = maximum(QS)
    c = build_code(3; npatch = Qw + 1); set_precision!(cutoff = 1e-6, maxdim = 4096)
    run_circuit_mq(c, Qw, xent(Qw); ec = false, R = 2)
end
println("warm-up done; timing now"); flush(stdout)

rows = Dict{String,Any}[]
println("Q  sites  chi   cap  maxErr    secs")
for Q in QS
    c = build_code(3; npatch = Q + 1); Random.seed!(21)
    set_precision!(cutoff = 1e-6, maxdim = 4096)
    t = @elapsed M = run_circuit_mq(c, Q, xent(Q); ec = false, R = 2)
    ψ = ref_state_mq(Q, xent(Q))
    err = maximum(abs(logical_expect(M, s) - refN(ψ, Q, s)) for s in specs_for(Q))
    push!(rows, Dict("Q" => Q, "sites" => length(c.sites), "chi" => CHIMAX[],
                     "cap" => CAP_HITS[], "err" => err, "secs" => t))
    @printf("%d  %-5d  %-4d  %-3d  %.1e  %8.2f\n", Q, length(c.sites), CHIMAX[], CAP_HITS[], err, t)
    flush(stdout)
end

open(joinpath(@__DIR__, "data", "deep_itensor_width.json"), "w") do io
    write(io, "{\n  \"engine\": \"itensor_mps\", \"d\": 3, \"note\": \"warm-timed extended width sweep\",\n")
    write(io, "  \"width\": [\n    " * join([jrow(r) for r in rows], ",\n    ") * "\n  ]\n}\n")
end
println("wrote data/deep_itensor_width.json")
