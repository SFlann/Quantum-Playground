# ============================================================================
#  trajectory_engine.jl  —  quantum-trajectory (Kraus / quantum-jump) noise on
#                           the d-parameterised MPS surface-code engine
#
#  This PULLS IN the scaling engine from ../8.Scaling verbatim (the Code object,
#  union-find decoder, transversal gates, magic-state teleportation, software
#  frame, and the exact non-collapsing readout oracle) and ADDS a channel-
#  agnostic *quantum* noise layer on top of it. Nothing in the scaling engine is
#  changed; every function below is additive.
#
#  The point of the series so far (notebooks 7–9) has been STOCHASTIC PAULI
#  noise: at each location, with some probability, insert an X / Y / Z. That is
#  exactly the noise a stabiliser simulator (tsim) can Monte-Carlo efficiently,
#  and for that regime tsim is far faster. This file goes where tsim structurally
#  cannot: ARBITRARY completely-positive trace-preserving (CPTP) channels —
#  amplitude damping (T1), phase damping (T2), coherent over-rotations, always-on
#  ZZ crosstalk, leakage out of the qubit subspace — carried on the real quantum
#  amplitudes of the MPS.
#
#  The vehicle is the QUANTUM-TRAJECTORY (a.k.a. quantum-jump / Monte-Carlo
#  wavefunction) unravelling: a channel  ρ → Σ_i K_i ρ K_i†  is realised on a
#  PURE state by, at each location, drawing one Kraus branch i with its Born
#  probability  p_i = ⟨ψ|K_i†K_i|ψ⟩,  applying K_i, and renormalising. The state
#  stays pure (2^n amplitudes, NOT a 2^{2n} density matrix), so it fits the
#  existing pure-state MPS engine unchanged. Averaging any observable over many
#  independent trajectories reconstructs the exact open-system expectation
#  Tr(ρ O). Many trajectories → embarrassingly parallel → HPC (see the notebook).
#
#  Scope: single-qubit CPTP channels + coherent unitaries + ZZ crosstalk on the
#  data qubits of the surface-code memory, plus a qutrit (|2⟩) leakage scaffold
#  showing the trajectory step is dimension-agnostic. See the notebook
#  `example_quantum_noise_trajectories.ipynb` for the narrative and the honest
#  limits (the bond-dimension wall of ../8.Scaling §7 still applies).
# ============================================================================

include(joinpath(@__DIR__, "..", "8.Scaling", "scaling_engine.jl"))

# ===========================================================================
#  §A  Single-qubit CPTP channels as Kraus sets
# ===========================================================================
# Each channel is a Vector of 2×2 complex matrices {K_i} with Σ_i K_i†K_i = I.
# We reuse the exact-reference Pauli matrices already defined in the scaling
# engine (_I, _X, _Y, _Z). None of these is a Pauli mixture in general — that is
# the entire point: they live outside the stabiliser formalism.

"Amplitude damping (T1 relaxation |1⟩→|0⟩) with per-application excited-state decay prob γ."
amp_damp(γ)   = [ComplexF64[1 0; 0 sqrt(1-γ)], ComplexF64[0 sqrt(γ); 0 0]]

"Phase damping (pure T2 dephasing) with per-application prob λ — decays coherence, not population."
phase_damp(λ) = [ComplexF64[1 0; 0 sqrt(1-λ)], ComplexF64[0 0; 0 sqrt(λ)]]

"Pauli-Z dephasing channel with flip prob p (the *stochastic-Pauli* dephasing tsim would use)."
dephase(p)    = [ComplexF64(sqrt(1-p))*_I, ComplexF64(sqrt(p))*_Z]

"Single-qubit depolarising channel with total error prob p (as a Kraus set, not a Pauli draw)."
depolarize(p) = [ComplexF64(sqrt(1-p))*_I, ComplexF64(sqrt(p/3))*_X,
                 ComplexF64(sqrt(p/3))*_Y, ComplexF64(sqrt(p/3))*_Z]

"""
    compose(A, B) -> Vector{Matrix}

Kraus set of channel `A` applied AFTER channel `B` (i.e. B first): {A_i B_j}. Lets you stack,
e.g. `compose(amp_damp(γ), phase_damp(λ))` for simultaneous T1+T2 in one location.
"""
compose(A, B) = [a*b for a in A for b in B]

"True iff the Kraus set is trace-preserving (Σ K†K = I) — a channel sanity check."
iscptp(K; atol=1e-10) = isapprox(sum(k'k for k in K), Matrix{ComplexF64}(I, size(K[1])); atol=atol)

# ===========================================================================
#  §B  Coherent (unitary) errors — the headline advantage over tsim
# ===========================================================================
# A systematic over-rotation or an always-on crosstalk term is a UNITARY, not a
# probabilistic Pauli mixture. tsim can only insert its Pauli TWIRL (the
# stochastic channel with matching average fidelity), discarding the coherence.
# Here we insert the real rotation — no twirl — and can also read out the exact
# logical channel to *quantify* what the twirl throws away.

"Coherent Z over-rotation by angle ε (calibration drift / miscalibrated phase gate)."
Rz(ε) = ComplexF64[exp(-im*ε/2) 0; 0 exp(im*ε/2)]

"""
    twirl_Rz(ε) -> Vector{Matrix}

The Pauli-twirl approximation of `Rz(ε)` — the stochastic-Z dephasing channel with the SAME
average fidelity that tsim's efficient noise layer would substitute. Since
Rz(ε)=cos(ε/2)I − i sin(ε/2)Z, its twirl is Z with probability sin²(ε/2). Compare the exact
logical channel of `Rz(ε)` against this to measure the coherent-vs-twirled gap.
"""
twirl_Rz(ε) = dephase(sin(ε/2)^2)

"Always-on ZZ crosstalk unitary exp(−iθ Z⊗Z) on an adjacent data pair (θ = coupling·time)."
zz_unitary(θ) = ComplexF64[exp(-im*θ) 0 0 0; 0 exp(im*θ) 0 0; 0 0 exp(im*θ) 0; 0 0 0 exp(-im*θ)]

"horizontally-adjacent data-qubit pairs of code `c` (for a nearest-neighbour crosstalk layer)."
function adjacent_pairs(c::Code)
    S = Set(c.data_coords)
    [(q, (q[1]+1, q[2])) for q in c.data_coords if (q[1]+1, q[2]) in S]
end

# ===========================================================================
#  §C  The quantum-jump step  (the trajectory unravelling itself)
# ===========================================================================
"""
    apply_channel!(psi, s, kraus) -> (psi, i)

Realise one single-site CPTP channel `kraus = {K_i}` on site `s` of MPS `psi` as ONE quantum
jump: draw branch `i` with its Born probability pᵢ = ⟨ψ|Kᵢ†Kᵢ|ψ⟩, apply Kᵢ, renormalise, and
return the collapsed state and the chosen branch. Averaging ψψ† over independent calls reproduces
Σᵢ Kᵢ ρ Kᵢ†. This is the whole trajectory method — and it is DIMENSION-AGNOSTIC: `s` may be a
qubit or a qutrit (leakage), the code is identical (see §F).

Cost: one `apply` per Kraus operator to score the branches (channels here have ≤4), then one more
to commit — all routed through the scaling engine's `_ap`, so bond-dimension diagnostics update.
"""
function apply_channel!(psi, s, kraus)
    ps = Float64[]
    for K in kraus
        Kpsi = apply(op(K, s), psi; cutoff = CUTOFF[])
        push!(ps, real(inner(Kpsi, Kpsi)))
    end
    tot = sum(ps); r = rand() * tot; acc = 0.0; idx = length(kraus)
    for (i, p) in enumerate(ps)
        acc += p
        if r <= acc; idx = i; break; end
    end
    phi = _ap(psi, op(kraus[idx], s))
    phi = phi / sqrt(real(inner(phi, phi)))
    phi, idx
end

# ===========================================================================
#  §D  A physical channel model + a trajectory epoch runner
# ===========================================================================
"""
    ChannelModel

A per-round *physical* (not necessarily Pauli) noise model for one idle epoch. Unlike notebook 7's
`NoiseModel`, whose faults can be pre-sampled into a Pauli list, these channels are applied as LIVE
quantum jumps because a jump's probability depends on the current amplitudes.

Fields (each optional; a default constructor leaves it off):
- `kraus1`    : a single-qubit CPTP channel `{Kᵢ}` applied to every data qubit each round.
- `coh`       : a coherent single-qubit unitary (e.g. `Rz(ε)`) applied to every data qubit each
                round — inserted as the *real* rotation, NOT twirled.
- `zz`        : ZZ-crosstalk angle θ applied to every adjacent data pair each round (0 = off).
- `q`         : classical measurement-flip probability (a readout bit-flip; stays classical).
"""
struct ChannelModel
    kraus1::Vector{Matrix{ComplexF64}}
    coh::Union{Nothing,Matrix{ComplexF64}}
    zz::Float64
    q::Float64
end
ChannelModel(; kraus1 = Matrix{ComplexF64}[], coh = nothing, zz = 0.0, q = 0.0) =
    ChannelModel(kraus1, coh, zz, q)

"""
    run_epoch_traj!(M, p, R, cm; C, B, use_AD, perfect_last_meas) -> M

One idle epoch on patch `p` under the *quantum* channel model `cm`, ONE trajectory. Each of the `R`
rounds, in order: (1) the coherent unitary `cm.coh` on every data qubit; (2) the ZZ-crosstalk layer;
(3) the stochastic single-qubit channel `cm.kraus1` as a live quantum jump on every data qubit;
(4) a raw syndrome round on the corrupted MPS; (5) classical measurement flips at rate `cm.q`. Then
sliding-decode the history into the software frame exactly as the noiseless engine does. Because the
real MPS is corrupted, the measured syndromes automatically reflect the channel — no outcome hacking.
"""
function run_epoch_traj!(M::Machine, p, R, cm::ChannelModel;
                         C = 2, B = 2, use_AD = true, perfect_last_meas = false)
    c = M.code
    zhist = Vector{Vector{Int}}(undef, R); xhist = Vector{Vector{Int}}(undef, R)
    U_zz = cm.zz == 0.0 ? nothing : zz_unitary(cm.zz)
    pairs = cm.zz == 0.0 ? Tuple{Tuple{Int,Int},Tuple{Int,Int}}[] : adjacent_pairs(c)
    for r in 1:R
        if cm.coh !== nothing
            for q in c.data_coords; M.psi = _ap(M.psi, op(cm.coh, sd(c, p, q))); end
        end
        if U_zz !== nothing
            for (qa, qb) in pairs; M.psi = _ap(M.psi, op(U_zz, sd(c, p, qa), sd(c, p, qb))); end
        end
        if !isempty(cm.kraus1)
            for q in c.data_coords; M.psi, _ = apply_channel!(M.psi, sd(c, p, q), cm.kraus1); end
        end
        z, x, M.psi = measure_raw_syndrome(M.psi, c, p)
        if !(perfect_last_meas && r == R)
            for i in 1:c.Nz; rand() < cm.q && (z[i] = 1 - z[i]); end
            for i in 1:c.Nx; rand() < cm.q && (x[i] = 1 - x[i]); end
        end
        zhist[r] = z; xhist[r] = x
    end
    decode_epoch_sliding!(M, p, zhist, xhist; C, B, use_AD)
    M
end

# ===========================================================================
#  §E  Trajectory Monte-Carlo estimators
# ===========================================================================
"""
    run_memory_traj(c, sym, R, cm; C, B, use_AD) -> Machine

One trajectory of the `R`-round memory of logical `sym` on patch 1 of code `c` under channel `cm`
(final round's readout clean). The single-trajectory analogue of `run_memory_experiment`.
"""
function run_memory_traj(c::Code, sym::Symbol, R::Int, cm::ChannelModel; C = 2, B = 2, use_AD = true)
    M = Machine(c); prepare_logical!(M, 1, sym)
    run_epoch_traj!(M, 1, R, cm; C, B, use_AD, perfect_last_meas = true)
    M
end

"""
    estimate_pL_traj(c, sym, L, R, cm, N; C, B, use_AD, seed) -> (pL, se)

Trajectory Monte-Carlo logical error probability under the *quantum* channel `cm`: run `N`
independent trajectories, count those whose frame-corrected ±1 oracle `logical_readout` comes back
negative. Returns the failure fraction and its binomial std error. Identical control flow to the
scaling engine's `estimate_pL`; only the noise is now a genuine CPTP channel rather than a Pauli
draw. Embarrassingly parallel over the trajectory index — see the notebook's HPC section.
"""
function estimate_pL_traj(c::Code, sym, L, R, cm::ChannelModel, N;
                          C = 2, B = 2, use_AD = true, seed = nothing)
    seed !== nothing && Random.seed!(seed)
    fails = 0
    for _ in 1:N
        M = run_memory_traj(c, sym, R, cm; C, B, use_AD)
        (logical_readout(M, 1, L) < 0) && (fails += 1)
    end
    pL = fails / N; (pL, sqrt(pL * (1 - pL) / N))
end

"""
    mean_correlator_traj(c, circuit, L1, L2, cm, N; R, C, B, use_AD, seed) -> (mean, se)

Reconstruct an EXACT open-system logical correlator ⟨L₁⊗L₂⟩ = Tr(ρ L₁L₂) by averaging the
non-collapsing oracle `corr2` over `N` trajectories that each run `circuit` with a noisy final idle
epoch under channel `cm`. This is the observable a stabiliser simulator cannot produce: it is not a
bit sample but the true ensemble expectation, so it captures the *coherent* part of the logical
error (e.g. how a coherent Rz(ε) shifts ⟨XX⟩ away from a Pauli-twirled prediction). Returns the
trajectory mean and its standard error of the mean.
"""
function mean_correlator_traj(c::Code, circuit, L1, L2, cm::ChannelModel, N;
                              R = 4, C = 2, B = 2, use_AD = true, seed = nothing)
    seed !== nothing && Random.seed!(seed)
    vals = Float64[]
    for _ in 1:N
        M = Machine(c)
        prepare_logical!(M, 1, :zero); prepare_logical!(M, 2, :zero)
        for gate in circuit; apply_logical!(M, gate, R; C, B, use_AD); end
        run_epoch_traj!(M, 1, R, cm; C, B, use_AD, perfect_last_meas = true)
        push!(vals, corr2(M, L1, L2))
    end
    m = sum(vals) / N
    se = sqrt(max(0.0, sum((v - m)^2 for v in vals)) / (N * max(1, N - 1)))
    (m, se)
end

# ===========================================================================
#  §F  Leakage scaffold — the trajectory step on a QUTRIT (|2⟩) site
# ===========================================================================
# Leakage is population escaping {|0⟩,|1⟩} into a third level |2⟩ (a transmon's
# second excited state, a neutral-atom Rydberg/hyperfine level). No Pauli — and
# no stabiliser formalism — can hold it; it needs a d≥3 local Hilbert space. The
# density-matrix cost of that is 3^{2n}; TRAJECTORIES keep it at 3^n amplitudes,
# which is exactly why the trajectory approach makes leakage *feasible* rather
# than merely definable. Crucially `apply_channel!` above needs NO change: it
# only multiplies matrices and takes inner products, so it works verbatim on a
# 3-level site with 3×3 Kraus operators. (A full Tier-C engine also raises the
# gate/measurement layer to 3 levels — see ../7.../Leakage-Errors-Design.md; here
# we ship the channel itself, the genuinely new physics.)

"""
    leak_channel(γL, γS) -> Vector{Matrix}

3-level leakage/seepage CPTP channel on {|0⟩,|1⟩,|2⟩}: with prob γL a |1⟩ leaks to |2⟩, with prob
γS a |2⟩ seeps back to |1⟩, else no jump. Trace-preserving on the qutrit (Σ K†K = I₃). Feed it to
`apply_channel!` on a `dim=3` site — the same jump machinery used for qubits.
"""
function leak_channel(γL, γS)
    K0 = ComplexF64[1 0 0; 0 sqrt(1-γL) 0; 0 0 sqrt(1-γS)]   # no jump
    Kl = ComplexF64[0 0 0; 0 0 0; 0 sqrt(γL) 0]              # |2⟩⟨1|  (leak)
    Ks = ComplexF64[0 0 0; 0 0 sqrt(γS); 0 0 0]              # |1⟩⟨2|  (seep)
    [K0, Kl, Ks]
end

"a `dim`-level ITensor site (dim=3 → qutrit) for standalone leakage-channel demos."
qudit_site(dim; n = 1) = Index(dim, "Qudit,Site,n=$n")

println("trajectory_engine.jl loaded — quantum-trajectory noise on the d-parameterised MPS engine")
println("  channels: amp_damp, phase_damp, dephase, depolarize, leak_channel; coherent: Rz, zz_unitary")
println("  estimators: estimate_pL_traj, mean_correlator_traj  (embarrassingly parallel over trajectories)")
