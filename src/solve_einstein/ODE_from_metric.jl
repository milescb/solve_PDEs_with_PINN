using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using Plots, LaTeXStrings

include("../utils/utils_einstein.jl")
# although the inverse of a diagonal matrix is just the inverse 
# of the diagonal elements, this allows for generalization if need-be
include("../utils/compute_inverse_4x4.jl")
include("../utils/general_utils.jl")

# A collection of constants to be used 
const G = 6.67e-11 # m²/kg²
const c = 3e8 # m/s²
const M = 1.989e30 #kg

params = @parameters ρ, θ
vars = @variables g00(..) g11(..)

const n = length(params)

Dρ = Differential(ρ)

diff_vec_1 = [0, Dρ, 0, 0]

# shape of entire matrix 
g_matrix_complete = [g00(ρ), 0            , 0           , 0            , #=
                  =# 0           , -g11(ρ), 0           , 0            , #= 
                  =# 0           , 0            , -ρ^2        , 0            , #=
                  =# 0           , 0            , 0           , -ρ^2 * (sin(θ))^2]

g_matrix_inverse = inverse_4x4(g_matrix_complete)

# ensure we do not have identical equations; matrix is symmetric
indices_eqns = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
eqns_temp = [ODE_equations(k[1],k[2],g_matrix_complete,g_matrix_inverse) ~ 0 for k in indices_eqns]

eqns = []
for i in eachindex(eqns_temp)
    if eqns_temp[i] != (0.0 ~ 0)
        push!(eqns,eqns_temp[i])
    end
end

# for now, use prior knowledge of what the metric looks like
ρ_limit = 1e4
bcs = [ # spherically symetric
        # g00(τ,ρ,θ,0) ~ g00(τ,ρ,θ,2π),
        # g11(τ,ρ,θ,0) ~ g11(τ,ρ,θ,2π),
        # flat space-time as r → ∞
        g00(ρ_limit) ~ 1,
        g11(ρ_limit) ~ 1,
        # knowledge about shape of solution
        #g00(τ,ρ,θ,ϕ) ~ -1/g11(τ,ρ,θ,ϕ),
        # imposing limit of weak gravity 
        # g00(τ,ρ,θ,ϕ) ~ -c^2 + 2*G*M/ρ
]

domains = [ ρ ∈ Interval(100, ρ_limit),
            θ ∈ Interval(0, π)]

@named pde_sys = PDESystem(eqns, bcs, domains, [ρ, θ], [g00(ρ), g11(ρ)])

@info "Equations and boundry conditions calculated"

numChains = length(vars)
dim = length(domains) # number of dimensions
activation = Lux.σ
chains = [Lux.Chain(Lux.Dense(dim, 32, activation), 
            Lux.Dense(32, 32, activation),
            Lux.Dense(32, 16, activation),
            Lux.Dense(16, 1)) for _ in 1:numChains]

strategy = QuasiRandomTraining(100)
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

i = 0
loss_history = []

res = Optimization.solve(prob, ADAM(1e-3); callback = callback, maxiters=700)
phi = discretization.phi