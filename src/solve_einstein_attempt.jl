#=
Author: Miles Cochran-Branson
Date: Fall 2022

Solve Einstein's fields equations to obtain the Schwarzschild metric. 
=#
using NeuralPDE, ModelingToolkit, Optimization, Lux, OptimizationOptimisers
import ModelingToolkit: Interval
using JLD2

include("../utils/utils_einstein.jl")
include("../utils/general_utils.jl")

# A collection of constants to be used 
n = 4 # number of dimensions in problem
const G = 6.67e-11 # m²/kg²
const c = 3e8 # m/s²
M = 1.989e30 #kg

params = @parameters τ ρ θ ϕ
vars = @variables g00(..) g01(..) g02(..) g03(..) #=
               =# g10(..) g11(..) g12(..) g13(..) #= 
               =# g20(..) g21(..) g22(..) g23(..) #=
               =# g30(..) g31(..) g32(..) g33(..)

Dτ = Differential(τ)
Dρ = Differential(ρ)
Dθ = Differential(θ)
Dϕ = Differential(ϕ)

diff_vec_1 = [Dτ, Dρ, Dθ, Dϕ]

vars_complete =   [g00(τ,ρ,θ,ϕ), g01(τ,ρ,θ,ϕ), g02(τ,ρ,θ,ϕ), g03(τ,ρ,θ,ϕ), #=
                =# g10(τ,ρ,θ,ϕ), g11(τ,ρ,θ,ϕ), g12(τ,ρ,θ,ϕ), g13(τ,ρ,θ,ϕ), #= 
                =# g20(τ,ρ,θ,ϕ), g21(τ,ρ,θ,ϕ), g22(τ,ρ,θ,ϕ), g23(τ,ρ,θ,ϕ), #=
                =# g30(τ,ρ,θ,ϕ), g31(τ,ρ,θ,ϕ), g32(τ,ρ,θ,ϕ), g33(τ,ρ,θ,ϕ)]

vars_complete_simple =   [g00(τ,ρ,θ,ϕ), 0,            0,            0, #=
                       =# 0           , g11(τ,ρ,θ,ϕ), 0,            0, #= 
                       =# 0           , 0           , -ρ^2        , 0, #=
                       =# 0           , 0           , 0           , -ρ^2 * (sin(θ))^2]

#inverse_complete = inverse_4x4(vars_complete)

#eqns = [Ricci(i,j,inverse_complete) ~ 0 for i in 0:(n-1) for j in 0:(n-1)]
eqns = [Ricci_simplified2(i,j,vars_complete_simple) ~ 0 for i in 0:(n-1) for j in 0:(n-1)]

# eqns = []
# for i in eachindex(eqns_temp)
#     if eqns_temp[i] != (0~0)
#         push!(eqns, eqns_temp[i])
#     end
# end

# for now, use prior knowledge of what the metric looks like
newton_limit(ρ) = -2*G*M/(c^2*ρ)
bcs = [#g01(τ,ρ,θ,ϕ) ~ 0,
#         g02(τ,ρ,θ,ϕ) ~ 0,
#         g03(τ,ρ,θ,ϕ) ~ 0,
#         g12(τ,ρ,θ,ϕ) ~ 0,
#         g13(τ,ρ,θ,ϕ) ~ 0,
#         g23(τ,ρ,θ,ϕ) ~ 0,
        # symetric 
        # g01(τ,ρ,θ,ϕ) ~ g10(τ,ρ,θ,ϕ),
        # g02(τ,ρ,θ,ϕ) ~ g20(τ,ρ,θ,ϕ),
        # g03(τ,ρ,θ,ϕ) ~ g30(τ,ρ,θ,ϕ),
        # g12(τ,ρ,θ,ϕ) ~ g21(τ,ρ,θ,ϕ),
        # g13(τ,ρ,θ,ϕ) ~ g31(τ,ρ,θ,ϕ),
        # g23(τ,ρ,θ,ϕ) ~ g32(τ,ρ,θ,ϕ),
        # spherically symetric
        g00(τ,ρ,θ,0) ~ g00(τ,ρ,θ,2π),
        g11(τ,ρ,θ,0) ~ g11(τ,ρ,θ,2π),
        #pior knowledge
        #g22(τ,ρ,θ,ϕ) ~ -ρ^2,
        #g33(τ,ρ,θ,ϕ) ~ -ρ^2 * (sin(θ))^2,
        # try to match newtonian gravity
        g00(τ,6e6,0,0) ~ newton_limit(ρ),
        g11(τ,6e6,0,0) ~ newton_limit(ρ)
]

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(6e6, 1e8),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], [vars_complete[i] for i in [1,6]])
#@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], vars_complete)

dim = length(domains) # number of dimensions
activation = Lux.σ
chains = [Lux.Chain(Lux.Dense(dim, 32, activation), 
            Lux.Dense(32, 16, activation),
            Lux.Dense(16, 1)) for _ in 1:2]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

i = 0
loss_history = []

res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(0.1); callback = callback, maxiters=1000)
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(0.01); callback = callback, maxiters=2000)
# prob = remake(prob,u0=res.minimizer)
# res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(0.0001); callback = callback, maxiters=5000)
phi = discretization.phi

save_training_files("trained_networks/EFE_simple")