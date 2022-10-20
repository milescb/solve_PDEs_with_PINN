#=
Author: Miles Cochran-Branson
Date: Fall 2022

Solve Einstein's fields equations to obtain the Schwarzschild metric. 
=#
@info "Loading Packages"
using NeuralPDE, ModelingToolkit, Optimization, Lux, OptimizationOptimisers, DiffEqFlux
using Plots
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

@info "Variables and parameters loaded" params vars

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

# should also be symmetric! Only 10 eqns; kinda ugly way of doing this but it works?
#eqns = [Ricci(i,j,inverse_complete) ~ 0 for i in 0:(n-1) for j in 0:(n-1)]
indices_eqns = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
eqns = [Ricci_simplified2(k[1],k[2],vars_complete_simple) ~ 0 for k in indices_eqns]

# eqns = []
# for i in eachindex(eqns_temp)
#     if eqns_temp[i] != (0~0)
#         push!(eqns, eqns_temp[i])
#     end
# end

# for now, use prior knowledge of what the metric looks like
newton_limit(ρ) = -2*G*M/(c^2*ρ)
ρ_limit = 1e8
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
        # flat space-time as r → ∞
        g00(τ,ρ_limit,θ,ϕ) ~ 1,
        g11(τ,ρ_limit,θ,ϕ) ~ -1,
        #g22(τ,Inf,θ,ϕ) ~ -ρ^2,
        #g33(τ,Inf,θ,ϕ) ~ -ρ^2 * (sin(θ))^2,
        # have to add all the other components going to zero
        # try to match newtonian gravity
        g00(τ,1,0,0) ~ 1 - 2*G*M/c^2,
        g11(τ,1,0,0) ~ -(1 - 2*G*M/c^2)^(-1)
]

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(1, ρ_limit),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], [vars_complete[i] for i in [1,6]])
#@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], vars_complete)

@info "Equations and boundry conditions calculated"

numChains = 2
dim = length(domains) # number of dimensions
activation = Lux.σ
chains = [Lux.Chain(Lux.Dense(dim, 32, activation), 
            Lux.Dense(32, 16, activation),
            Lux.Dense(16, 1)) for _ in 1:numChains]

# This fully does not work currently... ask how to use GPUs? 
# Tried on both M1 mac and lxplus node
# Problem when feeding in arguments to discretize function as initial_params
initθ = [DiffEqFlux.initial_params(chains[i]) |> Lux.gpu for i in 1:numChains]
@info "Using GPUs?" initθ

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

@info "discretization complete" prob

i = 0
loss_history = []

if ARGS != String[]
    learning_rates = [ARGS[1], ARGS[2], ARGS[3]]
else
    learning_rates = [0.7, 0.1, 0.0001]
end

@info "Beginning training with learning rate" learning_rates[1] 
res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(learning_rates[1]); callback = callback, maxiters=500)
loss1_history = loss_history
loss_history = []

@info "Beginning training with learning rate" learning_rates[2] 
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(learning_rates[2]); callback = callback, maxiters=500)
loss2_history = loss_history
loss_history = []

@info "Beginning training with learning rate" learning_rates[3]
prob = remake(prob,u0=res.minimizer) 
res = @time Optimization.solve(prob, OptimizationOptimisers.ADAM(learning_rates[3]); callback = callback, maxiters=500)
loss3_history = loss_history
loss_history = vcat(loss1_history, loss2_history, loss3_history)
phi = discretization.phi

@info "Training complete"

# Plot loss!
plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss",
        size=(400,400), dpi=200, label="")
plot!(1:length(loss1_history), loss1_history, 
        label="Learning rate $(learning_rates[1])")
plot!(length(loss1_history)+1:length(loss1_history+loss2_history), 
        loss2_history, label="Learning rate $(learning_rates[2])")
plot!(length(loss1_history+loss2_history)+1:length(loss1_history+loss2_history+loss3_history), 
        loss3_history, label="Learning rate $(learning_rates[3])")
savefig("./plots/EPE_simple_solution/loss.png")

save_training_files("trained_networks/EFE_simple")
