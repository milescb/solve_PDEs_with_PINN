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

include("./utils/utils_einstein.jl")
# although the inverse of a diagonal matrix is just the inverse 
# of the diagonal elements, this allows for generalization if need-be
include("./utils/compute_inverse_4x4.jl")
include("./utils/general_utils.jl")

# A collection of constants to be used 
const G = 6.67e-11 # m²/kg²
const c = 3e8 # m/s²
const M = 1.989e30 #kg

params = @parameters τ ρ θ ϕ
vars = @variables g00(..) g11(..)

const n = length(params)

@info "Variables and parameters loaded" params vars

Dτ = Differential(τ)
Dρ = Differential(ρ)
Dθ = Differential(θ)
Dϕ = Differential(ϕ)

diff_vec_1 = [Dτ, Dρ, Dθ, Dϕ]

# shape of entire matrix 
g_matrix_complete = [g00(τ,ρ,θ,ϕ), 0            , 0           , 0            , #=
                  =# 0           , g11(τ,ρ,θ,ϕ), 0           , 0            , #= 
                  =# 0           , 0            , -ρ^2        , 0            , #=
                  =# 0           , 0            , 0           , -ρ^2 * (sin(θ))^2]

g_matrix_inverse = inverse_4x4(g_matrix_complete)

# ensure we do not have identical equations; matrix is symmetric
indices_eqns = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
eqns = [PDE_equations(k[1],k[2],g_matrix_complete,g_matrix_inverse) ~ 0 for k in indices_eqns]

# for now, use prior knowledge of what the metric looks like
ρ_limit = 100
bcs = [ # spherically symetric
        g00(τ,ρ,θ,0) ~ g00(τ,ρ,θ,2π),
        g11(τ,ρ,θ,0) ~ g11(τ,ρ,θ,2π),
        # flat space-time as r → ∞
        g00(τ,ρ_limit,θ,ϕ) ~ 1,
        g11(τ,ρ_limit,θ,ϕ) ~ -1,
        # knowledge about shape of solution
        #g00(τ,ρ,θ,ϕ) ~ -1/g11(τ,ρ,θ,ϕ),
        # imposing limit of weak gravity 
        # g00(τ,ρ,θ,ϕ) ~ -c^2 + 2*G*M/ρ
]

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(1, ρ_limit),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], [g00(τ,ρ,θ,ϕ), g11(τ,ρ,θ,ϕ)])

@info "Equations and boundry conditions calculated"

numChains = length(vars)
dim = length(domains) # number of dimensions
activation = Lux.σ
chains = [Lux.Chain(Lux.Dense(dim, 32, activation), 
            Lux.Dense(32, 32, activation),
            Lux.Dense(32, 16, activation),
            Lux.Dense(16, 1)) for _ in 1:numChains]

# This fully does not work currently... ask how to use GPUs? 
# Tried on both M1 mac and lxplus node
# Problem when feeding in arguments to discretize function as initial_params
initθ = [DiffEqFlux.initial_params(chains[i]) |> Lux.gpu for i in 1:numChains]
@info "Using GPUs?" initθ

strategy = QuasiRandomTraining(100)
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

@info "discretization complete" prob

i = 0
loss_history = []

if ARGS != String[]
    learning_rates = [ARGS[1], ARGS[2], ARGS[3]]
else
    learning_rates = [1e-4, 1e-3, 1e-4]
end

@info "Beginning training with learning rate" learning_rates[1] 
#res = @time Optimization.solve(prob, BFGS(); callback = callback, maxiters=1000)
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
# plot!(1:length(loss1_history), loss1_history, 
#         label="Learning rate $(learning_rates[1])")
# plot!(length(loss1_history)+1:length(loss1_history+loss2_history), 
#         loss2_history, label="Learning rate $(learning_rates[2])")
# plot!(length(loss1_history+loss2_history)+1:length(loss1_history+loss2_history+loss3_history), 
#         loss3_history, label="Learning rate $(learning_rates[3])")
savefig("./plots/EPE_simple_solution/loss.png")

save_training_files("trained_networks/EFE_simple")
