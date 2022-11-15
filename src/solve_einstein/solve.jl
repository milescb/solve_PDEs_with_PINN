#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the Einstein field equations to obtain the Schwarzschild metrix. 

Here, we start from ODEs given via simplification analytically. In order to 
make our solution match Newtonian gravity, we consider a solution to the 
classical problem. We then attempt to match this with a differential equation
given by the network. The loss function for this is given in `additional_loss.jl`
and further explanation can be found there. 

Training files can be saved and used later for continued training with the file 
`solve_continue.jl`, or analysis and plots can be made with `plot.jl`. 
=#
using NeuralPDE, Lux, ModelingToolkit, Random
using DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using JLD2

include("../utils/general_utils.jl")

# constants for problem
include("constants.jl")

@parameters r
vars = @variables A(..) B(..)

Dr = Differential(r)
Drr = Differential(r)^2

eqns = [
    4.f0*Dr(A(r))*((B(r))^2) - 2.f0*r*Drr(B(r))*A(r)*B(r) + 2.f0*Dr(A(r))*Dr(B(r))*B(r) + r*((Dr(B(r)))^2)*A(r) ~ 0.f0,
    r*Dr(A(r))*B(r) + 2.f0*((A(r))^2)*B(r) - 2.f0*A(r)*B(r) - r*Dr(B(r))*A(r) ~ 0.f0,
    -2.f0*r*Drr(B(r))*A(r)*B(r) + r*Dr(A(r))*Dr(B(r))*B(r) + r*((Dr(B(r)))^2)*A(r) - 4.f0*Dr(B(r))*A(r)*B(r) ~ 0.f0
]

r_min = 1.f0
r_max = 60.f0
bcs = [
    B(r_max) ~ -1.f0,
    A(r_max) ~ 1.f0,
]

domains = [r ∈ Interval(r_min, r_max)]

@named pde_sys = PDESystem(eqns, bcs, domains, [r], [A(r), B(r)])

@info "Problem set-up complete"

# -------------------------------------------------------------------------------------
# Solve for Newtonian Gravity!
function newton_gravity(ddu,du,u,p,t)
    r_sqrd = (u[1]^2 + u[2]^2)
    ddu[1] = -GM*u[1]/(r_sqrd^1.5)
    ddu[2] = -GM*u[2]/(r_sqrd^1.5)
end

# initial conditions
x0 = [r0, 0.f0] # units of AU
dx0 = [0.f0, v0] # units of AU/yr
tspan = (0.f0, 0.1f0)

# solve problem
dx = 0.01
prob_newton = SecondOrderODEProblem(newton_gravity, dx0, x0, tspan)
@time sol_newton = solve(prob_newton, Tsit5(), dt=dx, saveat=dx)
sol_nts = [[sol_newton[i][3], sol_newton[i][4]] for i in eachindex(sol_newton)]

plot(getindex.(sol_nts,1),getindex.(sol_nts,2), label="", size=(400,400), dpi=200,
        xlabel="\$x\$ (AU)", ylabel="\$y\$ (AU)");
savefig("./plots/EPE_ODE_solution/newton_solution.png")
save_object("./trained_networks/EFE_ODE_diff/sol_nts.jld2", sol_nts)

@info "Solved ODE problem!"

# include additional loss function
include("additional_loss.jl")

# -------------------------------------------------------------------------------------
# define neural network
numChains = length(vars)
dim = length(domains) # number of dimensions
activation = Lux.σ
nnodes = 10
rng = Random.default_rng()
chains = [Lux.Chain(Lux.Dense(dim, nnodes, activation), 
            Lux.Dense(nnodes, 1)) for _ in 1:numChains]

#= 
Run training on GPU if availible.
    
Note that the operations above may not work on the GPU! If use of GPU is desired, 
uncomment the below code and add the argument `init_params = ps` to function 
`PhysicsInformedNN`.

DOES NOT WORK ON MAC M1 MACHINES
=#
# using Random, CUDA
# CUDA.allowscalar(false)
# ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:numChains]

# ensure initial parameters are of type Float32
ps = [Lux.setup(Random.default_rng(), chains[i])[1] |> Lux.ComponentArray .|> Float32 for i in 1:numChains]

@info "Discretization complete. Beginning training"

strategy = QuasiRandomTraining(20)
#strategy = GridTraining(0.1)
#strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy,
    additional_loss=additional_loss, init_params=ps)
@time prob = discretize(pde_sys, discretization)

# some decoration for reporting the loss. Required by callback function
i = 0
loss_history = []

res = Optimization.solve(prob, BFGS(); callback = callback, maxiters=15)
phi = discretization.phi

#save_training_files("./trained_networks/EFE_ODE_diff")