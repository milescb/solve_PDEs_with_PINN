#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the Einstein field equations to obtain the Schwarzschild metrix. 

This file is intended to be extention of the initial work done in `solve.jl`.
Training files are loaded and then further training is possible to obtain 
a better solution. Plotting and analysis can be done with the file `solve.jl`. 
=#
using NeuralPDE, Lux, ModelingToolkit
using DifferentialEquations
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using Plots, LaTeXStrings, JLD2

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

r_min = 0.1f0
r_max = 60.f0
bcs = [
    B(r_max) ~ -1.f0,
    A(r_max) ~ 1.f0,
]

domains = [r ∈ Interval(r_min, r_max)]

@named pde_sys = PDESystem(eqns, bcs, domains, [r], [A(r), B(r)])

@info "Problem set-up complete"

# -------------------------------------------------------------------------------------
# load previously stored newton solution
sol_nts = load_object("./trained_networks/EFE_ODE_diff/sol_nts.jld2")

# initial conditions
x0 = [r0, 0.f0] # units of AU
dx0 = [0.f0, v0] # units of AU/yr
tspan = (0.f0, 0.1f0)

# solve problem
dx = 0.01

include("additional_loss.jl")

# -------------------------------------------------------------------------------------
# Continue optimization problem

#= 
Run training on GPU if availible.

Note that the operations above may not work on the GPU! If use of GPU is desired, 
uncomment the below code and add the argument `init_params = ps` to function 
`PhysicsInformedNN`.

DOES NOT WORK ON MAC M1 MACHINES
=#
# using Random, CUDA
# CUDA.allowscalar(false)
# ps = [Lux.setup(Random.default_rng(), chains[i])[1] for i in 1:numChains]
# ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:numChains]

# load previous results
discretization, phi, res, loss_history, domains = load_training_files("./trained_networks/EFE_ODE_diff")

@time prob = discretize(pde_sys, discretization)

# some decoration for reporting the loss. Required by callback function
i = 0
loss_history = [] # uncomment to wipe current loss_history

# remake problem with previously trained parameters
prob = remake(prob, u0=res.minimizer)
res = Optimization.solve(prob, ADAM(1e-4); callback = callback, maxiters=25)
phi = discretization.phi

#save_training_files("./trained_networks/EFE_ODE_diff")