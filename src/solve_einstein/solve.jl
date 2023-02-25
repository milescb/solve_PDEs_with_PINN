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

using Plots, JLD2

include("../utils/general_utils.jl")
include("../utils/rk4.jl")

# constants for problem
include("constants.jl")

@parameters r
vars = @variables A(..) B(..)

Dr = Differential(r)
Drr = Differential(r)^2

eqns = [
    2.f0*r*A(r)*B(r)*Drr(A(r)) - r*A(r)*Dr(A(r))*Dr(B(r)) + 4.f0*A(r)*B(r)*Dr(A(r)) - r*B(r)*(Dr(A(r)))^2 ~ 0.f0,
    -2.f0*r*A(r)*B(r)*Drr(A(r)) + r*A(r)*Dr(A(r))*Dr(B(r)) + 4.f0*((A(r))^2)*Dr(B(r)) + r*B(r)*(Dr(A(r)))^2 ~ 0.f0,
    -2.f0*A(r)*B(r) + 2.f0*A(r)*(B(r))^2 - r*Dr(A(r))*B(r) + r*A(r)*Dr(B(r)) ~ 0.f0
]

r_min = 0.3f0
r_max = 60.f0
bcs = [
    B(r_max) ~ 1.f0,
    A(r_max) ~ 1.f0,
]

domains = [r ∈ Interval(r_min, r_max)]

@named pde_sys = PDESystem(eqns, bcs, domains, [r], [A(r), B(r)])

@info "Problem set-up complete"

# -------------------------------------------------------------------------------------
# Solve for Newtonian Gravity!
# input in form [x,dx,y,dy]
# output in form [dx, ddx, dy, ddy]
function newton_gravity(t,u)
    r_sqrd = (u[1]*u[1] + u[3]*u[3])
    du = [0.,0.,0.,0.]
    du[1] = u[2]
    du[2] = -GM*u[1]/(r_sqrd^1.5)
    du[3] = u[4]
    du[4] = -GM*u[3]/(r_sqrd^1.5)
    return du
end

# initial conditions
x0 = [15.f0, 0.f0] # units of AU
dx0 = [0.f0, 100.f0] # units of AU/yr
initial_conditions = [x0[1],dx0[1],x0[2],dx0[2]]
nPoints = 20

sol_newton = runge_kutta4(newton_gravity,0.0,1.0,initial_conditions,nPoints)
sol_nts = [[sol_newton[i][2][1], sol_newton[i][2][3]] for i in eachindex(sol_newton)]

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
chains = [Lux.Chain(Lux.Dense(dim, nnodes, activation), 
            Lux.Dense(nnodes, nnodes, activation),
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
# ps = [Lux.setup(Random.default_rng(), chains[i])[1] for i in 1:numChains]
# ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:numChains]

@info "Discretization complete. Beginning training"

strategy = QuasiRandomTraining(30)
discretization = PhysicsInformedNN(chains, strategy, 
    additional_loss=additional_loss)
@time prob = discretize(pde_sys, discretization)

# some decoration for reporting the loss. Required by callback function
i = 0
loss_history = []

res = Optimization.solve(prob, LBFGS(); callback = callback, maxiters=15)
phi = discretization.phi

save_training_files("./trained_networks/EFE_ODE_diff")