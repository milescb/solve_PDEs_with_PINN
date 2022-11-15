#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the Einstein field equations to obtain the Schwarzschild metrix. 

Here, we start from ODEs given via simplification analytically. In order to 
make our solution match Newtonian gravity, we consider a solution to classical
=#
using NeuralPDE, Lux, ModelingToolkit
using DifferentialEquations, Statistics, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using Plots, LaTeXStrings, JLD2

include("../utils/general_utils.jl")

# define some consts
const G = 6.67e-11 # m²/kg²
const M = 1.989e+35 #kg
const AU = 1.496e11 # meters
const yr = 3.154e7 #seconds
# change units, use AU and years
const GM = Float32(G*M*yr^2 / AU^3) # AU^3/yr^2; Kepler III says this is 4π^2
const c = Float32(3e8 * (yr/AU)) # AU / yr
const r0 = 40.f0 # AU
const v0 = 500.f0 # AU / year
const ricci_r = 2*GM/c^2 # AU

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

@info "Solved ODE problem!"

ϵ = 0.1f0 # smaller epsilon results in seg fault
"""
    additional_loss(phi,θ,p)

Compute additional loss when matching solution to newtonian gravity. 
"""
function additional_loss(phi, θ, p)

    # 00 component of metric from neural network
    g00(x,y) = phi[1](sqrt(x^2+y^2), θ.depvar[:A])[1]

    # set-up the problem using current metric
    function simple_geodesic(ddu,du,u,p,t)
        ddu[1] = (c^2/2) * ((g00(u[1]+ϵ,u[2]) - g00(u[1]-ϵ,u[2]))/2ϵ)
        ddu[2] = (c^2/2) * ((g00(u[1],u[2]+ϵ) - g00(u[1],u[2]-ϵ))/2ϵ)
    end

    # solve system of diff-eqs 
    prob_ode = SecondOrderODEProblem(simple_geodesic, dx0, x0, tspan)
    sol = solve(prob_ode, Rosenbrock32(), saveat=dx, dt=dx)

    # hack to avoid differing lengths when error dt less than dmin
    if length(sol) > length(sol_nts)
        @warn "length of temp solution too long"
        iter = length(sol_nts)
    elseif length(sol) < length(sol_nts)
        @warn "length of temp solution too short"
        iter = length(sol)
    else 
        iter = length(sol)
    end

    return 0.1 * sum(distance2(sol_nts[i][1],sol[i][3],
        sol_nts[i][2],sol[i][4]) for i in 1:iter)
end

# -------------------------------------------------------------------------------------
# define neural network
numChains = length(vars)
dim = length(domains) # number of dimensions
activation = Lux.σ
nnodes = 10
chains = [Lux.Chain(Lux.Dense(dim, nnodes, activation), 
            Lux.Dense(nnodes, 1)) for _ in 1:numChains]

#= 
Run training on GPU if availible.
Note that the operations above may not work on the GPU! If use of GPU is desired, 
uncomment the below code and add the argument `init_params = ps` to function 
`PhysicsInformedNN`.
=#
# using Random, CUDA
# CUDA.allowscalar(false)
# ps = [Lux.setup(Random.default_rng(), chains[i])[1] for i in 1:numChains]
# ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:numChains]

@info "Discretization complete. Beginning training"

strategy = QuasiRandomTraining(20)
#strategy = GridTraining(0.1)
#strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy,
    additional_loss=additional_loss)
@time prob = discretize(pde_sys, discretization)

# some decoration for reporting the loss. Required by callback function
i = 0
loss_history = []

res = Optimization.solve(prob, BFGS(); callback = callback, maxiters=15)
phi = discretization.phi

#save_training_files("./trained_networks/EFE_ODE_diff")