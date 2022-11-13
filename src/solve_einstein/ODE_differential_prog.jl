#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the Einstein field equations to obtain the Schwarzschild metrix. 

Here, we start from ODEs given via simplification analytically. In order to 
make our solution match Newtonian gravity, we consider a solution to classical
=#
using NeuralPDE, Lux, ModelingToolkit
using DifferentialEquations, Statistics, SciMLSensitivity
using Random, CUDA, StaticArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

using Plots, LaTeXStrings, JLD2

include("../utils/general_utils.jl")

# define some consts
const G = 6.67e-11 # m²/kg²
const M = 1.989e+30 #kg
const AU = 1.496e11 # meters
const yr = 3.154e7 #seconds
# change units
const GM = Float32(G*M*yr^2 / AU^3) # AU^3/yr^2; Kepler III says this is 4π^2
const c = Float32(3e8 * (yr/AU)) # AU / yr
const r_peri = 0.98f0 # AU
const semi_major = 1.f0
const v_peri = sqrt(GM * ((2.f0/r_peri) - (1.f0/semi_major)))
const ricci_r = 2*GM/c^2

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
r_max = 2.f0
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
x0 = [r_peri, 0.f0] # units of AU
dx0 = [0.f0, v_peri] # units of AU/yr
tspan = (0.f0, 1.f0)

# solve problem
dx = 0.1
prob_newton = SecondOrderODEProblem(newton_gravity, dx0, x0, tspan)
@time sol_newton = solve(prob_newton, Tsit5(), saveat=dx)
sol_nts = [[sol_newton[i][3], sol_newton[i][4]] for i in eachindex(sol_newton)]

scatter(getindex.(sol_nts,1),getindex.(sol_nts,2), label="", size=(400,400), dpi=200,
        xlabel="\$x\$ (AU)", ylabel="\$y\$ (AU)")
savefig("./plots/EPE_ODE_solution/newton_solution.png")

@info "Solved ODE problem!"

"""
    distance2(x1,x2,y1,y2)

Compute Euclidean distance between points (x1,y1) and (x2,y2)
"""
function distance2(x1,x2,y1,y2)
    return sqrt((x1-x2)^2 + (y1-y2)^2)
end

ϵ = sqrt(eps(Float32)) # machine epsilon for derivative
ϵ = 0.1f0
"""
    additional_loss(phi,θ,p)

Compute loss when matching solution to newtonian gravity. 
"""
function additional_loss(phi, θ, p)

    # 00 component of metric from neural network
    g00(x,y) = phi[1](sqrt(x^2+y^2), θ.depvar[:A])[1]

    # set-up the problem using current metric
    function simple_geodesic(ddu,du,u,p,t)
        # ddu[1] = -(c^2/2) * g00(u[1], u[2])
        # ddu[2] = -(c^2/2) * g00(u[1], u[2])
        ddu[1] = -(c^2/2) * ((g00(u[1]+ϵ,u[2]) - g00(u[1]-ϵ,u[2]))/2ϵ)
        ddu[2] = -(c^2/2) * ((g00(u[1],u[2]+ϵ) - g00(u[1],u[2]-ϵ))/2ϵ)
    end

    # solve system of diff-eqs 
    prob = SecondOrderODEProblem(simple_geodesic, dx0, x0, tspan)
    sol = solve(prob, Rosenbrock32(), reltol=1e-3, abstol=1e-3, saveat=dx)

    if length(sol) > length(sol_nts)
        iter = length(sol_nts)
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
#            Lux.Dense(nnodes, nnodes, activation),
            Lux.Dense(nnodes, 1)) for _ in 1:numChains]

# run training on GPU if availible
CUDA.allowscalar(false)
ps = [Lux.setup(Random.default_rng(), chains[i])[1] for i in 1:numChains]
ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:numChains]

# discretize
strategy = QuasiRandomTraining(20)
#strategy = GridTraining(0.1)
#strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy,
    additional_loss=additional_loss)
@time prob = discretize(pde_sys, discretization)

@info "Discretization complete. Beginning training"

# some decoration for reporting the loss
i = 0
loss_history = []

# solve the problem!
# maybe try LBFGS alg?
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters=15)
phi = discretization.phi

save_training_files("./trained_networks/EFE_ODE_diff")
save_object("./trained_networks/EFE_ODE_diff/init_params.jld2", ps)

@info "Training complete. Beginning analysis"

# -------------------------------------------------------------------------------------
## plot loss as a function of Epoch
plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss",
        size=(400,400), dpi=200, label="")
savefig("./plots/EPE_ODE_solution/loss.png")

## Compare solution to analytic!
u_analytic(ρ) = [1 - ricci_r/ρ, -1/(1 - ricci_r/ρ)]

dep_vars = [:A, :B]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

dr = 0.01
rs = [infimum(d.domain):dr:supremum(d.domain) for d in domains][1]

u_real = [[u_analytic(r)[i] for r in rs] for i in 1:numChains]
u_predict = [[phi[i]([r], minimizers[i])[1] for r in rs] for i in 1:numChains]

plot(rs, u_real[1], xlabel=L"r", ylabel=L"A(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:bottomright)
plot!(rs, u_predict[1], 
        label="Predicted Solution, \$\\chi^2/dof = $(round(χ²(u_predict[1], 
            u_real[1])/length(u_predict[1]),digits=2))\$")
savefig("./plots/EPE_ODE_solution/A.png")

plot(rs, u_real[2], xlabel=L"r", ylabel=L"B(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:right)
plot!(rs, u_predict[2], 
        label="Predicted Solution, \$\\chi^2/dof = $(round(χ²(u_predict[2], 
            u_real[2])/length(u_predict[2]),digits=2))\$")
savefig("./plots/EPE_ODE_solution/B.png")