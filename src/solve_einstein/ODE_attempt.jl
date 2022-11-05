#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the Einstein field equations to obtain the Schwarzschild metrix. 

Here, we start from ODEs given via simplification analytically. 
=#
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers
import ModelingToolkit: Interval, infimum, supremum

using Plots, LaTeXStrings

include("../utils/general_utils.jl")

# define some consts
const G = 6.67e-11 # m²/kg²
const c = 3e8 # m/s²
const M = 1.989e30 #kg
const ricci_r = 2*G*M/c^2 

@parameters r
vars = @variables A(..) B(..)

Dr = Differential(r)
Drr = Differential(r)^2

eqns = [
    4*Dr(A(r))*((B(r))^2) - 2*r*Drr(B(r))*A(r)*B(r) + 2*Dr(A(r))*Dr(B(r))*B(r) + r*((Dr(B(r)))^2)*A(r) ~ 0,
    r*Dr(A(r))*B(r) + 2*((A(r))^2)*B(r) - 2*A(r)*B(r) - r*Dr(B(r))*A(r) ~ 0,
    -2*r*Drr(B(r))*A(r)*B(r) + r*Dr(A(r))*Dr(B(r))*B(r) + r*((Dr(B(r)))^2)*A(r) - 4*Dr(B(r))*A(r)*B(r) ~ 0
]

r_limit = 1000
bcs = [
    B(r) ~ -1/A(r),
    A(r_limit) ~ 1,
    B(r_limit) ~ -1
]

domains = [r ∈ Interval(10, r_limit)]

@named pde_sys = PDESystem(eqns, bcs, domains, [r], [A(r), B(r)])

numChains = length(vars)
dim = length(domains) # number of dimensions
activation = Lux.σ
chains = [Lux.Chain(Lux.Dense(dim, 32, activation), 
            Lux.Dense(32, 32, activation),
            Lux.Dense(32, 1)) for _ in 1:numChains]

strategy = QuasiRandomTraining(100)
strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

i = 0
loss_history = []

res = Optimization.solve(prob, ADAM(1e-4); callback = callback, maxiters=1000)
phi = discretization.phi

## plot loss as a function of Epoch
plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss",
        size=(400,400), dpi=200, label="")
savefig("./plots/EPE_ODE_solution/loss.png")

## Compare solution to analytic!
r_temp = 1.0
u_analytic(ρ) = [1 - r_temp/ρ, -1/(1 - r_temp/ρ)]

dr = 0.1
rs = [infimum(d.domain):dr/10:supremum(d.domain) for d in domains][1]

dep_vars = [:A, :B]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

u_real = [[u_analytic(r)[i] for r in rs] for i in 1:numChains]
u_predict = [[phi[i]([r], minimizers[i])[1] for r in rs] for i in 1:numChains]

plot(rs, u_real[1], xlabel=L"r", ylabel=L"A(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:bottomright)
plot!(rs, u_predict[1], label="Predicted Solution")
savefig("./plots/EPE_ODE_solution/A.png")

plot(rs, u_real[2], xlabel=L"r", ylabel=L"B(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:bottomright)
plot!(rs, u_predict[2], label="Predicted Solution")
savefig("./plots/EPE_ODE_solution/B.png")