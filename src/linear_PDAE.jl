#=
Author: Miles Cochran-Branson
Date: Fall 2022

We solve the second test problem from the paper
https://springerplus.springeropen.com/articles/10.1186/2193-1801-3-137.
=#
using NeuralPDE, ModelingToolkit, Optimization, Lux, OptimizationOptimisers
using Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum

include("./utils/general_utils.jl")

@parameters x t
@variables u1(..) u2(..) u3(..)

Dx = Differential(x)
Dxx = Differential(x)^2
Dt = Differential(t)
Dtt = Differential(t)^2

eqns = [Dtt(u1(t,x)) ~ Dxx(u1(t,x)) + u3(t,x)*sin(π*x),
        Dtt(u2(t,x)) ~ Dxx(u2(t,x)) + u3(t,x)*cos(π*x),
        u1(t,x)*sin(π*x) + u2(t,x)*cos(π*x) - exp(-t) ~ 0]

bcs = [#initial conditions
        u1(0,x) ~ sin(π*x),
        u2(0,x) ~ cos(π*x),
        Dt(u1(0,x)) ~ -sin(π*x),
        Dt(u2(0,x)) ~ -cos(π*x),
        #boundry conditions
        u1(t,0) ~ 0,
        u1(t,1) ~ 0,
        u2(t,0) ~ exp(-t),
        u2(t,1) ~ -exp(-t)
]

domains = [x ∈ Interval(0.0, 1.0),
            t ∈ Interval(0.0, 1.0)]

@named pde_sys = PDESystem(eqns, bcs, domains, [t,x], [u1(t,x), u2(t,x), u3(t,x)])

dim = length(domains) # number of dimensions
n = 15
chains = [Lux.Chain(
            Dense(dim, n, Lux.σ), 
            Dense(n, n, Lux.σ), 
            Dense(n, 1)) for _ in 1:3]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

i = 0
loss = []

# Training
res = @time Optimization.solve(prob, ADAM(0.1); callback = callback, maxiters=5000)
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(0.01); callback = callback, maxiters=2000)
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(0.0001); callback = callback, maxiters=2000)
phi = discretization.phi

## ANALYSIS!!
u_real(t,x) = [exp(-t)*sin(π*x), exp(-t)*cos(π*x), (1 + π^2)*exp(-t)]

dx = 0.1
xs,ts = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]

# Have to use Lux in order to make this work!
dep_vars = [:u1, :u2, :u3]
minimizers = [res.u.depvar[dep_vars[i]] for i in 1:length(chains)]

u_predict = [[phi[i]([t,x], minimizers[i])[1] for t in ts for x in xs] for i in 1:length(chains)]
u_real_dat = [[u_real(t,x)[i] for t in ts for x in xs] for i in 1:length(chains)]
diff_u = [abs.(u_predict[i] .- u_real_dat[i]) for i in 1:length(chains)]

for i in 1:length(chains)
    plt1 = plot(xs, ts, u_predict[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, numerical")
    plt2 = plot(xs, ts, u_real_dat[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, analytic")
    plt3 = plot(xs, ts, diff_u[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("Difference")
    plot(plt1, plt2, plt3, dpi=200)
    savefig("plot$i.png")
end