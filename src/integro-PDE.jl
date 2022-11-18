#=
Author: Miles Cochran-Branson
Date: Fall 2022, senior capstone work

We solve the first example given in https://springerplus.springeropen.com/articles/10.1186/2193-1801-3-137,
namely the system of equations given by 

    ∂ₜu₁(t,x) = ∂ₓ²u₁(t,x) - u₂(t,x)∂ₓu₁(t,x) + u₁(t,x)² - 2 ∫ u₁(t,x)² dx
    0 = ∂ₓ u₂(t,x) - u₁(t,x)
    
subject to the initial conditions

    u₁(0,x) = cos(πx)   0 ≤ x ≤ 1

and the boundry conditions

    ∂ₓu₁(t,0) = ∂ₓu₁(t,1) = u₂(t,0) = u₂(t,1) = 0    t ≥ 0.

The paper notes the solution to this system is given by 

    u₁(t,x) = exp(-π²t) cos(πx)
    u₂(t,x) = (1/π) exp(-π²t) sin(πx).

The above paper computes the PDE analytically using series expansions. 

Below, we show how this system can be easily solved using PINN and the NeuralPDE.jl package!
=#
using NeuralPDE, ModelingToolkit, Optimization, Lux, DomainSets
using Random, CUDA
using Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum

include("./utils/general_utils.jl")

@parameters t x
@variables u1(..) u2(..)

Dx = Differential(x)
Dxx = Differential(x)^2
Dt = Differential(t)
Ix = Integral(x in DomainSets.ClosedInterval(0, 1))

eqns = [Dt(u1(t,x)) ~ Dxx(u1(t,x)) - u2(t,x)*Dx(u1(t,x)) + (u1(t,x))^2 - 2*Ix((u1(t,x))^2),
        0 ~ Dx(u2(t,x)) - u1(t,x)]

bcs = [u1(0,x) ~ cos(π*x),
        Dx(u1(t,0)) ~ 0,
        Dx(u1(t,1)) ~ 0,
        u2(t,0) ~ 0,
        u2(t,1) ~ 0
]

domains = [x ∈ Interval(0.0, 1.0),
            t ∈ Interval(0.0, 1.0)]

@named pde_sys = PDESystem(eqns, bcs, domains, [t,x], [u1(t,x), u2(t,x)])

dim = length(domains) # number of dimensions
n = 15
chains = [Lux.Chain(
            Dense(dim, n, Lux.σ), 
            Dense(n, n, Lux.σ), 
            Dense(n, 1)) for _ in 1:2]

# Use GPUs
ps = [Lux.setup(Random.default_rng(), chains[i])[1] for i in 1:2]
ps = [ps[i] |> Lux.ComponentArray |> gpu .|> Float32 for i in 1:2]

#strategy = QuadratureTraining()
strategy = QuasiRandomTraining(100)
discretization = PhysicsInformedNN(chains, strategy, init_params = ps)
@time prob = discretize(pde_sys, discretization)

# parameters for callback
i = 0
loss_history = []
learning_rates = [0.1, 0.01, 0.0001]

# Training
res = @time Optimization.solve(prob, ADAM(learning_rates[1]); callback=callback_every100, maxiters=500)
loss1_history = loss_history
loss_history = []
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(learning_rates[2]); callback=callback_every100, maxiters=2000)
loss2_history = loss_history
loss_history = []
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(learning_rates[3]); callback=callback_every100, maxiters=2000)
loss3_history = loss_history
loss_history = vcat(loss1_history, loss2_history, loss3_history)
phi = discretization.phi

## Analysis
u_analytic(t,x) = [exp(-π^2 * t) * cos(π*x), (1/π) * exp(-π^2*t) * sin(π*x)]

dx = 0.1
xs,ts = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]

dep_vars = [:u1, :u2]
minimizers = [res.u.depvar[dep_vars[i]] for i in 1:length(chains)]

u_predict = [[phi[i]([t,x], minimizers[i])[1] for t in ts for x in xs] for i in 1:length(chains)]
u_real = [[u_analytic(t,x)[i] for t in ts for x in xs] for i in 1:length(chains)]
diff_u = [abs.(u_predict[i] .- u_real[i]) for i in 1:length(chains)]

for i in 1:length(chains)
    plt1 = plot(xs, ts, u_predict[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, numerical")
    plt2 = plot(xs, ts, u_real[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, analytic")
    plt3 = plot(xs, ts, diff_u[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("Difference")
    plot(plt1, plt2, plt3, dpi=200)
    savefig("plots/integro_PDE/plot_u$i.png")
end

plt = plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss", 
            size=(400,400), dpi=300, label="", yaxis=:log)
plot!(1:length(loss1_history), loss1_history, 
        label="Learning rate $(learning_rates[1])")
plot!(length(loss1_history)+1:length(loss1_history)+length(loss2_history), 
        loss2_history, label="Learning rate $(learning_rates[2])")
plot!(length(loss1_history)+length(loss2_history)+1:length(loss1_history)+
        length(loss2_history)+length(loss3_history), 
        loss3_history, label="Learning rate $(learning_rates[3])")
savefig("plots/integro_PDE/loss.png")