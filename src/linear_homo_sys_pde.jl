using NeuralPDE, ModelingToolkit, Optimization, Lux, OptimizationOptimisers, DomainSets
using Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum

include("../utils/general_utils.jl")

@parameters t x
@variables u(..) v(..)

Dx = Differential(x)
Dt = Differential(t)

eqns = [Dt(u(t,x)) - Dx(v(t,x)) + (u(t,x) + v(t,x)) ~ 0,
        Dt(v(t,x)) - Dx(u(t,x)) + (u(t,x) + v(t,x)) ~ 0]

bcs = [u(0,x) ~ sinh(x), v(0,x) ~ cosh(x)]

domains = [x ∈ Interval(0.0, 1.0),
            t ∈ Interval(0.0, 1.0)]

@named pde_sys = PDESystem(eqns, bcs, domains, [t,x], [u(t,x), v(t,x)])

dim = length(domains) # number of dimensions
n = 15
chains = [Lux.Chain(
            Dense(dim, n, Lux.σ), 
            Dense(n, n, Lux.σ), 
            Dense(n, 1)) for _ in 1:2]

strategy = QuadratureTraining()
discretization = PhysicsInformedNN(chains, strategy)
@time prob = discretize(pde_sys, discretization)

i = 0
loss_history = []

learning_rates = [1e-3, 1e-4, 1e-7]
res = @time Optimization.solve(prob, ADAM(1e-3); callback = callback, maxiters=10000)
loss_history1 = loss_history
loss_history = []
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(1e-5); callback = callback, maxiters=7000)
loss_history2 = loss_history
loss_history = []
prob = remake(prob, u0=res.minimizer)
res = @time Optimization.solve(prob, ADAM(1e-7); callback = callback, maxiters=1000)
loss_history3 = loss_history
loss_history = vcat(loss_history1, loss_history2, loss_history3)
phi = discretization.phi

# ---------------------------------------------------------------------------------------------
## Evaluate results 
analytic_solution(t,x) = [sinh(x-t), cosh(x-t)]

dx = 0.1
xs,ts = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains]

dep_vars = [:u, :v]
minimizers = [res.u.depvar[dep_vars[i]] for i in 1:length(chains)]

u_predict = [[phi[i]([t,x], minimizers[i])[1] for t in ts for x in xs] for i in 1:length(chains)]
u_real = [[analytic_solution(t,x)[i] for t in ts for x in xs] for i in 1:length(chains)]
diff_u = [abs.(u_predict[i] .- u_real[i]) for i in 1:length(chains)]

for i in 1:length(chains)
    plt1 = plot(xs, ts, u_predict[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, numerical")
    plt2 = plot(xs, ts, u_real[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("\$u_{$(i)}(t,x)\$, analytic")
    plt3 = plot(xs, ts, diff_u[i], linetype=:contourf, xlabel=L"t", ylabel=L"x")
    title!("Difference")
    plot(plt1, plt2, plt3, dpi=200)
    savefig("plots/linear_homo_sys/solution_$i.png")
end

plt = plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss", 
            size=(400,400), dpi=300, label="", yaxis=:log)
plot!(1:length(loss_history1), loss_history1, 
        label="Learning rate $(learning_rates[1])")
plot!(length(loss_history1)+1:length(loss_history1)+length(loss_history2), 
        loss_history2, label="Learning rate $(learning_rates[2])")
plot!(length(loss_history1)+length(loss_history2)+1:length(loss_history1)+
        length(loss_history2)+length(loss_history3), 
        loss_history3, label="Learning rate $(learning_rates[3])")
savefig("plots/linear_homo_sys/loss.png")