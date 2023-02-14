#=
Author: Miles Cochran-Branson
Date: Winter 2023

A basic example to solve the Poisson equation given by 

    ∇²u(x,y) = -sin(πx)sin(πy)

with boudary conditions

    u(x,0) = u(x,1) = u(0,y) = u(1,y) = 0.

This has solution 

    u(x,y) = sin(πx)sin(πy)/2π². 
=#
using NeuralPDE, Lux, ModelingToolkit
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Plots
import ModelingToolkit: Interval, infimum, supremum

@parameters x y
@variables u(..)

Dxx = Differential(x)^2
Dyy = Differential(y)^2

eqn = [Dxx(u(x,y)) + Dyy(u(x,y)) ~ -sin(pi*x)*sin(pi*y)]

bcs = [u(x,0) ~ 0, 
        u(x,1) ~ 0,
        u(0,y) ~ 0,
        u(1,y) ~ 0]

domains = [x ∈ Interval(0, 1),
            y ∈ Interval(0, 1)]

@named pdesys = PDESystem(eqn, bcs, domains, [x,y], [u(x,y)])

dim = 2 # number of dimensions
chain = Lux.Chain(Lux.Dense(dim, 16, Lux.σ), 
                    Lux.Dense(16, 16, Lux.σ), 
                    Lux.Dense(16, 1))

discretization = PhysicsInformedNN(chain, QuadratureTraining())

prob = ModelingToolkit.discretize(pdesys,discretization)

i = 0
loss_history = []
callback = function (p,l)
    global i += 1
    if i % 500 == 0
        println("Current loss is: $l")
    end
    push!(loss_history, l)
    return false
end

## Learn solution
res = Optimization.solve(prob, BFGS(); callback = callback, maxiters=500)
phi = discretization.phi

## Plot results 
plot(1:length(loss_history), loss_history, size=(400,400),
    xlabel="epoch", ylabel="loss", yaxis=:log, label="")
savefig("plots/poisson/loss.png")

xs,ys = [infimum(d.domain):0.1/10:supremum(d.domain) for d in domains]
analytic_sol_func(x,y) = (sin(pi*x)*sin(pi*y))/(2pi^2)

u_predict = reshape([first(phi([x,y],res.minimizer)) for x in xs for y in ys],(length(xs),length(ys)))
u_real = reshape([analytic_sol_func(x,y) for x in xs for y in ys], (length(xs),length(ys)))
diff_u = abs.(u_predict .- u_real)

p1 = plot(xs, ys, u_real, linetype=:contourf,title = "Analytic");
p2 = plot(xs, ys, u_predict, linetype=:contourf,title = "Numeric");
p3 = plot(xs, ys, diff_u,linetype=:contourf,title = "Error");
plot(p1,p2,p3,dpi=300)

savefig("plots/poisson/solution.png")