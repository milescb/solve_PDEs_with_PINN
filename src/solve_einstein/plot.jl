#=
Author: Miles Cochran-Branson
Date: Fall 2022

    Given the trained network saved in `trained_networks`, we can plot how 
well our implementation matches the analytical solutions. Additionally, we 
evaluate how well our solution matches newtonian gravity, thus giving some 
idea on how well the condition required by the additional loss term is 
being met. 
    All plots are saved in the folder `plots`. 
=#
using Plots, LaTeXStrings, JLD2

# unfortunately, you have to load these for JLD2 to properly load objects given in training
using NeuralPDE, ModelingToolkit, Lux
using Optimization, OptimizationOptimisers, OptimizationOptimJL
import ModelingToolkit: Interval, infimum, supremum

@parameters r
@variables A(..) B(..)

# constants for problem
include("constants.jl")

include("../utils/general_utils.jl")
include("additional_loss.jl")

discretization, phi, res, loss_history, domains = load_training_files("./trained_networks/EFE_ODE_diff")

# -------------------------------------------------------------------------------------
## plot loss as a function of Epoch
plot(1:length(loss_history), loss_history, xlabel="Epoch", ylabel="Loss",
        size=(400,400), dpi=200, label="")
savefig("./plots/EPE_ODE_solution/loss.png")

# Define analytical solution
u_analytic(ρ) = [1 - ricci_r/ρ, 1/(1 - ricci_r/ρ)]

dep_vars = [:A, :B]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

dr = 0.01
rs = [infimum(d.domain):dr:supremum(d.domain) for d in domains][1]

u_real = [[u_analytic(r)[i] for r in rs] for i in 1:2]
u_predict = [[phi[i]([r], minimizers[i])[1] for r in rs] for i in 1:2]

plot(rs, u_real[1], xlabel=L"r", ylabel=L"A(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:topright)
plot!(rs, u_predict[1], 
        label="Predicted Solution, \$\\chi^2/dof = $(round(χ²(u_predict[1], 
            u_real[1]),digits=2))/$(length(u_predict[1]))\$")
savefig("./plots/EPE_ODE_solution/A.png")

plot(rs, u_real[2], xlabel=L"r", ylabel=L"B(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:topright)
plot!(rs, u_predict[2], 
        label="Predicted Solution,  \$\\chi^2/dof = $(round(χ²(u_predict[2], 
        u_real[1]),digits=2))/$(length(u_predict[2]))\$")
savefig("./plots/EPE_ODE_solution/B.png")

# -------------------------------------------------------------------------------------
# How well do we match Newton? Let's find out
g00_NN(x,y) = phi[1](sqrt(x^2+y^2), minimizers[1])[1]

function simple_geodesic(ddu,du,u,p,t)
    # take derivative with Zygote
    grad = Zygote.gradient((x,y) -> g00_NN(x,y), u[1], u[2])
    ddu[1] = -(c^2/2) * grad[1]
    ddu[2] = -(c^2/2) * grad[2]
end

# solve ODE problem 
prob_ode = SecondOrderODEProblem(simple_geodesic, dx0, x0, tspan)
sol = solve(prob_ode, Tsit5())

sol_ts = [[sol[i][3], sol[i][4]] for i in eachindex(sol)]

# plot and compare to newton solution!
plot(getindex.(sol_ts,1),getindex.(sol_ts,2), label="NN", size=(400,400), dpi=200,
        xlabel="\$x\$ (AU)", ylabel="\$y\$ (AU)")
plot!(getindex.(sol_nts,1),getindex.(sol_nts,2), label="Newton", legend=:topleft)
savefig("./plots/EPE_ODE_solution/compare_to_newton.png")