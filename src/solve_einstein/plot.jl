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

## Compare solution to analytic!
u_analytic(ρ) = [1 - ricci_r/ρ, -1/(1 - ricci_r/ρ)]

dep_vars = [:A, :B]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

dr = 0.01
rs = [infimum(d.domain):dr:supremum(d.domain) for d in domains][1]

u_real = [[u_analytic(r)[i] for r in rs] for i in 1:2]
u_predict = [[phi[i]([r], minimizers[i])[1] for r in rs] for i in 1:2]

plot(rs, u_real[1], xlabel=L"r", ylabel=L"A(r)", label="True Solution",
        size=(400,400), dpi=200, legend=:right)
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