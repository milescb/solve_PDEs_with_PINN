using NeuralPDE, ModelingToolkit, Optimization, Flux, OptimizationOptimisers
using JLD2, Plots
import ModelingToolkit: Interval, infimum, supremum

include("utils/general_utils.jl")

discretization, phi, res, loss = load_training_files("trained_networks/EFE_simple")

# Create grid for plotting
vars = @variables g00(..) g01(..) g02(..) g03(..) #=
               =# g10(..) g11(..) g12(..) g13(..) #= 
               =# g20(..) g21(..) g22(..) g23(..) #=
               =# g30(..) g31(..) g32(..) g33(..)

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(6e6, 1e8),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

τs, ρs, θs, ϕs = [domain.domain.lower:0.01:domain.domain.upper for domain in domains]

dep_vars = [:g00, :g11, :g22, :g33]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

u_predict = [[phi[i]([t,x], minimizers[i])[1] for t in ts for x in xs] for i in 1:length(dep_vars)]
u_real = [[u_analytic(t,x)[i] for t in ts for x in xs] for i in 1:length(dep_vars)]
diff_u = [abs.(u_predict[i] .- u_real[i]) for i in 1:length(dep_vars)]