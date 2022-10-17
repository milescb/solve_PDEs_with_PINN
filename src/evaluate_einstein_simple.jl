using ModelingToolkit, NeuralPDE
using JLD2, Plots, LaTeXStrings
import ModelingToolkit: Interval, infimum, supremum

include("../utils/general_utils.jl")

# define some consts
const G = 6.67e-11 # m²/kg²
const c = 3e8 # m/s²
const M = 1.989e30 #kg
const ricci_r = 2*G*M/c^2 

# load results of training
discretization, phi, res, loss = load_training_files("trained_networks/EFE_simple")

params = @parameters τ ρ θ ϕ
vars = @variables g00(..) g01(..) g02(..) g03(..) #=
               =# g10(..) g11(..) g12(..) g13(..) #= 
               =# g20(..) g21(..) g22(..) g23(..) #=
               =# g30(..) g31(..) g32(..) g33(..)

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(1, 100),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

τs, ρs, θs, ϕs = [domain.domain.lower:1.0:domain.domain.upper for domain in domains]

dep_vars = [:g00, :g11]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

u_analytic(τ,ρ,θ,ϕ) = [1 - ricci_r/ρ, -(1 - ricci_r/ρ)^(-1)]

u_predict = [[phi[i]([τ,ρ,θ,ϕ], minimizers[i])[1] for τ in τs for ρ in ρs for θ in θs for ϕ in ϕs] for i in 1:length(dep_vars)]
u_real = [[u_analytic(τ,ρ,θ,ϕ)[i] for τ in τs for ρ in ρs for θ in θs for ϕ in ϕs] for i in 1:length(dep_vars)]
diff_u = [abs.(u_predict[i] .- u_real[i]) for i in 1:length(dep_vars)]

# Plots them thangs!
for i in eachindex(dep_vars)
    plt1 = plot(τs, u_predict[i], label="Predicted Solution", xaxis=L"\tau", 
        yaxis=L"g_{00}", size=(300,300), dpi = 200)
    plot!(τs, u_real[i], label="True Solution")

    plt2 = plot(ρs, u_predict[i], label="Predicted Solution", xaxis=L"\rho", 
        yaxis=L"g_{00}", size=(300,300), dpi=200)
    plot!(ρs, u_real[i], label="True Solution")

    # plt3 = plot(θs, u_predict[i], label="Predicted Solution", xaxis=L"\theta", 
    #     yaixs=L"g_{00}", size=(300,300), dpi=200)
    # plot!(θs, u_real[i], label="True Solution")

    plt4 = plot(ϕs, u_predict[i])

    plot(plt1, plt2)
    savefig("../solve_PDEs_with_PINN/plots/EPE_simple_solution/g$(i)$(i)_solution.png")
end