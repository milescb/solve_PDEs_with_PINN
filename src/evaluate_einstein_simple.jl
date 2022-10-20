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

gran = 25
τs, ρs, θs, ϕs = [infimum(d.domain):((supremum(d.domain)-infimum(d.domain))/gran):supremum(d.domain) for d in domains]
τs = τs[1:gran]
ρs = ρs[1:gran]
θs = θs[1:gran]
ϕs = ϕs[1:gran]

dep_vars = [:g00, :g11]
minimizers = [res.u.depvar[dep_vars[i]] for i in eachindex(dep_vars)]

u_analytic(τ,ρ,θ,ϕ) = [1 - ricci_r/ρ, -(1 - ricci_r/ρ)^(-1)]

u_predict = [[[[phi[i]([τ,ρ,θ,ϕ], minimizers[i])[1] for ρ in ρs for θ in θs] for ϕ in ϕs] for τ in τs] for i in 1:length(dep_vars)]
u_real = [[[[u_analytic(τ,ρ,θ,ϕ)[i] for ρ in ρs for θ in θs] for ϕ in ϕs] for τ in τs] for i in 1:length(dep_vars)]
diff_u = [[[abs.(u_predict[i][j][k] .- u_real[i][j][k]) for k in eachindex(ϕs)] for j in eachindex(τs)] for i in eachindex(dep_vars)]

#xs, ys, zs = spherical_to_cartesian(ρs, θs, ϕs)

# Plots them thangs!
for g in eachindex(dep_vars)
    anim = @animate for i ∈ eachindex(τs)
        plt1 = plot(ρs, θs, u_predict[g][i][1], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Predicted(τ=$(τs[i]),ϕ=0)")
        plt2 = plot(ρs, θs, u_real[g][i][1], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Analytic(τ=$(τs[i]),ϕ=0)")
        plt3 = plot(ρs, θs, diff_u[g][i][1], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Difference(τ=$(τs[i]),ϕ=0)")
        plot(plt1, plt2, plt3)
    end
    gif(anim, "./plots/EPE_simple_solution/sol_tau_g$g$g.gif", fps = 5)

    anim = @animate for i ∈ eachindex(τs)
        plt1 = plot(ρs, θs, u_predict[g][1][i], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Predicted(τ=0,ϕ=$(round(ϕs[i])))")
        plt2 = plot(ρs, θs, u_real[g][1][i], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Analytic(τ=0,ϕ=$(round(ϕs[i])))")
        plt3 = plot(ρs, θs, diff_u[g][1][i], linetype=:contourf, dpi=200,
                xlabel=L"\rho", ylabel=L"\theta", zlabel=L"\tau")
        title!("Difference(τ=0,ϕ=$(round(ϕs[i])))")
        plot(plt1, plt2, plt3)
    end
    gif(anim, "./plots/EPE_simple_solution/sol_phi_g$g$g.gif", fps = 5)
end