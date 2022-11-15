#=
Author: Miles Cochran-Branson
Date: Fall 2022

We document here how extensions of the problem to higher dimensions are possible.
The required differential equations are derived directly from the Ricci tensor. 

Note that this is _not_ being using to solve currently and significant work would
need to be done in order to optimize the code. Future work would focus on

    - Ensuring that all differential equations are computed correctly
    - Applying appropriate boundary conditions
    - Match Newtonian gravity in the limit of low velocity and weak gravity 
      as in `solve.jl`

This file is merely here to provide an architecture with which to work in creating 
a more complex solution. 
=#
@info "Loading Packages"
using NeuralPDE, ModelingToolkit, Optimization, Lux, OptimizationOptimisers, DiffEqFlux
using Plots
import ModelingToolkit: Interval
using JLD2

include("../utils/utils_einstein.jl")
# although the inverse of a diagonal matrix is just the inverse 
# of the diagonal elements, this allows for generalization if need-be
include("../utils/compute_inverse_4x4.jl")
include("../utils/general_utils.jl")

# include constants of the problem
include("constants.jl")

params = @parameters τ ρ θ ϕ
vars = @variables g00(..) g11(..)

@info "Variables and parameters loaded" params vars

Dτ = Differential(τ)
Dρ = Differential(ρ)
Dθ = Differential(θ)
Dϕ = Differential(ϕ)

diff_vec_1 = [Dτ, Dρ, Dθ, Dϕ]

# shape of entire matrix 
g_matrix_complete = [g00(τ,ρ,θ,ϕ), 0            , 0           , 0            , #=
                  =# 0           , g11(τ,ρ,θ,ϕ) , 0           , 0            , #= 
                  =# 0           , 0            , -ρ^2        , 0            , #=
                  =# 0           , 0            , 0           , -ρ^2 * (sin(θ))^2]

g_matrix_inverse = inverse_4x4(g_matrix_complete)

# ensure we do not have identical equations; matrix is symmetric
indices_eqns = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
eqns = [PDE_equations(k[1],k[2],g_matrix_complete,g_matrix_inverse) ~ 0 for k in indices_eqns]

ρ_limit = 60.f0
bcs = [ # spherically symetric
        g00(τ,ρ,θ,0) ~ g00(τ,ρ,θ,2π),
        g11(τ,ρ,θ,0) ~ g11(τ,ρ,θ,2π),
        # flat space-time as r → ∞
        g00(τ,ρ_limit,θ,ϕ) ~ 1,
        g11(τ,ρ_limit,θ,ϕ) ~ -1,
        # also requirements on τ
        Dτ(g00(τ,ρ,θ,ϕ)) ~ 0,
        Dτ(g11(τ,ρ,θ,ϕ)) ~ 0,
]

domains = [τ ∈ Interval(0, 10.0),
            ρ ∈ Interval(0.1, ρ_limit),
            θ ∈ Interval(0, π),
            ϕ ∈ Interval(0, 2π)]

@named pde_sys = PDESystem(eqns, bcs, domains, [τ, ρ, θ, ϕ], [g00(τ,ρ,θ,ϕ), g11(τ,ρ,θ,ϕ)])