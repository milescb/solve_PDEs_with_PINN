#=
Author: Miles Cochran-Branson
Date: Fall 2022

Set-up differential equations to solve EFE to obtain 
Schwarzschild metric. 
=#

"""
    tensor_to_vector(i::Int, j::Int)

Find vector element of tensor when read left to right, top to bottom. 
"""
function tensor_to_vector(i::Int, j::Int)
    iter = 0
    for n in 0:3
        for m in 0:3
            iter += 1
            if n == i && m == j
                return iter
            end
        end
    end
end

# --------------------------------------------------------
# Compute Ricci tensor using connection coefficients

"""
    Γ(a,b,c)

Compute connection coefficients. Assume metric `g` is diagonal. 
"""
function Γ(c::Int, a::Int, b::Int, vars::Vector, invs::Vector) 
    eqn = 0
    for d in 0:(n-1)
        if c == d
            if b == d
                eqn += (diff_vec_1[a+1](vars[tensor_to_vector(b,d)]))*invs[tensor_to_vector(c,d)]
            end
            if a == d
                eqn += (diff_vec_1[b+1](vars[tensor_to_vector(a,d)]))*invs[tensor_to_vector(c,d)]
            end
            if a == b
                eqn += (diff_vec_1[d+1](vars[tensor_to_vector(a,b)]))*invs[tensor_to_vector(c,d)]
            end
        end
    end
    return 0.5 * eqn
end

"""
    PDE_equations(i,j,inverse)

Compute PDEs for finding Schwarzschild metric.
"""
function PDE_equations(i::Int, j::Int, vars::Vector, invs::Vector)
    eqn = 0
    for a in 0:(n-1)
        eqn += diff_vec_1[a+1](Γ(a,i,j,vars,invs))
    end
    for a in 0:(n-1)
        for b in 0:(n-1)
                eqn += Γ(a,i,b,vars,invs)*Γ(b,j,a,vars,invs)
        end
    end
    return eqn;
end

function Γ_ODE(c::Int, a::Int, b::Int, vars::Vector, invs::Vector) 
    eqn = 0.0
    for d in 0:(n-1)
        if c == d
            if (b == d && a == 1)
                eqn += (diff_vec_1[a+1](vars[tensor_to_vector(b,d)]))*invs[tensor_to_vector(c,d)]
            end
            if (a == d && b == 1)
                eqn += (diff_vec_1[b+1](vars[tensor_to_vector(a,d)]))*invs[tensor_to_vector(c,d)]
            end
            if (a == b && d == 1)
                eqn += (diff_vec_1[d+1](vars[tensor_to_vector(a,b)]))*invs[tensor_to_vector(c,d)]
            end
        end
    end
    return 0.5 * eqn
end

"""
    ODE_equations(i,j,inverse)

Compute ODEs for finbding Schwarzschild metric
"""
function ODE_equations(i::Int, j::Int, vars::Vector, invs::Vector)
    eqn = 0
    # if i == j
    #     eqn += diff_vec_1[2](Γ_ODE(2,i,j,vars,invs))
    # end
    if j == 1
        for a in 0:(n-1)
            eqn -= diff_vec_1[2](Γ_ODE(a,a,i,vars,invs))
        end
    end
    for a in 0:(n-1)
        for b in 0:(n-1)
                eqn += Γ_ODE(a,a,b,vars,invs)*Γ_ODE(b,i,j,vars,invs)
                eqn -= Γ_ODE(a,i,b,vars,invs)*Γ_ODE(b,a,j,vars,invs)
        end
    end
    return eqn;
end