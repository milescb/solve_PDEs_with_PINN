#=
Author: Miles Cochran-Branson
Date: Fall 2022

Functions to aid in set-up of system!
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

Compute connection coefficients.
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
                eqn += Γ(a,i,b,vars)*Γ(b,j,a,vars,invs)
        end
    end
    return eqn;
end