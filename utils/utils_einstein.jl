#=
Author: Miles Cochran-Branson
Date: Fall 2022

Functions to aid in set-up of system!
=#
using StaticArrays

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
# Compute inverse matrix 

"""
    minor_components(i::Int, j::Int)

Find elements of minors of a 4x4 matrix represented as a vector. 
"""
function minor_components(i::Int, j::Int)
    m1 = []
    for m in 1:4
        if m == i
            continue
        end
        for n in 1:4
            if n == j
                continue 
            end
            push!(m1, tensor_to_vector(m-1,n-1))
        end
    end
    return m1
end

# use function to define the minors in a static matrix
minors = @SMatrix [minor_components(i,j) for i in 1:4, j in 1:4]

## Determinants
"""
    det_3x3(A::Vector)

Compute determinant of 3x3 matrix represented as a vector. 
"""
function det_3x3(A::Vector)
    if length(A) != 9
        error("Matrix has incorrect size!")
    else
        return (A[1]*(A[5]*A[9] - A[6]*A[8]) -
                    A[2]*(A[4]*A[9] - A[6]*A[7]) +
                        A[3]*(A[4]*A[8] - A[5]*A[7]))
    end
end

"""
    det_4x4(A::Vector)

Compute determinant of 4x4 matrix represented as a vector. 
"""
function det_4x4(A::Vector)
    if length(A) != 16
        error("Matrix has incorrect size!")
    else
        return A[1] * det_3x3([A[i] for i in minors[1,1]]) -
                    A[2] * det_3x3([A[i] for i in minors[1,2]]) +
                        A[3] * det_3x3([A[i] for i in minors[1,3]]) -
                            A[4] * det_3x3([A[i] for i in minors[1,4]])
    end
end

"""
    adjugate_4x4(i::Int, j::Int)

Compute adjugate matrices of a 4x4 matrix represented as a vector. 
"""
function adjugate_4x4(i::Int, j::Int, A::Vector)
    return (-1)^(i+j) * det_3x3([A[n] for n in minors[i,j]])
end

## Finally, compute the inverse wrt each matrix component
"""
    inverse_4x4(A::Vector)

Compute the inverse of a 4x4 matrix represented as a vector. 
"""
function inverse_4x4(A::Vector)
    det = det_4x4(A)
    invs = []
    for i in 1:4
        for j in 1:4
            push!(invs, adjugate_4x4(i,j,A) / det)
        end
    end
    return invs
end

# --------------------------------------------------------
# Compute Ricci tensor using connection coefficients

"""
    Γ(a,b,c,inverse)

Compute connection coefficients.

Currently, I have added restrictions that account for the matrix 
being diagonal. These should eventually be removed. 
"""
function Γ(c::Int, a::Int, b::Int, inverse::Any) 
    eqn = 0
    for d in 0:(n-1)
        eqn += (diff_vec_1[a+1](vars[tensor_to_vector(b,d)](τ,ρ,θ,ϕ)) + 
                    diff_vec_1[b+1](vars[tensor_to_vector(a,d)](τ,ρ,θ,ϕ)) + 
                        diff_vec_1[d+1](vars[tensor_to_vector(a,b)](τ,ρ,θ,ϕ))) * 
                            inverse[tensor_to_vector(c,d)]
    end
    return 0.5 * eqn
end

"""
    Ricci(i, j, inverse)

Compute the [i,j] entry of the Ricci tensor. 
"""
function Ricci(i::Int, j::Int, inverse)
    eqn = 0
    for a in 0:(n-1)
        eqn += diff_vec_1[a+1](Γ(a,i,j,inverse))
    end
    for a in 0:(n-1)
        eqn -= diff_vec_1[j+1](Γ(a,a,i,inverse))
    end
    for a in 0:(n-1)
        for b in 0:(n-1)
            eqn += Γ(a,a,b,inverse)*Γ(b,i,j,inverse) - 
                        Γ(a,i,b,inverse)*Γ(b,a,j,inverse)
        end
    end
    return eqn
end

"""

"""
function Γ_simplified(c::Int, a::Int, b::Int) 
    eqn = 0
    for d in 0:(n-1)
        if c == d
            if b == d
                eqn += diff_vec_1[a+1](vars[tensor_to_vector(b,d)](τ,ρ,θ,ϕ))/vars[tensor_to_vector(c,d)](τ,ρ,θ,ϕ)
            end
            if a == d
                eqn += diff_vec_1[b+1](vars[tensor_to_vector(a,d)](τ,ρ,θ,ϕ))/vars[tensor_to_vector(c,d)](τ,ρ,θ,ϕ)
            end
            if a == b
                eqn += diff_vec_1[d+1](vars[tensor_to_vector(a,b)](τ,ρ,θ,ϕ))/vars[tensor_to_vector(c,d)](τ,ρ,θ,ϕ)
            end
        end
    end
    return 0.5 * eqn
end

"""
"""
function Ricci_simplified(i::Int, j::Int)
    eqn = 0
    for a in 0:(n-1)
        temp = Γ_simplified(a,i,j)
        if typeof(temp) == Num
            eqn += diff_vec_1[a+1](temp)
        end
    end
    for a in 0:(n-1)
        temp = Γ_simplified(a,a,i)
        if typeof(temp) == 0
            eqn -= diff_vec_1[j+1](temp)
        end
    end
    for a in 0:(n-1)
        for b in 0:(n-1)
            temp1 = Γ_simplified(a,a,b)
            temp2 = Γ_simplified(b,i,j)
            temp3 = Γ_simplified(a,i,b)
            temp = Γ_simplified(b,a,j)
            if typeof(temp1)==Num && typeof(temp2)==Num
                eqn += temp1*temp2
            end
            if typeof(temp3)==Num && typeof(temp)==Num
                eqn -= temp3*temp
            end
        end
    end
    return eqn
end

function Γ_simplified2(c::Int, a::Int, b::Int, vars::Any) 
    eqn = 0
    for d in 0:(n-1)
        if c == d
            eqn += (diff_vec_1[a+1](vars[tensor_to_vector(b,d)]) + 
                        diff_vec_1[b+1](vars[tensor_to_vector(a,d)]) + 
                            diff_vec_1[d+1](vars[tensor_to_vector(a,b)])) / 
                                vars[tensor_to_vector(c,d)]
        end
    end
    return 0.5 * eqn
end

"""
    Ricci(i, j, inverse)

Compute the [i,j] entry of the Ricci tensor. 
"""
function Ricci_simplified2(i::Int, j::Int, vars)
    eqn = 0
    for a in 0:(n-1)
        eqn += diff_vec_1[a+1](Γ_simplified2(a,i,j,vars))
    end
    for a in 0:(n-1)
        eqn -= diff_vec_1[j+1](Γ_simplified2(a,a,i,vars))
    end
    for a in 0:(n-1)
        for b in 0:(n-1)
            eqn += Γ_simplified2(a,a,b,vars)*Γ_simplified2(b,i,j,vars) - 
                        Γ_simplified2(a,i,b,vars)*Γ_simplified2(b,a,j,vars)
        end
    end
    return eqn
end