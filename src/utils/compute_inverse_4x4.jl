#=
Author: Miles Cochran-Branson
Date: Fall 2022

Functions to compute inverse of a 4x4 matrix. This can be used by
including this script via `include("compute_inverse_4x4.jl")` and 
then calling the function `inverse_4x4(A)`. This function takes as
input a vector, A, of length 16 which represents a 4x4 matrix and 
returns a vector where each element is a component of the inverse 
of A. 
=#

using StaticArrays

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
    if length(A) != 16
        error("Incorrect length of vector. Must be of length 16")
    end
    det = det_4x4(A)
    if det == 0
        error("Matrix is not invertible")
    end
    invs = []
    for i in 1:4
        for j in 1:4
            push!(invs, adjugate_4x4(i,j,A) / det)
        end
    end
    return invs
end