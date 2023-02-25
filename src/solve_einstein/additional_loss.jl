#=
Here, we define the additional loss function to match our solution to Newtonian 
gravity. All computations are done in Cartesian coordinates in the xy-plane. 

The differential equation we solve within the function is given by 

    d²ᵢ x(t) = c² Γⁱ₀₀
    
To compute the partial derivatives in Γⁱ₀₀, we use Zygote.jl to compute uing 
Automatic Differentiation. This gives us a system of ODEs which can be solved 
by DifferentialEquations.jl. After this solving is complete, we take the 
distance between the points found and those given by Newton and add the square 
of the distance to the loss function. 
=#
using SciMLSensitivity, Zygote

"""
    g00(x,y,phi,θ)

Extract g00 component from g matrix with input in Cartesian coords.

Arguments `phi` and `θ` are the prediction and weights of the network. 
Arguments `x` and `y` are Cartesian coords. Conversion to sphereical 
coords. is done within the function. 
"""
g00(x,y,phi,θ) = phi[1](sqrt(x^2+y^2), θ.depvar[:A])[1]

"""
    additional_loss(phi,θ,p)

Compute additional loss when matching solution to newtonian gravity. 

Parameters:
    - phi: prediction from network
    - θ: weights of network
    - p: parameters of PDE system. Unused here. 
"""
function additional_loss(phi, θ, p)

    # set-up the problem using current metric
    function simple_geodesic(t,u)
        # take derivative with Zygote
        grad = Zygote.gradient((x,y) -> g00(x,y,phi,θ), u[1], u[3])
        du = [0.,0.,0.,0.]
        du[1] = u[2]
        du[2] = -(c^2/2) * grad[1]
        du[3] = u[4]
        du[2] = -(c^2/2) * grad[2]
    end

    # solve system of diff-eqs. `saveat` and `dt` keywords are required to
    # avoid costly interpolation. 
    sol = runge_kutta4(simple_geodesic,0.0,1.0,initial_conditions,nPoints)

    # hack to avoid differing lengths when warning dt <= dtmin
    # usually not used / necessary 
    if length(sol) > length(sol_nts)
        @warn "length of temp solution too long"
        iter = length(sol_nts)
    elseif length(sol) < length(sol_nts)
        @warn "length of temp solution too short"
        iter = length(sol)
    else 
        iter = length(sol)
    end

    # take 10% of given loss so as to not have this term dominate
    return 0.01 * sum(distance2(sol_nts[i][1],sol[i][3],
        sol_nts[i][2],sol[i][4]) for i in 1:iter)
end