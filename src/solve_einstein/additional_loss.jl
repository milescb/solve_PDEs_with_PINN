#=
Here, we define the additional loss function to match our solution to Newtonian 
gravity. All computations are done in Cartesian coordinates in the xy-plane. 

The differential equation we solve within the function is given by 

    d²ᵢ x(t) = c² Γⁱ₀₀
    
To compute the partial derivatives in Γⁱ₀₀, we use a discrete, very approximate
derivate. This gives us a system of ODEs which can be solved by DifferentialEquations.jl. 
After this solving is complete, we take the distance between the points found and those 
given by Newton and add the square of the distance to the loss function. 
=#
using SciMLSensitivity

ϵ = 0.1f0 # smaller epsilon results in seg fault
"""
    additional_loss(phi,θ,p)

Compute additional loss when matching solution to newtonian gravity. 

Parameters:
    - phi: prediction from network
    - θ: weights of network
    - p: parameters of PDE system. Unused here. 
"""
function additional_loss(phi, θ, p)

    # g00 component of metric from neural network.
    # must convert from spherical to cartesian coords. 
    g00(x,y) = phi[1](sqrt(x^2+y^2), θ.depvar[:A])[1]

    # set-up the problem using current metric
    function simple_geodesic(ddu,du,u,p,t)
        # derivate is done with discrete methods
        ddu[1] = (c^2/2) * ((g00(u[1]+ϵ,u[2]) - g00(u[1]-ϵ,u[2]))/2ϵ)
        ddu[2] = (c^2/2) * ((g00(u[1],u[2]+ϵ) - g00(u[1],u[2]-ϵ))/2ϵ)
    end

    # solve system of diff-eqs. `saveat` and `dt` keywords are required to
    # avoid costly interpolation. 
    prob_ode = SecondOrderODEProblem(simple_geodesic, dx0, x0, tspan)
    sol = solve(prob_ode, Rosenbrock32(), saveat=dx, dt=dx)

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
    return 0.1 * sum(distance2(sol_nts[i][1],sol[i][3],
        sol_nts[i][2],sol[i][4]) for i in 1:iter)
end