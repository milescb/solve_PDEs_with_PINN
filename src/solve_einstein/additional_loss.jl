ϵ = 0.1f0 # smaller epsilon results in seg fault
"""
    additional_loss(phi,θ,p)

Compute additional loss when matching solution to newtonian gravity. 
"""
function additional_loss(phi, θ, p)

    # 00 component of metric from neural network
    g00(x,y) = phi[1](sqrt(x^2+y^2), θ.depvar[:A])[1]

    # set-up the problem using current metric
    function simple_geodesic(ddu,du,u,p,t)
        ddu[1] = (c^2/2) * ((g00(u[1]+ϵ,u[2]) - g00(u[1]-ϵ,u[2]))/2ϵ)
        ddu[2] = (c^2/2) * ((g00(u[1],u[2]+ϵ) - g00(u[1],u[2]-ϵ))/2ϵ)
    end

    # solve system of diff-eqs 
    prob_ode = SecondOrderODEProblem(simple_geodesic, dx0, x0, tspan)
    sol = solve(prob_ode, Rosenbrock32(), saveat=dx, dt=dx)

    # hack to avoid differing lengths when error dt less than dmin
    if length(sol) > length(sol_nts)
        @warn "length of temp solution too long"
        iter = length(sol_nts)
    elseif length(sol) < length(sol_nts)
        @warn "length of temp solution too short"
        iter = length(sol)
    else 
        iter = length(sol)
    end

    return 0.1 * sum(distance2(sol_nts[i][1],sol[i][3],
        sol_nts[i][2],sol[i][4]) for i in 1:iter)
end