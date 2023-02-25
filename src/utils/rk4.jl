
"""
    rk4(rhs,initial,final,yInitial,Npoints::Int)

Solve ODE problem of the form y' = f(t,x)

Uses Runga-Kutta method of order four. In order to solve
second order problems or systems, use the broadcast operater. 
Returns `Npoints`` solution points. 
"""
function runge_kutta4(rhs,initial,final,yInitial,Npoints::Int)
    step_size = (final-initial)/Npoints
    time = initial
    approx = yInitial
    points = [(time, approx)]
    for i in 1:Npoints
        k1 = step_size * rhs(time, approx)
        k2 = step_size * rhs(time+step_size/2, approx.+k1/2)
        k3 = step_size * rhs(time+step_size/2, approx.+k2/2)
        k4 = step_size * rhs(time+step_size/2, approx.+k3)
        approx += (k1 .+ 2*k2 .+ 2*k3 .+ k4)/6
        time = initial + i*step_size
        push!(points, (time, approx))
    end
    return points;
end