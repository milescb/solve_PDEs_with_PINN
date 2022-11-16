# ------------------------------------------------------------------------
# Book keeping and saving results 

"""
    save_training_files(path::String)

Save results of training to designated `path`.

Here, we save the functional prediction of the network `phi`, 
the result of the training `res`, the `loss_history`, the `discretization`,
and the `domains` used in training. 

We use `JLD2.jl` to save, so this package must be loaded before use. 
"""
function save_training_files(path::String)
    @info "Saving training files to" path
    save_object(path*"/phi.jld2", phi)
    save_object(path*"/res.jld2", res)
    save_object(path*"/loss.jld2", loss_history)
    save_object(path*"/discretization.jld2", discretization)
    save_object(path*"/domains.jld2", domains)
end

"""
    load_training_files(path::String)

Load results of training for analysis or continued training. 

This function returns the objects saved with `save_training_files`
in the order `discretization`, `phi`, `res`, `loss`, `domains`. 

We use `JLD2.jl` to save, so this package must be loaded before use. 
"""
function load_training_files(path::String)
    @info "Loading training file from" path
    discretization = load_object(path*"/discretization.jld2")
    phi = load_object(path*"/phi.jld2")
    res = load_object(path*"/res.jld2")
    loss = load_object(path*"/loss.jld2")
    domains = load_object(path*"/domains.jld2")
    return discretization, phi, res, loss, domains
end

# ------------------------------------------------------------------------
# Algebra, analysis, and conversion

"""
    spherical_to_cartesian(ρ,θ,ϕ)

Describe points in 3D space in cartesian coordinates given spherical.
"""
function spherical_to_cartesian(ρ,θ,ϕ)
    @info "Converting from spherical to cartesean coords."
    x = ρ.*sin.(θ).*cos.(ϕ)
    y = ρ.*sin.(θ).*sin.(ϕ)
    z = ρ.*cos.(θ)
    return x,y,z
end

"""
    distance2(x1,x2,y1,y2)

Compute Euclidean distance between points (x1,y1) and (x2,y2). 
"""
function distance2(x1,x2,y1,y2)
    return ((x1-x2)^2 + (y1-y2)^2)
end

"""
    χ²(obs::Vector, expt::Vector)

Compute chi squared metric. 

First argument is a vector of measured points while the second
contains the expected values. Output of the function is the 
un-normalized chi squared. To get chi squared per degrees of freedom
divide by `length(obs)`. 
"""
function χ²(obs::Vector, expt::Vector)
    out = 0;
    for i in eachindex(obs)
        out += abs((obs[i] - expt[i])^2 / expt[i])
    end
    return out
end

# ------------------------------------------------------------------------
# Definition of callback function for training neural networks
"""
    callback(p,l)

Return loss during optimization. 

Requires definition of a global interator, `i` and an empty 
vector, 'loss_history` to store loss history. 
"""
callback = function (p,l)
    global i += 1
    if i == 1
        @info "Initial loss is: $l"
    end
    #if i % 5 == 0
    @info "Loss at epoch $i is: $l"
    #end
    append!(loss_history, l)
    return false
end