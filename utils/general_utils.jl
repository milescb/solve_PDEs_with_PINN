function save_training_files(path)
    @info "Saving training files to" path
    save_object(path*"/phi.jld2", phi)
    save_object(path*"/res.jld2", res)
    save_object(path*"/loss.jld2", loss_history)
    save_object(path*"/discretization.jld2", discretization)
end

function load_training_files(path)
    @info "Loading training file from" path
    discretization = load_object(path*"/discretization.jld2")
    phi = load_object(path*"/phi.jld2")
    res = load_object(path*"/res.jld2")
    loss = load_object(path*"/loss.jld2")
    return discretization, phi, res, loss
end
# Definition of callback function
"""
    callback(p,l)

Return loss during optimization. 

Requires definition of a global interator, `i` and an empy 
vector to store loss history. 
"""
callback = function (p,l)
    global i += 1
    if i % 100 == 0
        @info "Current loss is: $l"
    end
    append!(loss_history, l)
    return false
end