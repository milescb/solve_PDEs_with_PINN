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
        println("Current loss is: $l")
    end
    append!(loss_history, l)
    return false
end