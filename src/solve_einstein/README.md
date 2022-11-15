# Solve Einstein's field equations to obtain Schwarzschild metric

This folder contains the work done on applying appropriate assumptions to solve the Einstein field equations numerically. These solutions are then compared to the known analytical solution. 

## Layout of the files

- `solve.jl` demonstrates solving this problem in the easiest possible manner. Notice that despite this working _most_ of the time, training may crash from time to time. Solutions are also highly dependent on initial conditions of the network, so running this file until a good result of loss is produced is a requirement. 
- `additional_loss.jl` provides the additional term to the loss function to make the solution in `solve.jl` match newtonian gravity. 
- `plot.jl` contains code for plotting and analysis. 
- `solve_continue.jl` allows for continued solving of this problem from `solve.jl` after initial training. 
- `extended_setup.jl` shows how this work may be extented to multiple dimensions. This is very much in developement currently. 