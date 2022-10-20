# Solving PDEs with Physics-Informed Neural Networks 

The following repository contains examples using Physics-Informed Neural Networks (PINN) to solve PDEs. We use the package `NeuralPDE.jl` to solve. 

## List of examples

Find a list of the example problems we have solved or are working on below

- Integral-PDE
- PDAE
- Einstein field equations to find Schwarzschild metric (in development)

## Running the Code

Several required packages are included in the `Project.toml` to allow one to run this code out of the box. You can use the environment in this repo to quickly load the correct versions of the packages by running

```
julia> using Pkg
julia> Pkg.instantiate()
```

Then, to run the code either activate the environment and run from the REPL, or run the scripts with 

```
julia --project <file_name>.jl
```
from terminal. Note that `Julia` needs to be in your path for this to work. 

## Contribution

Pull requests are encouraged!