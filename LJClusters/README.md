# LJClusters.jl

A simple Lennard-Jones cluster model.

## Usage

```julia
(@v1.8) pkg> add https://github.com/sunoru/mosimo-example.git:subdir=LJClusters

julia> using MosimoBase, LJClusters

# Declare a LJ cluster model of 8 particles.
julia> model = LJCluster3D(8);
# If `s` is a previously defined `MosiSystem` that has 8 particles,
# then the potential energy of `s` can be calculated as:
julia> potential_energy(s, model)
```
