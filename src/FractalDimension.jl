module FractalDimension

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end FractalDimension

using Reexport
@reexport using StateSpaceSets

include("includes/linear_regions.jl")
include("includes/generalized_dim.jl")
include("includes/correlationsum_vanilla.jl")
include("includes/correlationsum_boxassisted.jl")
include("includes/correlationsum_fixedmass.jl")
include("includes/molteno.jl")
include("includes/kaplanyorke.jl")
include("includes/takens_best_estimate.jl")
include("includes/higuchi.jl")

end