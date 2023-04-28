module FractalDimensions

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end FractalDimensions

using StateSpaceSets
export StateSpaceSet, SVector, AbstractStateSpaceSet

import Distances

using ComplexityMeasures
export entropy, Shannon, Renyi, ValueHistogram

include("linear_regions.jl")

include("entropy_based/generalized_dim.jl")
include("entropy_based/molteno.jl")

include("corrsum_based/correlationsum_vanilla.jl")
include("corrsum_based/correlationsum_boxassisted.jl")
include("corrsum_based/correlationsum_fixedmass.jl")
include("corrsum_based/takens_best_estimate.jl")

include("timeseries_roughness/higuchi.jl")

include("misc/kaplanyorke.jl")

end