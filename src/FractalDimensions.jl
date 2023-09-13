module FractalDimensions

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end FractalDimensions

# Define function for turning off progress bars
envprog() = get(ENV, "FRACTALDIMENSIONS_PROGRESS", "true") == "true"

using Reexport
@reexport using StateSpaceSets

import Distances

using ComplexityMeasures
export entropy, Shannon, Renyi, ValueHistogram

include("linear_fits/api.jl")
include("linear_fits/estimate_boxsizes.jl")
include("linear_fits/linear_regression.jl")
include("linear_fits/linear_regions.jl")
include("linear_fits/slopes_distribution.jl")

include("entropy_based/generalized_dim.jl")
include("entropy_based/molteno.jl")

include("corrsum_based/correlationsum_vanilla.jl")
include("corrsum_based/correlationsum_boxassisted.jl")
include("corrsum_based/correlationsum_fixedmass.jl")
include("corrsum_based/takens_best_estimate.jl")

include("extremes_based/extremesdim.jl")
include("extremes_based/confidence.jl")

include("timeseries_roughness/higuchi.jl")

include("misc/kaplanyorke.jl")

end