module FractalDimension

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end FractalDimension

using Reexport
@reexport using StateSpaceSets
using DelayEmbeddings: embed

end