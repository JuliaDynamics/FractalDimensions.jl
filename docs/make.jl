cd(@__DIR__)

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/apply_style.jl",
    joinpath(@__DIR__, "apply_style.jl")
)
include("apply_style.jl")

using FractalDimensions, ComplexityMeasures, StateSpaceSets

FRACTALDIMENSION_PAGES = [
    "index.md",
]

makedocs(
    modules = [FractalDimensions, ComplexityMeasures, StateSpaceSets],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        collapselevel = 3,
    ),
    sitename = "FractalDimensions.jl",
    authors = "George Datseris",
    pages = FRACTALDIMENSION_PAGES,
    doctest = false,
    draft = false,
)

if CI
    deploydocs(
        repo = "github.com/JuliaDynamics/FractalDimensions.jl.git",
        target = "build",
        push_preview = true
    )
end
