cd(@__DIR__)

using FractalDimensions, ComplexityMeasures, StateSpaceSets

FRACTALDIMENSION_PAGES = [
    "index.md",
]

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib");
    style = :authoryear
)

build_docs_with_style(FRACTALDIMENSION_PAGES, FractalDimensions, ComplexityMeasures, StateSpaceSets;
    expandfirst = ["index.md"], bib, warnonly = [:cross_references, :doctest, :missing_docs],
)
