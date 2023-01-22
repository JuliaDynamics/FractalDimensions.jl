# FractalDimensions.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDynamics.github.io/FractalDimension.jl/stable)
[![](https://img.shields.io/badge/DOI-10.1177/00375497211068820-purple)](https://journals.sagepub.com/doi/10.1177/00375497211068820)
[![CI](https://github.com/JuliaDynamics/FractalDimension.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/FractalDimension.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/FractalDimension.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/FractalDimension.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/FractalDimension)](https://pkgs.genieframework.com?packages=FractalDimension)

A Julia package that estimates fractal dimension(s) from data.
It can be used as a standalone package, or as part of
[DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/).

To install it, run `import Pkg; Pkg.add("FractalDimensions")`.

All further information is provided in the documentation, which you can either find [online](https://juliadynamics.github.io/FractalDimensions.jl/dev/) or build locally by running the `docs/make.jl` file.

_Previously, this package was part of ChaosTools.jl._


## Citation

If you use this package in a publication, please cite the paper below:
```
@ARTICLE{FractalDimensions.jl,
  title     = "Estimating the fractal dimension: a comparative review and open
               source implementations",
  author    = "Datseris, George and Kottlarz, Inga and Braun, Anton P and
               Parlitz, Ulrich",
  publisher = "arXiv",
  year      =  2021
}
```