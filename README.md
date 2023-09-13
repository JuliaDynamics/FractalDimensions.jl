# FractalDimensions.jl

[![](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/fractaldimensions/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/fractaldimensions/stable/)
[![](https://img.shields.io/badge/DOI-10.48550/ARXIV.2109.05937-purple)](https://arxiv.org/abs/2109.05937)
[![CI](https://github.com/JuliaDynamics/FractalDimensions.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/FractalDimensions.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/FractalDimensions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/FractalDimensions.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/FractalDimensions)](https://pkgs.genieframework.com?packages=FractalDimensions)

A Julia package that estimates various definitions of fractal dimension from data.
It can be used as a standalone package, or as part of
[DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/dev/).

To install it, run `import Pkg; Pkg.add("FractalDimensions")`.

All further information is provided in the documentation, which you can either find [online](https://juliadynamics.github.io/FractalDimensions.jl/stable/) or build locally by running the `docs/make.jl` file.

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
  year      =  2021,
  doi = {10.48550/ARXIV.2109.05937},
  url = {https://arxiv.org/abs/2109.05937},
}
```