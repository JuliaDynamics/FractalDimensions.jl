# FractalDimensions.jl

```@docs
FractalDimensions
```

## Introduction

!!! note
    This package is accompanying a review paper on estimating the fractal dimension: <https://arxiv.org/abs/2109.05937>. The paper is continuing the discussion of chapter 5 of [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7), Datseris & Parlitz, Springer 2022.


There are numerous methods that one can use to calculate a so-called "dimension" of a dataset which in the context of dynamical systems is called the [Fractal dimension](https://en.wikipedia.org/wiki/Fractal_dimension).
One way to do this is to estimate the **scaling behaviour of some quantity as a size/scale increases**. In the [Fractal dimension example](@ref) below, one finds the scaling of the correlation sum versus a ball radius. In this case, it approximately holds
$$
\log(C) \approx \Delta\log(\varepsilon)
$$
for radius $\varepsilon$. The scaling of many other quantities can be estimated as well, such as the generalized entropy, the Higuchi length, or others provided here.

To actually find $\Delta$, one needs to find a linearly scaling region in the graph $\log(C)$ vs. $\log(\varepsilon)$ and estimate its slope. Hence, **identifying a linear region is central to estimating a fractal dimension**. That is why, the section [Linear scaling regions](@ref) is of central importance for this documentation.


## Fractal dimension example

In this simplest example we will calculate the fractal dimension of the [chaotic attractor of the Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) (for default parameters). For this example, we will generate the data on the spot:

```@example MAIN
using DynamicalSystemsBase # for simulating dynamical systems
using CairoMakie           # for plotting

henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
u0 = zeros(2)
p0 = [1.4, 0.3]
henon = DeterministicIteratedMap(henon_rule, u0, p0)

X, t = trajectory(henon, 20_000; Ttr = 100)
scatter(X[:, 1], X[:, 2]; color = ("black", 0.01), markersize = 4)
```
instead of simulating the set `X` we could load it from disk, e.g., if there was a text file with two columns as x and y coordinates, we would load it as
```julia
using DelimitedFiles
file = "path/to/file.csv"
M = readdlm(file)    # here `M` is a metrix with two columns
X = StateSpaceSet(M) # important to convert to a state space set
```

After we have `X`, we can start computing a fractal dimension and for this example we will use the [`correlationsum`](@ref). Our goal is to compute the correlation sum of `X` for many different sizes/radii `ε`. This is as simple as
```@example MAIN
using FractalDimensions
ες = 2 .^ (-15:0.5:5) # semi-random guess
Cs = correlationsum(X, ες; show_progress = false)
```

For a fractal set `X` dynamical systems theory says that there should be an exponential relationship between the correlation sum and the sizes:
```@example MAIN
xs = log2.(ες)
ys = log2.(Cs)
scatterlines(xs, ys; axis = (ylabel = L"\log(C_2)", xlabel = L"\log (\epsilon)"))
```

The slope of the linear scaling region of the above plot is the fractal dimension (based on the correlation sum).

Given that we _see_ the plot, we can estimate where the linear scaling region starts and ends. This is generally done using [`LargestLinearRegion`](@ref) in [`slopefit`](@ref). But first, let's visualize what the method does, as it uses [`linear_regions`](@ref).

```@example MAIN
lrs, slopes = linear_regions(xs, ys, tol = 0.25)
fig = Figure()
ax = Axis(fig[1,1]; ylabel = L"\log(C_2)", xlabel = L"\log (\epsilon)")
for r in lrs
    scatterlines!(ax, xs[r], ys[r])
end
fig
```

The [`LargestLinearRegion`](@ref) method finds, and computes the slope of, the largest region:

```@example MAIN
Δ = slopefit(xs, ys, LargestLinearRegion())
```
This result is an approximation of _a_ fractal dimension.

The whole above pipeline we went through is bundled in [`grassberger_proccacia_dim`](@ref). Similar work is done by [`generalized_dim`](@ref) and many other functions.

!!! danger "Be wary when using `xxxxx_dim`"
    As stated clearly by the documentation strings, all pre-made dimension estimating functions (ending in `_dim`) perform a lot of automated steps, each having its own heuristic choices for function default values.
    They are more like convenient bundles with on-average good defaults, rather than precise functions. You should be careful
    when considering the validity of the returned number!


## Index (contents)

```@index
```

## Linear scaling regions

```@docs
slopefit
LinearRegression
linreg
```

```@docs
LargestLinearRegion
linear_regions
linear_region
```

```@docs
AllSlopesDistribution
estimate_boxsizes
minimum_pairwise_distance
```

## Generalized (entropy) dimension

Based on the definition of the generalized (Renyi) entropy, one can calculate an appropriate dimension, called *generalized dimension*:
```@docs
generalized_dim
molteno_dim
molteno_boxing
```

## Correlation sum based dimension

```@docs
grassberger_proccacia_dim
correlationsum
```

### Box-assisted version

```@docs
boxassisted_correlation_dim
boxed_correlationsum
prismdim_theiler
estimate_r0_buenoorovio
estimate_r0_theiler
```

## Fixed mass correlation sum

```@docs
fixedmass_correlation_dim
fixedmass_correlationsum
```

## Takens best estimate

```@docs
takens_best_estimate_dim
```

## Pointwise (local) correlation dimensions

```@docs
pointwise_dimensions
pointwise_correlationsums
```


## Kaplan-Yorke dimension

```@docs
kaplanyorke_dim
```

## Higuchi dimension

```@docs
higuchi_dim
```

## Extreme value value theory dimensions

The central function for this is [`extremevaltheory_dims_persistences`](@ref) which utilizes either [`Exceedances`](@ref) or [`BlockMaxima`](@ref).

### Main functions
```@docs
extremevaltheory_dims_persistences
extremevaltheory_dim
extremevaltheory_dims
extremevaltheory_local_dim_persistence
extremal_index_sueveges
```

### Exceedances estimator

```@docs
Exceedances
estimate_gpd_parameters
extremevaltheory_gpdfit_pvalues
```

### Block-maxima estimator
```@docs
BlockMaxima
estimate_gev_parameters
```

## Theiler window

The Theiler window is a concept that is useful when finding neighbors in a dataset that is coming from the sampling of a continuous dynamical system.
Itt tries to eliminate spurious "correlations" (wrongly counted neighbors) due to a potentially dense sampling of the trajectory. Typically a good choice for `w` coincides with the choice an optimal delay time, see `DelayEmbeddings.estimate_delay`, for any of the timeseries of the dataset.

For more details, see Chapter 5 of [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7), Datseris & Parlitz, Springer 2022.

## `StateSpaceSet` reference

```@docs
StateSpaceSet
standardize
```

## References

```@bibliography
```