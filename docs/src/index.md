# FractalDimensions.jl

```@docs
FractalDimensions
```

## Introduction

!!! note
    This package is accompanying a review paper on the fractal dimension: <https://arxiv.org/abs/2109.05937>. The paper is continuing the discussion of chapter 5 of [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7), Datseris & Parlitz, Springer 2022.


There are numerous methods that one can use to calculate a so-called "dimension" of a dataset which in the context of dynamical systems is called the [Fractal dimension](https://en.wikipedia.org/wiki/Fractal_dimension).
In the majority of cases, computing a fractal dimension means estimating the **scaling behaviour of some quantity as a size/scale increases**. In the [Fractal dimension example](@ref) below, one finds the scaling of the entropy of the histogram of some data, versus the width of the bins of the histogram. In this case, it approximately holds
$$
H \approx -\Delta\log(\varepsilon)
$$
for bin width $\varepsilon$. The scaling of many other quantities can be estimated as well, such as the correlation sum, the Higuchi length, or others provided here.

To actually find $\Delta$, one needs to find a linearly scaling region in the graph $H$ vs. $\log(\varepsilon)$ and estimate its slope. Hence, **identifying a linear region is central to estimating a fractal dimension**. That is why, the section [Linear scaling regions](@ref) is of central importance for this documentation.


## Fractal dimension example

In this simplest example we will calculate the fractal dimension of the [chaotic attractor of the Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) (for default parameters).

```@example MAIN
using DynamicalSystemsBase: DeterministicIteratedMap, trajectory
using CairoMakie

henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
u0 = zeros(2)
p0 = [1.4, 0.3]
henon = DeterministicIteratedMap(henon_rule, u0, p0)

X, t = trajectory(henon, 100_000; Ttr = 100)
scatter(X[:, 1], X[:, 2]; color = ("black", 0.01), markersize = 4)
```

Our goal is to compute entropies of the histogram of the above plot for many different partition sizes (bin widths) `ε`. Computing entropies is the job of [ComplexityMeasures.jl](https://github.com/JuliaDynamics/ComplexityMeasures.jl), but the two relevant names (`entropy, ValueHistogram`) are re-exported by FractalDimensions.jl.
```@example MAIN
using FractalDimensions
ες = 2 .^ (-15:0.5:5) # semi-random guess
Hs = [entropy(ValueHistogram(ε), X) for ε in ες]
```

```@example MAIN
xs = @. -log2(ες) # must use same base as `entropy`!!!
scatterlines(xs, Hs; axis = (ylabel = L"H_1", xlabel = L"-\log (\epsilon)"))
```

The slope of the linear scaling region of the above plot is the generalized dimension (of order q = 1) for the attractor of the Hénon map.

Given that we _see_ the plot, we can estimate where the linear scaling region starts and ends. However, we can use the function [`linear_region`](@ref) to get an estimate of the result as well. First let's visualize what it does, as it uses [`linear_regions`](@ref).

```@example MAIN
lrs, slopes = linear_regions(xs, Hs, tol = 0.25)
fig = Figure()
ax = Axis(fig[1,1]; ylabel = L"H_1", xlabel = L"-\log (\epsilon)")
for r in lrs
    scatterlines!(ax, xs[r], Hs[r])
end
fig
```

The [`linear_region`](@ref) function finds, and computes the slope of, the largest region:

```@example MAIN
Δ = linear_region(xs, Hs)[2]
```
This result is an approximation of the information dimension (because we used `q = 1`) of the attractor.

The whole above pipeline we went through is bundled in [`generalized_dim`](@ref). Similar pipeline is done by [`grassberger_proccacia_dim`](@ref) and many other functions.

!!! danger "Be wary when using `xxxxx_dim`"
    As stated clearly by the documentation strings, all pre-made dimension estimating functions (ending in `_dim`) perform a lot of automated steps, each having its own heuristic choices for function default values.
    They are more like convenient bundles with on-average good defaults, rather than precise functions. You should be careful
    when considering the validity of the returned number!

## Linear scaling regions

And other utilities, especially [`linreg`](@ref), used in both [`generalized_dim`] and [`grassberger_dim`](@ref).
```@docs
linear_regions
linear_region
linreg
estimate_boxsizes
minimum_pairwise_distance
```

## Generalized (entropy) dimension
Based on the definition of the Generalized entropy ([`genentropy`](@ref)), one can calculate an appropriate dimension, called *generalized dimension*:
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

## Kaplan-Yorke dimension
```@docs
kaplanyorke_dim
```

## Higuchi dimension
```@docs
higuchi_dim
```

## Extreme value value theory dimension
```@docs
extremevaltheory_dim
extremevaltheory_dims_persistences
extremevaltheory_local_dim_persistence
extremal_index_sueveges
estimate_gpd_parameters
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