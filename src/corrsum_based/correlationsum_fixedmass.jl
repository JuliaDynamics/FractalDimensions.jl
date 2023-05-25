export fixedmass_correlationsum, fixedmass_correlation_dim

using SpecialFunctions: digamma
using Neighborhood: bulksearch, KDTree, NeighborNumber, Theiler
using Random: randperm

"""
    fixedmass_correlation_dim(X [, max_j]; kwargs...)

Use the fixed mass algorithm for computing the correlation sum, and use
the result to compute the correlation dimension `Δ_M` of `X`.

This function does something extremely simple:
```julia
rs, ys = fixedmass_correlationsum(X, args...; kwargs...)
slopefit(rs, ys)
```
"""
function fixedmass_correlation_dim(X, args...; kwargs...)
    rs, ys = fixedmass_correlationsum(X, args...; kwargs...)
    return slopefit(rs, ys)
end

"""
    fixedmass_correlationsum(X [, max_j]; metric = Euclidean(), M = length(X)) → rs, ys

A fixed mass algorithm for the calculation of the [`correlationsum`](@ref),
and subsequently a fractal dimension ``\\Delta``,
with `max_j` the maximum number of neighbours that
should be considered for the calculation.

By default `max_j = clamp(N*(N-1)/2, 5, 32)` with `N` the data length.

## Keyword arguments

- `M` defines the number of points considered for the averaging of distances,
  randomly subsampling them from `X`.
- `metric = Euclidean()` is the distance metric.
- `start_j = 4` computes the equation below starting from `j = start_j`. Typically
  the first `j` values have not converged to the correct scaling of the fractal dimension.

## Description

"Fixed mass" algorithms mean that instead of trying to find all neighboring points
within a radius, one instead tries to find the max radius containing `j` points.
A correlation sum is obtained with this constrain, and equivalently the mean radius
containing `k` points.
Based on this, one can calculate ``\\Delta`` approximating the information dimension.
The implementation here is due to to [^Grassberger1988], which defines
```math
Ψ(j) - \\log N \\sim \\Delta \\times \\overline{\\log \\left( r_{(j)}\\right)}
```
where `` \\Psi(j) = \\frac{\\text{d} \\log Γ(j)}{\\text{d} j}
`` is the digamma function, `rs` = ``\\overline{\\log \\left( r_{(j)}\\right)}`` is the mean
logarithm of a radius containing `j` neighboring points, and
`ys` = ``\\Psi(j) - \\log N`` (``N`` is the length of the data).
The amount of neighbors found ``j`` range from 2 to `max_j`.
The numbers are also converted to base ``2`` from base ``e``.

``\\Delta`` can be computed by using `linear_region(rs, ys)`.

[^Grassberger1988]:
    Peter Grassberger (1988) [Finite sample Corrections to Entropy and Dimension Estimates,
    Physics Letters A 128(6-7)](https://doi.org/10.1016/0375-9601(88)90193-4)
"""
function fixedmass_correlationsum(X, max_j = _max_j_from_data(X);
    start_j = 4, metric = Euclidean(), M = length(X), w = 0)
    start_j < 2 && error("At least `j=1` must be skipped. Use `start_j > 1`.")
    N = length(X)
    @assert M ≤ N
    tree = KDTree(X, metric)
    searchdata = view(X, view(randperm(N), 1:M))
    _, distances = bulksearch(tree, searchdata, NeighborNumber(round(Int, max_j)), Theiler(w))
    # The ys define the left side of the equation
    ys = [digamma(j) - log(N) for j in start_j:max_j]
    # Holds the mean value of the logarithms of the distances.
    rs = zeros(length(ys))
    for dists in distances
        for (i, j) in enumerate(start_j:max_j)
            rs[i] += log(dists[j])
        end
    end
    change = log(2, MathConstants.e)
    return rs .* change ./ M, ys .* change
end

_max_j_from_data(X) = (N = length(X); clamp(round(Int, sqrt(N*(N-1)/2)), 5, 32))
