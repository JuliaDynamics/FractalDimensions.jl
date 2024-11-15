import ProgressMeter

using Distances: evaluate, Euclidean, pairwise, Metric
using Statistics:quantile

export correlationsum, boxed_correlationsum
export grassberger_proccacia_dim
export pointwise_dimensions, pointwise_correlationsums, local_correlation_dimension

"""
    grassberger_proccacia_dim(X::AbstractStateSpaceSet, εs = estimate_boxsizes(data); kwargs...)

Use the method of Grassberger and Proccacia [Grassberger1983](@cite), and the correction by
[Theiler1986](@cite), to estimate the correlation dimension `Δ_C` of  `X`.

This function does something extremely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
Δ_C = slopefit(rs, ys)(log2.(sizes), log2.(cm))[1]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus log(ε).

See [`correlationsum`](@ref) for the available keywords.
See also [`takens_best_estimate_dim`](@ref), [`boxassisted_correlation_dim`](@ref).
"""
function grassberger_proccacia_dim(X::AbstractStateSpaceSet, εs = estimate_boxsizes(X); kwargs...)
    cm = correlationsum(X, εs; kwargs...)
    return slopefit(log2.(εs), log2.(cm))[1]
end


"""
    correlationsum(X, ε::Real; w = 0, norm = Euclidean(), q = 2) → C_q(ε)

Calculate the `q`-order correlation sum of `X` (`StateSpaceSet` or timeseries)
for a given radius `ε` and `norm`. They keyword `show_progress = true` can be used
to display a progress bar for large `X`.

    correlationsum(X, εs::AbstractVector; w, norm, q) → C_q(ε)

If `εs` is a vector, `C_q` is calculated for each `ε ∈ εs` more efficiently.
Multithreading is also enabled over the available threads (`Threads.nthreads()`).
The function [`boxed_correlationsum`](@ref) is typically faster if the dimension of `X`
is small and if `maximum(εs)` is smaller than the size of `X`.

## Keyword arguments

- `q = 2`: order of the correlation sum
- `norm = Euclidean()`: distance norm
- `w = 0`: Theiler window
- `show_progress = true`: display a progress bar

## Description

The correlation sum is defined as follows for `q=2`:
```math
C_2(\\epsilon) = \\frac{2}{(N-w)(N-w-1)}\\sum_{i=1}^{N}\\sum_{j=1+w+i}^{N}
B(||X_i - X_j|| < \\epsilon)
```
for as follows for `q≠2`
```math
C_q(\\epsilon) = \\left[ \\sum_{i=1}^{N} \\alpha_i
\\left[\\sum_{j:|i-j| > w} B(||X_i - X_j|| < \\epsilon)\\right]^{q-1}\\right]^{1/(q-1)}
```
where
```math
\\alpha_i = 1 / (N (\\max(N-w, i) - \\min(w + 1, i))^{(q-1)})
```
with ``N`` the length of `X` and ``B`` gives 1 if its argument is
`true`. `w` is the [Theiler window](@ref).

See the article of Grassberger for the general definition [Grassberger2007](@cite) and
the book "Nonlinear Time Series Analysis" [Kantz2003](@cite), Ch. 6, for
a discussion around choosing best values for `w`, and Ch. 11.3 for the
explicit definition of the q-order correlationsum. Note that the formula in 11.3
is incorrect, but corrected here, indices are adapted to take advantage of all available
points and also note that we immediatelly exponentiate
``C_q`` to ``1/(q-1)``, so that it scales exponentially as
``C_q \\propto \\varepsilon ^\\Delta_q`` versus the size ``\\varepsilon``.
"""
correlationsum(X, ε::Real; kw...) = correlationsum(X, [ε]; kw...)[1]

function correlationsum(X, εs; q = 2, norm = Euclidean(), w = 0, show_progress = envprog())
    q ≤ 1 && @warn "The correlation sum is ill-defined for q ≤ 1."
    issorted(εs) || error("Sorted `ε` required for optimized version.")
    if q == 2
        correlationsum_2(X, εs, norm, w, show_progress)
    else
        correlationsum_q(X, εs, eltype(X)(q), norm, w, show_progress)
    end
end

# Many different algorithms were tested for a fast implementation of the vanilla
# correlation sum. At the end, we went full circle and returned to the simplest
# possible implementation, which becomes the fastest once multithreading is enabled.

function correlationsum_2(X, εs::AbstractVector{<:Real}, norm, w, show_progress)
    N = length(X)
    progress = ProgressMeter.Progress(N; desc="Correlation sum: ", enabled=show_progress)
    Css = [zeros(Int, length(εs)) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in Base.OneTo(N)
        x = X[i]
        Cs = Css[Threads.threadid()]
        for j in i+1+w:N
            dist = norm(x, X[j])
            lastidx = searchsortedfirst(εs, dist)
            Cs[lastidx:end] .+= 1
        end
        ProgressMeter.next!(progress)
    end
    C = sum(Css)
    return C .* (2 / ((N-w-1)*(N-w)))
end

function correlationsum_q(X, εs::AbstractVector{<:Real}, q, norm, w, show_progress)
    N, C = length(X), zero(eltype(X))
    irange = 1:N
    progress = ProgressMeter.Progress(N;
        desc = "Correlation sum: ", enabled = show_progress
    )
    Css = [zeros(eltype(X), length(εs)) for _ in 1:Threads.nthreads()]
    Css_dum = [zeros(eltype(X), length(εs)) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for i in irange
        x = X[i]
        normalisation = (max(N-w, i) - min(w+1, i))
        Cs = Css[Threads.threadid()]
        Cs_dum = Css_dum[Threads.threadid()]
        Cs_dum .= zero(eltype(X))
        # computes all distances from 0 up to i-w
        for j in 1:i-w-1
            dist = norm(x, X[j])
            lastidx = searchsortedfirst(εs, dist)
            Cs_dum[lastidx:end] .+= 1
        end
        # computes all distances after i+w till the end
        for j in i+w+1:N
            dist = norm(x, X[j])
            lastidx = searchsortedfirst(εs, dist)
            Cs_dum[lastidx:end] .+= 1
        end
        @. Cs += (Cs_dum / normalisation)^(q-1)
        ProgressMeter.next!(progress)
    end

    C = sum(Css)
    return @. (C / N) ^ (1 / (q-1))
end

#######################################################################################
# Pointwise dimension
#######################################################################################
"""
    pointwise_dimensions(X::StateSpaceSet, εs::AbstractVector; kw...) → Δloc

Return the pointwise dimensions for each point in `X`, i.e.,
the exponential scaling of the inner correlation sum

```math
c_q(\\epsilon) = \\left[\\sum_{j:|i-j| > w} B(||X_i - X_j|| < \\epsilon)\\right]^{q-1}
```
versus ``\\epsilon``. `Δloc[i]` is the exponential scaling (deduced by a call to
[`linear_region`](@ref)) of ``c_q`` versus ``\\epsilon`` for the `i`th point of `X`.

Keywords are the same as in [`correlationsum`](@ref).
To obtain the inner correlation sums without doing the exponential scaling fit
use `FractalDimensions.pointwise_correlationsums` with same inputs.
"""
function pointwise_dimensions(X, εs::AbstractVector = estimate_boxsizes(X); kw...)
    Cs = pointwise_correlationsums(X, εs; kw...)
    # Linear region loop
    x = log.(εs)
    function local_slope(x, C)
        L = length(x)
        i = findfirst(c -> c > 0, C)
        z = view(x, i:L)
        y = log.(C)[i:end]
        return linear_region(z, y; warning = false)[2]
    end
    Dlocs = map(C -> local_slope(x, C), Cs)
    return Dlocs
end

function pointwise_correlationsums(X, εs::AbstractVector;
        norm = Euclidean(), w = 0, q = 2, show_progress = true
    )
    E, T = length(εs), eltype(X)
    Cs = [zeros(T, E) for _ in eachindex(X)]
    progress = ProgressMeter.Progress(length(X);
        desc="Pointwise corrsum: ", dt=1, enabled=show_progress
    )

    Threads.@threads for i in eachindex(X)
        C = Cs[i]
        # distances of points in the range of indices around `i`
        distances = pointwise_distances_fast(X, i, w, norm)
        for k in E:-1:1
            C[k] = count(<(εs[k]), distances)^(q-1)
        end
        ProgressMeter.next!(progress)
    end
    # We do not do any normalization here on the correlation sums
    return Cs
end

# similar function as used in the q-order correlation sum
@inbounds function pointwise_distances_fast(X, i, w, norm)
    r1 = 1:i-w-1
    lr1 = length(r1)
    r2 = i+w+1:length(X)
    v = X[i]
    out = zeros(eltype(X), lr1+length(r2))
    for j in eachindex(r1)
        ξ = r1[j]
        out[j] = norm(v, X[ξ])
    end
    for j in eachindex(r2)
        ξ = r2[j]
        out[j+lr1] = norm(v, X[ξ])
    end
    return out
end

"""
    local_correlation_dimension(X, ζ [, εs]; kw...) → Δ_ζ

Return the local dimension `Δ_ζ` around state space point `ζ` given a set of state space
points `X` which is assumed to surround (or be sufficiently near to) `ζ`.
The local dimension is the exponential scaling of the correlation sum for point `ζ` versus
some radii `εs`. `εs` can be a vector of reals, or it can be an integer,
in which space that many points are equi-spaced logarithmically between the minimum and
maximum distance of `X` to `ζ`.

## Keyword arguments

- `q = 2, norm = Euclidean()`: same as in [`correlationsum`](@ref).
- `fit = LinearRegression()`: given to [`slopefit`](@ref) to estimate the dimension.
  This default assumes that the set `X` is already sufficiently close to `ζ`.
"""
function local_correlation_dimension(args...; fit = LinearRegression(), kw...)
    C, es = local_correlation_sum(args...; kw...)
    ΔGP = slopefit(log2.(es), log2.(C), fit)[1]
    return ΔGP
end

function local_correlation_sum(X, ζ, k = 8; norm = Euclidean(), q = 2)
    # First estimate distances
    dists = map(x -> norm(x, ζ), X)
    εs = _generate_boxsizes(k, dists)
    C = zeros(Int, length(εs))
    # Then convert to correlation sum
    for i in eachindex(εs)
        C[i] = count(<(εs[i]), dists)^(q-1)
    end
    return C/length(dists), εs
end

function _generate_boxsizes(k::Int, dists)
    dmin, dmax = quantile(dists, [0.1, 0.9]) # second smallest
    base = MathConstants.e
    lower = log(base, dmin)
    upper = log(base, dmax)
    return float(base) .^ range(lower, upper; length = k)
end

_generate_boxisizes(es::AbstractVector, dists) = es