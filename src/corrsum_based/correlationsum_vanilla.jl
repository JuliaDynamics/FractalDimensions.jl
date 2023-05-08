import ProgressMeter

using Distances: evaluate, Euclidean, pairwise, Metric

export correlationsum, boxed_correlationsum
export grassberger_proccacia_dim
export pointwise_dimensions, pointwise_correlationsums

"""
    grassberger_proccacia_dim(X::AbstractStateSpaceSet, εs = estimate_boxsizes(data); kwargs...)

Use the method of Grassberger and Proccacia[^Grassberger1983], and the correction by
Theiler[^Theiler1986], to estimate the correlation dimension `Δ_C` of  `X`.

This function does something extremely simple:
```julia
cm = correlationsum(data, εs; kwargs...)
Δ_C = linear_region(log2.(sizes), log2.(cm))[2]
```
i.e. it calculates [`correlationsum`](@ref) for various radii and then tries to find
a linear region in the plot of the log of the correlation sum versus log(ε).

See [`correlationsum`](@ref) for the available keywords.
See also [`takens_best_estimate`](@ref), [`boxassisted_correlation_dim`](@ref).

[^Grassberger1983]:
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)
[^Theiler1986]:
    Theiler, [Spurious dimension from correlation algorithms applied to limited time-series
    data. Physical Review A, 34](https://doi.org/10.1103/PhysRevA.34.2427)
"""
function grassberger_proccacia_dim(X::AbstractStateSpaceSet, εs = estimate_boxsizes(X); kwargs...)
    cm = correlationsum(X, εs; kwargs...)
    return linear_region(log2.(εs), log2.(cm))[2]
end


"""
    correlationsum(X, ε::Real; w = 0, norm = Euclidean(), q = 2) → C_q(ε)
Calculate the `q`-order correlation sum of `X` (`StateSpaceSet` or timeseries)
for a given radius `ε` and `norm`. They keyword `show_progress = true` can be used
to display a progress bar for large `X`.

    correlationsum(X, εs::AbstractVector; w, norm, q) → C_q(ε)

If `εs` is a vector, `C_q` is calculated for each `ε ∈ εs` more efficiently.
If also `q=2`, we attempt to do further optimizations, if the allocation of
a matrix of size `N×N` is possible.

The function [`boxed_correlationsum`](@ref) is typically faster and should be preferred.

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
C_q(\\epsilon) = \\left[\\frac{1}{\\alpha} \\sum_{i=w+1}^{N-w}
\\left[\\sum_{j:|i-j| > w} B(||X_i - X_j|| < \\epsilon)\\right]^{q-1}\\right]^{1/(q-1)}
```
where
```math
\\alpha = (N-2w)(N-2w-1)^{(q-1)}
```
with ``N`` the length of `X` and ``B`` gives 1 if its argument is
`true`. `w` is the [Theiler window](@ref).

See the article of Grassberger for the general definition [^Grassberger2007] and
the book "Nonlinear Time Series Analysis" [^Kantz2003], Ch. 6, for
a discussion around choosing best values for `w`, and Ch. 11.3 for the
explicit definition of the q-order correlationsum. Note that the formula in 11.3
is incorrect, but corrected here, and also note that we immediatelly exponentiate
``C_q`` to ``1/(q-1)``, so that it scales exponentially as
``C_q \\propto \\varepsilon ^\\Delta_q`` versus the size ``\\varepsilon``.

[^Grassberger2007]:
    Peter Grassberger (2007) [Grassberger-Procaccia algorithm. Scholarpedia,
    2(5):3043.](http://dx.doi.org/10.4249/scholarpedia.3043)

[^Kantz2003]:
    Kantz, H., & Schreiber, T. (2003). [Nonlinear Time Series Analysis,
    Cambridge University Press.](https://doi.org/10.1017/CBO9780511755798)
"""
function correlationsum(X, ε; q = 2, norm = Euclidean(), w = 0, show_progress = false)
    q ≤ 1 && @warn "The correlation sum is ill-defined for q ≤ 1."
    if q == 2
        correlationsum_2(X, ε, norm, w, show_progress)
    else
        correlationsum_q(X, ε, eltype(X)(q), norm, w, show_progress)
    end
end

#######################################################################################
# Real ε implementations (in case matrix doesn't fit in memory)
#######################################################################################
function correlationsum_2(X, ε::Real, norm, w, show_progress)
    N = length(X)
    progress = ProgressMeter.Progress(N; desc="Correlation sum: ", enabled=show_progress)
    C = zero(eltype(X))
    @inbounds Threads.@threads for i in Base.OneTo(N)
        x = X[i]
        for j in i+1+w:N
            C += evaluate(norm, x, X[j]) < ε
        end
        ProgressMeter.next!(progress)
    end
    return C * 2 / ((N-w-1)*(N-w))
end

function correlationsum_q(X, ε::Real, q, norm, w, show_progress)
    N, C = length(X), zero(eltype(X))
    progress = ProgressMeter.Progress(length(1+w:N-w);
        desc = "Correlation sum: ", enabled = show_progress
    )
    @inbounds Threads.@threads for i in 1+w:N-w
        x = X[i]
        C_current = zero(eltype(X))
        # computes all distances from 0 up to i-w
        for j in 1:i-w-1
            C_current += evaluate(norm, x, X[j]) < ε
        end
        # computes all distances after i+w till the end
        for j in i+w+1:N
            C_current += evaluate(norm, x, X[j]) < ε
        end
        C += C_current^(q - 1)
        ProgressMeter.next!(progress)
    end
    normalisation = (N-2w)*(N-2w-1)^(q-1)
    return (C / normalisation) ^ (1 / (q-1))
end

#######################################################################################
# Vector ε implementations: q = 2
#######################################################################################
# Many different algorithms were tested for a fast implementation of the vanilla
# correlation sum. The fastest was the one currently here. It pre-computes
# the distances as a vector (as not all distances need be computed
# due to the theiler window and duplicity). Then, for each vector of distances
# we simply count how many are less that ε.
# We also tested sorting the distances and then using `searchsortedfirst`.
# However, oddly, this was consistently slower. I guess it is because
# counting <(ε) has different scaling with N than sort has.

function correlationsum_2(X, εs::AbstractVector, norm, w, show_progress)
    issorted(εs) || error("Sorted `ε` required for optimized version.")
    Cs = zeros(eltype(X), length(εs))
    N = length(X)
    factor = 2/((N-w)*(N-1-w))
    lower_ε_range = length(εs)÷2:-1:1
    upper_ε_range = (length(εs)÷2 + 1):length(εs)

    progress = ProgressMeter.Progress(N;
        desc = "Correlation sum: ", dt = 1.0, enabled = show_progress
    )

    Threads.@threads for i in 1:N
        dist = _fast_distance_2(X, i, w, norm)
        # First loop: mid-way ε until lower saturation point (C=0)
        @inbounds for k in lower_ε_range
            ε = εs[k]
            for i in 1:N
                Cs[k] += count(<(ε), dist)
            end
            Cs[k] == 0 && break
        end
        # Second loop: mid-way ε until higher saturation point (C=max)
        @inbounds for k in upper_ε_range
            ε = εs[k]
            for i in 1:N
                Cs[k] += count(<(ε), dist)
            end
            if Cs[k] ≈ 1/factor
                Cs[k:end] .= 1/factor
                break
            end
        end
        ProgressMeter.next!(progress)
    end
    return Cs .* factor
end

@inbounds function _fast_distance_2(X, i, w, norm)
    r = i+1+w:length(X)
    v = X[i]
    out = zeros(eltype(X), length(r))
    for j in eachindex(r)
        ξ = r[j]
        out[j] = norm(v, X[ξ])
    end
    return out
end

#######################################################################################
# Vector ε implementations: q ≠ 2
#######################################################################################
function correlationsum_q(X, εs::AbstractVector, q, norm, w, show_progress)
    issorted(εs) || error("Sorted `ε` required for optimized version.")
    E, T, N = length(εs), eltype(X), length(X)
    C_currents = [zeros(T, E) for _ in 1:Threads.nthreads()]
    Css = [zeros(T, E) for _ in 1:Threads.nthreads()]
    factor = (N-2w)*(N-2w-one(T))^(q-1)
    irange = 1+w:N-w
    progress = ProgressMeter.Progress(length(irange);
        desc="Correlation sum: ", dt=1, enabled=show_progress
    )

    Threads.@threads for i in irange
        C_current = C_currents[Threads.threadid()]
        Cs = Css[Threads.threadid()]
        fill!(C_current, 0)
        dist = _fast_distance_q(X, i, w, norm)
        for k in E:-1:1
            C_current[k] = count(<(εs[k]), dist)
        end
        Cs .+= C_current .^ (q-1)
        ProgressMeter.next!(progress)
    end
    Cs = sum(Css)
    return (Cs ./ factor) .^ (1/(q-1))
end

@inbounds function _fast_distance_q(X, i, w, norm)
    r1 = 1:i-w-1
    lr1 = length(r1)
    r2 = i+w+1:length(X)
    v = X[i]
    out = zeros(eltype(X), lr1+length(r2))
    i = 1
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
use `pointwise_correlationsums`.
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