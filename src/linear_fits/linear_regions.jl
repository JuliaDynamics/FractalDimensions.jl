export linear_region, linear_regions, estimate_boxsizes, linreg
export minimum_pairwise_distance

#####################################################################################
# Type
#####################################################################################
"""
    LargestLinearRegion <: SlopeFit
    LargestLinearRegion(; dxi::Int = 1, tol = 0.25)

Identify regions where the curve `y(x)` is linear, by scanning the
`x`-axis every `dxi` indices sequentially
(e.g. at `x[1]` to `x[5]`, `x[5]` to `x[10]`, `x[10]` to `x[15]` and so on if `dxi=5`).

If the slope (calculated via linear regression) of a region of width `dxi` is
approximatelly equal to that of the previous region,
within relative tolerance `tol` and absolute tolerance `0`,
then these two regions belong to the same linear region.

The largest such region is then used to estimate the slope via standard
linear regression of all points belonging to the largest linear region.
"Largest" here means the region that covers the more extent along the `x`-axis.

Use [`linear_regions`](@ref) if you wish to obtain the decomposition into linear regions.
"""
struct LargestLinearRegion
    dxi::Int
    tol::Float64
    method::Symbol
end
LargestLinearRegion(; dxi = 1, tol = 0.25) = LargestLinearRegion(dxi, tol, :sequential)

function _slopefit(x, y, llr::LargestLinearRegion, ci)
    lregions, tangents = linear_regions(x, y;
        dxi = llr.dxi, tol = llr.tol, method = llr.method
    )
    # Find biggest linear region:
    j = findmax(length, lregions)[2]
    xfit, yfit = x[lregions[j]], y[lregions[j]]
    return _slopefit(xfit, yfit, LinearRegression(), ci)
end

#####################################################################################
# Functions
#####################################################################################
"""
    LargestLinearRegion <: SlopeFit
    LargestLinearRegion(; dxi::Int = 1, tol = 0.25)

Identify regions where the curve `y(x)` is linear, by scanning the
`x`-axis every `dxi` indices sequentially
(e.g. at `x[1]` to `x[5]`, `x[5]` to `x[10]`, `x[10]` to `x[15]` and so on if `dxi=5`).

If the slope (calculated via standard linear regression) of a region of width `dxi` is
approximatelly equal to that of the previous region,
within tolerance `tol`,
then these two regions belong to the same linear region.

Return the indices of `x` that correspond to the linear regions, `lrs`,
and the correct `tangents` at each region
(obtained via a second linear regression at each accumulated region).
`lrs` is hence a vector of `UnitRange`s.
"""
function linear_regions(
        x::AbstractVector, y::AbstractVector;
        method = :sequential, dxi::Int = method == :overlap ? 3 : 1, tol = 0.25,
    )
    @assert length(x) == length(y)
    return if method == :overlap
        # TODO: Implement this...
        linear_regions_overlap(x, y, dxi, tol)
    elseif method == :sequential
        linear_regions_sequential(x, y, dxi, tol)
    end
end

function linear_regions_sequential(x, y, dxi, tol)
    maxit = length(x) ÷ dxi
    prevtang = slope(view(x, 1:max(dxi, 2)), view(y, 1:max(dxi, 2)))
    lrs = Int[1] #start of first linear region is always 1
    lastk = 1

    # Start loop over all partitions of `x` into `dxi` intervals:
    for k in 1:maxit-1
        r = k*dxi:(k+1)*dxi
        tang = slope(view(x, r), view(y, r))
        if isapprox(tang, prevtang, rtol=tol, atol = 0)
            # Tanget is similar with initial previous one (based on tolerance)
            continue
        else
            # Tangent is not similar enougn
            # Set the START of a new linear region
            # which is also the END of the previous linear region
            push!(lrs, k*dxi)
            lastk = k
            # Set new previous tangent (only if it was not the same as current)
            prevtang = tang
        end
    end
    # final linear region always ends here:
    push!(lrs, length(x))
    # Reformat into ranges
    lranges = [lrs[i]:lrs[i+1] for i in 1:length(lrs)-1]
    # create new tangents that do have linear regression weighted
    tangents = [slope(view(x, r), view(y, r)) for r in lranges]
    return lranges, tangents
end


# This function is deprecated in the new `SlopeFit` interface.
# It is kept here because it was used extensively in our review paper,
# due to its convenience of marking the largest region for the scatter markers.
"""
    linear_region(x, y; kwargs...) -> (region, slope)

Call [`linear_regions`](@ref) and identify and return the largest linear region
(a `UnitRange` of the indices of `x`) and its corresponding slope.

The keywords `dxi, tol` are propagated as-is to [`linear_regions`](@ref).
The keyword `ignore_saturation = true` ignores saturation that (sometimes) happens
at the start and end of the curve `y(x)`, where the curve flattens.
The keyword `sat = 0.01` decides what saturation is (while `abs(y[i]-y[i+1])<sat` we
are in a saturation regime).

The keyword `warning = true` prints a warning if the linear region is less than 1/3
of the available x-axis.
"""
function linear_region(x::AbstractVector, y::AbstractVector;
    dxi::Int = 1, tol::Real = 0.2, ignore_saturation = true, warning = true, sat = 0.01)

    isat = 0
    if ignore_saturation
        j = findfirst(i -> abs(y[i] - y[i-1]) > sat, length(y):-1:2)
        if !isnothing(j)
            i = (length(y):-1:2)[j]
            x, y = x[1:i], y[1:i]
        end
        k = findfirst(i -> abs(y[i+1] - y[i]) > sat, 1:length(y)-1)
        if !isnothing(k)
            x, y = x[k:end], y[k:end]
            isat = k-1
        end
    end

    lregions, tangents = linear_regions(x, y; dxi, tol)
    # Find biggest linear region:
    j = findmax(length, lregions)[2]
    if length(lregions[j]) ≤ length(x)÷3 - isat && warning
        @warn "Found linear region spans less than a 3rd of the available x-axis "*
              "and might imply inaccurate slope or insufficient data. "*
              "Recommended: plot `x` vs `y`."
    end
    return lregions[j] .+ isat, tangents[j]
end

#####################################################################################
# Autotomatic estimation for proper `ε` from a StateSpaceSet
#####################################################################################
"""
    estimate_boxsizes(X::AbstractStateSpaceSet; kwargs...) → εs

Return `k` exponentially spaced values: `εs = base .^ range(lower + w, upper + z; length = k)`,
that are a good estimate for sizes ε that are used in calculating a [Fractal Dimension](@ref).
It is strongly recommended to [`standardize`](@ref) input dataset before using this
function.

Let `d₋` be the minimum pair-wise distance in `X`, `d₋ = dminimum_pairwise_distance(X)`.
Let `d₊` be the average total length of `X`, `d₊ = mean(ma - mi)` with
`mi, ma = minmaxima(X)`.
Then `lower = log(base, d₋)` and `upper = log(base, d₊)`.
Because by default `w=1, z=-1`, the returned sizes are an order of mangitude
larger than the minimum distance, and an order of magnitude smaller than the maximum
distance.

## Keywords
* `w = 1, z = -1, k = 16` : as explained above.
* `base = MathConstants.e` : the base used in the `log` function.
* `warning = true`: Print some warnings for bad estimates.
* `autoexpand = true`: If the final estimated range does not cover at least 2 orders of
  magnitude, it is automatically expanded by setting `w -= we` and `z -= ze`.
  You can set different default values to the keywords `we = w, ze = z`.
"""
function estimate_boxsizes(
        X::AbstractStateSpaceSet;
        k::Int = 16, z = -1, w = 1, base = MathConstants.e,
        warning = true, autoexpand = true, ze = z, we = w
    )

    mi, ma = minmaxima(X)
    max_d = Statistics.mean(ma - mi)
    min_d, _ = minimum_pairwise_distance(X)
    if min_d == 0 && warning
        @warn(
        "Minimum distance in the dataset is zero! Probably because of having data "*
        "with low resolution, or duplicate data points. Setting to `d₊/base^4` for now.")
        min_d = max_d/(base^4)
    end

    lower = log(base, min_d)
    upper = log(base, max_d)

    if lower ≥ upper
        error("`lower ≥ upper`. There must be something fundamentally wrong with dataset.")
    elseif lower+w ≥ upper+z && warning
        @warn(
        "Automatic boxsize determination was inappropriate: `lower+w` was found ≥ than "*
        "`upper+z`. Returning `base .^ range(lower, upper; length = k)`. "*
        "Please adjust keywords or provide a bigger dataset.")
        εs = float(base) .^ range(lower, upper; length = k)
    elseif abs(upper+z - (lower+w)) < 2 && autoexpand
        if warning
            @warn(
            "Boxsize limits do not differ by at least 2 orders of magnitude. "*
            "Setting `w-=$(we)` and `z+=$(ze)`, please adjust keywords `w, z` otherwise.")
        end
        εs = float(base) .^ range(lower+w-we, upper+z-ze; length = k)
    else
        εs = float(base) .^ range(lower+w, upper+z; length = k)
    end
    return εs
end


import Neighborhood
"""
    minimum_pairwise_distance(X::StateSpaceSet, kdtree = dimension(X) < 10, metric = Euclidean())

Return `min_d, min_pair`: the minimum pairwise distance
of all points in the dataset, and the corresponding point pair.
The third argument is a switch of whether to use KDTrees or a brute force search.
"""
function minimum_pairwise_distance(
        X::AbstractStateSpaceSet, kdtree = dimension(X) < 10, metric = Euclidean()
    )
    if kdtree
        _mpd_kdtree(X, metric)
    else
        _mpd_brute(X, metric)
    end
end

function _mpd_kdtree(X::AbstractStateSpaceSet, metric = Euclidean())
    tree = Neighborhood.KDTree(X, metric)
    min_d = eltype(X[1])(Inf)
    min_pair = (0, 0)
    theiler = Neighborhood.Theiler(0)
    for i in eachindex(X)
        inds, dists = Neighborhood.knn(tree, X[i], 1, theiler(i); sortds=false)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (i, ind)
        end
    end
    return min_d, min_pair
end

function _mpd_brute(X::AbstractStateSpaceSet, metric = Euclidean())
    min_d = eltype(X)(Inf)
    min_pair = (0, 0)
    @inbounds for i in eachindex(X)
        for j in (i+1):length(X)
            dist = metric(X[i], X[j])
            if dist < min_d
                min_d = dist
                min_pair = (i, j)
            end
        end
    end
    return min_d, min_pair
end