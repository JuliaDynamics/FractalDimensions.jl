export estimate_boxsizes

"""
    estimate_boxsizes(X::AbstractStateSpaceSet; kwargs...) → εs

Return `k` exponentially spaced values: `εs = base .^ range(lower + w, upper + z; length = k)`,
that are a good estimate for sizes ε that are used in calculating a fractal Dimension.
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
    min_d = eltype(eltype(X))(Inf)
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
    min_d = eltype(eltype(X))(Inf)
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