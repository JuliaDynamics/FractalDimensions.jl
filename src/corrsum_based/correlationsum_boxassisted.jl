import ProgressMeter
export boxed_correlationsum, boxassisted_correlation_dim
export estimate_r0_buenoorovio, prismdim_theiler, estimate_r0_theiler

################################################################################
# Boxed correlation sum docstrings
################################################################################
"""
    boxassisted_correlation_dim(X::AbstractStateSpaceSet; kwargs...)

Use the box-assisted optimizations of [BuenoOrovio2007](@cite)
to estimate the correlation dimension `Δ_C` of `X`.

This function does something extremely simple:
```julia
εs, Cs = boxed_correlationsum(X; kwargs...)
slopefit(log2.(εs), log2.(Cs))[1]
```

and hence see [`boxed_correlationsum`](@ref) for more information and available keywords.
"""
function boxassisted_correlation_dim(X::AbstractStateSpaceSet; kwargs...)
    εs, Cs = boxed_correlationsum(X; kwargs...)
    return slopefit(log2.(εs), log2.(Cs))[1]
end

"""
    boxed_correlationsum(X::AbstractStateSpaceSet, εs, r0 = maximum(εs); kwargs...) → Cs

Estimate the [`correlationsum`](@ref) for each size `ε ∈ εs` using an optimized algorithm
that first distributes data into boxes of size `r0`, and then computes the correlation sum
for each box and each neighboring box of each box.
This method is much faster than [`correlationsum`](@ref), **provided that** the
box size `r0` is significantly smaller than the attractor length.
Good choices for `r0` are [`estimate_r0_buenoorovio`](@ref) or
[`estimate_r0_theiler`](@ref).

    boxed_correlationsum(X::AbstractStateSpaceSet; kwargs...) → εs, Cs

In this method the minimum inter-point distance and [`estimate_r0_buenoorovio`](@ref)
of `X` are used to estimate suitable `εs` for the calculation, which are also returned.

## Keyword arguments

* `q = 2` : The order of the correlation sum.
* `P = 2` : The prism dimension.
* `w = 0` : The [Theiler window](@ref).
* `show_progress = false` : Whether to display a progress bar for the calculation.
* `norm = Euclidean()` : Distance norm.

## Description

`C_q(ε)` is calculated for every `ε ∈ εs` and each of the boxes to then be
summed up afterwards. The method of splitting the data into boxes was
implemented according to [Theiler1987](@cite). `w` is the [Theiler window](@ref).
`P` is the prism dimension. If `P` is unequal to the dimension of the data, only the
first `P` dimensions are considered for the box distribution (this is called the
prism-assisted version). By default `P` is 2, which is the version
suggested by [^Bueno2007]. Alternative for `P` is the [`prismdim_theiler`](@ref).
Note that only when `P = dimension(X)` the boxed version is guaranteed to be
exact to the original [`correlationsum`](@ref). For any other `P`, some
point pairs that should have been included may be skipped due to having smaller
distance in the remaining dimensions, but larger distance in the first `P` dimensions.
"""
function boxed_correlationsum(X::AbstractStateSpaceSet; P = 2, kwargs...)
    P = min(P, dimension(X))
    r0, ε0 = estimate_r0_buenoorovio(X, P)
    εs = 2.0 .^ range(log2(ε0), log2(r0); length = 16)
    Cs = boxed_correlationsum(X, εs, r0; P, kwargs...)
    return εs, Cs
end

boxed_correlationsum(X, e::Real, r0 = e; kwargs...) = boxed_correlationsum(X, [e], r0; kwargs...)[1]

function boxed_correlationsum(
        X, εs, r0 = maximum(εs); q = 2, P = 2, kwargs...
    )
    P ≤ dimension(X)   || error("Prism dimension has to be ≤ than `X` dimension.")
    issorted(εs)     || error("Sorted `εs` required for optimized version.")
    if r0 < maximum(εs)
        @warn("Box size `r0` has to be ≥ than `maximum(εs)`.")
        r0 = maximum(εs)
    end

    boxes, contents = data_boxing(X, r0, P)
    Cs = if q == 2
        boxed_correlationsum_2(boxes, contents, X, εs; kwargs...)
    else
        boxed_correlationsum_q(boxes, contents, X, εs, q; kwargs...)
    end
    return Cs
end

"""
    prismdim_theiler(X)

An algorithm to find the ideal choice of a prism dimension for
[`boxed_correlationsum`](@ref) using Theiler's original suggestion.
"""
function prismdim_theiler(X)
    D = dimension(X)
    N = length(X)
    if D > 0.75 * log2(N)
        return max(2, ceil(0.5 * log2(N)))
    else
        return D
    end
end

################################################################################
# Data boxing
################################################################################
"""
    data_boxing(X, r0 [, P]) → boxes, contents

Distribute `X` into boxes of size `r0`. Return box positions (in cartesian coordinates)
and the contents of each box as two separate vectors. Implemented according to
the paper by [Theiler1987](@cite) improving the algorithm by Grassberger and
Procaccia [Grassberger1983](@cite). If `P` is smaller than the dimension of the data,
only the first `P` dimensions are considered for the distribution into boxes.
If `P` is not given, all data dimensions are used.

The returned values are sorted (and this is crucial for optimal implementation
of the boxed correlation sum).

See also: [`boxed_correlationsum`](@ref).
"""
function data_boxing(X, r0, P)
    Xreduced = P == dimension(X) ? X : X[:, SVector{P, Int}(1:P)]
    data_boxing(Xreduced, r0)
end
function data_boxing(X, r0)
    mini = minima(X)

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point - mini)/r0), X)
    permutations = sortperm(bins; alg=QuickSort)

    boxes = unique(bins[permutations])
    contents = Vector{Vector{Int}}()
    sizehint!(contents, length(boxes))

    prior, prior_perm = 1, permutations[1]
    # distributes all permutation indices into boxes
    for (index, perm) in enumerate(permutations)
        if bins[perm] ≠ bins[prior_perm]
            push!(contents, permutations[prior:index-1])
            prior, prior_perm = index, perm
        end
    end
    push!(contents, permutations[prior:end])

    boxes, contents
end

################################################################################
# Concrete implementation, q = 2
################################################################################
using ChunkSplitters: chunks

"""
    boxed_correlationsum_2(boxes, contents, X, εs; w = 0)

For a vector of `boxes` and the indices of their `contents` inside of `X`,
calculate the classic correlationsum of a radius or multiple radii `εs`.
`w` is the Theiler window, for explanation see [`boxed_correlationsum`](@ref).
"""
function boxed_correlationsum_2(boxes, contents, X, εs; norm = Euclidean(), w = 0, show_progress = envprog())
    N = length(X)
    M = length(boxes)
    progress = ProgressMeter.Progress(M;
        desc = "Boxed correlation sum: ", enabled = show_progress
    )

    # parallelized via chunksplitting
    Css = [zeros(Int, length(εs)) for _ in 1:Threads.nthreads()]
    Threads.@threads for (threadid, chunk) in enumerate(chunks(1:M; n = Threads.nthreads()))
        local Cs = Css[threadid]
        for index in chunk
            indices_neighbors = find_neighborboxes_2(index, boxes, contents)
            indices_box = contents[index]
            inner_correlationsum_2!(Cs, indices_box, indices_neighbors, X, εs; w, norm)
            ProgressMeter.next!(progress)
        end
    end

    C = .+(Css...,)
    return C .* (2 / ((N - w) * (N - w - 1)))
end

"""
    find_neighborboxes_2(index, boxes, contents) → indices::Vector{Int}

Return all `indices` of the points in the boxes around the box that has `index`
in `boxes` (`boxes, contents` are the output of `data_boxing`).
"""
function find_neighborboxes_2(index, boxes, contents)
    indices = Int[]
    box = boxes[index]
    N_box = length(boxes)
    # Note here the search range: it explicitly uses the knowledge that q=2,
    # and hence knows that we only need to store indices _after_ the index
    # we are currently looping over, as future indices are included,
    # but no reason to scan for previous indices: distance between points
    # only needs to be computed once!
    for index2 in index:N_box
        # Since the boxes are in cartesian coordinates in integer space,
        # We know guaranteed the max distance in cartesian coordinates: it is ± 1.
        if evaluate(Chebyshev(), box, boxes[index2]) < 2
            append!(indices, contents[index2])
        end
    end
    indices
end

"""
    inner_correlationsum_2(idxs_box, idxs_neigh, X, εs; norm = Euclidean(), w = 0)

Calculate the classic correlation sum for values `X` inside a box,
(which contains indices `idxs_box`), while considering as neighbors only the indices
in `idxs_neigh`, which have been pre-calculated in `find_neighborboxes_2`.

Compute for all distances `ε ∈ εs` using `norm`. `w` is the Theiler window.

See [`boxed_correlationsum`](@ref)
"""
function inner_correlationsum_2!(Cs, idxs_box, idxs_neigh, X, εs; norm = Euclidean(), w = 0)
    Ny, Nε = length(idxs_neigh), length(εs)
    @inbounds for (i, index_in_X) in enumerate(idxs_box)
    	x = X[index_in_X]
        for j in i+1:Ny
            neigh_index_in_X = idxs_neigh[j]
            # Check for Theiler window.
            if abs(neigh_index_in_X - index_in_X) > w
                # Calculate distance.
		        dist = evaluate(norm, X[neigh_index_in_X], x)
		        for k in Nε:-1:1
		            if dist < εs[k]
		                Cs[k] += 1
		            else
		                break
		            end
		        end
		    end
        end
    end
    return Cs
end


################################################################################
# Concrete implementation, q != 2
################################################################################
# As the code is very similar to the one above, no docstirngs here.

function boxed_correlationsum_q(boxes, contents, X, εs, q; norm = Euclidean(), w = 0, show_progress = envprog())
    q ≤ 1 && @warn "This function is currently not specialized for q ≤ 1" *
    " and may show unexpected behaviour for these values."
    Css = [zeros(eltype(eltype(X)), length(εs)) for _ in 1:Threads.nthreads()]
    C_currents = [zeros(Int, length(εs)) for _ in 1:Threads.nthreads()]
    M = length(boxes)
    progress = ProgressMeter.Progress(M;
        desc = "Boxed correlation sum: ", enabled = show_progress
    )
    Threads.@threads for (threadid, chunk) in enumerate(chunks(1:M; n = Threads.nthreads()))
        local Cs = Css[threadid]
        local C_current = C_currents[threadid]
        for index in chunk
            indices_neighbors = find_neighborboxes_q(index, boxes, contents)
            indices_box = contents[index]
            inner_correlationsum_q!(Cs, C_current, indices_box, indices_neighbors, X, εs, q; w, norm)
            ProgressMeter.next!(progress)
        end
    end
    C = .+(Css...,)
    return clamp.(C, 0, Inf) .^ (1 / (q-1))
end

function find_neighborboxes_q(index, boxes, contents)
    indices = Int[]
    box = boxes[index]
    # Notice how here we cannot do the same optimization as in the q=2
    # case where we already start from i. :(
    for (index2, box2) in enumerate(boxes)
        if evaluate(Chebyshev(), box, box2) < 2
            append!(indices, contents[index2])
        end
    end
    indices
end

function inner_correlationsum_q!(
        Cs, C_current, idxs_box, idxs_neigh, data, εs, q::Real; norm = Euclidean(), w = 0
    )
    N, Nε = length(data), length(εs)
    for i in idxs_box
        # The normalisation is index dependent since the number of total points considered varies.
        normalisation = (N * (max(N - w, i) - min(w + 1, i))^(q - 1))
        C_current .= 0
        x = data[i]
        for j in idxs_neigh
            # Check that this index is not whithin the Theiler window
        	if abs(i - j) > w
		        dist = evaluate(norm, x, data[j])
		        for k in Nε:-1:1
		            if dist < εs[k]
		                C_current[k] += 1
		            else
		                break
		            end
		        end
		    end
        end
        Cs .+= C_current .^ (q-1) ./ normalisation
    end
    return Cs
end

#######################################################################################
# Good boxsize estimates for boxed correlation sum
#######################################################################################
using Statistics: mean
"""
    estimate_r0_theiler(X::AbstractStateSpaceSet) → r0, ε0
Estimate a reasonable size for boxing the data `X` before calculating the
[`boxed_correlationsum`](@ref) proposed by [Theiler1987](@cite).
Return the boxing size `r0` and minimum inter-point distance in `X`, `ε0`.

To do so the dimension is estimated by running the algorithm by Grassberger and
Procaccia [Grassberger1983](@cite) with `√N` points where `N` is the number of total
data points. Then the optimal boxsize ``r_0`` computes as
```math
r_0 = R (2/N)^{1/\\nu}
```
where ``R`` is the size of the chaotic attractor and ``\\nu`` is the estimated dimension.
"""
function estimate_r0_theiler(data)
    N = length(data)
    mini, maxi = minmaxima(data)
    R = mean(maxi .- mini)
    # Sample √N datapoints for a rough estimate of the dimension.
    data_sample = data[unique(rand(1:N, ceil(Int, sqrt(N))))] |> StateSpaceSet
    # Define radii for the rough dimension estimate
    min_d, _ = minimum_pairwise_distance(data)
    if min_d == 0
        @warn(
        "Minimum distance in the dataset is zero! Probably because of having data "*
        "with low resolution, or duplicate data points. Setting to `d₊/1000` for now.")
        min_d = R/(10^3)
    end
    lower = log10(min_d)
    εs = 10 .^ range(lower, stop = log10(R), length = 12)
    # Actually estimate the dimension.
    cm = correlationsum(data_sample, εs)
    ν = linear_region(log.(εs), log.(cm), tol = 0.5, warning = false)[2]
    # The combination yields the optimal box size
    r0 = R * (2/N)^(1/ν)
    return r0, min_d
end


using Random: shuffle!

"""
    estimate_r0_buenoorovio(X::AbstractStateSpaceSet, P = 2) → r0, ε0

Estimate a reasonable size for boxing `X`, proposed by
Bueno-Orovio and Pérez-García [BuenoOrovio2007](@cite), before calculating the correlation
dimension as presented by [Theiler1987](@cite).
Return the size `r0` and the minimum interpoint distance `ε0` in the data.

If instead of boxes, prisms
are chosen everything stays the same but `P` is the dimension of the prism.
To do so the dimension `ν` is estimated by running the algorithm by Grassberger
and Procaccia [Grassberger1983](@cite) with `√N` points where `N` is the number of
total data points.
An effective size `ℓ` of the attractor is calculated by boxing a small subset
of size `N/10` into boxes of sidelength `r_ℓ` and counting the number of filled
boxes `η_ℓ`.
```math
\\ell = r_\\ell \\eta_\\ell ^{1/\\nu}
```
The optimal number of filled boxes `η_opt` is calculated by minimising the number
of calculations.
```math
\\eta_\\textrm{opt} = N^{2/3}\\cdot \\frac{3^\\nu - 1}{3^P - 1}^{1/2}.
```
`P` is the dimension of the data or the number of edges on the prism that don't
span the whole dataset.

Then the optimal boxsize ``r_0`` computes as
```math
r_0 = \\ell / \\eta_\\textrm{opt}^{1/\\nu}.
```
"""
function estimate_r0_buenoorovio(X, P = 2)
    mini, maxi = minmaxima(X)
    N = length(X)
    R = mean(maxi .- mini)
    ν = zero(eltype(X))
    min_d, _ = minimum_pairwise_distance(X)
    if min_d == 0
        @warn(
        "Minimum distance in the dataset is zero! Probably because of having data "*
        "with low resolution, or duplicate data points. Setting to `d₊/1000` for now.")
        min_d = R/(10^3)
    end
    # Define logarithmic series of radii
    εs = 10.0 .^ range(log10(min_d), log10(R); length = 12)

    # Sample N/10 datapoints out of data for rough estimate of effective size.
    allidxs = collect(1:N)
    idxs = shuffle!(allidxs)[1:N÷10]
    sample1 = X[idxs]
    r_ℓ = R / 10
    η_ℓ = length(data_boxing(sample1, r_ℓ, P)[1])
    r0 = zero(eltype(X))

    # The possibility of a bad pick exists, if so, the calculation is repeated.
    while true
        # Sample √N datapoints for rough dimension estimate
        idxs = shuffle!(allidxs)[1:ceil(Int, sqrt(N))]
        sample2 = X[idxs]
        # Estimate ν from a sample using the Grassberger Procaccia algorithm.
        cm = correlationsum(sample2, εs)
        ν = linear_region(log.(εs), log.(cm); tol = 0.5, warning = false)[2]
        # Estimate the effictive size of the chaotic attractor.
        ℓ = r_ℓ * η_ℓ^(1/ν)
        # Calculate the optimal number of filled boxes according to Bueno-Orovio
        η_opt = N^(2/3) * ((3^ν - 1/2) / (3^P - 1))^(1/2)
        # The optimal box size is the effictive size divided by the box number
        # to the power of the inverse dimension.
        r0 = ℓ / η_opt^(1/ν)
        !isnan(r0) && break
    end

    if r0 < min_d
        warn("The calculated `r0` box size was smaller than the minimum interpoint " *
        "distance. Please provide `r0` manually. For now, setting `r0` to "*
        "average attractor length divided by `MathConstants.e^3`.")
        r0 = max(MathConstants.e*min_d, R/MathConstants.e^3)
    end
    return r0, min_d
end
