import ProgressMeter
export boxed_correlationsum, boxassisted_correlation_dim
export estimate_r0_buenoorovio, autoprismdim, estimate_r0_theiler

################################################################################
# Boxed correlation sum main API functions
################################################################################
"""
    boxassisted_correlation_dim(X::AbstractStateSpaceSet; kwargs...)

Use the box-assisted optimizations of [^Bueno2007]
to estimate the correlation dimension `Δ_C` of `X`.

This function does something extremely simple:
```julia
εs, Cs = boxed_correlationsum(X; kwargs...)
return linear_region(log2.(Cs), log2.(εs))[2]
```

and hence see [`boxed_correlationsum`](@ref) for more information and available keywords.

[^Bueno2007]:
    Bueno-Orovio and Pérez-García, [Enhanced box and prism assisted algorithms for
    computing the correlation dimension. Chaos Solitons & Fractrals, 34(5)
    ](https://doi.org/10.1016/j.chaos.2006.03.043)
"""
function boxassisted_correlation_dim(X::AbstractStateSpaceSet; kwargs...)
    εs, Cs = boxed_correlationsum(X; kwargs...)
    return linear_region(log2.(εs), log2.(Cs))[2]
end

"""
    boxed_correlationsum(X::AbstractStateSpaceSet, εs, r0 = maximum(εs); kwargs...) → Cs

Estimate the box assisted q-order correlation sum `Cs` of `X` for each radius in `εs`,
by splitting the data into boxes of size `r0`
beforehand. This method is much faster than [`correlationsum`](@ref), **provided that** the
box size `r0` is significantly smaller than the attractor length.
Good choices for `r0` are [`estimate_r0_buenoorovio`](@ref) and
[`estimate_r0_theiler`](@ref).

See [`correlationsum`](@ref) for the definition of the correlation sum.

Initial implementation of the algorithm was according to [^Theiler1987].
However, current implementation has been re-written and utilizes histogram handling
from ComplexityMeasures.jl and nearest neighbor searches in discrete spaces from Agents.jl.

    boxed_correlationsum(X::AbstractStateSpaceSet; kwargs...) → εs, Cs

In this method the minimum inter-point distance and [`estimate_r0_buenoorovio`](@ref)
of `X` are used to estimate good `εs` for the calculation, which are also returned.

## Keyword arguments

* `q = 2` : The order of the correlation sum.
* `P = autoprismdim(X)` : The prism dimension.
* `w = 0` : The [Theiler window](@ref).
* `show_progress = false` : Whether to display a progress bar for the calculation.

## Description

`C_q(ε)` is calculated for every `ε ∈ εs` and each of the boxes to then be
summed up afterwards. The method of splitting the data into boxes was
implemented according to Theiler[^Theiler1987]. `w` is the [Theiler window](@ref).
`P` is the prism dimension. If `P` is unequal to the dimension of the data, only the
first `P` dimensions are considered for the box distribution (this is called the
prism-assisted version). By default `P` is choosen automatically.

The function is explicitly optimized for `q = 2` but becomes quite slow for `q ≠ 2`.

See [`correlationsum`](@ref) for the definition of `C_q`.

[^Theiler1987]:
    Theiler, [Efficient algorithm for estimating the correlation dimension from a set
    of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)
"""
function boxed_correlationsum(X; P = 2, kwargs...)
    r0, ε0 = estimate_r0_buenoorovio(X, P)
    εs = MathConstants.e .^ range(log(ε0), log(r0); length = 16)
    Cs = boxed_correlationsum(X, εs, r0; P, kwargs...)
    return εs, Cs
end

function boxed_correlationsum(
        X, εs, r0 = maximum(εs);
        q = 2, P = autoprismdim(X), w = 0,
        show_progress = false,
    )
    boxes_to_contents, hist_size = data_boxing(X, r0, P)

    Cs = if q == 2
        boxed_correlationsum_2(boxes_to_contents, offsets, X, εs; w, show_progress)
    else
        boxed_correlationsum_q(boxes, contents, X, εs, q; w, show_progress)
    end
    return Cs
end

"""
    autoprismdim(X, version = :bueno)

An algorithm to find the ideal choice of a prism dimension for
[`boxed_correlationsum`](@ref). `version = :bueno` uses `P=2`, while
`version = :theiler` uses Theiler's original suggestion.
"""
function autoprismdim(X, version = :bueno)
    D = dimension(X)
    N = length(X)
    if version == :bueno
        return min(D, 2)
    elseif version == :theiler
        if D > 0.75 * log2(N)
            return max(2, ceil(0.5 * log2(N)))
        else
            return D
        end
    else
        error("Unknown method.")
    end
end

################################################################################
# Data boxing and iterating over boxes for neighboring points
################################################################################
"""
    data_boxing(X::StateSpaceSet, r0 [, P::Int]) → boxes_to_contents, hist_size

Distribute `X` into boxes of size `r0`. Return a dictionary, mapping tuples
(cartesian indices of the histogram boxes) into point indices of `X` in the boxes
and the (maximum) size of the boxing scheme (i.e., max dimensions of the histogram).
If `P` is given, only the first `P` dimensions of `X` are considered for constructing
the boxes and distributing the points into them.

Used in: [`boxed_correlationsum`](@ref).

[^Theiler1987]:
    Theiler, [Efficient algorithm for estimating the correlation dimension from a set
    of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)
"""
function data_boxing(X, r0::AbstractFloat, P::Int = autoprismdim(X))
    P ≤ dimension(X) || error("Prism dimension has to be ≤ than data dimension.")
    Xreduced = P == dimension(X) ? X : X[:, SVector{P, Int}(1:P)]
    encoding = RectangularBinEncoding(RectangularBinning(r0, false), Xreduced)
    return _data_boxing(Xreduced, encoding), encoding.histsize
end

function _data_boxing(X, encoding)
    # Output is a dictionary mapping cartesian indices to vector of data point indices
    # in said cartesian index bin
    boxes_to_contents = Dict{NTuple{dimension(X), Int}, Vector{Int}}()
    for (j, x) in enumerate(X)
        i = encode(encoding, x) # linear index of box in histogram
        ci = Tuple(encoding.ci[i]) # cartesian index of box in histogram
        if !haskey(boxes_to_contents, ci)
            boxes_to_contents[ci] = Int[]
        end
        push!(boxes_to_contents[ci], j)
    end
    return boxes_to_contents
end

################################################################################
# Correlation sum computation code
################################################################################
"""
    boxed_correlationsum_2(boxes, contents, X, εs; w = 0)
For a vector of `boxes` and the indices of their `contents` inside of `X`,
calculate the classic correlationsum of a radius or multiple radii `εs`.
`w` is the Theiler window, for explanation see [`boxed_correlationsum`](@ref).
"""
function boxed_correlationsum_2(boxes_to_contents, hist_size, X, εs;
        w = 0, show_progress = false
    )
    # Note that the box index is also its cartesian index in the histogram
    Cs = zeros(eltype(X), length(εs))
    progress = ProgressMeter.Progress(length(boxes_to_contents);
        desc = "Boxed correlation sum: ", dt = 1.0, enabled = show_progress
    )
    # Skip predicate (theiler window): if `true` skip current point index.
    # Notice that the predicate depends on `q`, because if `q = 2` we can safely
    # skip all points with index `j` less or equal to `i`
    skip = if q == 2
        (i, j) -> j ≤ w + i
    else
        (i, j) -> abs(i - j) ≤ w
    end
    offsets = chebyshev_1_offsets(length(hist_size))
    # We iterate over all existing boxes; for each box, we iterate over
    # all points in the box and all neighboring boxes (offsets added to box coordinate)
    # TODO: Threading
    for box_index in keys(boxes_to_contents)
        # This is a special iterator; for the given box, it iterates over
        # all points in this box and in the neighboring boxes (offsets added)
        indices_in_box = boxes_to_contents[box_index]
        nearby_indices_iter = PointsInBoxesIterator(boxes_to_contents, hist_size, offsets, box_index)
        add_to_corrsum!(Cs, εs, X, indices_in_box, nearby_indices_iter, skip)
        ProgressMeter.next!(progress)
    end
    # Normalize accordingly
    if q == 2
        return Cs .* (2 / ((N - w) * (N - w - 1)))
    else
        return clamp.((Cs ./ ((N - 2w) * (N - 2w - 1) ^ (q-1))), 0, Inf) .^ (1 / (q-1))
    end
end

function chebyshev_1_offsets(P)
    # Offsets, which are required by the nearest neighbor algorithm, are constants
    # and depend only on `P` (histogram dimension = prism). We compute them here:
    # 1 is the radius (distance) of boxes in Chebyshev metric that we need to include
    hypercube = Iterators.product(repeat([-1:1], P)...)
    offsets = vec([β for β ∈ hypercube])
    # make it guaranteed so that (0s...) offset is first in order
    z = ntuple(i -> 0, Val(P))
    filter!(x -> x ≠ z, offsets)
    pushfirst!(offsets, z)
    return offsets
end

@inbounds function add_to_corrsum!(Cs, εs, X, indices_in_box, nearby_indices_iter, skip)
    for j in nearby_indices_iter
        # It is crucial that the second loop are the origin box indices because the
        # loop over the custom iterator does not reset!
        for i in indices_in_box
            skip(i, j) && continue
            dist = norm(X[i], X[j])
            for k in length(εs):-1:1
                if dist < εs[k]
                    Cs[k] += 1
                else
                    break
                end
            end
        end
    end
    return Cs
end

################################################################################
# Extremely optimized custom iterator for nearby boxes
################################################################################

# Notice that from creation we know the first box index, and its nubmer
# in the box offseting sequence is by construction 1

# For optinal performance and design we need a different method of starting the iteration
# and another one that continues iteration. Second case uses the explicitly
# known knowledge of `box_number` being a valid position index.

struct PointsInBoxesIterator{D}
    boxes_to_contents::Dict{NTuple{D, Int}, Vector{Int}}
    hist_size::NTuple{D,Int}          # size of histogram
    hist_axes::NTuple{D, Base.OneTo{Int}} # axes, for using in check bounds
    origin::NTuple{D,Int}             # origin (box we started from)
    origin_indices::Vector{Int}       # point indices in box we started from
    offsets::Vector{NTuple{D,Int}}    # Result of `offsets_within_radius` pretty much
    L::Int                            # length of `offsets`
end

function PointsInBoxesIterator(
        boxes_to_contents, hist_size, offsets, origin::CartesianIndex{D}
    ) where {D}
    L = length(offsets)
    hist_axes = map(n -> Base.OneTo(n), hist_size)
    origin_indices = boxes_to_contents[origin]
    return GridSpaceIdIterator{D}(
        boxes_to_contents, hist_Size, hist_axes, origin, origin_indices, offsets, L
    )
end

@inbounds function Base.iterate(
        iter::PointsInBoxesIterator, state = (1, 1, iter.origin_indices)
    )
    offsets, L, origin = getproperty.(Ref(iter), (:offsets, :L, :origin))
    box_number, inner_i, idxs_in_box = state
    X = length(idxs_in_box)
    if inner_i > X
        # we have exhausted IDs in current position, so we reset and go to next
        box_number += 1
        # Stop iteration if `box_index` exceeded the amount of positions
        box_number > L && return nothing
        inner_i = 1
        box_index = offsets[box_number] .+ origin
        # Of course, we need to check if we have valid index
        while invalid_access(box_index, iter.boxes_to_contents, iter.hist_axes)
            box_number += 1
            box_number > L && return nothing
            box_index = offsets[box_number] .+ origin
        end
        idxs_in_box = iter.boxes_to_contents[box_index]
    end
    # We reached the next valid position and non-empty position
    id = idxs_in_box[inner_i]
    return (id, (box_number, inner_i + 1, idxs_in_box))
end


################################################################################
# Old stuff
################################################################################



"""
    inner_correlationsum_2(indices_X, indices_Y, X, εs; norm = Euclidean(), w = 0)
Calculates the classic correlation sum for values `X` inside a box, considering
`Y` consisting of all values in that box and the ones in neighbouring boxes for
all distances `ε ∈ εs` calculated by `norm`. To obtain the position of the
values in the original time series `X`, they are passed as `indices_X` and
`indices_Y`.

`w` is the Theiler window. Each index to the original array is checked for the
distance of the compared index. If this absolute value is not higher than `w`
its element is not used in the calculation of the correlationsum.

See also: [`correlationsum`](@ref)
"""
function inner_correlationsum_2(indices_X, indices_Y, X, εs; norm = Euclidean(), w = 0)
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs, Ny, Nε = zeros(length(εs)), length(indices_Y), length(εs)
    for (i, index_X) in enumerate(indices_X)
    	x = X[index_X]
        for j in i+1:Ny
            index_Y = indices_Y[j]
            # Check for Theiler window.
            if abs(index_Y - index_X) > w
                # Calculate distance.
		        dist = evaluate(norm, X[index_Y], x)
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


"""
    find_neighborboxes_2(index, boxes, contents) → indices
For an `index` into `boxes` all neighbouring boxes beginning from the current
one are searched. If the found box is indeed a neighbour, the `contents` of
that box are added to `indices`.
"""
function find_neighborboxes_2(index, boxes, contents)
    indices = Int[]
    box = boxes[index]
    N_box = length(boxes)
    for index2 in index:N_box
        if evaluate(Chebyshev(), box, boxes[index2]) < 2
            append!(indices, contents[index2])
        end
    end
    indices
end





"""
    boxed_correlationsum_q(boxes, contents, X, εs, q; w = 0)
For a vector of `boxes` and the indices of their `contents` inside of `X`,
calculate the `q`-order correlationsum of a radius or radii `εs`.
`w` is the Theiler window, for explanation see [`boxed_correlationsum`](@ref).
"""
function boxed_correlationsum_q(boxes, contents, X, εs, q; w = 0, show_progress = false)
    q <= 1 && @warn "This function is currently not specialized for q <= 1" *
    " and may show unexpected behaviour for these values."
    Cs = zeros(eltype(X), length(εs))
    N = length(X)
    M = length(boxes)
    if show_progress
        progress = ProgressMeter.Progress(M; desc = "Boxed correlation sum: ", dt = 1.0)
    end
    for index in 1:M
        indices_neighbors = find_neighborboxes_q(index, boxes, contents)
        indices_box = contents[index]
        Cs .+= inner_correlationsum_q(indices_box, indices_neighbors, X, εs, q; w)
        show_progress && ProgressMeter.update!(progress, index)
    end
    clamp.((Cs ./ ((N - 2w) * (N - 2w - 1) ^ (q-1))), 0, Inf) .^ (1 / (q-1))
end


"""
    inner_correlationsum_q(indices_X, indices_Y, data, εs, q::Real; norm, w)
Calculates the `q`-order correlation sum for values `X` inside a box,
considering `Y` consisting of all values in that box and the ones in
neighbouring boxes for all distances `ε ∈ εs` calculated by `norm`. To obtain
the position of the values in the original time series `data`, they are passed
as `indices_X` and `indices_Y`.

`w` is the Theiler window. The first and last `w` points of this data set are
not used by themselves to calculate the correlationsum.

See also: [`correlationsum`](@ref)
"""
function inner_correlationsum_q(
        indices_X, indices_Y, data, εs, q::Real; norm = Euclidean(), w = 0
    )
    @assert issorted(εs) "Sorted εs required for optimized version."
    Cs = zeros(length(εs))
    N, Ny, Nε = length(data), length(indices_Y), length(εs)
    for i in indices_X
        # Check that this index is not within Theiler window of the boundary
        # This step is neccessary for easy normalisation.
        (i < w + 1 || i > N - w) && continue
        C_current = zeros(Nε)
        x = data[i]
        for j in indices_Y
            # Check that this index is not whithin the Theiler window
        	if abs(i - j) > w
                # Calculate the distance for the correlationsum
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
        Cs .+= C_current .^ (q-1)
    end
    return Cs
end



"""
    find_neighborboxes_q(index, boxes, contents) → indices
For an `index` into `boxes` all neighbouring boxes are searched. If the found
box is indeed a neighbour, the `contents` of that box are added to `indices`.
"""
function find_neighborboxes_q(index, boxes, contents)
    indices = Int[]
    box = boxes[index]
    for (index2, box2) in enumerate(boxes)
        if evaluate(Chebyshev(), box, box2) < 2
            append!(indices, contents[index2])
        end
    end
    indices
end


#######################################################################################
# Good boxsize estimates for boxed correlation sum
#######################################################################################
using Statistics: mean
"""
    estimate_r0_theiler(X::AbstractStateSpaceSet) → r0, ε0
Estimate a reasonable size for boxing the data `X` before calculating the
[`boxed_correlationsum`](@ref) proposed by Theiler[^Theiler1987].
Return the boxing size `r0` and minimum inter-point distance in `X`, `ε0`.

To do so the dimension is estimated by running the algorithm by Grassberger and
Procaccia[^Grassberger1983] with `√N` points where `N` is the number of total
data points. Then the optimal boxsize ``r_0`` computes as
```math
r_0 = R (2/N)^{1/\\nu}
```
where ``R`` is the size of the chaotic attractor and ``\\nu`` is the estimated dimension.

[^Theiler1987]:
    Theiler, [Efficient algorithm for estimating the correlation dimension from a set
    of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]:
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)
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

"""
    estimate_r0_buenoorovio(X::AbstractStateSpaceSet, P = autoprismdim(X)) → r0, ε0

Estimate a reasonable size for boxing `X`, proposed by
Bueno-Orovio and Pérez-García[^Bueno2007], before calculating the correlation
dimension as presented by Theiler[^Theiler1983].
Return the size `r0` and the minimum interpoint distance `ε0` in the data.

If instead of boxes, prisms
are chosen everything stays the same but `P` is the dimension of the prism.
To do so the dimension `ν` is estimated by running the algorithm by Grassberger
and Procaccia[^Grassberger1983] with `√N` points where `N` is the number of
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

[^Bueno2007]:
    Bueno-Orovio and Pérez-García, [Enhanced box and prism assisted algorithms for
    computing the correlation dimension. Chaos Solitons & Fractrals, 34(5)
    ](https://doi.org/10.1016/j.chaos.2006.03.043)

[^Theiler1987]:
    Theiler, [Efficient algorithm for estimating the correlation dimension from a set
    of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]:
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function estimate_r0_buenoorovio(X, P = autoprismdim(X))
    mini, maxi = minmaxima(X)
    N = length(X)
    R = mean(maxi .- mini)
    # The possibility of a bad pick exists, if so, the calculation is repeated.
    ν = zero(eltype(X))
    min_d, _ = minimum_pairwise_distance(X)
    if min_d == 0
        @warn(
        "Minimum distance in the dataset is zero! Probably because of having data "*
        "with low resolution, or duplicate data points. Setting to `d₊/1000` for now.")
        min_d = R/(10^3)
    end

    # Sample N/10 datapoints out of data for rough estimate of effective size.
    sample1 = X[unique(rand(1:N, N÷10))] |> StateSpaceSet
    r_ℓ = R / 10
    η_ℓ = length(data_boxing(sample1, r_ℓ, P)[1])
    r0 = zero(eltype(X))
    while true
        # Sample √N datapoints for rough dimension estimate
        sample2 = X[unique(rand(1:N, ceil(Int, sqrt(N))))] |> StateSpaceSet
        # Define logarithmic series of radii.
        εs = 10.0 .^ range(log10(min_d), log10(R); length = 16)
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
        "average attractor length divided by 16")
        r0 = max(4min_d, R/16)
    end
    return r0, min_d
end
