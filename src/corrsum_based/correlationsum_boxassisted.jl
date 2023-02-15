import ProgressMeter
export boxed_correlationsum, boxassisted_correlation_dim
export estimate_r0_buenoorovio, autoprismdim, estimate_r0_theiler

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

################################################################################
# Boxed Correlation sum (we distribute data to boxes beforehand)
################################################################################
"""
    boxed_correlationsum(X::AbstractStateSpaceSet, εs, r0 = maximum(εs); kwargs...) → Cs

Estimate the box assisted q-order correlation sum `Cs` out of a
dataset `X` for each radius in `εs`, by splitting the data into boxes of size `r0`
beforehand. This method is much faster than [`correlationsum`](@ref), **provided that** the
box size `r0` is significantly smaller than then the attractor length.
Good choices for `r0` are [`estimate_r0_buenoorovio`](@ref) and
[`estimate_r0_theiler`](@ref).

See [`correlationsum`](@ref) for the definition of the correlation sum.

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
    εs = 2.0 .^ range(log2(ε0), log2(r0); length = 16)
    Cs = boxed_correlationsum(X, εs, r0; P, kwargs...)
    return εs, Cs
end

function boxed_correlationsum(
        X, εs, r0 = maximum(εs);
        q = 2, P = autoprismdim(X), w = 0,
        show_progress = false,
    )
    @assert P ≤ size(X, 2) "Prism dimension has to be ≤ than X dimension."
    boxes, contents = data_boxing(X, r0, P)
    Cs = if q == 2
        boxed_correlationsum_2(boxes, contents, X, εs; w, show_progress)
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

"""
    data_boxing(X_i, r0, P) → boxes, contents

Distribute `X` into boxes of size `r0`. Return box positions
and the contents of each box as two separate vectors. Implemented according to
the paper by Theiler[^Theiler1987] improving the algorithm by Grassberger and
Procaccia[^Grassberger1983]. If `P` is smaller than the dimension of the data,
only the first `P` dimensions are considered for the distribution into boxes.

See also: [`boxed_correlationsum`](@ref).

[^Theiler1987]:
    Theiler, [Efficient algorithm for estimating the correlation dimension from a set
    of discrete points. Physical Review A, 36](https://doi.org/10.1103/PhysRevA.36.4456)

[^Grassberger1983]:
    Grassberger and Proccacia, [Characterization of strange attractors, PRL 50 (1983)
    ](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.346)
"""
function data_boxing(X, r0, P)
    @assert P ≤ size(X, 2) "Prism dimension has to be ≤ than data dimension."
    mini = minima(X)[1:P]

    # Map each datapoint to its bin edge and sort the resulting list:
    bins = map(point -> floor.(Int, (point[1:P] - mini)/r0), X)
    permutations = sortperm(bins, alg=QuickSort)

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

    StateSpaceSet(boxes), contents
end

"""
    boxed_correlationsum_2(boxes, contents, X, εs; w = 0)
For a vector of `boxes` and the indices of their `contents` inside of `X`,
calculate the classic correlationsum of a radius or multiple radii `εs`.
`w` is the Theiler window, for explanation see [`boxed_correlationsum`](@ref).
"""
function boxed_correlationsum_2(boxes, contents, X, εs; w = 0, show_progress = false)
    Cs = zeros(eltype(X), length(εs))
    N = length(X)
    M = length(boxes)
    if show_progress
        progress = ProgressMeter.Progress(M; desc = "Boxed correlation sum: ", dt = 1.0)
    end
    for index in 1:M
        indices_neighbors = find_neighborboxes_2(index, boxes, contents)
        indices_box = contents[index]
        Cs .+= inner_correlationsum_2(indices_box, indices_neighbors, X, εs; w)
        show_progress && ProgressMeter.update!(progress, index)
    end
    Cs .* (2 / ((N - w) * (N - w - 1)))
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
