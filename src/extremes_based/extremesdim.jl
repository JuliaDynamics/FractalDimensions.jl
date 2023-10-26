export extremevaltheory_dims_persistences, extremevaltheory_dims, extremevaltheory_dim
export extremevaltheory_local_dim_persistence
export extremal_index_sueveges
export estimate_gpd_parameters
export extremevaltheory_gpdfit_pvalues
export BlockMaxima
export Exceedances
using Distances: euclidean
using Statistics: mean, quantile, var
import ProgressMeter
include("gpd.jl")
include("gev.jl")

"""
    BlockMaxima(blocksize::Int, p::Real)

This struct contains the parameters needed to perform an estimation
of the local dimensions through the block maxima method of extreme
value theory. This method divides the input data into blocks of length
`blocksize` and fits the maxima of each block to a Generalized Extreme
Value distribution. The parameter `p` is a number between 0 and 1 that
determines the p-quantile for the computation of the extremal index. 
"""
struct BlockMaxima
    blocksize::Int
    p::Real
end

"""
    Exceedances(p::Real, estimator::Symbol)

This struct contains the parameters needed to perform an estimation
of the local dimensions through the peaks over threshold method of extreme
value theory. This method sets a threshold and fits the exceedances to
Generalized Pareto Distribution. The parameter `p` is a number between
0 and 1 that determines the p-quantile for the threshold and computation
of the extremal index. The argument `estimator` is a symbol that can take
the values `:exp, :pwm, :mm`, as in [`estimate_gpd_parameters`](@ref).
"""
struct Exceedances
    p::Real
    estimator::Symbol
end


"""
    extremevaltheory_dim(X::StateSpaceSet, p::type; kwargs...) → Δ

Convenience syntax that returns the mean of the local dimensions of
[`extremevaltheory_dims_persistences`](@ref), which approximates
a fractal dimension of `X` using extreme value theory. The type `p`
tells the function which approach to use when Computing the dimension, see 
`BlockMaxima` and `Exceedances`.

See also [`extremevaltheory_gpdfit_pvalues`](@ref) for obtaining confidence on the results.
"""
function extremevaltheory_dim(X, p; kw...)
    Δloc, θloc = extremevaltheory_dims_persistences(X, p; compute_persistence = false, kw...)
    return mean(Δloc)
end

"""
    extremevaltheory_dims(X::StateSpaceSet, p::type; kwargs...) → Δloc

Convenience syntax that returns the local dimensions of
[`extremevaltheory_dims_persistences`](@ref).
"""
function extremevaltheory_dims(X, p; kw...)
    Δloc, θloc = extremevaltheory_dims_persistences(X, p; compute_persistence = false, kw...)
    return Δloc
end

"""
    extremevaltheory_dims_persistences(x::AbstractStateSpaceSet, p::type; kwargs...)

Return the local dimensions `Δloc` and the persistences `θloc` for each point in the
given set. The type `p` tells the function which approach to use when computing the
dimension, see `BlockMaxima` and `Exceedances`. The exceedances approach follows the
estimation done via extreme value theory [Lucarini2016](@cite).
The computation is parallelized to available threads (`Threads.nthreads()`).

See also [`extremevaltheory_gpdfit_pvalues`](@ref) for obtaining confidence on the results.

## Keyword arguments

- `show_progress = true`: displays a progress bar.
- `compute_persistence = true:` whether to aso compute local persistences
  `θloc` (also called extremal index). If `false`, `θloc` are `NaN`s.

## Description

For each state space point ``\\mathbf{x}_i`` in `X` we compute
``g_i = -\\log(||\\mathbf{x}_i - \\mathbf{x}_j|| ) \\; \\forall j = 1, \\ldots, N`` with
``||\\cdot||`` the Euclidean distance. Next, we choose an extreme quantile probability
``p`` (e.g., 0.99) for the distribution of ``g_i``. We compute ``g_p`` as the ``p``-th
quantile of ``g_i``. Then, we collect the exceedances of ``g_i``, defined as
``E_i = \\{ g_i - g_p: g_i \\ge g_p \\}``, i.e., all values of ``g_i`` larger or equal to
``g_p``, also shifted by ``g_p``. There are in total ``n = N(1-q)`` values in ``E_i``.
According to extreme value theory, in the limit ``N \\to \\infty`` the values ``E_i``
follow a two-parameter Generalized Pareto Distribution (GPD) with parameters
``\\sigma,\\xi`` (the third parameter ``\\mu`` of the GPD is zero due to the
positive-definite construction of ``E``). Within this extreme value theory approach,
the local dimension ``\\Delta^{(E)}_i`` assigned to state space point ``\\textbf{x}_i``
is given by the inverse of the ``\\sigma`` parameter of the
GPD fit to the data[^Lucarini2012], ``\\Delta^{(E)}_i = 1/\\sigma``.
``\\sigma`` is estimated according to the `estimator` keyword.

A more precise description of this process is given in the review paper [Datseris2023](@cite).
"""

function extremevaltheory_dims_persistences(X::AbstractStateSpaceSet, type;
    show_progress = envprog(), kw...
)
    # The algorithm in the end of the day loops over points in `X`
    # and applies the local algorithm.
    N = length(X)
    Δloc = zeros(eltype(X), N)
    θloc = copy(Δloc)
    progress = ProgressMeter.Progress(
        N; desc = "Extreme value theory dim: ", enabled = show_progress
    )
    logdists = [copy(Δloc) for _ in 1:Threads.nthreads()]
    Threads.@threads for j in eachindex(X)
        logdist = logdists[Threads.threadid()]
        logdist = map(x -> -log(euclidean(x, X[j])), vec(X))
        deleteat!(logdist, j)
        D, θ = extremevaltheory_local_dim_persistence(logdist, type; kw...)
        Δloc[j] = D
        θloc[j] = θ
        ProgressMeter.next!(progress)
    end
    return Δloc, θloc
end


"""
    extremevaltheory_dims_persistences(X::AbstractStateSpaceSet, p::Real;
    estimator = :exp, kw...
)

This function works without structs. Since a method is not specified, it
defaults to computing the local dimension through the exceedances method
with the :exp estimator, see [`estimate_gpd_parameters`](@ref).
"""
function extremevaltheory_dims_persistences(X::AbstractStateSpaceSet, p::Real;
    estimator = :exp, kw...
)
type = Exceedances(p, estimator)
extremevaltheory_dims_persistences(X, type; kw...)
end



function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::Exceedances; compute_persistence = true, estimator = :mm
    )
    estimator = type.estimator
    p = type.p
    σ, ξ, E, thresh = extremevaltheory_local_gpd_fit(logdist, p, estimator)
    # The local dimension is the reciprocal σ
    Δ = 1/σ
    # Lastly, obtain θ if asked for
    if compute_persistence
        θ = extremal_index_sueveges(logdist, p, thresh)
    else
        θ = NaN
    end
    return Δ, θ
end


function extremevaltheory_local_gpd_fit(logdist, p, estimator)
    # Here `logdist` is already the -log(euclidean) distance of one point
    # to all other points in the set.
    # Extract the threshold corresponding to the quantile defined
    thresh = quantile(logdist, p)
    # Filter to obtain Peaks Over Threshold (PoTs)
    # PoTs = logdist[findall(≥(thresh), logdist)]
    PoTs = filter(≥(thresh), logdist)
    # We need to filter, because one entry will have infinite value,
    # because one entry has 0 Euclidean distance in the set.
    filter!(isfinite, PoTs)
    # We re-use to PoTs vector do define the exceedances (save memory allocations)
    exceedances = (PoTs .-= thresh)
    # We need to ensure that the minimum of the exceedences is zero,
    # and sometimes it can be very close, but not exactly, zero
    minE = minimum(exceedances)
    if minE > 0
        exceedances .-= minE
    end
    # Extract the GPD parameters.
    σ, ξ = estimate_gpd_parameters(exceedances, estimator)
    return σ, ξ, exceedances, thresh
end

"""
    extremal_index_sueveges(y::AbstractVector, p)

Compute the extremal index θ of `y` through the Süveges formula for quantile probability `p`,
using the algorithm of [Sveges2007](@cite).
"""
function extremal_index_sueveges(y::AbstractVector, p::Real,
        # These arguments are given for performance optim; not part of public API
        thresh::Real = quantile(y, p)
    )
    p = 1 - p
    Li = findall(x -> x > thresh, y)
    Ti = diff(Li)
    Si = Ti .- 1
    Nc = length(findall(x->x>0, Si))
    N = length(Ti)
    θ = (sum(p.*Si)+N+Nc - sqrt( (sum(p.*Si) +N+Nc).^2 - 8*Nc*sum(p.*Si)) )./(2*sum(p.*Si))
    return θ
end

# convenience function
"""
    extremevaltheory_local_dim_persistence(X::StateSpaceSet, ζ, p::Real; kw...)

Return the local values `Δ, θ` of the fractal dimension and persistence of `X` around a
state space point `ζ`. `p` and `kw` are as in [`extremevaltheory_dims_persistences`](@ref).
"""
function extremevaltheory_local_dim_persistence(
        X::AbstractStateSpaceSet, ζ::AbstractVector, p::Real; kw...
    )
    logdist = map(x -> -log(euclidean(x, ζ)), X)
    Δ, θ = extremevaltheory_local_dim_persistence(logdist, p; kw...)
    return Δ, θ
end

function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::Exceedances; compute_persistence = true)
    p = type.p
    estimator = type.estimator

    σ, ξ, E, thresh = extremevaltheory_local_gpd_fit(logdist, p, estimator)
    # The local dimension is the reciprocal σ
    Δ = 1/σ
    # Lastly, obtain θ if asked for
    if compute_persistence
        θ = extremal_index_sueveges(logdist, p, thresh)
    else
        θ = NaN
    end
    return Δ, θ
end


"""
    extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::Blockmaxima; compute_persistence = true,
        estimator = :mm
    )

This function computes the local dimensions Δ and the extremal index θ for each observation
in the trajectory x. It uses the block maxima approach: divides the data in N/blocksize blocks
of length blocksize, where N is the number of data, and takes the maximum of those bloks as
samples of the maxima of the process. In order for this method to work correctly, both the
blocksize and the number of blocks must be high. Note that there are data points that are not
used by the algorithm. Since it is not always possible to express the number of input data
poins as N = blocksize * nblocks + 1. To reduce the number of unused data, chose an N equal or
 superior to blocksize * nblocks + 1. 
The extremal index can be interpreted as the inverse of the persistance of the extremes around
that point.
"""
function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::BlockMaxima; compute_persistence = true, estimator = :mm
    )
    p = type.p
    N = length(logdist)
    blocksize = type.blocksize
    nblocks = floor(Int, N/blocksize)
    newN = blocksize*nblocks
    firstindex = N - newN + 1
    Δ = zeros(newN)
    θ = zeros(newN)
    progress = ProgressMeter.Progress(
        N - firstindex; desc = "Extreme value theory dim: ", enabled = true
    )

    duplicatepoint = !isempty(findall(x -> x == Inf, logdist))
    if duplicatepoint
        error("Duplicated data point on the input")
    end
    if compute_persistence
        θ = extremal_index_sueveges(logdist, p)
    else
        θ = NaN
    end
    # Extract the maximum of each block
    maxvector = maximum(reshape(logdist[firstindex:N],(blocksize,nblocks)),dims= 1)
    σ = estimate_gev_scale(maxvector)
    Δ = 1 / σ
    return Δ, θ
end