export extremevaltheory_dims_persistences, extremevaltheory_dim
export extremevaltheory_local_dim_persistence
using Distances: euclidean
using Statistics: mean, quantile
import ProgressMeter

# The functions in this section are versions inspired from the code
# for MATLAB given in the following papers:

# Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent
# drivers of weather extremes: application to hot and cold days in North
# America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

# Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies
# of North Atlantic predictability and extremes. Scientific Reports, 7,
# 41278, doi: 10.1038/srep41278

# Süveges, Mária. 2007. Likelihood estimation of the extremal index.
# Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2

"""
    extremevaltheory_dim(X::StateSpaceSet, q::Real) → Δ

Convenience syntax that returns the mean of the local dimensions of
[`extremevaltheory_dims_persistences`](@ref), which approximates
a fractal dimension of `X` using extreme value theory and quantile `q`.
"""
function extremevaltheory_dim(X, q)
    Δloc, θloc = extremevaltheory_dims_persistences(X, q; compute_persistence = false)
    return mean(Δloc)
end


"""
    extremevaltheory_dims_persistences(x::AbstractStateSpaceSet, q::Real; kw...) -> Δloc, θ

Computation of the local dimensions `Δloc` and the extremal indices `θ` for each point in the
given set for a given quantile `q`. The extremal index can be interpreted as the
inverse of the persistence of the extremes around each point point.

The keyword `show_progress = true` displays a progress bar.
"""
function extremevaltheory_dims_persistences(X::AbstractStateSpaceSet, q::Real;
        show_progress = true, kw...
    )
    # The algorithm in the end of the day loops over points in `X`
    # and applies the local algorithm.
    # However, we write two different loop functions; one can
    # compute the distance matrix directly from the get go.
    # However, this is likely to not fit in memory even for a moderately high
    # amount of points in `X`. So, we make an alternative that goes row by row.
    N = length(X)
    Δloc = zeros(eltype(X), N)
    θloc = copy(Δloc)
    progress = ProgressMeter.Progress(
        N; desc = "Extreme value theory dim: ", enabled = show_progress
    )
    try
        # `vec(X)` gives the underlying `Vector{SVector}` for which `pairwise`
        # is incredibly optimized for!
        logdistances = -log.(pairwise(Euclidean(), vec(X)))
        _loop_over_matrix!(Δloc, θloc, progress, logdistances, q; kw...)
    catch
        @warn "Couldn't create $(N)×$(N) distance matrix; using slower algorithm..."
        _loop_and_compute_logdist!(Δloc, θloc, progress, X, q; kw...)
    end
    return Δloc, θloc
end

# TODO: Threading
function _loop_over_matrix!(Δloc, θloc, progress, logdistances, q; kw...)
    for (j, logdist) in enumerate(eachcol(logdistances))
        D, θ = extremevaltheory_local_dim_persistence(logdist, q)
        Δloc[j] = D
        θloc[j] = θ
        ProgressMeter.next!(progress)
    end
end
function _loop_and_compute_logdist!(Δloc, θloc, progress, X, q; kw...)
    logdist = copy(Δloc)
    for j in eachindex(X)
        map!(x -> -log(euclidean(x, X[j])), logdist, vec(X))
        D, θ = extremevaltheory_local_dim_persistence(logdist, q)
        Δloc[j] = D
        θloc[j] = θ
        ProgressMeter.next!(progress)
    end
end

function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, q::Real; compute_persistence = true
    )
    # Here `logdist` is already the -log(euclidean) distance of one point
    # to all other points in the set.
    # Extract the threshold corresponding to the quantile defined
    thresh = quantile(logdist, q)
    # Compute the extremal index
    if compute_persistence
        θ = extremal_index_sueveges(logdist, q, thresh)
    else
        θ = NaN
    end
    # Sort the time series and find all the Peaks Over Threshold (PoTs)
    PoTs = logdist[findall(≥(thresh), logdist)]
    # We need to filter, because one entry will have infinite value,
    # because one entry has 0 Euclidean distance in the set.
    filter!(isfinite, PoTs)
    exceedances = PoTs .- thresh
    # Extract the GPD parameters.
    # Assuming that the distribution is exponential, the
    # average of the PoTs is the unbiased estimator, which is just the mean
    # of the exceedances.
    # The local dimension is the reciprocal of the exceedances of the PoTs
    Δ = 1 ./ mean(exceedances)
    return Δ, θ
end

"""
    extremal_index_sueveges(logdist::AbstractVector, q, thresh)

Compute the extremal index θ through the Süveges formula.
"""
function extremal_index_sueveges(logdist::AbstractVector, q::Real, thresh::Real)
    # TODO: This is wrong for now
    q = 1 - q
    Li = findall(x -> x > thresh, logdist)
    Ti = diff(Li)
    Si = Ti .- 1
    Nc = length(findall(x->x>0, Si))
    N = length(Ti)
    θ = (sum(q.*Si)+N+Nc - sqrt( (sum(q.*Si) +N+Nc).^2 - 8*Nc*sum(q.*Si)) )./(2*sum(q.*Si))
    return θ
end

# convenience function
"""
    extremevaltheory_local_dim_persistence(X::StateSpaceSet, ζ, q::Real; kw...)

Return the local values `Δ, θ` of the fractal dimension and persistence of `X` around a
state space point `ζ`. `q` and `kw` are as in [`extremevaltheory_dims_persistences`](@ref).
"""
function extremevaltheory_local_dim_persistence(
        X::AbstractStateSpaceSet, ζ::AbstractVector, q::Real; kw...
    )
    logdist = map(x -> -log(euclidean(x, ζ)), X)
    Δ, θ = extremevaltheory_local_dim_persistence(logdist, q; kw...)
    return Δ, θ
end
