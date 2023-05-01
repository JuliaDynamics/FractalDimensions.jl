export extremevaltheory_dims_persistences, extremevaltheory_dim
export extremevaltheory_local_dim_persistence
export extremal_index_sueveges
using Distances: euclidean
using Statistics: mean, quantile, var
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
    extremevaltheory_dim(X::StateSpaceSet, p::Real; kwargs...) → Δ

Convenience syntax that returns the mean of the local dimensions of
[`extremevaltheory_dims_persistences`](@ref), which approximates
a fractal dimension of `X` using extreme value theory and quantile probability `p`.
"""
function extremevaltheory_dim(X, p)
    Δloc, θloc = extremevaltheory_dims_persistences(X, p; compute_persistence = false)
    return mean(Δloc)
end


"""
    extremevaltheory_dims_persistences(x::AbstractStateSpaceSet, p::Real; kwargs)

Return the local dimensions `Δloc` and the persistences `θloc` for each point in the
given set for quantile probability `p`, according to the estimation done via extreme value
theory [^Lucarini2016] [^Faranda2011].
The computation is parallelized to available threads (`Threads.nthreads()`).

## Keyword arguments

- `show_progress = true`: displays a progress bar.
- `estimator = :mm`: how to estimate the `σ` parameter of the
  Generalized Pareto Distribution. The local fractal dimension is `1/σ`.
  The possible values are: `:exp, :mm`, as in [`estimate_gpd_parameters`](@ref).
- `compute_persistence = true:` whether to aso compute local persistences
  `θloc` (also called extremal index). If `false`, `θloc` are `NaN`s.

## Description

For each state space point ``\\mathbf{x}_i`` in `X` we compute
``g_j = -\\log(||\\mathbf{x}_i - \\mathbf{x}_j|| ) \\; \\forall j = 1, \\ldots, N``
``||\\cdot||`` the Euclidean distance. Next, we choose an extreme quantile probability
``p`` (e.g., 0.99) for the distribution of ``g_j``. We compute ``g_p`` as the ``p``-th
quantile of ``g_j``. Then, we collect the exceedances of ``g_j``, defined as
``E = \\{ g_j - g_q: g_j \\ge g_q \\}``, i.e., all values of ``g_j`` larger or equal to
``g_q``, also shifted by ``g_q``. There are in total ``n = N(1-q)`` values in ``E``.
According to extreme value theory, in the limit ``N \\to \\infty`` the values ``E``
follow a two-parameter Generalized Pareto Distribution (GPD) with parameters
``\\sigma,\\xi`` (the third parameter ``\\mu`` of the GPD is zero due to the
positive-definite construction of ``E``). Within this extreme value theory approach,
the local dimension ``\\Delta^{(E)}_i`` assigned to state space point ``\\textbf{x}_i``
is given by the inverse of the ``\\sigma`` parameter of the
GPD fit to the data[^Faranda2011], ``\\Delta^{(E)}_i = /\\sigma``.
``\\sigma`` is estimated according to the `estimator` keyword.

[^Lucarini2016]:
    Lucarini et al., [Extremes and Recurrence in Dynamical Systems](
    https://www.wiley.com/en-gb/Extremes+and+Recurrence+in+Dynamical+Systems-p-9781118632192)

[^Faranda2011]:
    Faranda et al., [J. Stat. Phys., 605 145(5):1156-1180.](
    https://link.springer.com/article/10.1007/s10955-011-0234-7)
"""
function extremevaltheory_dims_persistences(X::AbstractStateSpaceSet, p::Real;
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
        _loop_over_matrix!(Δloc, θloc, progress, logdistances, p; kw...)
    catch
        @warn "Couldn't create $(N)×$(N) distance matrix; using slower algorithm..."
        _loop_and_compute_logdist!(Δloc, θloc, progress, X, p; kw...)
    end
    return Δloc, θloc
end

function _loop_over_matrix!(Δloc, θloc, progress, logdistances, p; kw...)
    Threads.@threads for (j, logdist) in enumerate(eachcol(logdistances))
        D, θ = extremevaltheory_local_dim_persistence(logdist, p)
        Δloc[j] = D
        θloc[j] = θ
        ProgressMeter.next!(progress)
    end
end
function _loop_and_compute_logdist!(Δloc, θloc, progress, X, p; kw...)
    logdists = [copy(Δloc) for _ in 1:Threads.nthreads()]
    Threads.@threads for j in eachindex(X)
        logdist = logdists[Threads.threadid()]
        @inbounds map!(x -> -log(euclidean(x, X[j])), logdist, vec(X))
        D, θ = extremevaltheory_local_dim_persistence(logdist, p)
        Δloc[j] = D
        θloc[j] = θ
        ProgressMeter.next!(progress)
    end
end

function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, p::Real; compute_persistence = true, estimator = :mm
    )
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
    exceedances = PoTs .- thresh
    # Extract the GPD parameters.
    σ = estimate_gpd_parameters(exceedances, estimator)[1]
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


################################################################################
# Fitting Pareto
################################################################################
"""
    estimate_gpd_parameters(X::AbstractVector{<:Real}, estimator::Symbol = :mm)

Estimate and return the parameters `σ, ξ` of a Generalized Pareto Distribution
fit to `X`, assuming that `minimum(X) == 0` and hence the parameter `μ` is 0
(if not, simply shift `X` by its minimum), according to the methods provided
in [^Pons2023].

Optionally choose the estimator, which can be:

- `:exp`: Assume the distribution is exponential instead of GP and
  get `σ` from sample mean and `ξ = 0`.
- `mm`: Standing for "method of moments", estimants are given by
  ```math
  \\xi = (\\bar{x}^2/s^2 - 1)/2, \\quad \\sigma = \\bar{x}(\\bar{x}^2/s^2 + 1)/2
  ```
  with ``\\bar{x}`` the sample mean and ``s^2`` the sample variance. This estimator
  only exists if the true distribution `ξ` value is < 0.5.
- `:pwm`: standing for "probability weighted moments" TODO.
"""
function estimate_gpd_parameters(X, estimator)
    if estimator == :exp
        # Assuming that the distribution is exponential, the
        # average of the PoTs is the unbiased estimator, which is just the mean
        # of the exceedances.
        return mean(X), zero(eltype(X))
    elseif estimator == :mm
        # for whatever reason the authors don't use the corrected versions
        x̄ = mean(X)
        s² = var(X; corrected = false, mean = x̄)
        ξ = (1/2)*((x̄^2/s²) - 1)
        σ = (x̄/2)*((x̄^2/s²) + 1)
        return σ, ξ
    else
        error("Unknown estimator for Pareto distribution")
    end
    return σ
end

"""
    extremal_index_sueveges(y::AbstractVector, p)

Compute the extremal index θ of `y` through the Süveges formula for quantile probability `p`.
"""
function extremal_index_sueveges(y::AbstractVector, p::Real,
        # These arguments are given for performance optim; not part of public API
        thresh::Real = quantile(y, p)
    )
    # TODO: This is wrong for now
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
