export Exceedances, estimate_gpd_parameters

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

# Extension of core function
function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::Exceedances; compute_persistence = true
    )
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
    estimate_gpd_parameters(X::AbstractVector{<:Real}, estimator::Symbol)

Estimate and return the parameters `σ, ξ` of a Generalized Pareto Distribution
fit to `X`, assuming that `minimum(X) ≥ 0` and hence the parameter `μ` is 0
(if not, simply shift `X` by its minimum), according to the methods provided
in [Pons2023](@ref).

The estimator can be:

- `:exp`: Assume the distribution is exponential instead of GP and
  get `σ` from mean of `X` and set `ξ = 0`.
- `mm`: Standing for "method of moments", estimants are given by
  ```math
  \\xi = (\\bar{x}^2/s^2 - 1)/2, \\quad \\sigma = \\bar{x}(\\bar{x}^2/s^2 + 1)/2
  ```
  with ``\\bar{x}`` the sample mean and ``s^2`` the sample variance.
  This estimator only exists if the true distribution `ξ` value is < 0.5.
"""
function estimate_gpd_parameters(X, estimator)
    if estimator == :exp
        # Assuming that the distribution is exponential, the
        # average of the PoTs is the unbiased estimator, which is just the mean
        # of the exceedances.
        σ = mean(X)
        ξ = zero(eltype(X))
    elseif estimator == :mm
        x̄ = mean(X)
        s² = var(X; corrected = true, mean = x̄)
        ξ = (1/2)*((x̄^2/s²) - 1)
        σ = (x̄/2)*((x̄^2/s²) + 1)
    elseif estimator == :pwm
        a0 = mean(X)
        n = length(X)
        xsorted = sort(X)
        a1 = sum(i -> xsorted[i]*(1 - (i - 0.35)/n), 1:n)/n
        ξ = 2 - a0/(a0 - 2a1)
        σ = 2*a0*a1/(a0 - 2a1)
        if -σ/ξ < xsorted[end]
            ξ = -σ/xsorted[end]
        end
    else
        throw(ArgumentError("Unknown estimator $(estimator) for Generalized Pareto distribution fit."))
    end
    return σ, ξ
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