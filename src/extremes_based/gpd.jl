export estimate_gpd_parameters

"""
    estimate_gpd_parameters(X::AbstractVector{<:Real}, estimator::Symbol = :mm)

Estimate and return the parameters `σ, ξ` of a Generalized Pareto Distribution
fit to `X`, assuming that `minimum(X) ≥ 0` and hence the parameter `μ` is 0
(if not, simply shift `X` by its minimum), according to the methods provided
in [^Flavio2023].

Optionally choose the estimator, which can be:

- `:exp`: Assume the distribution is exponential instead of GP and
  get `σ` from mean of `X` and set `ξ = 0`.
- `mm`: Standing for "method of moments", estimants are given by
  ```math
  \\xi = (\\bar{x}^2/s^2 - 1)/2, \\quad \\sigma = \\bar{x}(\\bar{x}^2/s^2 + 1)/2
  ```
  with ``\\bar{x}`` the sample mean and ``s^2`` the sample variance.
  This estimator only exists if the true distribution `ξ` value is < 0.5.

[^Flavio2023]:
    Flavio et al., Stability of attractor local dimension estimates in
    non-Axiom A dynamical systems, [preprint](https://hal.science/hal-04051659)
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
