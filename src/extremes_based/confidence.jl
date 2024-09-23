export extremevaltheory_gpdfit_pvalues
export CramerVonMises

# for confidence testing
using HypothesisTests: OneSampleADTest, ApproximateOneSampleKSTest, pvalue
using Distributions: GeneralizedPareto, pdf
using ComplexityMeasures

"""
    extremevaltheory_gpdfit_pvalues(X, p; kw...)

Return various computed quantities that may quantify the significance of the results of
[`extremevaltheory_dims_persistences`](@ref)`(X, p; kw...)`, terms of quantifying
how well a Generalized Pareto Distribution (GPD) describes exceedences
in the input data.

## Keyword arguments

- `show_progress = true`: display a progress bar.
- `TestType = ApproximateOneSampleKSTest`: the test type to use. It can be
  `ApproximateOneSampleKSTest, ExactOneSampleKSTest, CramerVonMises`.
  We noticed that `OneSampleADTest` sometimes yielded nonsensical results:
  all p-values were equal and were very small ≈ 1e-6.
- `nbins = round(Int, length(X)*(1-p)/20)`: number of bins to use when computing
  the histogram of the exceedances for computing the NRMSE.
  The default value will use equally spaced
  bins that are equal to the length of the exceedances divided by 20.

## Description

The function computes the exceedances ``E_i`` for each point ``x_i \\in X`` as in
[`extremevaltheory_dims_persistences`](@ref).
It returns 5 quantities, all being vectors of length `length(X)`:

- `Es`, all exceedences, as a vector of vectors.
- `sigmas, xis` the fitted σ, ξ to the GPD fits for each exceedance
- `nrmses` the normalized root mean square distance of the fitted GPD
  to the histogram of the exceedances
- `pvalues` the pvalues of a statistical test of the appropriateness of the GPD fit

The output `nrmses` quantifies the distance between the fitted GPD and the empirical
histogram of the exceedances. It is computed as
```math
NRMSE = \\sqrt{\\frac{\\sum{(P_j - G_j)^2}{\\sum{(P_j - U)^2}}
```
where ``P_j`` the empirical (observed) probability at bin ``j``, ``G_j`` the fitted GPD
probability at the midpoint of bin ``j``, and ``U`` same as ``G_j`` but for the uniform
distribution. The divisor of the equation normalizes the expression, so that the error
of the empirical distribution is normalized to the error of the empirical distribution
with fitting it with the uniform distribution. It is expected that NRMSE < 1.
The smaller it is, the better the data are approximated by GPD versus uniform distribution.

The output `pvalues` is a vector of p-values. `pvalues[i]` corresponds to the p-value
of the hypothesis: _"The exceedences around point `X[i]` are sampled from a GPD"_ versus
the alternative hypothesis that they are not.
To extract the p-values, we perform a one-sample hypothesis via HypothesisTests.jl
to the fitted GPD.
Very small p-values then indicate
that the hypothesis should be rejected and the data are not well described by a GPD.
This can be an indication that we do not have enough data, or that we choose
too high of a quantile probability `p`, or that the data are not suitable in general.
This p-value based method for significance has been used in [^Faranda2017],
but it is unclear precisely how it was used.

For more details on how these quantities may quantify significance, see our review paper.

[^Faranda2017]:
    Faranda et al. (2017), Dynamical proxies of North Atlantic predictability and extremes,
    [Scientific Reports, 7](https://doi.org/10.1038/srep41278)
"""
function extremevaltheory_gpdfit_pvalues(X::AbstractStateSpaceSet, p::Real;
        estimator = :mm, kw...
    )
    @warn "Using `p::Real` is deprecated. Explicitly create `Exceedances(p, estimator)`."
    type = Exceedances(p, estimator)
    return extremevaltheory_gpdfit_pvalues(X, type; kw...)
end

function extremevaltheory_gpdfit_pvalues(X::AbstractStateSpaceSet, type::Exceedances;
        show_progress = envprog(), TestType = ApproximateOneSampleKSTest,
        nbins = max(round(Int, length(X)*(1-type.p)/20), 10),
    )
    (; p, estimator) = type
    N = length(X)
    progress = ProgressMeter.Progress(
        N; desc = "Extreme value theory p-values: ", enabled = show_progress
    )
    logdists = [zeros(eltype(X), N) for _ in 1:Threads.nthreads()]
    pvalues = zeros(N)
    sigmas = zeros(N)
    xis = zeros(N)
    nrmses = zeros(N)
    Es = [Float64[] for _ in 1:N]

    Threads.@threads for j in eachindex(X)
        logdist = logdists[Threads.threadid()]
        @inbounds map!(x -> -log(euclidean(x, X[j])), logdist, vec(X))
        σ, ξ, E = extremevaltheory_local_gpd_fit(logdist, p, estimator)
        sigmas[j] = σ
        xis[j] = ξ
        Es[j] = E
        # Note that exceedances are defined with 0 as their minimum
        gpd = GeneralizedPareto(0, σ, ξ)
        test = TestType(E, gpd)
        pvalues[j] = pvalue(test)
        nrmses[j] = gpd_nrmse(E, gpd, nbins)
        ProgressMeter.next!(progress)
    end
    return Es, nrmses, pvalues, sigmas, xis
end

function gpd_nrmse(E, gpd, nbins)
    # Compute histogram of E
    bins = range(0, nextfloat(maximum(E), 2); length = nbins)
    binning = FixedRectangularBinning(bins)
    allprobs = allprobabilities_and_outcomes(ValueHistogram(binning), E)[1]
    width = step(bins)

    # We will calcuate the GPD pdf at the midpoint of each bin
    midpoints = bins .+ width/2
    rmse = zero(eltype(E))
    mmse = zero(eltype(E))
    # value of uniform density (equal probability at each bin)
    meandens = (1/length(allprobs))/width

    for (j, prob) in enumerate(allprobs)
        dens = prob / width # density value, to compare with pdf
        dens_gpd = pdf(gpd, midpoints[j])
        rmse += (dens - dens_gpd)^2
        mmse += (dens - meandens)^2
    end
    nrmse = sqrt(rmse/mmse)
    if isinf(nrmse)
        error("inf nrmse. Allprobs; $(allprobs). E; $(E)")
    end
    return nrmse
end

"""
    CramerVonMises(X, dist)

A crude implementation of the Cramer Von Mises test that yields a `p` value
for the hypothesis that the data in `X` are sampled from the distribution `dist`.
"""
struct CramerVonMises{E, D}
    e::E
    d::D
end

import HypothesisTests
using Distributions: cdf, Normal

# I got this test from:
# https://www.youtube.com/watch?v=pCz8WlKCJq8

function HypothesisTests.pvalue(test::CramerVonMises)
    X = test.e
    gpd = test.d
    n = length(X)
    xs = sort!(X)

    T = 1/(12n) + sum(i -> (cdf(gpd, xs[i]) - (2i - 1)/2n)^2, 1:n)

    # An approximation of the T statistic is normally distributed
    # with std of approximately sqrt(1/45)
    # From there the z statistic
    zstat = T/sqrt(1/45)

    p = 2*(1 - cdf(Normal(0,1), zstat))
    return p
end