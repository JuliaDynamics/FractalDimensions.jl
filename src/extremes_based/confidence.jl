export extremevaltheory_gpdfit_pvalues

# for confidence testing
using HypothesisTests: OneSampleADTest, ApproximateOneSampleKSTest, pvalue
using Distributions: GeneralizedPareto

"""
    extremevaltheory_gpdfit_pvalues(X, p; kw...) → pvalues, sigmas, xis

Quantify significance of the results of
[`extremevaltheory_dims_persistences`](@ref)`(X, p; kw...)` by
quantifying how well a Generalized Pareto Distribution (GPD) describes exceedences
in the input data using the approach described in [^Faranda2017].

Return the p-values of the statistical test, and the fitted `σ, ξ` values to each
of the GPD fits of the exceedances for each point in `X`.

## Description

The output `pvalues` is a vector of p-values. `pvalues[i]` corresponds to the p-value
of the hypothesis: _"The exceedences around point `X[i]` are sampled from a GPD"_ versus
the alternative hypothesis that they are not. Very small p-values then indicate
that the hypothesis should be rejected and the data are not well described by a GPD.
This can be a good indication that we do not have enough data, or that we choose
too high of a quantile probability `p`.

Alternatively, if the majority of `pvalues` are sufficiently high, e.g., higher
than 0.05, then we have some confidence that the probability `p` and/or amount of
data follow well the theory of FD from EVT.

To extract the p-values, we perform a one-sample hypothesis via HypothesisTests.jl
to the fitted GPD.

## Keyword arguments

- `show_progress, estimator` as in [`extremevaltheory_dims_persistences`](@ref)
- `TestType = ApproximateOneSampleKSTest`: the test type to use. It can be any
  of the [one-sample non-parametric tests from HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/nonparametric/#Nonparametric-tests)
  however we noticed that `OneSampleADTest` sometimes yielded nonsensical results:
  all p-values were equal and were very small ≈ 1e-6.

[^Faranda2017]:
    Faranda et al. (2017), Dynamical proxies of North Atlantic predictability and extremes,
    [Scientific Reports, 7](https://doi.org/10.1038/srep41278)
"""
function extremevaltheory_gpdfit_pvalues(X::AbstractStateSpaceSet, p;
        estimator = :mm, show_progress = envprog(), TestType = ApproximateOneSampleKSTest
    )
    N = length(X)
    progress = ProgressMeter.Progress(
        N; desc = "Extreme value theory p-values: ", enabled = show_progress
    )
    logdists = [zeros(eltype(X), N) for _ in 1:Threads.nthreads()]
    pvalues = zeros(N)
    sigmas = zeros(N)
    xis = zeros(N)

    Threads.@threads for j in eachindex(X)
        logdist = logdists[Threads.threadid()]
        @inbounds map!(x -> -log(euclidean(x, X[j])), logdist, vec(X))
        σ, ξ, E = extremevaltheory_local_gpd_fit(logdist, p, estimator)
        sigmas[j] = σ
        xis[j] = ξ
        # Note that exceedances are defined with 0 as their minimum
        gpd = GeneralizedPareto(0, σ, ξ)
        test = TestType(E, gpd)
        pvalues[j] = pvalue(test)
        ProgressMeter.next!(progress)
    end
    return pvalues, sigmas, xis
end
