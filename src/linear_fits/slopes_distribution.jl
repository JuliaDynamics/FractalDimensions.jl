# TODO:

"""
    AllSlopesDistribution <: SlopeFit
    AllSlopesDistribution()

Estimate a slope by computing the distribution of all possible slopes that
can be estimated from the curve `y(x)`, according to the method by [^Deshmukh2021].
The returned slope is the distribution mean and the confidence intervals are
simply the corresponding quantiles of the distribution.

Not implemented yet, the method is here as a placeholder.

[^Deshmukh2021]:
    Deshmukh et al., Toward automated extraction and characterization of scaling regions
    in dynamical systems, [Chaos 31, 123102 (2021)](https://doi.org/10.1063/5.0069365).
"""
struct AllSlopesDistribution <: SlopeFit end

function _slopefit(x, y, ::AllSlopesDistribution, ci::Real; kw...)
    error("Not implemented yet, PRs welcomed!")
end
