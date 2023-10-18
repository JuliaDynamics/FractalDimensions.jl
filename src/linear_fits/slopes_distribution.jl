# TODO:

"""
    AllSlopesDistribution <: SlopeFit
    AllSlopesDistribution()

Estimate a slope by computing the distribution of all possible slopes that
can be estimated from the curve `y(x)`, according to the method by [Deshmukh2021](@cite).
The returned slope is the distribution mean and the confidence intervals are
simply the corresponding quantiles of the distribution.

Not implemented yet, the method is here as a placeholder.
"""
struct AllSlopesDistribution <: SlopeFit end

function _slopefit(x, y, ::AllSlopesDistribution, ci::Real; kw...)
    error("Not implemented yet, PRs welcomed!")
end
