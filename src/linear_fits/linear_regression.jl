import Statistics
# The following function comes from a version in StatsBase that is now deleted
# StatsBase is copyrighted under the MIT License with
# Copyright (c) 2012-2016: Dahua Lin, Simon Byrne, Andreas Noack, Douglas Bates,
# John Myles White, Simon Kornblith, and other contributors.
"""
    linreg(x, y) -> a, b
Perform a linear regression to find the best coefficients so that the curve:
`z = a + b*x` has the least squared error with `y`.
"""
function linreg(x::AbstractVector, y::AbstractVector)
    # Least squares given
    # Y = a + b*X
    # where
    # b = cov(X, Y)/var(X)
    # a = mean(Y) - b*mean(X)
    if size(x) != size(y)
        throw(DimensionMismatch("x has size $(size(x)) and y has size $(size(y)), " *
            "but these must be the same size"))
    end
    mx = Statistics.mean(x)
    my = Statistics.mean(y)
    # don't need to worry about the scaling (n vs n - 1)
    # since they cancel in the ratio
    b = Statistics.covm(x, mx, y, my)/Statistics.varm(x, mx)
    a = my - b*mx
    return a, b
end

function _slopefit(x, y, ::LinearRegression)
    a, s = linreg(x, y)
    n = length(y)
    # CI computed via https://stattrek.com/regression/slope-confidence-interval
    # standard error of slope
    df = max(n - 2, 1)
    yhat = @. a + s*x
    standard_error = sqrt((sum((y .- yhat).^2) ./ df)) / sqrt(sum((x .- mean(x)).^2))
    ci = 0.95 # 95% confidence interval
    α = 1 - ci
    pstar = 1 - α/2
    tdist = TDist(df)
    critical_value = quantile(tdist, pstar)
    margin_of_error = critical_value * standard_error
    s05 = s - margin_of_error
    s95 = s + margin_of_error
    return s, s05, s95
end
