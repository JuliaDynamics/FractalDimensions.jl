# module LinearScalingRegion

export linreg
export slopefit
export LinearRegression, LargestLinearRegion, AllSlopesDistribution

"""
    SlopeFit

Supertype of types used in [`slopefit`](@ref).
"""
abstract type SLopeFit end



"""
    slopefit(x, y [, t::SLopeFit]; kw...) → s, s05, s95

Fit a linear scaling region in the curve of the two `AbstractVectors` `y` versus `x`.
Return the estimated slope, as well as the confidence intervals for it.

The methods `t` that can be used for the estimation are:

- [`LinearRegression`](@ref)
- [`LargestLinearRegion`](@ref) (default)
- [`AllSlopesDistribution`](@ref)

The keyword `ignore_saturation = true` ignores saturation that (sometimes) happens
at the start and end of the curve `y(x)`, where the curve flattens.
The keyword `sat_threshold = 0.01` decides what saturation is:
while `abs(y[i]-y[i+1]) < sat_threshold` we are in a saturation regime.
Said differently, slopes with value `sat_threshold/dx` with `dx = x[i+1] - x[i]`
are neglected.

The keyword `ci = 0.95` specifies which quantile (and the 1 - quantile) the confidence
interval values are returned at, and by defualt it is 95% (and hence also 5%).
"""
function slopefit(x, y, t::SlopeFit = LargestLinearRegion();
        ignore_saturation = true, sat_threshold = 0.01, ci = 0.95,
    )
    if ignore_saturation
        j = findfirst(i -> abs(y[i] - y[i-1]) > sat_threshold, length(y):-1:2)
        if !isnothing(j)
            i = (length(y):-1:2)[j]
            x, y = x[1:i], y[1:i]
        end
        k = findfirst(i -> abs(y[i+1] - y[i]) > sat_threshold, 1:length(y)-1)
        if !isnothing(k)
            x, y = x[k:end], y[k:end]
        end
    end
    return _slopefit(x, y, t, ci)
end



# ends