export BlockMaxima, estimate_gev_parameters

"""
    BlockMaxima(blocksize::Int, p::Real)

This struct contains the parameters needed to perform an estimation
of the local dimensions through the block maxima method of extreme
value theory. This method divides the input data into blocks of length
`blocksize` and fits the maxima of each block to a Generalized Extreme
Value distribution. The parameter `p` is a number between 0 and 1 that
determines the p-quantile for the computation of the extremal index.
"""
struct BlockMaxima
    blocksize::Int
    p::Real
end

"""
    estimate_gev_parameters(X::AbstractVector{<:Real}, θ::Real)

Estimate and return the parameters `σ, μ` of a Generalized Extreme Value distribution
fit to `X`, assuming that the parameter `ξ` is 0, and that the extremal index θ
is a known constant, and can be estimated through the function
extremal_index_sueveges(...).
The estimators through the method of moments are given by
    σ = √((̄x²-̄x^2)/(π^2/6))
    μ = ̄x - σ(log(θ) + γ)
where γ is the constant of Euler-Mascheroni.
"""
function estimate_gev_parameters(X, θ)
        γ = 0.57721 # Euler-Mascheroni constant
        moment1 = mean(X)
        moment2 = mean(x -> x^2, X)
        σ = √((moment2-moment1^2)/(π^2/6))
        μ = moment1 - σ*(log(θ) + γ)
    return σ, μ
end
export estimate_gev_parameters

"""
    estimate_gev_scale(X::AbstractVector{<:Real})

Estimate and return the scale parameter σ of a Generalized Extreme Value distribution
fit to `X`, assuming that the parameter `ξ` is 0.
The estimator through the method of moments is given by
    σ = √((̄x²-̄x^2)/(π^2/6))
This function is given to improve performance, since for the computation of the
local dimension and the location parameter are not necesary to estimate the dimension.
"""
function estimate_gev_scale(X)
    moment1 = mean(X)
    moment2 = mean(x -> x^2, X)
    σ = √(abs(moment2-moment1^2)/(π^2/6))
    return σ
end
export estimate_gev_scale