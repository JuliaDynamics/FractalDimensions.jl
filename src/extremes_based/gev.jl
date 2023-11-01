export BlockMaxima, estimate_gev_parameters

"""
    BlockMaxima(blocksize::Int, p::Real)

Instructions type for [`extremevaltheory_dims_persistences`](@ref) and related functions.
This method divides the input data into blocks of length
`blocksize` and fits the maxima of each block to a Generalized Extreme
Value distribution. In order for this method to work correctly, both the
`blocksize` and the number of blocks must be high. Note that there are data points that are not
used by the algorithm. Since it is not always possible to express the number of input data
poins as `N = blocksize * nblocks + 1`. To reduce the number of unused data, chose an `N` equal or
superior to `blocksize * nblocks + 1`. This method and several variants of it has been studied
in [faranda2011numerical](@cite)

The parameter `p` is a number between 0 and 1 that
determines the p-quantile for the computation of the extremal index
and hence is irrelevant if `compute_persistences = false` in
[`extremevaltheory_dims_persistences`](@ref).

See also [`estimate_gev_parameters`](@ref).
"""
struct BlockMaxima
    blocksize::Int
    p::Real
end


function extremevaltheory_local_dim_persistence(
        logdist::AbstractVector{<:Real}, type::BlockMaxima; compute_persistence = true
    )
    p = type.p
    N = length(logdist)
    blocksize = type.blocksize
    nblocks = floor(Int, N/blocksize)
    newN = blocksize*nblocks
    firstindex = N - newN + 1
    Δ = zeros(newN)
    θ = zeros(newN)
    progress = ProgressMeter.Progress(
        N - firstindex; desc = "Extreme value theory dim: ", enabled = true
    )

    duplicatepoint = !isempty(findall(x -> x == Inf, logdist))
    if duplicatepoint
        error("Duplicated data point on the input")
    end
    if compute_persistence
        θ = extremal_index_sueveges(logdist, p)
    else
        θ = NaN
    end
    # Extract the maximum of each block
    maxvector = maximum(reshape(logdist[firstindex:N],(blocksize,nblocks)),dims= 1)
    σ = estimate_gev_scale(maxvector)
    Δ = 1 / σ
    return Δ, θ
end


"""
    estimate_gev_parameters(X::AbstractVector{<:Real}, θ::Real)

Estimate and return the parameters `σ, μ` of a Generalized Extreme Value distribution
fit to `X` (which typically is the collected block maxima of the log distances of a state space set),
assuming that the parameter `ξ` is 0, and that the extremal index θ
is a known constant, and can be estimated through the function
`extremal_index_sueveges`.
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