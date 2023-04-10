export loc_dimension_persistence, extremal_index_sueveges, extremesdim
using Distances: euclidean
using Statistics: mean, quantile

# The functions in this section are versions inspired from the code
# for MATLAB given in the following papers:

# Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent
# drivers of weather extremes: application to hot and cold days in North
# America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

# Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies
# of North Atlantic predictability and extremes. Scientific Reports, 7,
# 41278, doi: 10.1038/srep41278

# Süveges, Mária. 2007. Likelihood estimation of the extremal index.
# Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2

"""
    extremevaltheory_dim(X::StateSpaceSet, q::Real) → Δ

Convenience syntax that returns the mean of the local dimensions of
[`extremevaltheory_dims_persistences`](@ref), which approximates
a fractal dimension of `X` using extreme value theory and quantile `q`.
"""
function extremevaltheory_dim(args...; kwargs...)
    Dloc, θloc = extremevaltheory_dims_persistences(args...; compute_persistence = false)
    return mean(Dloc)
end


"""
    loc_dimension_persistence(x::AbstractStateSpaceSet, q::Real) -> Dloc, θ

Computation of the local dimensions `Dloc` and the extremal indices `θ` for each point in the
given set for a given quantile `q`. The extremal index can be interpreted as the
inverse of the persistence of the extremes around each point point.
"""
function loc_dimension_persistence(x::AbstractStateSpaceSet, q::Real)

    println("Computing dynamical quantities")

    if !(0 < q < 1)
        error("The quantile has to be between 0 and 1")
    end

    N = length(x)
    D1 = zeros(eltype(x), N)
    θ = copy(D1)

    for j in 1:N
        # Compute the observables
        logdista = -log.([euclidean(x[j,:],x[i,:]) for i in 1:N])
        # Extract the threshold corresponding to the quantile defined
        thresh = quantile(logdista, q)
        # Compute the extremal index # TODO: This is wrong for now
        # θ[j] = extremal_index_sueveges(logdista, q, thresh)
        # Sort the time series and find all the Peaks Over Threshold (PoTs)
        PoTs = logdista[findall(x -> x > thresh, logdista)]
        filter!(isfinite, PoTs)
        exceedances = PoTs .- thresh
        # Extract the GPD parameters.
        # Assuming that the distribution is exponential, the
        # average of the PoTs is the unbiased estimator, which is just the mean
        # of the exceedances.
        # The local dimension is the reciprocal of the exceedances of the PoTs
        D1[j] = 1 ./ mean(exceedances)
    end
    return D1, θ
end

"""
    extremal_index_sueveges(logdista::AbstractVector, q, thresh)

Compute the extremal index θ through the Süveges formula.
"""
function extremal_index_sueveges(logdista::AbstractVector, q::Real, thresh::Real)
    q = 1 - q
    Li = findall(x -> x > thresh, logdista)
    Ti = diff(Li)
    Si = Ti .- 1
    Nc = length(findall(x->x>0, Si))
    N = length(Ti)
    θ = (sum(q.*Si)+N+Nc - sqrt( (sum(q.*Si) +N+Nc).^2 - 8*Nc*sum(q.*Si)) )./(2*sum(q.*Si))
    return θ
end
