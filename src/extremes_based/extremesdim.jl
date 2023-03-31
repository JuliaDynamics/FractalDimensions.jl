
"""
    The functions in this section are versions adapted to Julia from the code 
    for MATLAB given in the following papers:
    
        REFERENCES
    Please cite:
    
    Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
    drivers of weather extremes: application to hot and cold days in North 
    America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3
    Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
    of North Atlantic predictability and extremes. Scientific Reports, 7, 
    41278, doi: 10.1038/srep41278

    Süveges, Mária. 2007. Likelihood estimation of the extremal index.
    Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2
     

"""




"""
loc_dimension_persistence(x:: AbstractVector, quanti:: Real) -> D1, θ
    Computation of the local dimensions D1 and the extremal index θ for each observation in the 
    trajectory x, for a given quantile "quanti". The extremal index can be interpreted as the 
    inverse of the persistence of the extremes around that point.
    
"""
function loc_dimension_persistence(x:: AbstractVector, quanti:: Real)

    println("Computing dynamical quantities")

    if !(0 < quanti < 1)
        error("The number quanti represents a quantile and has to be between 0 and 1")
    end

    D1 = zeros(size(x[:,1]));
    θ = zeros(size(x[:,1]));
    for j in range(1,length(D1))
        # Compute the observables
        logdista = -log.([euclidean(x[j,:],x[i,:]) for i in range(1,length(x[:,1]))]);
        # Extract the threshold corresponding to the quantile defined   
        thresh = quantile(logdista, quanti);
        # Compute the extremal index, use the external function extremal_Sueveges
        θ[j] = fun_extremal_index_sueveges(logdista, quanti, thresh);
        #Sort the time series and find all the PoTs
        logextr = logdista[findall(x -> x > thresh, logdista)];
        filter!(isfinite,logextr);
        #Extract the GPD parameters; since the distribution is exponential, the 
        #average of the PoTs is the unbiased estimator, which is just the mean 
        #of the exceedances.
        #The local dimension is the reciprocal of the exceedances of the PoTs
        D1[j] = 1 ./ mean(logextr .- thresh);
    end
    return [D1, θ]
end
export loc_dimension_persistence

"""
    extremal_index_sueveges(Y:: AbstractVector, u:: Real, quanti:: Real)
    
    This function computes the extremal index θ through the Süveges formula for 
    a time series Y, given the quantile p and the corresponding threshold p.
    
"""
function extremal_index_sueveges(Y:: AbstractVector, u:: Real, quanti:: Real)

    # Compute theta
    
    if !(0 < quanti < 1)
        error("The number quanti represents a quantile and has to be between 0 and 1")
    end

    q = 1 - quanti;
    Li = findall(x -> x > u, Y);
    Ti = diff(Li); 
    Si = Ti .- 1;
    Nc = length(findall(x->x>0,Si));
    N = length(Ti);
    
    θ = (sum(q.*Si)+N+Nc - sqrt( (sum(q.*Si) +N+Nc).^2-8*Nc*sum(q.*Si)) )./(2*sum(q.*Si))
end

export extremal_index_sueveges




"""
    extremesdim(x:: AbstractVector, quanti:: Real) -> D

    Computes an estimation of the dimension D of the attractor of a system 
    given a trajectory along said attractor. 



"""
function extremesdim(x:: AbstractVector, quanti:: Real)

    println("Computing dynamical quantities")

    if !(0 < quanti < 1)
        error("The number quanti represents a quantile and has to be between 0 and 1")
    end

    D1 = zeros(size(x[:,1]));
    for j in range(1,length(D1))
        # Compute the observables
        logdista = -log.([euclidean(x[j,:],x[i,:]) for i in range(1,length(x[:,1]))]);
        # Extract the threshold corresponding to the quantile defined   
        thresh = quantile(logdista, quanti);
        #Sort the time series and find all the PoTs
        logextr = logdista[findall(x -> x > thresh, logdista)];
        filter!(isfinite,logextr);
        #Extract the GPD parameters; since the distribution is exponential, the 
        #average of the PoTs is the unbiased estimator, which is just the mean 
        #of the exceedances.
        #The local dimension is the reciprocal of the exceedances of the PoTs
        D1[j] = 1 ./ mean(logextr .- thresh);
    end
    return mean(D1)
end
export extremesdim
