export kaplanyorke_dim

"""
    kaplanyorke_dim(λs::AbstractVector)

Calculate the Kaplan-Yorke dimension, a.k.a. Lyapunov dimension [Kaplan1979](@cite)
from the given Lyapunov exponents `λs`.

## Description

The Kaplan-Yorke dimension is simply the point where
`cumsum(λs)` becomes zero (interpolated):
```math
 D_{KY} = k + \\frac{\\sum_{i=1}^k \\lambda_i}{|\\lambda_{k+1}|},\\quad k = \\max_j \\left[ \\sum_{i=1}^j \\lambda_i > 0 \\right].
```

If the sum of the exponents never becomes negative the function
will return the length of the input vector.

Useful in combination with `lyapunovspectrum` from ChaosTools.jl.
"""
function kaplanyorke_dim(λs::AbstractVector{<:Real})
    λs = sort(λs; rev = true)
    s, k =  cumsum(λs), length(λs)
    # Find k such that sum(λ_i for i in 1:k) is still possitive
    for i in eachindex(s)
        if s[i] < 0
            k = i-1
            break
        end
    end

    if k == 0
        return zero(λs[1])
    elseif k < length(λs)
        return k + s[k]/abs(λs[k+1])
    else
        return typeof(λs[1])(length(λs))
    end
end
