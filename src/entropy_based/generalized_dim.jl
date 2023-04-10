using ComplexityMeasures: Renyi, ValueHistogram, entropy
export generalized_dim

"""
    generalized_dim(X::StateSpaceSet [, sizes]; q = 1, base = 2) -> Δ_q

Return the `q` order generalized dimension of `X`,
by calculating  its histogram-based Rényi entropy for each `ε ∈ sizes`.

The case of `q = 0` is often called "capacity" or "box-counting" dimension, while
`q = 1` is the "information" dimension.

## Description

The returned dimension is approximated by the
(inverse) power law exponent of the scaling of the Renyi entropy ``H_q``,
versus the box size `ε`, where `ε ∈ sizes`:

```math
H_q \\approx -\\Delta_q\\log_{b}(\\varepsilon)
```

``H_q`` is calculated using `ComplexityMeasures: Renyi, ValueHistogram, entropy`,
i.e., by doing a histogram of the data with a given box size.

Calling this function performs a lot of automated steps:

  1. A vector of box sizes is decided by calling `sizes = estimate_boxsizes(dataset)`,
     if `sizes` is not given.
  2. For each element of `sizes` the appropriate entropy is calculated as
     ```julia
     H = [entropy(Renyi(; q, base), ValueHistogram(ε), data) for ε ∈ sizes]
     ```
     Let `x = -log.(sizes)`.
  3. The curve `H(x)` is decomposed into linear regions,
     using [`linear_regions`](@ref)`(x, h)`.
  4. The biggest linear region is chosen, and a fit for the slope of that
     region is performed using the function [`linear_region`](@ref),
     which does a simple linear regression fit using [`linreg`](@ref).
     This slope is the return value of `generalized_dim`.

By doing these steps one by one yourself, you can adjust the keyword arguments
given to each of these function calls, refining the accuracy of the result.
The source code of this function is only 3 lines of code.
"""
function generalized_dim(data::AbstractStateSpaceSet, sizes = estimate_boxsizes(data);
        base = 2, q = 1.0
    )
    H = [entropy(Renyi(; q, base), ValueHistogram(ε), data) for ε ∈ sizes]
    x = -log.(base, sizes)
    return linear_region(x, H)[2]
end
