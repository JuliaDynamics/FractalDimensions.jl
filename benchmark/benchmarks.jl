using BenchmarkTools
using Random
using FractalDimensions

const SUITE = BenchmarkGroup()
CORRSUMSUITE = SUITE["CorrelationSum"] = BenchmarkGroup()
ENTROPYSUITE = SUITE["EntropyBased"] = BenchmarkGroup()

const EPSILON_AMOUNT = 16 # range of 16 ε values
const DIMENSIONS = [3, 7]
const LENGTHS = [10^3, 10^5]
const LOGBASE = MathConstants.e
const RNG = Random.MersenneTwister(1234)

# Convenience functions that given an ε range they do the computations
genentropyf(X, εs::AbstractVector) = genentropy.(Ref(X), εs)
buenooroviof(X, εs) = estimate_r0_buenoorovio(X)
takensf(X, εs) = takens_best_estimate(X, maximum(εs))
fixedmassf(X, εs) = correlationsum_fixedmass(X, length(εs))
moltenof(X, εs) = molteno_boxing(X)

for D in DIMENSIONS
    for N in LENGTHS
        # dataset:
        X = StateSpaceSet(randn(RNG, N, D))
        r0, ε0 = estimate_r0_buenoorovio(X)
    end
end


# %% Entropy

# %% Correlation sum
