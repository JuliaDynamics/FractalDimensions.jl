using FractalDimensions
using Random
using BenchmarkTools

EPSILON_AMOUNT = 16 # range of 16 ε values
DIMENSIONS = [3]
LENGTHS = [2*10^4]
LOGBASE = MathConstants.e
RNG = Random.MersenneTwister(1234)

for D in DIMENSIONS
    for N in LENGTHS
        # dataset:
        X = StateSpaceSet(randn(RNG, N, D))
        es = estimate_boxsizes(X; w = 0.5, z = 0, base = LOGBASE)
        bm = @benchmark correlationsum($X, $es; show_progress = false)
        t = median(bm).time/1e9
        a = median(bm).allocs
        println("Corr.sum, D=$(D), N=$(N): $(t) sec., $(a) allocs")
        # @time "Corr.sum, D=$(D), N=$(N):" correlationsum(X, es; show_progress = false)
    end
end
