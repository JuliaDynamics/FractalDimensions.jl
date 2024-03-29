using FractalDimensions
using Random
using BenchmarkTools

EPSILON_AMOUNT = 16 # range of 16 ε values
DIMENSIONS = [3]
LENGTHS = [10^4]
LOGBASE = MathConstants.e
RNG = Random.MersenneTwister(1234)

function display_benchmark(bm, desc)
    t = median(bm).time/1e9
    a = median(bm).allocs
    m = median(bm).memory
    println(desc, ": $(t) sec., $(a) allocs, $(m) memory")
end

for D in DIMENSIONS
    for N in LENGTHS
        # dataset:
        X = StateSpaceSet(randn(RNG, N, D))
        println("Corr.sum, D=$(D), N=$(N)")
        es = estimate_boxsizes(X; w = 0.5, z = 0, base = LOGBASE, k = 16)
        # for q in [2, 3]
        #     bm = @benchmark correlationsum($X, $es; show_progress = false, q = $(q), w = 5)
        #     t = median(bm).time/1e9
        #     a = median(bm).allocs
        #     m = median(bm).memory
        #     println("q=$(q): $(t) sec., $(a) allocs, $(m) memory")
        #     # @time "Corr.sum, D=$(D), N=$(N):" correlationsum(X, es; show_progress = false)
        # end


        # boxed
        r0 = es[end]/10
        ε0 = es[1]
        es = MathConstants.e .^ range(log(ε0), log(r0); length = 12)

        # bm = @benchmark FractalDimensions.data_boxing($X, $r0)
        # display_benchmark(bm, "data boxing")

        # for q in [2, 3]
        for q in [2]
            bm = @benchmark boxed_correlationsum($X, $es; show_progress = false, q = $(q), w = 5)
            display_benchmark(bm, "boxed q=$(q)")
        end
    end
end


# Before:
# data boxing: 0.0110387 sec., 119615 allocs, 7211040 memory
# data boxing: 0.0110672 sec., 119615 allocs, 7211040 memory

# After:
# data boxing: 0.01072235 sec., 129737 allocs, 7576688 memory