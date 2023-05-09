using FractalDimensions
using Test

@testset "min pair dist" begin
    X = StateSpaceSet([1, 2, 3.1])
    mind, minpair = minimum_pairwise_distance(X, true)
    @test mind == 1
    @test minpair == (1, 2)
    mind, minpair = minimum_pairwise_distance(X, false)
    @test mind == 1
    @test minpair == (1, 2)
end