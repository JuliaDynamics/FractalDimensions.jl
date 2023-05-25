using FractalDimensions
using Test
using Random: MersenneTwister

@testset "min pair dist" begin
    X = StateSpaceSet([1, 2, 3.1])
    mind, minpair = minimum_pairwise_distance(X, true)
    @test mind == 1
    @test minpair == (1, 2)
    mind, minpair = minimum_pairwise_distance(X, false)
    @test mind == 1
    @test minpair == (1, 2)
end

@testset "linreg" begin
    x = 0:0.01:1
    y = @. 2.5x + 1
    s, s05, s95 = slopefit(x, y, LinearRegression())
    @test all(x -> isapprox(x, 2.5; atol = 0, rtol = 1e-12), (s, s05, s95))

    y = y .+ 0.01randn(MersenneTwister(1234), 101)
    s, s05, s95 = slopefit(x, y, LinearRegression())
    @test !all(x -> isapprox(x, 2.5; atol = 0, rtol = 1e-12), (s, s05, s95))
    @test all(x -> isapprox(x, 2.5; atol = 0, rtol = 1e-2), (s, s05, s95))
end

@testset "linear regions" begin
    # fixed linear
    x = 1:10
    y = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    regions, tangents = linear_regions(x, y)
    @test regions[1] == 1:3
    @test tangents[1] == 0
    @test regions[2] == 3:8
    @test tangents[2] == 1
    @test regions[3] == 8:10
    @test tangents[3] == 0
    # sigmoid
    x = -5:0.25:5
    y = tanh.(x)
    regions, tangents = linear_regions(x, y)
    # In relative tolerance, the initial regions are very different
    @test regions[1] == 1:2
    @test regions[2] == 2:3
    # The largest region is in the middle with slope close to 1
    val, idx = findmax(length, regions)
    @test val > 1
    @test 0.9 < tangents[idx] < 1.1

    s, s05, s95 = slopefit(x, y, LargestLinearRegion())
    @test 0.9 < s < 1.1
    @test 0.8 < s05 < 1.0
    @test 0.1 < s95 < 1.2

    # We now decrease tolerance drastically,
    # and the first found region increases
    regions, tangents = linear_regions(x, y; tol = 0.5)
    @test regions[1] == 1:3

    # Lastly, we check the `linear_region` function for test coverage
    region, tangent = linear_region(x, y)
    @test 0.9 < tangent < 1.1
end

@testset "all slopes distr" begin
    x = 1:10
    y = [0, 0, 0, 1, 2, 3, 4, 5, 5, 5]
    @test_throws ErrorException slopefit(x, y, AllSlopesDistribution())
end
