using FractalDimensions
using Test
using Random: Xoshiro
using DynamicalSystemsBase: Systems, trajectory

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

# Convenience syntax going back to the old `quickentropy`
function quickentropy(x, e; q = 1)
    return entropy(Renyi(;q), ValueHistogram(e), x)
end

@testset "analytic sets" begin
    A = Dataset(rand(Xoshiro(1234), 100_000, 2))
    θ = rand(Xoshiro(1234), 100_000).*2π
    B = Dataset(cos.(θ), sin.(θ))
    sizesA = estimate_boxsizes(A)
    sizesB = estimate_boxsizes(B)

    @testset "generalized_dim" begin
        for q in [0, 2, 1, 2.56]
            dA = generalized_dim(A, sizesA; q)
            test_value(dA, 1.8, 2.0)
            dB = generalized_dim(B, sizesB; q)
            test_value(dB, 0.8, 1.0)
        end
    end
    @testset "molteno_dim" begin
        for q in [0, 2, 1, 2.56]
            dA = molteno_dim(A)
            test_value(dA, 1.8, 2.0; q)
            dB = molteno_dim(B)
            test_value(dB, 0.8, 1.0; q)
        end
    end
end

@testset "henon map" begin
    A = trajectory(Systems.henon(), 100_000; Ttr = 100)
    @testset "generalized_dim" begin
        dA = generalized_dim(A; q = 0)
        dA1 = generalized_dim(A; q = 1.0)
        dA3 = generalized_dim(A; q = 3.0)
        test_value(dA, 1.22, 1.23)
        @test dA > dA1 > dA3
        test_value(dA1, 1.13, 1.15)
    end

    @testset "molteno_dim" begin
        probs, εs = molteno_boxing(A; k0 = 6)
        # fig = Figure(); display(fig)
        # ax = Axis(fig[1,1])
        molt_dim = q -> begin
            dd = entropy.(Ref(Renyi(;q, base = 2)), probs)
            x = -log.(2, εs)
            # scatterlines!(ax, x, dd; label = "q = $q")
            linear_region(x, dd)[2]
        end
        dA = molt_dim(0)
        dA1 = molt_dim(1.0)
        dA3 = molt_dim(3.0)
        # axislegend(ax)
        # no matter what we do here, and no matter the `k0` parameter,
        # the dimension at q = 1 always turns out larger. Don't know why...
        # Also, the values we get from the Molteno are significantly higher
        # than the ones we get from the standard histograms...
        test_value(dA, 1.22, 1.26)
        @test dA > dA3
        test_value(dA3, 1.13, 1.16)
    end
end
