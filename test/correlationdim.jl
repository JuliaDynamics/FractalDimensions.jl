using FractalDimensions
using Test
using Random: Xoshiro
using DynamicalSystemsBase

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

# Random with Δ ≈ 2
A = StateSpaceSet(rand(Xoshiro(1234), 10_000, 2))
sizesA = estimate_boxsizes(A; z = -2)
# Circle with Δ ≈ 1
θ = rand(Xoshiro(1234), 10_000) .* 2π
B = StateSpaceSet(cos.(θ), sin.(θ))
sizesB = estimate_boxsizes(B; z = -2)
# Circle with exact points
θ = collect(range(0, 2π; length = 101))
pop!(θ)
C = StateSpaceSet(cos.(θ), sin.(θ))
# Henon with Δ ≈ 1.2
henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
H = standardize(trajectory(henon, 10_000; Ttr = 100)[1])
sizesH = estimate_boxsizes(H; z = -2)

@testset "correlation sums analytic" begin
    X = StateSpaceSet([SVector(0.0, 0.0), SVector(0.5, 0.5)])
    εs = [0.1, 1.0]
    @testset "two-point q=$(q)" for q in [2, 3]
        Cs = correlationsum(X, εs; q)
        Csb = boxed_correlationsum(X, εs, 1.0; q)
        Csb2 = boxed_correlationsum(X, εs, 1.5; q)
        @test Cs == Csb == Csb2 == [0, 1]
    end
    # If max radious, all points are in
    # q shouldn't matter here; we're just checking the correlation sum formula
    X = SVector{2, Float64}.(vec(collect(Iterators.product(0:0.05:0.99, 0:0.05:0.99))))
    X = StateSpaceSet(X)
    @testset "norm, q = $q" for q in [2, 2.5, 4.5]
        @testset "vanilla" begin
            @test correlationsum(X, [5]; q)[1] ≈ 1
        end
        @testset "boxed" begin
            @test boxed_correlationsum(X, [5]; q)[1] ≈ 1
        end
    end
    # Okay, now let's use the `C` set where we can analytically compute correlation sums
    # for `r = 0.7` each point has 2 neighbors
    # and for `r = 0.13` each point has 4 neighbors
    # For q=3 we have 2 (the neighbors) to the power of 2, so total is 4
    # while for `r = 0.13` the 4 neighbors become 16
    @testset "circle exact" begin
        N = length(C)
        normal2 = (N * (N - 1))
        normal3 = N*(N-1)^2
        vanilla(C, r, q, w=0) = correlationsum(C, r; q, w)
        boxed(C, r, q, w=0) = boxed_correlationsum(C, r; q, w)
        @testset "version: $(f)" for f in (vanilla,  boxed)
            for (r, total2, total3) in zip((0.07, 0.13), (2N, 4N), (4N, 16N))
                @test f(C, r, 2) ≈ total2 / normal2
                @test f(C, r, 3) ≈ (total3/normal3)^(0.5)
            end
        end
        # Boxed-assisted corrsum shouldn't care about `r0`
        # And just to be extra safe, let's check the equivalence between the boxed
        # and unboxed version of the corrsums
        @testset "irrelevance from r0" begin
            for r0 in [0.2, 4.0, 5.0]
                @test boxed_correlationsum(C, 0.07, r0) ≈ 2N / normal2
                @test boxed_correlationsum(C, 0.13, r0) ≈ 4N / normal2
            end
        end
    end
    # A significantly different theiler window should have significantly
    # different correlation sum for data that close in space
    # is also close in time;
    @testset "theiler" begin
        θ = 0:0.01:2π
        C = StateSpaceSet(cos.(θ), sin.(θ))
        @testset "q = $q" for q in [2, 2.5, 4.5]
            @test correlationsum(C, 0.1; q) > correlationsum(C, 0.1; q, w = 50)
            @test boxed_correlationsum(C, 0.1; q) > boxed_correlationsum(C, 0.1; q, w = 50)
        end
    end
    # Lastly, ensure things make sense also in high dimensional spaces
    F = StateSpaceSet(rand(Xoshiro(1234), SVector{4, Float64}, 10_000))
    @testset "4D, prism=$(P)" for P in [2, 4]
        @test boxed_correlationsum(F, 5.1; P) ≈ 1
    end

    @testset "Bueno-orovio r0" begin
        r0 = estimate_r0_buenoorovio(F)[1]
        @test 0 < r0 < 1
        r0 = estimate_r0_buenoorovio(F, 4)[1]
        @test 0 < r0 < 1
    end

    @testset "Theiler r0" begin
        r0 = estimate_r0_theiler(F)[1]
        @test 0 < r0 < 1
        r0 = estimate_r0_theiler(F[:, 1:2])[1]
        @test 0 < r0 < 1
    end

end

@testset "Correlation dims; automated" begin
    # We can't test q != 1 here; it doesn't work. It doesn't give correct results.
    @testset "Grassberger" begin
        dA = grassberger_proccacia_dim(A; q = 2.0)
        test_value(dA, 1.9, 2.1)
        dA = grassberger_proccacia_dim(A, sizesA; q = 2.0)
        test_value(dA, 1.9, 2.1)
        dB = grassberger_proccacia_dim(B, sizesB; q = 2.0)
        test_value(dB, 0.9, 1.1)
        dH = grassberger_proccacia_dim(H)
        test_value(dH, 1.2, 1.3)
    end
    @testset "Boxed" begin
        # We use the internal method because the sizes don't work out well
        Cs = boxed_correlationsum(A, sizesA)
        dA = linear_region(log.(sizesA), log.(Cs))[2]
        test_value(dA, 1.9, 2.1)
        Cs = boxed_correlationsum(B, sizesB)
        dB = linear_region(log.(sizesB), log.(Cs))[2]
        test_value(dB, 0.9, 1.1)
        dH = boxassisted_correlation_dim(H)
        test_value(dH, 1.2, 1.3)
    end
end

@testset "Fixed mass correlation sum" begin
    rs, ys = fixedmass_correlationsum(A, 64)
    @test all(<(0), rs) # since max is 1, log(r) must be negative
    reg, tan = linear_region(rs, ys)
    test_value(tan, 1.8, 2.0)
    reg, tan = linear_region(rs[40:end], ys[40:end])
    test_value(tan, 1.9, 2.0)
    dB = fixedmass_correlation_dim(B)
    test_value(dB, 0.9, 1.0)
    dH = fixedmass_correlation_dim(H)
    test_value(dH, 1.11, 1.31)
end

@testset "Takens best est" begin
    D_C, D_C_95u, D_C_95l = FractalDimensions.takens_best_estimate(A, 0.1)
    test_value(D_C, 1.9, 2.0)
    @test D_C_95u < 1.05*D_C
    @test D_C_95l > 0.95*D_C

    D_C = takens_best_estimate_dim(A, 0.01)
    test_value(D_C, 1.9, 2.0)

    D_C = takens_best_estimate_dim(B, 0.1)
    test_value(D_C, 0.9, 1.1)

    D_C = takens_best_estimate_dim(H, 0.05)
    test_value(D_C, 1.2, 1.26)
end
