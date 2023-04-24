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
# Henon with Δ ≈ 1.2
henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
H = standardize(trajectory(henon, 10_000; Ttr = 100)[1])
sizesH = estimate_boxsizes(X; z = -2)

@testset "correlation sums analytic" begin
    X = StateSpaceSet([SVector(0.0, 0.0), SVector(0.5, 0.5)])
    εs = [0.1, 1.0]
    Cs = correlationsum(X, εs)
    Csb = boxed_correlationsum(X, εs, 1.0)
    Csb2 = boxed_correlationsum(X, εs, 1.5)
    @test Cs == Csb == Csb2 == [0, 1]
    # If max radious, all points are in
    X = StateSpaceSet(rand(Xoshiro(1234), 1000, 2))
    @test correlationsum(X, 5) ≈ boxed_correlationsum(X, [5])[1] ≈ 1
    # q shouldn't matter here; we're just checking the correlation sum formula
    @testset "norm, q = $q" for q in [2, 2.5, 4.5]
        @test correlationsum(X, 5; q) ≈ boxed_correlationsum(X, [5]; q)[1] ≈ 1
    end
    # And just to be extra safe, let's check the equivalence between the boxed
    # and unboxed version of the corrsums
    @testset "equiv. q = $q" for q in [2, 2.5, 4.5]
        @test correlationsum(X, 0.1; q) ≈ boxed_correlationsum(X, 0.1; q)
        @test correlationsum(X, 0.1; q, w = 10) ≈ boxed_correlationsum(X, 0.1; q, w = 10)
        @test correlationsum(X, [0.1, 0.5]; q) ≈ boxed_correlationsum(X, [0.1, 0.5]; q)
    end

end

@testset "Correlation dim; automated" begin
    # We can't test q != 1 here; it doesn't work. It doesn't give correct results.
    @testset "Grassberger" begin
        dA = grassberger_proccacia_dim(A; q = 2.0)
        test_value(dA, 1.9, 2.1)
        dA = grassberger_proccacia_dim(A, sizesA; q = 2.0)
        test_value(dA, 1.9, 2.1)
        dB = grassberger_proccacia_dim(B, sizesB; q = 2.0)
        test_value(dB, 0.9, 1.1)
    end
    @testset "Boxed" begin
        # We use the internal method because the sizes don't work out well
        Cs = boxed_correlationsum(A, sizesA)
        dA = linear_region(log.(sizesA), log.(Cs))[2]
        test_value(dA, 1.9, 2.1)
        Cs = boxed_correlationsum(B, sizesB, 0.1)
        dB = linear_region(log.(sizesB), log.(Cs))[2]
        test_value(dB, 0.9, 1.1)
    end
    @testset "Boxed, Henon" begin
        dH = boxassisted_correlation_dim(H)
        test_value(dX, 1.2, 1.3)
        # Also ensure that the values match with vanilla version
        r0 = estimate_r0_buenoorovio(H)[1]
        es = r0 .* 10 .^ range(-2; stop = 0, length = 10)
        C = correlationsum(H, es)
        @test boxed_correlationsum(X, es, r0) ≈ C
    end
end

@testset "Fixed mass correlation sum" begin
    @testset "Analytic" begin
        rs, ys = fixedmass_correlationsum(A, 64)
        @test all(<(0), rs) # since max is 1, log(r) must be negative
        reg, tan = linear_region(rs, ys)
        test_value(tan, 1.8, 2.0)
        reg, tan = linear_region(rs[40:end], ys[40:end])
        test_value(tan, 1.9, 2.0)

        dB = fixedmass_correlation_dim(B)
        test_value(dB, 0.9, 1.0)
    end
    @testset "Henon" begin
        dX = fixedmass_correlation_dim(X)
        test_value(dX, 1.11, 1.31)
    end
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

    D_C = takens_best_estimate_dim(X, 0.05)
    test_value(D_C, 1.2, 1.26)
end
