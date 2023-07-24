using Test, FractalDimensions
using Statistics
using Random: Xoshiro
ENV["FRACTALDIMENSIONS_PROGRESS"] = false

@testset "Circle" begin
    θ = collect(range(0, 2π; length = 5001))
    θ .+= 1e-9randn(Xoshiro(1234), 5001)
    pop!(θ)
    A = StateSpaceSet(cos.(θ), sin.(θ))

    @testset "validity" begin
        Δloc, θ = extremevaltheory_dims_persistences(A, 0.95;
            compute_persistence = false,
            show_progress = false,
        )
        avedim = mean(Δloc)
        sigma = std(Δloc)
        # Here are some totally arbitrary criteria for accuracy
        # note that normally the dimensions of every single point
        # should have been exactly the same. Not sure why they aren't...
        @test 0.98 < avedim < 1.02
        @test sigma < 0.01
    end

    @testset "Convenience API" begin
        D = extremevaltheory_dim(A, 0.95;
        show_progress = false,
        allocate_matrix = true)
        @test 0.9 < D < 1.1
    end

    # TODO: These tests need to be made legitimate!
    @testset "pvalues" begin
        pvalues = extremevaltheory_gpdfit_pvalues(A, 0.95)[1]
        @test all(p -> 0 ≤ p ≤ 1, pvalues)
    end

    @testset "wrong estimator" begin
        @test_throws ArgumentError extremevaltheory_dims_persistences(A, 0.95;
            estimator = :wrong,
        )
    end
end

@testset "Random 2D" begin
    A = StateSpaceSet(rand(Xoshiro(1234), 10_000, 2))
    sizesA = estimate_boxsizes(A; z = -2)
    qs = [0.98, 0.995]
    estimators = [:mm, :pwm, :exp]

    @testset "q=$(q)" for q in qs
        @testset "est=$(estimator)" for est in estimators
            Δloc, θ = extremevaltheory_dims_persistences(A, q;
                compute_persistence = false, show_progress = false,
                estimator = est
            )
            avedim = mean(Δloc)
            @test 1.9 < avedim < 2.1
            @test any(>(2), Δloc)
        end
    end

    # TODO: These tests need to be made legitimate!
    @testset "pvalues" begin
        pvalues = extremevaltheory_gpdfit_pvalues(A, 0.95)[1]
        @test all(p -> 0 ≤ p ≤ 1, pvalues)
    end
end

@testset "persistence analytic" begin
    x = rand(100001)
    y = [max(x[i],x[i+1]) for i in 1:length(x)-1]
    p = 0.98
    θ = extremal_index_sueveges(y, p)
    @test mean(θ) ≈ 0.5 atol = 1e-2
end