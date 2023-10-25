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
        estimator_evt = Exceedances(0.95, :exp)
        Δloc, θ = extremevaltheory_dims_persistences(A, estimator_evt;
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
        estimator_evt = Exceedances(0.95, :exp)
        D = extremevaltheory_dim(A, estimator_evt;
        show_progress = false)
        @test 0.9 < D < 1.1
    end

end

@testset "Random 2D" begin
    A = StateSpaceSet(rand(Xoshiro(1234), 10_000, 2))
    sizesA = estimate_boxsizes(A; z = -2)
    qs = [0.98, 0.995]
    estimators = [:mm, :pwm, :exp]

    @testset "q=$(q)" for q in qs
        @testset "est=$(est)" for est in estimators
            evt_estimator = Exceedances(q, est)
            Δloc, θ = extremevaltheory_dims_persistences(A, evt_estimator;
                compute_persistence = false, show_progress = false,
                )
            avedim = mean(Δloc)
            if est == :pwm
                @test 1.9 < avedim < 2.2
            else
                @test 1.9 < avedim < 2.1
                @test any(>(2), Δloc)
            end          
        end

        @testset "block maxima" begin    
            evt_estimator = BlockMaxima(100, q)
            Δloc, θ = extremevaltheory_dims_persistences(A, evt_estimator;
                compute_persistence = false, show_progress = false,
                )
            avedim = mean(Δloc)
            @test 1.9 < avedim < 2.1
            @test any(>(2), Δloc)            
        end

    end

    @testset "significance" begin
        Es, nrmses, pvalues, sigmas, xis = extremevaltheory_gpdfit_pvalues(A, 0.99)
        @test all(p -> 0 ≤ p ≤ 1, pvalues)
        badcount = count(<(0.05), pvalues)/length(pvalues)
        @test badcount < 0.1
        @test all(p -> 0 ≤ p ≤ 1, nrmses)
        @test mean(nrmses) < 0.5
        @test all(E -> all(≥(0), E), Es)
        @test all(≥(0), sigmas)
    end
end

@testset "persistence analytic" begin
    x = rand(100001)
    y = [max(x[i],x[i+1]) for i in 1:length(x)-1]
    p = 0.98
    θ = extremal_index_sueveges(y, p)
    @test mean(θ) ≈ 0.5 atol = 1e-2
end