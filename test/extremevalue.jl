using Test, FractalDimensions
using Statistics, DynamicalSystemsBase

@testset "Lorenz63" begin
    @inline @inbounds function lorenz_rule(u, p, t)
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end

    ρ = 28.0
    ds = CoupledODEs(lorenz_rule, [0, 10, 0.0], [10.0, ρ, 8/3])

    tr, tvec = trajectory(ds, 1000; Δt = 0.05, Ttr = 100)
    estimator = :mm
    qs = [0.97, 0.98, 0.99, 0.995]
    N = length(tr)

    @testset "q=$(q)" for q in qs
        Δloc, θ = extremevaltheory_dims_persistences(tr, q;
            compute_persistence = false, estimator
        )
        avedim = mean(Δloc)
        @test 1.95 < avedim < 2.15
    end
end

# %% Analytic test for persistence
@testset "analytic uniform noise" begin
    x = rand(100001)
    y = [max(x[i],x[i+1]) for i in 1:length(x)-1]
    p = 0.98
    θ = extremal_index_sueveges(y, p)
    @test mean(θloc) ≈ 0.5 atol = 1e-2
end