using FractalDimensions, Test

@testset "kaplan yorke" begin
    ls = [1, 0, -1]
    @test kaplanyorke_dim(ls) == 2
    ls = [1, 1, 1]
    @test kaplanyorke_dim(ls) == 3
    ls = -[1, 1, 1]
    @test kaplanyorke_dim(ls) == 0
end