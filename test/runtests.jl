using Test
using FractalDimension

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "FractalDimension" begin
    # include("multiscale/multiscale.jl")
end