using Test
using FractalDimensions

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "FractalDimensions" begin
    include("entropydim.jl")
    include("correlationdim.jl")
    include("roughness.jl")
    include("misc.jl")
end