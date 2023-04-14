using Test
using FractalDimensions

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "FractalDimensions" begin
    testfile("entropydim.jl")
    testfile("correlationdim.jl")
    testfile("roughness.jl")
    testfile("misc.jl")
    testfile("extremevalue.jl")
end