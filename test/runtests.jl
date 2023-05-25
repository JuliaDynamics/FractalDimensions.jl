using Test
using FractalDimensions

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
ENV["FRACTALDIMENSIONS_PROGRESS"] = false

@testset "FractalDimensions" begin
    testfile("slope_fits.jl")
    testfile("entropydim.jl")
    testfile("correlationdim.jl")
    testfile("roughness.jl")
    testfile("misc.jl")
    testfile("extremevalue.jl")
end