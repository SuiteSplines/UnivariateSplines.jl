using Test

tests = [
    "base",
    "bsplinebasisfuns",
    "bsplines",
    "dersbsplinebasisfuns",
    "gaussquad",
    "gengaussquad",
    "interpolation",
    "iterators",
    "mappings",
    "quasiinterpolation",
    "refinement",
    "spaces",
    "weightedquad"
]

@testset "UnivariateSplines" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        println("$fp ...")
        include(fp)
    end
end # @testset
