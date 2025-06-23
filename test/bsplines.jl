# SafeTestsets does not support macros. Hence, here we
# use a module to create a safe environment
module BsplineEvaluationTest

using Test 

using IgaBase
using UnivariateSplines
using LinearAlgebra: norm

space = SplineSpace(2, IncreasingVector([0.0,1.0,3.0,4.0,5.5,6.0]), [3,1,1,2,1,3])

spline = Bspline(space)
x = grevillepoints(space)

@testset "splinespace supports" begin
    x = grevillepoints(space)
    @test UnivariateSplines.update!(spline.cache, x) == true
    @test UnivariateSplines.update!(spline.cache, x) == false
    @test UnivariateSplines.update!(spline.cache, IncreasingRange(1.0,3.0,3)) == true
end

@testset "Bspline contruction" begin
    @test dimension(spline) == 1
    @test codimension(spline) == (1,1)
end

@testset "Inplace evaluation" begin
    spline.coeffs .= 1.0
    @evaluate y = spline(x)
    @test all(y.≈1.0)

    @evaluate! y += spline(x)
    @test all(isapprox.(y, 2.0, atol=1e-15))

    @evaluate! y -= spline(x)
    @test all(isapprox.(y, 1.0, atol=1e-15))
end

@testset "Consistent 1d-interpolation" begin
    # initialize
    space = SplineSpace(2, IncreasingVector([0.0,1.0,3.0,4.0,5.5,6.0]), [3,1,1,2,1,3])
    s = Bspline(space)

    # project a smooth function
    g = ScalarFunction(x -> sin(x))
    project!(s, onto=g, method=Interpolation)
    x = grevillepoints(space)
    @evaluate y = g(x)
    @evaluate! y -= s(x)

    # test residual at the interpolation nodes is zero
    @test isapprox(norm(y), 0.0, atol=1e-12)

    # project another spline function
    sref = Bspline(refine(space, kRefinement(2,3)))
    project!(sref, onto=s, method=Interpolation)
    x = grevillepoints(sref.space)
    @evaluate y = s(x)
    @evaluate! y -= sref(x)
    @test isapprox(norm(y), 0.0, atol=1e-12)
end

end # module
