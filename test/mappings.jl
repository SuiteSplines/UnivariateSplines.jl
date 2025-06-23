# SafeTestsets does not support macros. Hence, here we
# use a module to create a safe environment
module BsplineMappingsTest

using Test
using IgaBase
using UnivariateSplines

space = SplineSpace(Degree(2), Interval(0.0,1.0), 4)
F = GeometricMapping(Bspline, space; codimension=2)

@testset "Mapping contruction" begin
    @test dimension(F) == 1
    @test codimension(F) == (1,2)
end

x = grevillepoints(space)

@testset "Mapping partition of unity" begin
    F[1].coeffs .= F[2].coeffs .= 1.0
    @evaluate y = F(x)
    
    @test y[1] ≈ [1 1] && y[2] ≈ [1 1] && y[3] ≈ [1 1]
end

@testset "Mapping inplace operations" begin
    @evaluate y = F(x)
    @evaluate! y += F(x)
    @test y[1] ≈ [2 2] && y[2] ≈ [2 2] && y[3] ≈ [2 2]

    @evaluate! y -= F(x)
    @test y[1] ≈ [1 1] && y[2] ≈ [1 1] && y[3] ≈ [1 1]
end

@testset "gradient dim==1, codim==3" begin

    # construct spline
    space = SplineSpace(Degree(2), Interval(0.0,1.0), 4)
    s = GeometricMapping(Bspline, space; codimension=3)
    ∇s = Gradient(s)

    # test gradient of Field
    @test dimension(∇s) == 1
    @test codimension(∇s) == (1,3)
    @test ∇s[1] isa Bspline && ∇s[1].ders == 1
    @test ∇s[2] isa Bspline && ∇s[2].ders == 1
    @test ∇s[3] isa Bspline && ∇s[3].ders == 1

    # perform projection and test accuracy w.r.t analytical solution
    g = GeometricMapping(Interval(0.0,1.0), x -> x^2 + x, x -> -x^2 -2 + x, x -> x + 3 )
    ∇g = Gradient(g)

    project!(s, onto=g, method=Interpolation)
    x = grevillepoints(s[1].space)

    # perform evaluation of Gradient
    @evaluate z = ∇s(x)

    # commpare with exact function
    @evaluate y = ∇g(x)
    @test y ≈ z

    @evaluate! y -= ∇s(x)
    @test isapprox(y[1], [0.0 0.0 0.0], atol=1e-12) && isapprox(y[4], [0.0 0.0 0.0], atol=1e-12)
end

end # module
