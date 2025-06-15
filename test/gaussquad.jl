using Test, SafeTestsets

@safetestset "Gauss quadrature" begin

using UnivariateSplines, LinearAlgebra

@testset "Gauss rules" begin

    @test_throws AssertionError GaussRule{Legendre}(IncreasingVector([-1.0,0.0,1.0]),[1.0,1.0])

    @test length(GaussRule(3))==3
    @test length(GaussRule(Legendre, 3))==3
    @test length(GaussRule(Lobatto, 3))==3

    Q = GaussRule(Lobatto, 3, 0.0, 3.0)
    @test Q.x ≈ [0.0,1.5,3.0]
    @test sum(Q.w) ≈ 3.0

    # A Gauss Legendre rule of n points can exactly integrate polynomials
    # up to degree 2n-1
    Q = GaussRule(Legendre, 3, 0.0, 2.0)
    f(x,p) = x.^p
    for k in 0:5
        @test dot(Q.w, f(Q.x, k)) ≈ 2.0^(k+1) / (k+1)
    end
    @test !(dot(Q.w, f(Q.x, 6)) ≈ 2.0^(7) / 7)

    # A Gauss Lobatto rule of n points can exactly integrate polynomials
    # up to degree 2n-3
    Q = GaussRule(Lobatto, 3, 0.0, 2.0)
    f(x,p) = x.^p
    for k in 0:3
        @test dot(Q.w, f(Q.x, k)) ≈ 2.0^(k+1) / (k+1)
    end
    @test !(dot(Q.w, f(Q.x, 4)) ≈ 2.0^(5) / 5)

    # Quadrature rules can be copied
    Q = GaussRule(Lobatto, 3)
    Qcopy = copy(Q)
    Qcopy.w[1] = 5.0
    @test !(Qcopy.w ≈ Q.w)

    # quadrature rules can be affinely mapped to a new interval
    Q = affine_transform!(GaussRule(Lobatto, 3), Interval(-1.0,1.0),  Interval(0.0,3.0))
    @test Q.x ≈ [0.0,1.5,3.0]
    @test sum(Q.w) ≈ 3.0
end

@testset "Global Gaussian rules" begin
    # test polynomial reproduction
    f(x,p) = x.^p

    Q = PatchRule(IncreasingVector([0.0,0.5,2.0]); npoints=3, method=Legendre)
    for k in 0:5
        @test dot(Q.w[:], f(Q.x[:], k)) ≈ 2.0^(k+1) / (k+1)
    end
    @test !(dot(Q.w[:], f(Q.x[:], 6)) ≈ 2.0^(7) / 7)

    Q = PatchRule(IncreasingVector([0.0,0.5,2.0]); npoints=3, method=Lobatto)
    for k in 0:3
        @test dot(Q.w[:], f(Q.x[:], k)) ≈ 2.0^(k+1) / (k+1)
    end
    @test !(dot(Q.w[:], f(Q.x[:], 4)) ≈ 2.0^(5) / 5)
end

@testset "Convenience constructors" begin
    S = SplineSpace(2, IncreasingVector([0.0,0.5,2.0]), [3,1,3])
    Q₁ = PatchRule(S; method=Legendre, npoints=3)
    Q₂ = PatchRule(S; method=Legendre)
    Q₃ = PatchRule(S)
    @test Q₁.x == Q₂.x == Q₃.x
    @test Q₁.w == Q₂.w == Q₃.w

    Q = PatchRule(S; method=Legendre, npoints=3)
    Q₁ = GaussRule(Legendre, 3, 0.0, 0.5)
    Q₂ = GaussRule(Legendre, 3, 0.5, 2.0)
    @test GaussRule(Q,1).x == Q₁.x && GaussRule(Q,1).w == Q₁.w
    @test GaussRule(Q,2).x == Q₂.x && GaussRule(Q,2).w == Q₂.w

end

end # module
