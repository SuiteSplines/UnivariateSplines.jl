using Test, SafeTestsets

@safetestset "Spline basic types" begin

    using UnivariateSplines

    # test initialization of Degree and KnotVector
    @test Degree==Integer
end

@safetestset "Spline basic functionality" begin

    using LinearAlgebra
    using SortedSequences, UnivariateSplines

    # initialize
    p = Degree(2)
    u = IncreasingVector([0.0,1.0,2.0])
    m = [3,1,3]
    U = KnotVector(u,m)

    @testset "Find span indices" begin
        @test findspan(p, U, 0.0) == 3
        @test findspan(p, U, 2.0) == 4
        @test findspan(p, U, 0.5) == 3
        @test findspan(p, U, 1.5) == 4
    end

    @testset "Greville-points" begin
        # test computation of greville abcissea
        @test grevillepoints(p, U) == [0.0,0.5,1.5,2.0]
    end

    # test dimension of spline space
    @testset "Dimensions" begin
        num_elements(U) == 2
        @test dimsplinespace(p, U) == 4
    end

    # test unit-preserving rescaling of B-spline basis functions
    @testset "Normalization weights" begin
        Q = PatchRule(u; npoints=p+1, method=Legendre)
        B = bspline_interpolation_matrix(p, U, Q.x, 1)[1]

        @test dot(Q.w, unit_integral_rescaling(p, U, 1) * B[:,1]) ≈ 1
        @test Q.w' * B .* unit_integral_rescaling(p, U)' ≈ ones(1,dimsplinespace(p,U))
    end
end
