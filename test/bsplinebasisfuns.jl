using Test, SafeTestsets

@safetestset "B-spline basis functions" begin

    using UnivariateSplines

    # test evaluation of a single basis function
    @test onebasisfuneval(KnotVector([0.0,1.0,2.0,3.0]), 0.0) == 0.0
    @test onebasisfuneval(KnotVector([0.0,1.0,2.0,3.0]), 1.0) == 0.5
    @test onebasisfuneval(KnotVector([0.0,1.0,2.0,3.0]), 1.5) == 3/4
    @test onebasisfuneval(KnotVector([0.0,1.0,2.0,3.0]), 2.0) == 0.5
    @test onebasisfuneval(KnotVector([0.0,1.0,2.0,3.0]), 3.0) == 0.0

    # test B-spline evaluation using the Cox-DeBoor algorithm
    p = Degree(2)
    U = KnotVector([0.0,0.0,0.0,1.0,2.0,2.0,2.0])
    u = 1.5
    span = findspan(p, U, u)
    B = bsplinebasisfuns(p, U, span, u, p+1)

    # test values
    @test B[1,3] == 1.0
    @test B[1,2] == 0.5
    @test B[2,2] == 0.5
    @test B[1,1] == 1/8
    @test B[2,1] == 5/8
    @test B[3,1] == 1/4

    # test supports
    @test B[2,3] == 1.0
    @test B[3,3] == 2.0
    @test B[3,2] == 1.0

    # test B-spline evaluation at a set of points
    # check Partition of unity at Greville points
    g = grevillepoints(p, U)
    span = findspan(p, U, g)
    B = bsplinebasisfuns(p, U, span, g, p+1)
    for i in 1:p+1
        @test sum(B[1:p+2-i,:,i], dims=(1,)) ≈ ones(1,dimsplinespace(p,U))
    end
end
