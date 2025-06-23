using Test, SafeTestsets

@safetestset "Interpolation" begin

    using IgaBase
    using UnivariateSplines

    p = Degree(2)
    U = KnotVector([0.0,1.0,1.5,2.5,3.0], [3,1,1,2,3])
    n = dimsplinespace(p, U)
    x = grevillepoints(p, U)

    # test the b-spline interpolation matrices
    B = bspline_interpolation_matrix(p, U, x, p+1)
    @test length(B) == p+1
    for i in 1:p+1
        @test size(B[i]) == (n,n+1-i)
        @test sum(B[i], dims=(2,)) ≈ ones(n)
    end

    # test polynomial reproduction
    y = global_insert(x, 4)
    C = bspline_interpolation_matrix(p, U, y, p+1)
    for i in 1:p+1
        for k in 0:p+1-i
            α = B[i] \ x.^k         # perform projection
            @test C[i] * α ≈ y.^k   # test polynomial reproduction on refined partition
        end
    end

    # test the b-spline derivatives interpolation matrices
    B = ders_bspline_interpolation_matrix(p, U, x, p+1)
    @test length(B) == p+1
    @test size(B[1]) == (n,n)
    @test size(B[2]) == (n,n)
    @test size(B[3]) == (n,n)
    @test sum(B[1], dims=(2,)) ≈ ones(n)
    @test sum(B[2], dims=(2,)) ≈ zeros(n)

end
