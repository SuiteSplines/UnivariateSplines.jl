using Test, SafeTestsets

@safetestset "Spline spaces functionality" begin

using UnivariateSplines
using SparseArrays, LinearAlgebra

@testset "splinespace" begin
    
    # test sub and super spaces
    S = SplineSpace(2, Interval(0.0,3.0), 3)
    @test Degree(S) == 2
    @test KnotVector(S) == [0.0,0.0,0.0,1.0,2.0,3.0,3.0,3.0]

    # test h-refinement
    V = refine(S, hRefinement(1))
    u, m = deconstruct_vector(KnotVector(V))
    @test Degree(V) == 2
    @test u == [0.0,0.5,1.0,1.5,2.0,2.5,3.0] && m == [3,1,1,1,1,1,3]

    # test p-refinement
    V = refine(S, pRefinement(2))
    u, m = deconstruct_vector(KnotVector(V))
    @test Degree(V) == 4
    @test u == [0.0,1.0,2.0,3.0] && m == [5,3,3,5]

    # test hp-refinement
    V = refine(S, hpRefinement(1,2))
    u, m = deconstruct_vector(KnotVector(V))
    @test Degree(V) == 4
    @test u == [0.0,0.5,1.0,1.5,2.0,2.5,3.0] && m == [5,3,3,3,3,3,5]

    # test k-refinement
    V = refine(S, kRefinement(1,2))
    u, m = deconstruct_vector(KnotVector(V))
    @test Degree(V) == 4
    @test u == [0.0,0.5,1.0,1.5,2.0,2.5,3.0] && m == [5,1,3,1,3,1,5]

    # subspace of gradients
    V = differentiate(S)
    @test V.p==S.p-1
    @test dimsplinespace(V) == dimsplinespace(S)-1

    # superspace of functions that are less smooth
    V = roughen(S)
    @test V.p==S.p
    u1, m1 = deconstruct_vector(S.U)
    u2, m2 = deconstruct_vector(V.U)
    @test m1[1]==m2[1]
    @test m1[2:end-1].+1==m2[2:end-1]
    @test m1[end]==m2[end]
end

import UnivariateSplines: extraction_operator, left_boundary_constraints, right_boundary_constraints, periodic_boundary_constraints

@testset "Space constraints" begin
    p = Degree(3)
    U = KnotVector(IncreasingVector([0.0,1.5,2.0,3.0,5.5,6.0]), [p+1,1,1,2,1,p+1])

    constraints = [1,2]
    A = left_boundary_constraints(p, U; con=constraints)
    C = extraction_operator(p, U; cleft=[1,2])
    @test size(C,2) == dimsplinespace(p, U) - 2
    @test isapprox(norm(A * C), 0.0; atol=1e-14)
    @test isposdef(C' * C)

    constraints = [1,2]
    A = right_boundary_constraints(p, U; con=constraints)
    C = extraction_operator(p, U; cright=[1,2])
    @test size(C,2) == dimsplinespace(p, U) - 2
    @test isapprox(norm(A * C), 0.0; atol=1e-14)
    @test isposdef(C' * C)

    constraints = [1,2,3]
    A = periodic_boundary_constraints(p, U; con=constraints)
    C = extraction_operator(p, U; cperiodic=[1,2,3])
    @test size(C,2) == dimsplinespace(p, U) - 3
    @test isapprox(norm(A * C), 0.0; atol=1e-14)
    @test isposdef(C' * C)
end

@testset "Subspaces" begin
    S = SplineSpace(3, 10; cperiodic=[1:3...], cleft=[2])
    @test dimsplinespace(S) == dimsplinespace(S.p, S.U) - 4
    @test isposdef(S.C' * S.C) # check extraction operator is of full rank
    @test sum(S.C; dims=2) ≈ ones(dimsplinespace(S.p, S.U))
end

@testset "Basis" begin

    # initialize
    p = Degree(2)
    u = IncreasingVector([0.0,1.0,3.0,4.0,5.5,6.0])
    m = [3,1,1,2,1,3]
    U = KnotVector(u,m)
    S = SplineSpace(p, U)

    # global interpolation matrix
    x = global_insert(u, 2)
    span = findspan(p, U, x)
    B = ders_bspline_interpolation_matrix(p, U, span, x, 3)

    # compute basis
    A = BsplineBasis(S, x, 3)

    @test size(A) == (dimsplinespace(p, U), length(x), 3)
    @test eltype(A) == Float64
    @test length(A) == prod(size(A))
    @test A[1,1,1] == 1.0

    for k in 1:3
        Z = [A[i,j,k] for i in 1:size(A,1), j in 1:size(A,2)]
        @test isapprox(norm(B[k]' - Z), 0.0, atol=1e-15)
    end

    # setindex
    A[1,1,1] = 5.0
    @test A[1,1,1] == 5.0

    # setindex allows only certain values to be set. The sparsity of the original
    # array may not change.
    @test_throws BoundsError A[10,1,1]
    @test A[8,1,1] == 0.0
end

end
