using Test, SafeTestsets

@safetestset "Weighted quadrature" begin

    using UnivariateSplines

    using UnivariateSplines, SortedSequences, LinearAlgebra
    using UnivariateSplines: unit_integral_rescaling, table_required_points, nquadpoints
    using UnivariateSplines: distribute_points, sparse_difference_matrix, system_matrix

    # Several tests are based on the examples in:
    # Hiemstra, René R., et al. "Fast formation and assembly of finite element
    # matrices with application to isogeometric linear elasticity." Computer Methods
    # in Applied Mechanics and Engineering 355 (2019): 234-260.

    # test unit-preserving rescaling of B-spline basis functions
    p = Degree(2)
    u = IncreasingVector([0.0,1.0,2.0,3.0])
    m = [p+1,1,2,p+1]
    U = KnotVector(u,m)
    Q = PatchRule(u; npoints=p+1, method=Legendre)
    B = bspline_interpolation_matrix(p, U, Q.x, 1)[1]

    @test dot(Q.w, unit_integral_rescaling(p, U, 1) * B[:,1]) ≈ 1.0
    @test Q.w' * B .* unit_integral_rescaling(p, U)' ≈ ones(1,dimsplinespace(p,U))

    # compute table of required points
    S = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,2,2,3])
    V = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,1,1,3])
    A = table_required_points(S, V)
    @test A[1,1] == 3; @test A[1,2] == 5;
    @test A[2,1] == 2; @test A[2,2] == 5
    @test A[3,1] == 3

    nq = nquadpoints(S,V)
    @test nq[1] == 3
    @test nq[2] == 2
    @test nq[3] == 3

    X = distribute_points(S, V)
    @test X[1] ≈ 1/6
    @test X[2] ≈ 0.5
    @test X[3] ≈ 5/6
    @test X[4] ≈ 1.25
    @test X[5] ≈ 1.75
    @test X[6] ≈ 13/6
    @test X[7] ≈ 2.5
    @test X[8] ≈ 17/6

    # Compute and check weighted quadrature rule
    B = bspline_interpolation_matrix(Degree(S), KnotVector(S), X, 1)[1] # target space
    M = system_matrix(S,V)
    Q = WeightedQuadrule(S,V)
    @test Q.w' * B ≈ M

    M = system_matrix(S,V,1,2)
    Q = WeightedQuadrule(S, differentiate(V), X; gradient=true)
    @test Q.w' * B ≈ M

end # safetestset


@safetestset "Weighted quadrature accuracy" begin

    using Test
    using UnivariateSplines

    using UnivariateSplines, SortedSequences, LinearAlgebra
    using UnivariateSplines: unit_integral_rescaling, table_required_points, nquadpoints
    using UnivariateSplines: distribute_points, sparse_difference_matrix, system_matrix

    for m=[2,3,5,8,13,21]

        partition = IncreasingRange(Interval(0.0,1.0), m)

        for p=2:10

            # define trial and testspace
            S = SplineSpace(p, partition)
            V = SplineSpace(p, partition)

            # check quadrature rule in integrating entries of the mass matrix
            x = distribute_points(S, differentiate(V))

            Q = (
                    WeightedQuadrule(S, V, x), 
                    WeightedQuadrule(S, differentiate(V), x; gradient=true)
                )

            # evaluate trial functions at the quadrature points
            B = bspline_interpolation_matrix(S, x, 1)[1]

            # system matrices
            M = system_matrix(S,V) # mass matrix
            A = system_matrix(S,V,1,2) # matrix with derivatives on the test functions

            # check quadrature rule with a derivative on the test function
            @test Q[1].w' * B ≈ M
            @test Q[2].w' * B ≈ A

        end
    end

end # safetestset