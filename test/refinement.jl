using Test, SafeTestsets

@safetestset "Refinement" begin

    using IgaBase
    using UnivariateSplines, SparseArrays
    import SortedSequences: Unique

    @testset "bezierdecomposition" begin
        p = Degree(3)
        kts = KnotVector([0.0,1.0,2.0,3.0,4], [4,1,1,1,4])
        n = dimsplinespace(p, kts)
        m = length(Unique(kts))-1
        C = bezier_extraction_operator(p, kts)

        # test dimensions
        @test size(C,1)==p+1
        @test size(C,2)==p+1
        @test size(C,3)==m

        # test values element 1
        @test C[1,1,1] ≈ 1
        @test C[2,1,1] ≈ 0
        @test C[3,1,1] ≈ 0
        @test C[4,1,1] ≈ 0
        @test C[1,2,1] ≈ 0
        @test C[2,2,1] ≈ 1
        @test C[3,2,1] ≈ 1/2
        @test C[4,2,1] ≈ 1/4
        @test C[1,3,1] ≈ 0
        @test C[2,3,1] ≈ 0
        @test C[3,3,1] ≈ 1/2
        @test C[4,3,1] ≈ 7/12
        @test C[1,4,1] ≈ 0
        @test C[2,4,1] ≈ 0
        @test C[3,4,1] ≈ 0
        @test C[4,4,1] ≈ 1/6

        # test values element 2
        @test C[1,1,2] ≈ 1/4
        @test C[2,1,2] ≈ 0
        @test C[3,1,2] ≈ 0
        @test C[4,1,2] ≈ 0
        @test C[1,2,2] ≈ 7/12
        @test C[2,2,2] ≈ 2/3
        @test C[3,2,2] ≈ 1/3
        @test C[4,2,2] ≈ 1/6
        @test C[1,3,2] ≈ 1/6
        @test C[2,3,2] ≈ 1/3
        @test C[3,3,2] ≈ 2/3
        @test C[4,3,2] ≈ 2/3
        @test C[1,4,2] ≈ 0
        @test C[2,4,2] ≈ 0
        @test C[3,4,2] ≈ 0
        @test C[4,4,2] ≈ 1/6

        # test values element 3
        @test C[1,1,3] ≈ 1/6
        @test C[2,1,3] ≈ 0
        @test C[3,1,3] ≈ 0
        @test C[4,1,3] ≈ 0
        @test C[1,2,3] ≈ 2/3
        @test C[2,2,3] ≈ 2/3
        @test C[3,2,3] ≈ 1/3
        @test C[4,2,3] ≈ 1/6
        @test C[1,3,3] ≈ 1/6
        @test C[2,3,3] ≈ 1/3
        @test C[3,3,3] ≈ 2/3
        @test C[4,3,3] ≈ 7/12
        @test C[1,4,3] ≈ 0
        @test C[2,4,3] ≈ 0
        @test C[3,4,3] ≈ 0
        @test C[4,4,3] ≈ 1/4

        # test values element 4
        @test C[1,1,4] ≈ 1/6
        @test C[2,1,4] ≈ 0
        @test C[3,1,4] ≈ 0
        @test C[4,1,4] ≈ 0
        @test C[1,2,4] ≈ 7/12
        @test C[2,2,4] ≈ 1/2
        @test C[3,2,4] ≈ 0
        @test C[4,2,4] ≈ 0
        @test C[1,3,4] ≈ 1/4
        @test C[2,3,4] ≈ 1/2
        @test C[3,3,4] ≈ 1
        @test C[4,3,4] ≈ 0
        @test C[1,4,4] ≈ 0
        @test C[2,4,4] ≈ 0
        @test C[3,4,4] ≈ 0
        @test C[4,4,4] ≈ 1
    end

    @testset "knotinsertion" begin
        p = Degree(2)
        U = KnotVector([0.0,1.0,2.0,3.0,4], [3,1,1,2,3])
        C = h_refinement_operator!(p, U, [0.5, 1.5, 2.5, 3.5])

        # test knot vector
        @test U == construct_vector([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], [3,1,1,1,1,1,2,1,3])

        # test dimensions subdivision matrix
        @test size(C,1) == 11
        @test size(C,2) == 7
        @test nnz(C) == 19

        # test values
        @test C[1,1] ≈ 1
        @test C[2,1] ≈ 1/2
        @test C[2,2] ≈ 1/2
        @test C[3,2] ≈ 3/4
        @test C[4,2] ≈ 1/4
        @test C[3,3] ≈ 1/4
        @test C[4,3] ≈ 3/4
        @test C[5,3] ≈ 3/4
        @test C[6,3] ≈ 1/4
        @test C[5,4] ≈ 1/4
        @test C[6,4] ≈ 3/4
        @test C[7,4] ≈ 1/2
        @test C[7,5] ≈ 1/2
        @test C[8,5] ≈ 1
        @test C[9,5] ≈ 1/2
        @test C[9,6] ≈ 1/2
        @test C[10,6] ≈ 1/2
        @test C[10,7] ≈ 1/2
        @test C[11,7] ≈ 1
    end

    @testset "two-scale relation" begin
        # B-spline space for checking 2-scale relation operator
        p = Degree(2)
        U = KnotVector([0.0,1.0,2.0,3.0,4], [3,1,1,2,3])
        Δp, k = 2, 2 # increase degree by Δp and add k new knots to U

        # brute force computation of two-scale-operator for comparison
        u, m = deconstruct_vector(U)
        q = p + Δp
        V = global_insert(KnotVector(u, m.+Δp), k)
        x = grevillepoints(q, V)
        B = bspline_interpolation_matrix(p, U, x, 1)[1]
        A = bspline_interpolation_matrix(q, V, x, 1)[1]
        C = A \ Matrix(B)

        # verify result 2-scale relation
        D,  q, V = refinement_operator(p, U, kRefinement(k, Δp))
        Qp, q, V = refinement_operator(p, U, pRefinement(Δp))
        Qh, q, V = refinement_operator(q, V, hRefinement(k))
        E = Qh * Qp

        for i in 1:length(D)
            @test isapprox(D[i], C[i]; atol=1e-12) # check every value
            @test isapprox(E[i], C[i]; atol=1e-12) # check every value
        end
    end
end
