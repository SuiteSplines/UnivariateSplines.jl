using Test, SafeTestsets

@safetestset "Quasi-interpolation" begin

    using UnivariateSplines, LinearAlgebra

    @testset "quasi_interpolation_lyche" begin
        p  = Degree(3)
        U  = KnotVector([0.0,1.0,1.5,2.5,3.0], [p+1,1,1,2,p+1])
        k1 = p+1
        y  = grevillepoints(p, U)

        A = approximate_collocation_inverse(p, U, k1)   # quasi-interpolation operator
        B = bspline_interpolation_matrix(p, U, y, 1)[1] # b-spline collocation

        # test polynomial reproduction (up to degree k1-1)
        for k in 0:k1-1
            x = y.^k
            @test B * A * x ≈ x
        end
    end

    @testset "deboor-fix-dual-functionals p=2" begin
        p = 2
        kts = KnotVector([0,0,0,1,1.5,3.0,3.0,3.0])
        gp = grevillepoints(p, kts)
        B = ders_bspline_interpolation_matrix(p, kts, gp, p+1)
        A  = reshape([Matrix(B[1]) Matrix(B[2]) Matrix(B[3])], size(B[1])..., 3)
        for k in 1:dimsplinespace(p,kts)
            μ = UnivariateSplines.deboor_fix_dual_functional_coeffs(kts[k:k+p+1], gp[k])
            v = A[k,:,:] * μ
            @test v[k] ≈ 1
            @test norm(v) ≈ 1
        end
    end

    @testset "deboor-fix-dual-functionals p=3" begin
        p = 3
        kts = KnotVector([0,0,0,0,1,1.5,3.0,3.0,3.0,3.0])
        gp  = grevillepoints(p, kts)
        B   = ders_bspline_interpolation_matrix(p, kts, gp, p+1)
        A   = reshape([Matrix(B[1]) Matrix(B[2]) Matrix(B[3]) Matrix(B[4])], size(B[1])..., 4) 
        for k in 1:dimsplinespace(p,kts)
            μ = UnivariateSplines.deboor_fix_dual_functional_coeffs(kts[k:k+p+1], gp[k])
            v = A[k,:,:] * μ
            @test v[k] ≈ 1
            @test norm(v) ≈ 1
        end
    end

    @testset "deboor-fix-dual-functionals p=4" begin
        p = 4
        kts = KnotVector([0,0,0,0,0,1,1.5,3.0,3.0,3.0,3.0,3.0])
        gp  = grevillepoints(p, kts)
        B   = ders_bspline_interpolation_matrix(p, kts, gp, p+1)
        A   = reshape([Matrix(B[1]) Matrix(B[2]) Matrix(B[3]) Matrix(B[4]) Matrix(B[5])], size(B[1])..., 5)
        for k in 1:dimsplinespace(p,kts)
            μ = UnivariateSplines.deboor_fix_dual_functional_coeffs(kts[k:k+p+1], gp[k])
            v = A[k,:,:] * μ
            @test v[k] ≈ 1
            @test norm(v) ≈ 1
        end        
    end

    import UnivariateSplines: moment, blossom_label, inverse_mass_scaling

    @testset "quasi_interpolation_l2" begin

        # test generalized Blossom labels
        r = 3
        σ = collect(1.0:1.0:10.0)
        @test blossom_label(Val(0), r, σ) ≈ 1.0
        @test blossom_label(Val(1), r, σ) ≈ 18.0
        @test blossom_label(Val(2), r, σ) ≈ 18.0
        @test blossom_label(Val(3), r, σ) ≈ -54.0
        @test blossom_label(Val(4), r, σ) ≈ 54.0
        @test blossom_label(Val(5), r, σ) ≈ -19440

        p  = Degree(2)
        U  = KnotVector([0.0,1.0,1.5,2.5,3.0], [p+1,1,1,2,p+1])
        k1 = p+1

        # test moments with respect to anchor of function
        m = moment(U[4:3+p+k1], k1)
        @test m[1] ≈ 0
        @test m[2] ≈ 0.540000000000000
        @test m[3] ≈ -0.138000000000000
        @test m[4] ≈ 0.460200000000000
        @test m[5] ≈ -0.215460000000000
        @test m[6] ≈ 0.471570000000000

        # test inverse mass scaling
        u = inverse_mass_scaling(p, p, U)
        @test u[1] ≈ 0.0187500000000000
        @test u[2] ≈ 0.0317708333333333
        @test u[3] ≈ 0.0156250000000000
        @test u[4] ≈ 0.00260416666666667
        @test u[5] ≈ 0.000868055555555556

        # test values of approximate inverse
        S = approximate_l2_inverse(p, U, p+1)
        # diagonal
        @test S[1,1] ≈ 6.20000000000000
        @test S[2,2] ≈ 5.51288888888889
        @test S[3,3] ≈ 2.46720000000000
        @test S[4,4] ≈ 4.77700000000000
        @test S[5,5] ≈ 3.39043209876543
        @test S[6,6] ≈ 18.2222222222222
        @test S[7,7] ≈ 14

        # second diagonal
        @test S[1,2] ≈ -2.61333333333333
        @test S[2,3] ≈ -1.25760000000000
        @test S[3,4] ≈ -1.16640000000000
        @test S[4,5] ≈ -1.21388888888889
        @test S[5,6] ≈ -1.35185185185185
        @test S[6,7] ≈ -8.66666666666667

        # third diagonal
        @test S[1,3] ≈ 0.288000000000000
        @test S[2,4] ≈ 0.325333333333333
        @test S[3,5] ≈ 0.120000000000000
        @test S[4,6] ≈ 0.166666666666667
        @test S[5,7] ≈ 0.222222222222222

        # test symmetry
        @test norm(S - S') ≈ 0.0

        # ToDo: test polynomial reproduction (need quadrature)
    end
end
