using Test, SafeTestsets

@safetestset "Generalized Gaussian rules" begin

    using IgaBase
    using UnivariateSplines, LinearAlgebra

    C = [(3,0), (4,0), (4,1), (5,1), (6,1), (7,2), (8,2), (9,3), (10,3)]
    for (q,r) in C

        for e in 1:3:50
            U  = KnotVector([0.0:1.0:e...], [q+1, fill(1,e-1)..., q+1])

            # compute analytical integrals
            n = dimsplinespace(q, U)
            target = zeros(n,1)
            for k=1:n
                target[k] = (U[k+q+1] - U[k]) / (q+1)
            end

            # compute numerical integrals
            Q = GeneralizedGaussrule(q, r, e, 0.0, Float64(e))
            B = bspline_interpolation_matrix(q, U, Q.x, 1)[1]
            err = norm(target - (B' * Q.w)) / n
            @test isapprox(err, 0.0, atol=1e-15)
        end
    end

end
