using Test, SafeTestsets

@safetestset "dersbsplinebasisfuns" begin

    using UnivariateSplines

    v = dersbsplinebasisfuns(2, KnotVector([0.0,0.0,0.0,1.0,2.0,2.0,2.0]), 4, 1.5, 3)
    @test v[1,1] == 0.125
    @test v[2,1] == 0.625
    @test v[3,1] == 0.25
    @test v[1,2] == -0.5
    @test v[2,2] == -0.5
    @test v[3,2] == 1.0
    @test v[1,3] == 1.0
    @test v[2,3] == -3.0
    @test v[3,3] == 2.0
end
