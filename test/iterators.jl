using Test, SafeTestsets

@safetestset "Iterators" begin

using UnivariateSplines

@testset "B-spline element supports iterator" begin
    S = SplineSpace(2, [0.0,1.0,3.0,4.0,5.5,6.0], [3,1,1,2,1,3])
    @test dimsplinespace(S) == 8
  
    α = BsplineSupport(S)
    @test length(α) == 8
    @test eltype(α) == UnitRange{Int}
    @test size(α) == (8,)
    @test size(α,1) == 8


    @test α[1] == 1:1
    @test α[2] == 1:2
    @test α[3] == 1:3
    @test α[4] == 2:3
    @test α[5] == 3:4
    @test α[6] == 4:5
    @test α[7] == 4:5
    @test α[8] == 5:5
end

@testset "spline element supports iterator with constrained functions" begin
    S = SplineSpace(2, [0.0,1.0,3.0,4.0,5.5,6.0], [3,1,1,2,1,3]; cleft=2:2, cright=1:1)
    @test dimsplinespace(S) == 6

    α = Support(S)
    @test length(α) == 6
    @test eltype(α) == Vector{Int}
    @test size(α) == (6,)
    @test size(α,1) == 6

    @test α[1] == [1,2]
    @test α[2] == [1,2,3]
    @test α[3] == [2,3]
    @test α[4] == [3,4]
    @test α[5] == [4,5]
    @test α[6] == [4,5]
end

@testset "spline element supports iterator with periodic constrained functions" begin
    S = SplineSpace(2, [0.0,1.0,3.0,4.0,5.5,6.0], [3,1,1,2,1,3]; cperiodic=1:2)
    @test dimsplinespace(S) == 6

    α = collect(Support(S))
    @test α[1] == [1,4,5]
    @test α[2] == [1,2,5]
    @test α[3] == [1,2,3]
    @test α[4] == [2,3]
    @test α[5] == [3,4]
    @test α[6] == [4,5]
end

@testset "Span indices iterator" begin
    S = SplineSpace(2, [0.0,1.0,2.0,3.0,4.0], [3,1,2,1,3])

    α =  collect(SpanIndex(S))
    @test length(α) == 4
    @test α[1] == 3
    @test α[2] == 4
    @test α[3] == 6
    @test α[4] == 7
end

@testset "Span indices iterator" begin
    S = SplineSpace(2, [0.0,1.0,2.0,3.0,4.0], [3,1,2,1,3])
    
    α = collect(SpanIndices(S))
    @test length(α) == 4
    @test α[1] == 1:3
    @test α[2] == 2:4
    @test α[3] == 4:6
    @test α[4] == 5:7

end

end # safetestset
