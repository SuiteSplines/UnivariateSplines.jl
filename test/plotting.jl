using Test

using UnivariateSplines
using Plots

@testset "Project and plot a spline function" begin
    # create spline function
    space = SplineSpace(Degree(2), Interval(0.0,2*π), Dimension(10))
    bspline = Bspline(space)

    # project function onto spline space
    g = ScalarFunction(x -> sin.(x.^2 + x))
    project!(bspline, onto=g, method=Interpolation)

    # plot b-spline function, its control points and knots
    h1 = plot(bspline; density=30, label="B-spline interpolant of g(x)")
    plot!(bspline; seriestype=:controlpolygon)
    plot!(bspline; seriestype=:knots)
    plot!(bspline; seriestype=:grevillepoints)

    # plot B-spline interpolant
    x = IncreasingRange(0.0,2*π,200)
    plot!(x, g(x); label="g(x)")

    # plot B-spline space
    h2 = plot(bspline; seriestype=:space)
    plot(h1, h2, layout=(2,1))
end

@testset "Plot a spline space and quadrature rule" begin
    space = SplineSpace(Degree(2), Interval(0.0,5.0), Dimension(5))
    Q = PatchRule(space; method=Legendre, npoints=3)
    h1 = plot(space)
    h2 = plot(Q; seriestype=:qpoints, label="Gauss-rule")
    plot!(Q; seriestype=:qweights, ylim=[0,1], xlabel="x", ylabel="y")
    plot(h1, h2, layout=(2,1))
end

@testset "Plot a 2D or 3D spatial spline curve" begin 
    space = SplineSpace(3, [0.0,1.0,2.0,3.0,4.0,5.0], [4,3,3,3,3,4])
    curve = GeometricMapping(Bspline, space; codimension=2)

    # Manualy set the coefficient values
    curve[1].coeffs .= Float64[  152,151,150,
                            150,110,50,
                            10,90,130,
                            140,113,70,
                            45,70,110,
                            152]
    curve[2].coeffs .= Float64[  14,13,12,
                            10,55,15,
                            95,100,65,
                            27,50,44,
                            75,35,53,
                            14]
    # plot the curve
    plot(curve; density=300,
        label="",
        linewidth=3,
        background=RGB(0.2, 0.2, 0.2),
        linecolor=:white,
        axis = :off,
        aspect_ratio=:equal,
        ticks=false)

    # plot the controlpolygon and the knots

    plot!(curve; seriestype=:controlpolygon)
    plot!(curve; seriestype=:knots, markercolor=:black, markersize=4)
end