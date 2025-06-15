export controlpoints

using RecipesBase

AbstractMappings.visualization_grid(space::SplineSpace; density::Int) = global_insert(breakpoints(space), density)
AbstractMappings.visualization_grid(spline::AbstractSpline{1}; density::Int) = visualization_grid(spline.space, density=density)

controlpoints(spline::AbstractSpline{1}) = spline.coeffs

# plot 'controlpolygon' of a univariate function
@recipe function f(spline::AbstractSpline{1}, ::Val{:controlpolygon})
    return grevillepoints(spline.space), controlpoints(spline)
end

# plot 'controlpolygon' of a univariate function
@recipe function f(spline::AbstractMapping{1,1,2}, ::Val{:controlpolygon})
    return controlpoints(spline[1]), controlpoints(spline[2])
end

@recipe function f(spline::AbstractMapping{1,1,3}, ::Val{:controlpolygon})
    return controlpoints(spline[1]), controlpoints(spline[2]), controlpoints(spline[3])
end

# series recipe 'controlpolygon'
@recipe function f(::Type{Val{:controlpolygon}}, x, y, z)
    primary     --> false
    seriestype  := :path
    linewidth   --> 0.5
    linecolor   --> :grey
    markersize  --> 3
    markercolor --> :white
    markershape --> :circle
end

# plot recipe b-spline 'knots'
@recipe function f(spline::AbstractSpline{1}, ::Val{:knots})
    x = breakpoints(spline.space)
    @evaluate y = spline(x)
    return x, y
end

# plot recipe b-spline 'knots'
@recipe function f(spline::AbstractMapping{1,1,2}, ::Val{:knots})
    @assert spline[2].space == spline[2].space
    x = breakpoints(spline[1].space)
    @evaluate y = spline(x)
    return y.data[1], y.data[2]
end

@recipe function f(spline::AbstractMapping{1,1,3}, ::Val{:knots})
    @assert spline[2].space == spline[2].space == spline[3].space
    x = breakpoints(spline[1].space)
    @evaluate y = spline(x)
    return y.data[1], y.data[2], y.data[3]
end

# series recipe 'knots'
@recipe function f(::Type{Val{:knots}}, x, y, z)
    primary     --> false
    seriestype  := :scatter
    markersize  --> 2
    markercolor --> :black
end

@recipe function f(spline::AbstractSpline{1}, ::Val{:grevillepoints})
    x = grevillepoints(spline.space)
    @evaluate y = spline(x)
    return x, y
end


@recipe function f(::Type{Val{:grevillepoints}}, x, y, z)
    primary     --> false
    seriestype  := :scatter
    markersize  --> 3
    markercolor --> :black
end

@recipe function f(spline::Bspline, ::Val{:space})
    return spline.space
end

@recipe function f(space::SplineSpace; density::Int=20)
    x = visualization_grid(space, density=density)
    y = bspline_interpolation_matrix(space, x, 1)[1]
    return x, y
end

@recipe function f(::Type{Val{:space}}, x, y, z)
    label       --> ""
    xguide      --> "x"
    yguide      --> "y"
    linewidth   --> 1
    seriestype  :=  :path
end

##region Plotting of quadrature rules
@recipe function f(Q::AbstractQuadrule; seriestype=:qpoints)
    return Q, Val(seriestype)
end

@recipe function f(Q::AbstractQuadrule{1}, ::Val{:qpoints})
    return Q.x, zeros(length(Q.x))
end

@recipe function f(Q::AbstractQuadrule{1}, ::Val{:qweights})
    return Q.x, Q.w
end

# series recipe 'quadrature-points'
@recipe function f(::Type{Val{:qpoints}}, x, y, z)
    label       --> "quadrature rule"
    seriestype  := :scatter
    markershape --> :circle
    markercolor --> :black
    markersize  --> 2
end

# series recipe 'quadrature-weights'
@recipe function f(::Type{Val{:qweights}}, x, y, z)
    label       --> ""
    seriestype  :=  :stem
    linewidth   --> 1
    linecolor   --> :black
    marker      --> :circle
    markersize  --> 3
    markercolor --> :black
end
