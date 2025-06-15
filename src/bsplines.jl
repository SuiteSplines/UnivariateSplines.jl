export Bspline, domain, dimension, codimension, @evaluate, @evaluate!, refine
export Interpolation, QuasiInterpolation, project!, l2_error
export partial_derivative, gradient, jacobian, hessian

"""
    BsplineEvaluationCache <: EvaluationCache{1}

Cache that caches basis function evaluation at a set of evaluation points.
"""
mutable struct BsplineEvaluationCache <: EvaluationCache{1}
    func::Function
    basis::AbstractArray
    grid::AbstractVector
    isinit::Bool
    function BsplineEvaluationCache(f::Function)
        eval = new()
        eval.func = f
        eval.isinit = false
        return eval
    end
end

# overloading of update! to update! cache on the fly
function IgaBase.update!(eval::BsplineEvaluationCache, x)
    if isinitialized(eval) && (eval.grid==x)
        return false # no update
    end
    eval.grid = x
    eval.basis = eval.func(x)
    eval.isinit = true
    return true
end

"""
    Bspline{T} <: ScalarMapping{1}

A univariate B-spline function of field `T`.
"""
struct Bspline{T<:Real,S<:AbstractVector{T}, I<:AbstractVector{Int}} <: AbstractSpline{1}
    space::SplineSpace{T}
    coeffs::S
    indices::I
    ders::Int
    cache::BsplineEvaluationCache
    orientation::Int
    function Bspline(space::SplineSpace{T}, coeffs::S, indices::I, cache::BsplineEvaluationCache; ders::Int, orientation::Int=1) where {T, S, I}
        @assert orientation==1 || orientation==-1
        @assert length(coeffs) == dimsplinespace(space) "Size coefficient array is inconsistent with dimension of spline-space."
        return new{T,S,I}(space, coeffs, indices, ders, cache, orientation)
    end
end

function Bspline(space::SplineSpace{T}, coeffs::AbstractVector{T}, indices::AbstractVector{Int}; ders=0, orientation::Int=1) where {T}
    cache = BsplineEvaluationCache(x -> Matrix(ders_bspline_interpolation_matrix(space, x, ders+1)[ders+1]) )
    return Bspline(space, coeffs, indices, cache; ders=ders, orientation=orientation)
end

function Bspline(space::SplineSpace{T}, coeffs::AbstractVector{T}, cache::BsplineEvaluationCache; ders=0, orientation::Int=1) where {T}
    n = dimsplinespace(space)
    indices = LinearIndices(Base.OneTo(n))
    return Bspline(space, coeffs, indices, cache; ders=ders, orientation=orientation)
end

function Bspline(space::SplineSpace{T}, coeffs::AbstractVector{T}; ders=0, orientation::Int=1) where {T}
    n = dimsplinespace(space)
    indices = LinearIndices(Base.OneTo(n))    
    return Bspline(space, coeffs, indices; ders=ders, orientation=orientation)
end

function Bspline(space::SplineSpace{T}; ders=0, orientation::Int=1) where {T}
    n = dimsplinespace(space)
    coeffs = zeros(T, n)
    indices = LinearIndices(Base.OneTo(n))
    return Bspline(space, coeffs, indices; ders=ders, orientation=orientation)
end

IgaBase.domain(spline::Bspline) = domain(spline.space)

Base.eltype(::Bspline{T}) where T = T

function Base.show(io::IO, spline::Bspline)
    T = eltype(spline)
    S = spline.space
    print(io, "Bspline{$T}($S)")
end

IgaBase.orientation(spline::Bspline) = spline.orientation

function Base.similar(spline::Bspline; coeffs=similar(spline.coeffs))
    return Bspline(spline.space, coeffs, spline.indices, spline.cache; ders=spline.ders, orientation=spline.orientation)
end

# overloading of AbstractMappings.evalkernel enables computation
# evaluation of a Bspline using the  @evaluate! macro.
for (OP,op) in [(:(=), :(.=)), (:(+=), :(.+=)), (:(-=), :(.-=)), (:(*=), :(.*=)), (:(/=), :(./=))]
    local S = Val{OP}
    local ex = Expr(op, :y, Expr(:call, :*, :A, :c))
    @eval function AbstractMappings.evalkernel_imp!(op::$S, y, x, spline::Bspline)
        IgaBase.update!(spline.cache, x)
        A = spline.cache.basis
        c = spline.coeffs
        $ex  # y .= A * x, y .+= A * x, etc
    end

end

function IgaBase.refine_imp(spline::Bspline, method::AbstractRefinement)
    C, space = refinement_operator(spline.space, method)
    coeffs = C * spline.coeffs
    return Bspline(space, coeffs; ders=spline.ders, orientation=spline.orientation)
end

function IgaBase.project!(spline::Bspline; onto, method::Type{<:AbstractInterpolation})
    x = grevillepoints(spline.space)
    update!(spline.cache, x)
    @evaluate! spline.coeffs = onto(x)
    IgaBase.project_imp!(method, spline, spline.coeffs)
    nothing
end

@inline function IgaBase.project_imp!(::Type{Interpolation}, spline::Bspline, y::AbstractVector)
    spline.coeffs .= spline.cache.basis \ y
end

@inline function IgaBase.project_imp!(::Type{QuasiInterpolation}, spline::Bspline, y::AbstractVector)
    A = Matrix(approximate_collocation_inverse(spline.space))
    spline.coeffs .= A * y
end

IgaBase.standard_quadrature_rule(f, spline::Bspline) = PatchRule(spline.space)

# dir is a dummy variable, needed for a uniform interface with the multidimensional case
@inline function IgaBase.partial_derivative(spline::Bspline, ders::Int, dir::Int=1)
    @assert dir==1
    return Bspline(spline.space, spline.coeffs; ders=spline.ders+ders, orientation=spline.orientation)
end
