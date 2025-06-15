export GaussRule, PatchRule, affine_transform!, Legendre, Lobatto, Interval

struct GaussRule{S,T<:Real} <: AbstractQuadrule{1}
    x::IncreasingVector{T}
    w::Vector{T}
    function GaussRule{S}(x, w) where {S}
        @assert length(x)==length(w) "Number of points are not consistent with number of weights."
        T = eltype(w)
        return new{S,T}(IncreasingVector(x, false),w)
    end
end

function GaussRule(::Type{Legendre}, n::Dimension)
    x, w = gausslegendre(n)
    return GaussRule{Legendre}(x, w)
end

function GaussRule(::Type{Lobatto}, n::Dimension)
    x, w = gausslobatto(n)
    return GaussRule{Lobatto}(x, w)
end

GaussRule(n::Dimension) = GaussRule(Legendre, n)

function GaussRule(method, n::Dimension, a::T, b::T) where {T<:Real}
    Q = GaussRule(method, n)
    return affine_transform!(Q, Interval(T(-1),T(1)), Interval(a,b))
end

struct PatchRule{S,T<:Real} <: AbstractQuadrule{1}
    i::IncreasingVector{Int}
    x::Vector{T}
    w::Vector{T}
    function PatchRule{S}(i::IncreasingVector{Int}, x::Vector{T}, w::Vector{T}) where{S,T}
        @assert length(x)==length(w) "Number of points are not consistent with number of weights."
        return new{S,T}(i,x,w)
    end
end

num_elements(qr::PatchRule) = length(qr.i)-1

function PatchRule(u::IncreasingSequence{T}; npoints::Int, method::Type{<:GaussianRule}=Legendre) where {T<:Real}

    m = length(u)-1 # number of elements
    I = Interval(T(-1), T(1))
    Q = GaussRule(method, npoints)
    Qsave = copy(Q)

    dim = dim_global_gauss_rule(method, npoints, m)
    n_add = increase_counter_by(method, npoints)
    J, X, W = zeros(Int,m+3), zeros(T, dim), zeros(T, dim)
    J[1] = 1; J[2] = 2;
    X[1] = u[1]; X[end] = u[end]
    W[1] = 0.0; W[end] = 0.0
    j = 1
    for i in 1:m
        # affinely transform quadrature rule to element
        affine_transform!(Q, I, Interval(u[i], u[i+1]))

        # save points and weights
        indices = j+1:j+npoints
        X[indices] = Q.x[:]
        W[indices] += Q.w[:]

        # reinitialize local element quadrature rule
        Q.x.data[:], Q.w[:] = Qsave.x, Qsave.w
        j += n_add
        J[i+2] = J[i+1] + n_add # global indices
    end
    J[end] = J[end-1]+1

    return PatchRule{method}(IncreasingVector(J, false), X, W)
end

function PatchRule(space::SplineSpace; npoints::Int=space.p+1, method::Type{<:GaussianRule}=Legendre)
    return PatchRule(breakpoints(space); npoints=npoints, method=method)
end

function affine_transform!(Q::AbstractQuadrule{1}, I::Interval, J::Interval)
    LI = I.b - I.a
    LJ = J.b - J.a
    s = LJ / LI
    for i in 1:length(Q)
        alpha = (Q.x[i] - I.a) / LI
        Q.x.data[i] = (1.0 - alpha) * J.a + alpha * J.b
        Q.w[i] *= s
    end
    return Q
end

function affine_transform(Q::AbstractQuadrule{1}, I::Interval, J::Interval)
    return affine_transform!(copy(Q), I, J)
end

function GaussRule(Q::PatchRule{S}, k::Int) where {S}
    indices = Q.i[k+1]:Q.i[k+2]-1
    return GaussRule{S}(Q.x[indices], Q.w[indices])
end

dim_global_gauss_rule(::Type{Legendre}, n_points::Dimension, n_elements::Dimension) = n_points * n_elements + 2  # +2 due to added boundary pts
dim_global_gauss_rule(::Type{Lobatto}, n_points::Dimension, n_elements::Dimension) = (n_points-1) * n_elements + 3 # +2 due to added boundary pts
increase_counter_by(::Type{Legendre}, n_points::Dimension) = n_points
increase_counter_by(::Type{Lobatto}, n_points::Dimension) = n_points-1