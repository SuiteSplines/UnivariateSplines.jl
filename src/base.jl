export Degree, Regularity, Dimension
export IncreasingVector, IncreasingRange, NonDecreasingVector
export global_insert, deconstruct_vector, construct_vector
export KnotVector, KnotSpanIndices
export findspan, dimsplinespace, num_elements, grevillepoints
export unit_integral_rescaling, bspline_integral_value

import SortedSequences: IncreasingVector, IncreasingRange, NonDecreasingVector
import SortedSequences: global_insert, deconstruct_vector, construct_vector

"""
    KnotVector{T<:Real}

A knot-vector is a non-decreasing sequence of real numbers.
(typeallias of NonDecreasingVector{T})
"""
const KnotVector = NonDecreasingVector

"""
    KnotSpanIndices

A sorted sequence of integers that refer to the corresponding non-zero knot span
of a `KnotVector`.
"""
const KnotSpanIndices = SortedSequence{Int}

"""
    dimsplinespace(p, U)

Compute the dimension of the spline space defined by degree ``p`` and
knotsvector ``U``.
"""
dimsplinespace(p::Degree, U::KnotVector) = length(U)-p-1

"""
    num_elements(U::KnotVector)

Count the number of non-zero knot spans or elements in the knotvector.
"""
num_elements(U::KnotVector) = length(Unique(U)) - 1

"""
    grevillepoints(p, U, i)
    grevillepoints(p, U)

Compute the ``i``'th Greville Absissa corresponding to the ``i``'th B-spline
defined by degree 'p' and knotvector 'U' of spline space.

# Examples:
```jldoctest
julia> p = Degree(2);

julia> U = KnotVector([0.0,1.0,2.5,3.0], [3,1,2,1])
7-element NonDecreasingVector{Float64}:
 0.0
 0.0
 0.0
 1.0
 2.5
 2.5
 3.0

julia> grevillepoints(p, U, 2)
0.5

julia> grevillepoints(p, U)
4-element IncreasingVector{Float64}:
 0.0
 0.5
 1.75
 2.5
```
"""
grevillepoints(p::Degree, U::KnotVector, i::Integer) = sum(U[i+1:i+p]) / p

function grevillepoints(p::Degree, U::KnotVector{T}) where {T<:Real}
    n = dimsplinespace(p, U)
    g = T[grevillepoints(p, U, j) for j in 1:n]
    return IncreasingVector(g, false)
end

"""
    findspan(p, U, u)

Given polynomial degree ``p`` and knotvector ``U`` determine the knot span
index ``i`` of a point ``x`` such that ``u \\in [U_i, U_{i+1})``.

# Examples:
```jldoctest findspan
julia> p = Degree(2)
2

julia> U = KnotVector([0.0,1.0,1.5,2.5,3.0],[3,1,1,2,3])
10-element NonDecreasingVector{Float64}:
 0.0
 0.0
 0.0
 1.0
 1.5
 2.5
 2.5
 3.0
 3.0
 3.0

julia> x = 2.0
2.0

julia> span = findspan(p, U, x)
5

julia> U[span] ≤ x < U[span+1]
true
```
It is also possible to compute the knot-spans of all values in an
'IncreasingVector{T}'
```jldoctest findspan
julia> x = IncreasingVector([0.25, 0.75, 1.0, 1.25, 2.5, 2.75]);

julia> span = findspan(p, U, x)
6-element NonDecreasingVector{Int64}:
 3
 3
 4
 4
 7
 7
```
"""
function findspan(p::Degree, U::KnotVector{T}, u::T) where {T<:Real}

    if u<U[p+1] || u>U[end-p]
        a = U[p+1]
        b = U[end-p]
        error("Boundserror findspan(p::Degree, U::KnotVector{T}, u::T). $a <= $u <= $b")
    end

    if u==U[end-p]
        span = length(U)-p-1
        while U[span]==U[span+1]
            span-=1
        end
    else
        low = 0
        high = length(U)
        mid = round(Int64,(low + high)/2)

        while u < U[mid] || u >= U[mid + 1]
          if u < U[mid]
            high = mid
          else
            low = mid
          end
          mid = round(Int64,(low + high)/2)
        end
        span = round(Int64,(low + high)/2)
    end
    return span
end

function findspan(p::Degree, U::KnotVector{T}, u::AbstractVector{T}) where {T<:Real}

    # allocate space for output
    n = length(U)-p-1
    m = length(u)
    span = zeros(Int64,m)
    span[1] = findspan(p,U,u[1])
    span[m] = findspan(p,U,u[m])

    # find span at u[i]
    for i in 2:m-1
        span[i] = span[i-1]
        while u[i] >= U[span[i]+1]
            span[i] += 1
        end
    end
    return NonDecreasingVector(span, false)
end


"""
    bspline_integral_value(p, U, i::Int)
    bspline_integral_value(p, U)

Compute the integral of a b-spline function defined by degree `p` and
knotvector `U`.
"""
@inline bspline_integral_value(p::Degree, U::KnotVector, i::Integer) = (U[i+p+1] - U[i]) / (p+1)
bspline_integral_value(p::Degree, U::KnotVector) = [bspline_integral_value(p, U, i) for i in 1:dimsplinespace(p, U)]

"""
    unit_integral_rescaling(p, U, i::Int)
    unit_integral_rescaling(p, U)

Compute the scaling factors that normalize the B-spline basis functions to
have unit integral.
"""
@inline unit_integral_rescaling(p::Degree, U::KnotVector, i::Integer) = (p+1) / (U[i+p+1] - U[i])
unit_integral_rescaling(p::Degree, U::KnotVector) = [unit_integral_rescaling(p,U,i) for i in 1:dimsplinespace(p,U)]
