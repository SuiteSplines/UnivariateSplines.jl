export SplineSpace, ExtractionOperator, BsplineBasis
export differentiate, roughen, refinement_operator
export refine, hRefinement, pRefinement, kRefinement, hpRefinement
export breakpoints, domain, parentfuns

using LinearAlgebra, SparseArrays

const ExtractionOperator{T} = SparseMatrixCSC{T,Int}

"""
    SplineSpace(p, U)
    SplineSpace(p, x, m)
    SplineSpace(p, Interval(a, b), num_elements)

Definition of a spline space by means of the polynomial degree ``p``, a
sequence of break-points ``x`` and the knot-multiplicity ``m`` or by
prescribing the knot-vector ``U``.

# Examples:
```jldoctest
julia> S = SplineSpace(2, [0.0,2.0,3.0], [3,1,3])
SplineSpace(degree = 2, interval = [0.0, 3.0], dimension = 4)

julia> dimsplinespace(S)
4

julia> Degree(S)
2

julia> KnotVector(S)
7-element NonDecreasingVector{Float64}:
 0.0
 0.0
 0.0
 2.0
 3.0
 3.0
 3.0
```
"""
struct SplineSpace{T<:Real}
    p::Degree
    U::KnotVector{T}
    C::ExtractionOperator{T}
    s::IncreasingVector{Int}
    function SplineSpace(p::Degree, U::KnotVector{T}; kwargs...) where {T}
        C = extraction_operator(p, U; kwargs...)
        s = IncreasingVector(collect(SpanIndex(p, U)), false)
        return new{T}(p, U, C, s)
    end
end

Base.ndims(::SplineSpace) = 1

function check_multiplicity_vector(p::Degree, mult::Vector{Int})
    @assert(mult[1]==mult[end]==p+1, "Use p+1 knots at boundary knots.")
    for k in 2:length(mult)-1
         @assert(0 < mult[k] < p+1, "Knot multiplicity does not satisfy 0 < m <= p.")
    end
end

function SplineSpace(p, ukts, mult::Vector{Int}; kwargs...)
    check_multiplicity_vector(p, mult)
    SplineSpace(p, KnotVector(IncreasingVector(ukts), mult); kwargs...)
end

function SplineSpace(p::Degree, ukts::IncreasingSequence{<:Real}; kwargs...)
    num_elements = length(ukts) - 1
    mult = ones(Int, num_elements+1); mult[1] = mult[end] = p+1
    return SplineSpace(p, ukts, mult; kwargs...)
end

function SplineSpace(p::Degree, domain::Interval, num_elements::Dimension; kwargs...)
    ukts = IncreasingRange(domain, num_elements+1)
    return SplineSpace(p, ukts; kwargs...)
end

function SplineSpace(p::Degree, num_elements::Dimension; kwargs...)
    return SplineSpace(p, Interval(0.0,1.0), num_elements; kwargs...)
end

function Base.show(io::IO, space::SplineSpace)
    p = space.p
    a = space.U[1]
    b = space.U[end]
    n = dimsplinespace(space)
    print(io, "SplineSpace(degree = $p, interval = [$a, $b], dimension = $n)")
end

# overloading Base.length to work with type SplineSpace
# such that we can use TensorProduct{Dim,SplineSpace{T}}
@inline Base.length(S::SplineSpace) = dimsplinespace(S)
@inline Base.size(S::SplineSpace) = (length(S),)
@inline IgaBase.numbertype(::SplineSpace{T}) where T = T

IgaBase.domain(space::SplineSpace) = Interval(space.U[1], space.U[end])

# return the parent B-spline function indices of the `kth`
# spline function
function parentfuns(S::SplineSpace, k::Int)
    j = S.C.colptr[k]:S.C.colptr[k+1]-1
    return S.C.rowval[j]
end

function IgaBase.differentiate(S::SplineSpace; kwargs...)
    return SplineSpace(Degree(S)-1, KnotVector(S)[2:end-1]; kwargs...)
end

function roughen(S::SplineSpace, k::Int=1; kwargs...)
    u, m = deconstruct_vector(KnotVector(S))
    m[2:end-1].+=k
    return SplineSpace(Degree(S), u, m; kwargs...)
end

# ToDo: Make available to constraint spaces
function IgaBase.refine(space::SplineSpace, R::AbstractRefinement)
    p = refine(Degree(space), R)
    U = refine(KnotVector(space), R)
    return SplineSpace(p, U)
end

IgaBase.Degree(S::SplineSpace) = S.p
KnotVector(S::SplineSpace) = S.U
ExtractionOperator(S::SplineSpace) = S.C

"""
    extraction_operator(p::Degree, U::KnotVector; [cperiodic, cleft, cright])

Compute an extraction operator that creates a subspace of the `SplineSpace` with
for example peroidicity built-in.
"""
function extraction_operator(p::Degree, U::KnotVector; cperiodic=Int[], cleft=Int[], cright=Int[])
    n = dimsplinespace(p, U)
    C = Matrix(I, n, n)
    if !isempty(cperiodic)
        A = periodic_boundary_constraints(p, U; con=cperiodic)
        C *= nullspace(A * C; perm=a->[length(a), 1:length(a)-1...])
    end
    if !isempty(cleft)
        A = left_boundary_constraints(p, U; con=cleft)
        C *=  nullspace(A * C; perm=permute_none)
    end
    if !isempty(cright)
        A = right_boundary_constraints(p, U; con=cright)
        C *=  nullspace(A * C; perm=permute_none)
    end
    return C
end

function periodic_boundary_constraints(p, U; con=[1:p...])
    n, m = dimsplinespace(p, U), length(con)
    A = zeros(m, n)
    A[1:m,1:p] = dersbsplinebasisfuns(p, U, U[1], p)[1:p,con]'
    A[1:m,end-p+1:end] = -dersbsplinebasisfuns(p, U, U[end], p)[2:p+1,con]'
    return A
end

function left_boundary_constraints(p, U; con=[1])
    n, m = dimsplinespace(p, U), length(con)
    A = zeros(m, n)
    A[1:m,1:p] = dersbsplinebasisfuns(p, U, U[1], p)[1:p,con]'
    return A
end

function right_boundary_constraints(p, U; con=[1])
    n, m = dimsplinespace(p, U), length(con)
    A = zeros(m, n)
    A[1:m,end-p+1:end] = dersbsplinebasisfuns(p, U, U[end], p)[2:p+1,con]'
    return A
end

num_elements(S::SplineSpace) = num_elements(S.U)
dimsplinespace(S::SplineSpace) = size(S.C,2)
grevillepoints(S::SplineSpace) = IncreasingVector(S.C' * grevillepoints(Degree(S), KnotVector(S)), false)
breakpoints(S::SplineSpace) = IncreasingVector(KnotVector(S))
dersbsplinebasisfuns(S::SplineSpace, x, nout=1) = dersbsplinebasisfuns(Degree(S), KnotVector(S), x, nout)

bspline_integral_value(S::SplineSpace) = S.C' * bspline_integral_value(S.p, S.U)
unit_integral_rescaling(S::SplineSpace) = 1 ./ bspline_integral_value(S)

bspline_interpolation_matrix(S::SplineSpace, u, nout) = map(B -> B * S.C, bspline_interpolation_matrix(Degree(S), KnotVector(S), u, nout))
bspline_interpolation_matrix(S::SplineSpace, u) = bspline_interpolation_matrix(S, u, 1)[1]

ders_bspline_interpolation_matrix(S::SplineSpace, u, nout) = map(B -> B * S.C, ders_bspline_interpolation_matrix(Degree(S), KnotVector(S), u, nout))
ders_bspline_interpolation_matrix(S::SplineSpace, u) = ders_bspline_interpolation_matrix(S, u, 1)[1]

approximate_collocation_inverse(S::SplineSpace, k1::Int=S.p+1) = S.C' * approximate_collocation_inverse(Degree(S), KnotVector(S), k1)

approximate_l2_inverse(S::SplineSpace, k1::Int=S.p+1) = S.C' * approximate_l2_inverse(Degree(S), KnotVector(S), k1)

bezier_extraction_operator(S::SplineSpace) = bezier_extraction_operator(Degree(S), KnotVector(S))

# ToDo: Make available to periodic and other constraint spaces
function refinement_operator(S::SplineSpace, method::AbstractRefinement)
    C, p, U = refinement_operator(Degree(S), KnotVector(S), method)
    return C, SplineSpace(p, U)
end

abstract type Basis{T<:Real} <: AbstractArray{T,3} end

"""
    BsplineBasis{T<:Real} <: Basis{T,3}

Special array type for storing evaluating B-spline and their derivatives.
"""
struct BsplineBasis{T} <: Basis{T}
    n::Int
    span::NonDecreasingSequence{Int}
    data::Array{T,3}
end
BsplineBasis(span::NonDecreasingSequence{Int}, data::Array{T,3}) where T = BsplineBasis(size(data,1), span, data)

function BsplineBasis(p::Degree, U::KnotVector, span, x, nout::Dimension) 
    BsplineBasis(span, dersbsplinebasisfuns(p, U, span, x, nout))
end

function BsplineBasis(p::Degree, U::KnotVector, x, nout::Dimension) 
    span = findspan(p, U, x)
    BsplineBasis(span, dersbsplinebasisfuns(p, U, span, x, nout))
end

BsplineBasis(S::SplineSpace, x, nout::Dimension) = BsplineBasis(Degree(S), KnotVector(S), x, nout)
BsplineBasis(S::SplineSpace, span, x, nout::Dimension) = BsplineBasis(Degree(S), KnotVector(S), span, x, nout)


Base.size(A::Basis) = (A.span[end], size(A.data,2), size(A.data,3))
Base.size(A::Basis, i) = size(A)[i]
Base.eltype(A::Basis{T}) where T = T

using Base: Indices, @propagate_inbounds, @boundscheck, checkbounds

@inline @propagate_inbounds function Base.getindex(A::Basis{T}, i::Int, j::Int, k::Int) where T
    @boundscheck checkbounds(A, i, j, k)
    s = A.span[j]
    I = i - (s-A.n)
    if I in 1:A.n
        return A.data[I,j,k]
    else
        return T(0)
    end
end

@inline @propagate_inbounds function Base.setindex!(A::Basis{T}, V, I::Int, J::Int, K::Int) where T
    @boundscheck checkbounds(A, I, J, K)
    s = A.span[J]
    Indices = I - (s-A.n)
    if I in 1:A.n
        A.data[Indices,J,K] = V
    else
        BoundsError(A, (I,J,K))
    end
end