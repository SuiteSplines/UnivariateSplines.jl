export num_elements, BsplineSupport, Support, SpanIndex, SpanIndices, EvalIndices, BasisIndices, ElementIterator

"""
    BsplineSupport(S::SplineSpace)

Iterator that outputs the supporting elements of each B-spline basis-function
as a `UnitRange{Int64}`.

# Example:
```jldoctest
julia> S = SplineSpace(2, [0.0,2.0,3.0], [3,1,3]);

julia> for α in BsplineSupport(S)
           @show α
       end
α = 1:1
α = 1:2
α = 1:2
α = 2:2
```
"""
struct BsplineSupport <: AbstractVector{UnitRange{Int}}
    p::Int
    ia::NonDecreasingSequence{Int}
end

function BsplineSupport(S::SplineSpace)
    ~, ~, ia, ~ = deconstruct_vector(S.U)
    return BsplineSupport(S.p, ia)
end

Base.size(s::BsplineSupport) = (length(s.ia) - s.p-1,)
Base.length(s::BsplineSupport) = length(s.ia) - s.p-1
Base.eltype(::BsplineSupport) = UnitRange{Int}
num_elements(s::BsplineSupport) = s.ia[end]-1

Base.checkbounds(s::BsplineSupport, k) = Base.checkbounds(s.ia, k)

@inline function Base.getindex(s::BsplineSupport, k::Int)
    @boundscheck checkbounds(s, k)
    return s.ia[k]:s.ia[k+s.p+1]-1
end

"""
    BsplineSupport(S::SplineSpace)

Iterator that outputs the supporting elements of each B-spline basis-function
as a `UnitRange{Int64}`.
"""
struct Support{S<:SplineSpace}  <: AbstractVector{UnitRange{Int}}
    space::S
    supports::BsplineSupport
end

function Support(s::SplineSpace)
    supports = BsplineSupport(s)
    return Support(s, supports)
end

Base.size(s::Support) = (dimsplinespace(s.space),)
Base.length(s::Support) = dimsplinespace(s.space)
Base.eltype(::Support) = Vector{Int}
num_elements(s::Support) = num_elements(s.supports)


@inline function Base.getindex(s::Support, k::Int)
    # @boundscheck checkbounds(s, k)
    return union(s.supports[parentfuns(s.space, k)]...)
end

"""
    SpanIndex(S::SplineSpace)

Iterator that outputs the span index corresponding to each
element in the partition

# Example:
```jldoctest
julia> S = SplineSpace(2, [0.0,1.0,2.0,3.0,4.0], [3,1,2,1,3]);

julia> for s in SpanIndex(S)
           @show s
       end
s = 3
s = 4
s = 6
s = 7
```
"""
struct SpanIndex{T}
    p::Degree
    U::KnotVector{T}
end

SpanIndex(S::SplineSpace) = SpanIndex(S.p, S.U)

Base.eltype(::SpanIndex) = Int64
Base.IteratorSize(itertype::SpanIndex) = Base.SizeUnknown()

function Base.iterate(iter::SpanIndex)
    v = iter.U
    mult = SortedSequences.count_multiplicity_down(v, 1)
    return (mult, mult)
end

function Base.iterate(iter::SpanIndex, state)
    v = iter.U
    if !(state+iter.p+2 > length(v))
        mult = SortedSequences.count_multiplicity_down(v, state+1)
        state+=mult
        return (state, state)
    end
end

struct SpanIndices{T}
    p::Degree
    U::KnotVector{T}
end

SpanIndices(S::SplineSpace) = SpanIndices(S.p, S.U)

Base.eltype(::SpanIndices) = UnitRange{Int64}
Base.IteratorSize(itertype::SpanIndices) = Base.SizeUnknown()

function Base.iterate(iter::SpanIndices)
    p = iter.p
    v = iter.U
    mult = SortedSequences.count_multiplicity_down(v, 1)
    return (mult-p:mult, mult)
end

function Base.iterate(iter::SpanIndices, state)
    p = iter.p
    v = iter.U
    if !(state+p+2 > length(v))
        mult = SortedSequences.count_multiplicity_down(v, state+1)
        state+=mult
        return (state-p:state, state)
    end
end