export onebasisfuneval, bsplinebasisfuns

"""
    onebasisfuneval(U, u)

Compute value of a single B-spline basis-function defined by local
knotvector 'U' evaluated at point 'x'.

The implementation of this function is taken from

Piegl, Les, and Wayne Tiller. The NURBS book. Springer Science & Business Media, 2012.

"""
function onebasisfuneval(U::KnotVector{T}, u::T) where {T<:Real}

    # initialize
    p = length(U)-2
    funs = zeros(T,p+1)

    if u<U[1] || u > U[p+2]
      funs = 0.0
    elseif sum(u.==U)==p+1
    	funs = 1.0
    else
      # initialise zeroth degree basisfunction
      for j in 1:p+1
        funs[j] = 0.0
        if (u>=U[j] && u<U[j+1])
          funs[j] = 1.0
        end
      end

      # compute triangular table
      for k in 1:p
        if funs[1]==0.0
          saved = 0.0
        else
          saved = ((u-U[1])*funs[1]) / (U[k+1]-U[1])
        end

        for j in 1:p+1-k
          Uleft  = U[j+1]
          Uright = U[j+k+1]

          if funs[j+1]==0.0
            funs[j] = saved
            saved = 0.0
          else
            temp = funs[j+1] / (Uright - Uleft)
            funs[j] = saved + (Uright - u) * temp
            saved = (u - Uleft) * temp
          end
        end
      end
    end
  return funs[1]
end

onebasisfuneval(p::Degree, U::KnotVector{T}, u::Vector{T}) where {T<:Real} = T[onebasisfuneval(p,U,u[i]) for i in 1:length(u)]

"""
    bsplinebasisfuns(p, U, i::Integer, x::T, nout=1)
    bsplinebasisfuns(p, U, x::T, nout=1)

Compute the ``p+1`` non-zero B-spline basis-functions at site ``x \\in [U_{i},U_{i+1})``.
The output is a matrix where column ``j`` (rows ``1:p+2-j``), for ``j=1,...,n_{out}``
correspond to the ``p+2-j`` non-zero B-spline basis functions of degree ``p+1-j``.
The remaining entries correspond to supports. If the active knot-span is not provided
then it will be computed.

# Examples:
```jldoctest bsplines
julia> p = Degree(2);

julia> U = KnotVector([0.0,0.0,0.0,1.0,2.0,2.0,2.0]);

julia> x = 1.5
1.5

julia> span = findspan(p, U, x)
4

julia> B = bsplinebasisfuns(p, U, span, x, p+1)
3×3 Matrix{Float64}:
 0.125  0.5  1.0
 0.625  0.5  1.0
 0.25   1.0  2.0
```
The knot span index can be omited, in which case it is computed
on the fly.
```jldoctest bsplines
julia> bsplinebasisfuns(p, U, x, p+1)
3×3 Matrix{Float64}:
 0.125  0.5  1.0
 0.625  0.5  1.0
 0.25   1.0  2.0
```

By prescribing a vector of ``m`` sites ``x \\in \\mathbb{R}^m`` we can compute the
B-splines at more than one point. The function can be called with or without
prescribing the knot-span index

    bsplinebasisfuns(p, U, span::Vector{Integer}, u::Vector{T}, nout=1)
    bsplinebasisfuns(p, U, u::SortedSequence{T}, nout=1)

The output is a 3-dimensional array ``B \\in \\mathbb{R}^{(p+1) \\times m \\times n_{out}}``.

# Examples:
```jldoctest bsplines
julia> x = IncreasingVector([0.5, 1.5]);

julia> span = findspan(p, U, x)
2-element NonDecreasingVector{Int64}:
 3
 4

julia> bsplinebasisfuns(p, U, span, x, p+1)
3×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.25   0.125
 0.625  0.625
 0.125  0.25

[:, :, 2] =
 0.5  0.5
 0.5  0.5
 2.0  1.0

[:, :, 3] =
 1.0  1.0
 1.0  1.0
 1.0  2.0
```
Alternatively, the non-vanishing B-splines up to and including degree ``p`` can
be computed at ``n_{el}`` non-zero knot-spans (elements) with ``m`` sites each,
that is, ``x \\in \\mathbb{R}^{m \\times n_{el}}``. Again, the function can be
called with or without prescribing the knot-span index

    bsplinebasisfuns(p, U, span::Vector{Int64}, u::Matrix{T}, nout=1)
    bsplinebasisfuns(p, U, u::Matrix{T}, nout=1)

The output is a 4-dimensional array ``B \\in \\mathbb{R}^{(p+1) \\times m \\times n_{el} \\times n_{out}}``.

# Examples:
```jldoctest bsplines
julia> x = [0.25 1.25;
            0.75 1.75];

julia> bsplinebasisfuns(p, U, x, p+1)
3×2×2×3 Array{Float64, 4}:
[:, :, 1, 1] =
 0.5625   0.0625
 0.40625  0.65625
 0.03125  0.28125

[:, :, 2, 1] =
 0.28125  0.03125
 0.65625  0.40625
 0.0625   0.5625

[:, :, 1, 2] =
 0.75  0.25
 0.25  0.75
 2.0   2.0

[:, :, 2, 2] =
 0.75  0.25
 0.25  0.75
 1.0   1.0

[:, :, 1, 3] =
 1.0  1.0
 1.0  1.0
 1.0  1.0

[:, :, 2, 3] =
 1.0  1.0
 1.0  1.0
 2.0  2.0
```
"""
function bsplinebasisfuns end # implementation follows below

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, span::Int, u::T, nout::Int=1) where {T<:Real}
    funs = zeros(T,p+1,p+1); funs[1,p+1] = 1

    for i in 1:p
        # compute support
        for j in 1:i
            funs[i+1,p+2-j] = U[span+j] - U[span+j-i]
        end

        # compute B-pline basis functions
        for j in 1:i
            alfa = (u - U[span+j-i]) / funs[i+1,p+2-j]
            funs[j,  p+1-i] += (1-alfa) * funs[j,p+2-i]
            funs[j+1,p+1-i] +=    alfa  * funs[j,p+2-i]
        end
    end
    return funs[:,1:nout]
end

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, u::T, nout::Int=1) where {T<:Real}
    return bsplinebasisfuns(p, U, findspan(p, U, u), u, nout)
end

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, span::KnotSpanIndices, u::AbstractVector{T}, nout::Int=1) where {T<:Real}

    m = length(u)
    Nu = zeros(T,p+1,m,nout)

    for i in 1:m
        Nu[:,i,:] = bsplinebasisfuns(p, U, span[i], u[i], nout)
    end
    return Nu
end

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, u::AbstractVector{T}, nout::Int=1) where {T<:Real}
    return bsplinebasisfuns(p, U, findspan(p, U, u), u, nout)
end

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, span::KnotSpanIndices, u::Matrix{T}, nout::Int=1) where {T<:Real}

    m, ne = size(u)
    Nu = zeros(T,p+1,m,ne,nout)

    for j in 1:ne
        for i in 1:m
            Nu[:,i,j,:] = bsplinebasisfuns(p, U, span[j], u[i,j], nout)
        end
    end
    return Nu
end

function bsplinebasisfuns(p::Degree, U::KnotVector{T}, u::Matrix{T}, nout::Int=1) where {T<:Real}
    span = findspan(p, U, IncreasingVector(u[1,:]))
    return bsplinebasisfuns(p, U, span, u, nout)
end
