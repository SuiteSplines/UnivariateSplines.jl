export dersbsplinebasisfuns

"""

    dersbsplinebasisfuns(p, U, span::Int, u::T, nout=1)
    dersbsplinebasisfuns(p, U, u::T, nout=1)

Compute the non-vanishing B-spline basis-functions of degree ``p`` and their ``n_{out}-1``'th order
derivatives at the site ``x``. The output is a matrix ``B \\in \\mathbb{R}^{(p+1) \\times n_{out}}`` where
the ``i``'th row corresponds to the ``i``'th non-vanishing B-spline at ``x`` and the ``j``'th column
corresponds to the ``j-1``'th derivative.

The implementation of this function is based on

Piegl, Les, and Wayne Tiller. The NURBS book. Springer Science & Business Media, 2012.

# Examples:
```jldoctest derivatives
julia> p = Degree(2);

julia> U = KnotVector([0.0,0.0,0.0,1.0,2.0,2.0,2.0]);

julia> x = 1.5;

julia> span = findspan(p, U, x)
4

julia> dersbsplinebasisfuns(p, U, span, x, p+1)
3×3 Matrix{Float64}:
 0.125  -0.5   1.0
 0.625  -0.5  -3.0
 0.25    1.0   2.0
```
`dersbsplinebasisfuns` can also be called without providing the span-indices, in
which case they will be calculated on the fly.
```jldoctest derivatives
julia> dersbsplinebasisfuns(p, U, x, p+1)
3×3 Matrix{Float64}:
 0.125  -0.5   1.0
 0.625  -0.5  -3.0
 0.25    1.0   2.0
```

By prescribing a vector of ``m`` sites ``x \\in \\mathbb{R}^m`` we can compute the B-splines
and their derivatives at more than one point.  The function can be called with or without
prescribing the knot-span index

    dersbsplinebasisfuns(p, U, span::Vector{Integer}, u::Vector{T}, nout=1)
    dersbsplinebasisfuns(p, U, u::SortedSequence{T}, nout=1)

The output is a 3-dimensional array ``B \\in \\mathbb{R}^{(p+1) \\times m \\times n_{out}}``.

# Examples:
```jldoctest derivatives
julia> x = IncreasingVector([0.5, 1.5]);

julia> span = findspan(p, U, x)
2-element NonDecreasingVector{Int64}:
 3
 4

julia> dersbsplinebasisfuns(p, U, span, x, p+1)
3×2×3 Array{Float64, 3}:
[:, :, 1] =
 0.25   0.125
 0.625  0.625
 0.125  0.25

[:, :, 2] =
 -1.0  -0.5
  0.5  -0.5
  0.5   1.0

[:, :, 3] =
  2.0   1.0
 -3.0  -3.0
  1.0   2.0
```
Alternatively, the non-vanishing derivatives of the B-splines can be computed at ``n_{el}`` non-zero
knot-spans (elements) with ``m`` sites each, that is, ``x \\in \\mathbb{R}^{m \\times n_{el}}``. Again,
the function can be called with or without prescribing the knot-span index

    dersbsplinebasisfuns(p, U, span::Vector{Int64}, u::Matrix{T}, nout=1)
    dersbsplinebasisfuns(p, U, u::Matrix{T}, nout=1)

The output is a 4-dimensional array ``B \\in \\mathbb{R}^{(p+1) \\times m \\times n_{el} \\times n_{out}}``.

# Examples:
```jldoctest derivatives
julia> x = [0.25 1.25;
            0.75 1.75];

julia> dersbsplinebasisfuns(p, U, x, p+1)
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
 -1.5   -0.5
  1.25  -0.25
  0.25   0.75

[:, :, 2, 2] =
 -0.75  -0.25
  0.25  -1.25
  0.5    1.5

[:, :, 1, 3] =
  2.0   2.0
 -3.0  -3.0
  1.0   1.0

[:, :, 2, 3] =
  1.0   1.0
 -3.0  -3.0
  2.0   2.0
```
"""
function dersbsplinebasisfuns end   # implementation follows below

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, span::Integer, u::T, nout::Int=1) where {T<:Real}

    # initialize
    n = dimsplinespace(p,U)

    # compute triangular table of basis functions
    ndu   = zeros(T,p+1,p+1)
    left  = zeros(T,p+1)
    right = zeros(T,p+1)
    ndu[1,1] = 1
    for j in 1:p
        left[j+1]  = u - U[span+1-j]
        right[j+1] = U[span+j] - u
        saved = 0.0

        for r in 0:j-1
            # lower triangle
            ndu[j+1,r+1] = right[r+2]+ left[j-r+1]
            temp = ndu[r+1,j] / ndu[j+1,r+1]

            # upper triangle
            ndu[r+1,j+1] = saved + right[r+2] * temp
            saved = left[j-r+1]*temp
        end
        ndu[j+1,j+1] = saved
    end

    # load the basisfunctions
    ders = zeros(T,p+1,nout)
    ders[:,1] = ndu[:,end]

    # This section computes the derivatives
    a = zeros(nout,p+1)
    for r in 0:p
        s1=0; s2=1; a[1,1] = 1

        # loop to compute kth derivative
        for k in 1:nout-1
            d=0
            rk=r-k; pk = p-k
            if (r >= k)
                a[s2+1,1] = a[s1+1,1] / ndu[pk+2,rk+1]
                d = a[s2+1,1] * ndu[rk+1,pk+1]
            end

            if (rk >= -1)
                j1 = 1
            else
                j1 = -rk
            end

            if (r-1 <= pk)
                j2 = k-1
            else
                j2 = p-r
            end

            for j in j1:j2
                a[s2+1,j+1] = (a[s1+1,j+1] - a[s1+1,j]) / ndu[pk+2,rk+j+1]
                d = d + a[s2+1,j+1] * ndu[rk+j+1,pk+1]
            end

            if (r <= pk)
                a[s2+1,k+1] = - a[s1+1,k] / ndu[pk+2,r+1]
                d = d + a[s2+1,k+1] * ndu[r+1,pk+1]
            end
            ders[r+1,k+1] = d
            j = s1; s1 = s2; s2 = j
        end
    end

    # multiply by the correct factors
    r = p
    for k in 1:nout-1
        for j=0:p
            ders[j+1,k+1] = ders[j+1,k+1] * r
        end
        r = r*(p-k);
    end

    return ders
end

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, u::T, nout::Int=1) where {T<:Real}
    span = findspan(p, U, u)
    return dersbsplinebasisfuns(p, U, span, u, nout)
end

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, span::AbstractVector, u::AbstractVector{T}, nout::Int=1) where {T<:Real}

    m = length(u)
    Nu = zeros(T,p+1,m,nout)

    for i in 1:m
        Nu[:,i,:] = dersbsplinebasisfuns(p, U, span[i], u[i], nout)
    end
    return Nu
end

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, u::SortedSequence{T}, nout::Int=1) where {T<:Real}
    span = findspan(p, U, u)
    return dersbsplinebasisfuns(p, U, span, u, nout)
end

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, span::AbstractVector, u::Matrix{T}, nout::Int=1) where {T<:Real}

    m, ne = size(u)
    Nu = zeros(T,p+1,m,ne,nout)

    for j in 1:ne
        for i in 1:m
            Nu[:,i,j,:] = dersbsplinebasisfuns(p, U, span[j], u[i,j], nout)
        end
    end
    return Nu
end

function dersbsplinebasisfuns(p::Degree, U::KnotVector{T}, u::Matrix{T}, nout::Int=1) where {T<:Real}
    span = findspan(p, U, IncreasingVector(u[1,:]))
    return dersbsplinebasisfuns(p, U, span, u, nout)
end
