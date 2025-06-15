export WeightedQuadrule

const TargetSpace = SplineSpace
const TestSpace = SplineSpace

function system_matrix(S::TargetSpace, V::TestSpace, k::Int=1, l::Int=1)

    # initialize
    @assert (k in 1:Degree(S)+1) && (l in 1:Degree(V)+1)
    u = IncreasingVector(KnotVector(S))
    v = IncreasingVector(KnotVector(V))
    @assert u == v

    # construct global quadrature rule
    nq = ceil(Int, 0.5*(Degree(S) + Degree(V) + 1))
    Q = PatchRule(u; npoints=nq, method=Legendre)
    
    # compute B-spline basisfunctions of the test space
    Nu = ders_bspline_interpolation_matrix(Degree(S), KnotVector(S), Q.x, k)[k]

    # compute B-spline basisfunctions of the trial space
    Nv = ders_bspline_interpolation_matrix(Degree(V), KnotVector(V), Q.x, l)[l]

    return Nv' * (Q.w .* Nu)
end

function sparse_difference_matrix(n::Int, T=Float64)
    return spdiagm(n, n-1, -1 => fill(T(1), n-1), 0 => fill(T(-1), n-1))
end

"""
    table_required_points(S, V)

Determine the number of exactness conditions in subsets of elements. The output
is a matrix ``\\mathsf{A}`` where ``\\mathsf{A}_{ij}`` denotes the minimum number
of points required in interval ``i:j`` to perform exact weighted quadrature.

Examples:
```jldoctest weighted_rule
julia> S = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,2,2,3]);

julia> V = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,1,1,3]);

julia> A = UnivariateSplines.table_required_points(S, V)
3×3×1 Array{Int64, 3}:
[:, :, 1] =
 3  5  7
 2  5  0
 3  0  0
```
"""
function table_required_points(S::TargetSpace, V::TestSpace)

    # determine elements in support of each basis function
    supp_s = collect(Support(S))

    # allocate space for array that describes the number of
    # exactness conditions
    n = dimsplinespace(V)
    nquad = zeros(Int, num_elements(KnotVector(V)), Degree(V)+1, n)
    nquadacc = zeros(Int, num_elements(KnotVector(V)), Degree(V)+1, n)


    # loop over testfunctions
    for (i,supp_v) in enumerate(Support(V))
        # fill table
        for (α,μ) in UnivariateSplines.Unique(intersect.((supp_v,), supp_s))
            if !isempty(α)
                nquad[α[1], length(α), i] = μ
            end
        end
    end

    # accumulate the entries in nquad to determine the total number
    # of exactness conditions in a particular sub-interval
    for j in 1:size(nquadacc,2)
        for i in 1:size(nquadacc,1)+1-j
            for l in 1:j
                for k in i:min(i+j-l, size(nquad,1))
                    nquadacc[i,j,:] += nquad[k,l,:]
                end
            end
        end
    end

    # compute the number of points in subset of elements
    nquadacc = maximum(nquadacc,dims=3)
    return nquadacc
end

"""
    nquadpoints(S::SplineSpace, V::SplineSpace; add_boundary_points=false, add_additional_points=0, min_points=1)

Determine the minimum number of quadrature points corresponding to a weighted
quadrature rule with test-space ``V`` and target-space ``S``.

Examples:
```jldoctest weighted_rule
julia> S = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,2,2,3]);

julia> V = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,1,1,3]);

julia> A = UnivariateSplines.nquadpoints(S, V; add_boundary_points=false)
3-element Vector{Int64}:
 3
 2
 3
```
"""
function nquadpoints(S::TargetSpace, V::TestSpace; add_boundary_points::Bool=false, add_additional_points::Int=0, min_points::Int=2)

    # compute table of required quadrature points
    nreq = table_required_points(S, V)
    nelms = size(nreq,1)
    p = size(nreq,2)-1

    # determine placement of quadrature points
    for k in 1:p
        for i in 1:nelms-k

            # initialize
            a = nreq[i:i+k,1]
            b = nreq[i,k+1]

            # number of points to insert
            m = max(0, b-sum(a)-add_boundary_points*k)

            # loop over points
            for j in 1:m
                K = sortperm(a,rev=true)
                a[K[end]]+=1
            end

            # update required points per element
            nreq[i:i+k,1] = a
        end
    end

    # add additional points
    nreq[:,1] .+= add_additional_points

    # set minimum number of points
    for i in 1:size(nreq,1)
        if nreq[i,1]<min_points
            nreq[i,1] = min_points
        end
    end

    return  nreq[:,1]
end

"""
    distribute_points(S::SplineSpace, V::SplineSpace; add_boundary_points::Bool=false)

Compute distribution of quadrature points in weigthed quadrature with a test-space ``V``
and target space for quadrature ``S``. The nodes are distribute in such a way that all
conditions for exact quadrature are satisfied with a minimum number of points.

The distribution of quadrature points in weighted quadrature is based on the
following paper.

Hiemstra, René R., et al. "Fast formation and assembly of finite element matrices
with application to isogeometric linear elasticity." Computer Methods in Applied
Mechanics and Engineering 355 (2019): 234-260.

# Examples:
```jldoctest weighted_rule
julia> S = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,2,2,3]);

julia> V = SplineSpace(2, IncreasingVector([0.0,1.0,2.0,3.0]), [3,1,1,3]);

julia> UnivariateSplines.distribute_points(S, V)
8-element IncreasingVector{Float64}:
 0.16666666666666666
 0.5
 0.8333333333333334
 1.25
 1.75
 2.1666666666666665
 2.5
 2.8333333333333335
```
"""
function distribute_points(S::TargetSpace, V::TestSpace; add_boundary_points::Bool=false, add_additional_points::Int=0, min_points::Int=2)

    # initialize
    nquad = nquadpoints(S, V; add_boundary_points=add_boundary_points, add_additional_points=add_additional_points, min_points=min_points)
    n = length(nquad)

    # define distribution function
    if add_boundary_points
        h = (a,b,n) -> LinRange(a + (b-a)/(n+1), b - (b-a)/(n+1), n)
    else
        h = (a,b,n) -> LinRange(a + 0.5*(b-a)/n, b - 0.5*(b-a)/n, n)
    end

    # initialize
    u = IncreasingVector(KnotVector(S))
    T = eltype(u)
    m = sum(nquad)
    if add_boundary_points
        m += n+1
    end
    v = zeros(T,m)

    # loop over elements
    l = 0
    for k in 1:n
        # treat boundary points
        if add_boundary_points
            l += 1
            v[l] = u[k]
        end

        # treat internal points
        v[l+1:l+nquad[k]] = h(u[k], u[k+1], nquad[k])
        l = l+nquad[k]
    end

    # add final boundary point
    if add_boundary_points
        v[end] = u[end]
    end

    return IncreasingVector(v)
end

"""
    normalization_weights(X)

Determine a set of normalization weights that result in a more uniform set oftype(
weighted quadrature weights when solving for the least norm solution.
)
"""
function normalization_weights(X::IncreasingVector{T}) where {T<:Real}
    m = length(X)
    c = zeros(T, m)
    c[1] = X[2] - X[1]
    for k in 2:length(X)-1
        c[k] = 0.5*(X[k+1] - X[k-1])
    end
    c[end] = X[end] - X[end-1]
    return c
end

"""
    solve_leastnorm_without_constraints(A, b, S=BigFloat)

Solve underdetermined system of equation ``A x = b`` using QR factorization with
arbitrary precision.
"""
function solve_leastnorm_without_constraints(A::Matrix{T}, b::Vector{T}, S=BigFloat) where {T<:Real}

    # check input
    @assert size(A,2)>=length(b)

    # compute least norm solution in high precision
    w = qr(convert(Matrix{S}, A)) \ convert(Vector{S}, b)

    return convert(Vector{T},w), convert(T,norm(A*w-b))
end

"""
    wquadweights!(M, B, W, C)

Compute the quadrature weights in weighted quadrature such that ``` M ≈ W' * B```.
"""
function wquadweights!(M::SparseMatrixCSC{T}, B::SparseMatrixCSC{T}, W::SparseMatrixCSC{T}, C::Vector{T}) where {T<:Real}

    # number of test testfunctions
    n = size(M,1)
    res = zeros(T,n)
    for k in 1:n
        I, b = findnz(M[k,:])   # determine indices of nonzero entries of row k of matrix A
        K, w = findnz(W[:,k])   # find quadrature points in support of test function i

        c = C[K] .* w
        A = Matrix(B[K,I])' .* c'

        w, res[k] = solve_leastnorm_without_constraints(A, b, BigFloat)

        W[K,k] = w.*c
    end
    e = norm(res)
    @assert norm(res)<eps(T) "The quadrature rules are not up to machine precision."
end

"""
    WeightedQuadrule{T<:Real} <: AbstractQuadrule{1}

Weighted quadrature rules are test function specific quadrature rules that exactly
integrate all functions in a targetspace. The rules are computed in high precision
and are accurate up to 16 digits.
"""
struct WeightedQuadrule{T<:Real} <: AbstractQuadrule{1}
    x::IncreasingVector{T}
    w::SparseMatrixCSC{T}
    function WeightedQuadrule(x::IncreasingSequence{T}, w::SparseMatrixCSC{T}) where T
        @assert length(x)==size(w,1) "Number of points are not consistent with number of weights."
        return new{T}(x,w)
    end
end

function WeightedQuadrule(S::TargetSpace, V::TestSpace, x::IncreasingVector{T}; gradient::Bool=false) where {T<:Real}

    # initialize test TestSpace
    p = Degree(V)
    U = KnotVector(V)
    n = dimsplinespace(p, U)

    # compute quadrature weights
    M = system_matrix(S, V)
    B = bspline_interpolation_matrix(Degree(S), KnotVector(S), x, 1)[1]
    W = bspline_interpolation_matrix(p, U, x, 1)[1]
    c = normalization_weights(x)
    wquadweights!(M, B, W, c)

    # apply differential operator
    if gradient
        c = unit_integral_rescaling(p, U)
        D = sparse_difference_matrix(n+1)
        W = (W .* c') * D'
    end

    return WeightedQuadrule(x, W)
end

function WeightedQuadrule(S::TargetSpace, V::TestSpace; gradient::Bool=false, add_boundary_points::Bool=false)
    x = distribute_points(S, V; add_boundary_points=add_boundary_points)
    return WeightedQuadrule(S, V, x; gradient=gradient)
end
