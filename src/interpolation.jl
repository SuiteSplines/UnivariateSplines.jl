using SparseArrays

export bspline_interpolation_matrix, ders_bspline_interpolation_matrix

"""
    bspline_interpolation_matrix(p, U, i, x, nout=1)

Given the polynomial degree ``p`` and knot-vector ``U`` compute the collocation
matrices of the B-spline basis functions sampled at a set of points ``x``. The output
is a set of sparse matrices whose rows correspond to the evaluation points and
whose columns are associated with the B-spline degrees of freedom.

# Examples:
```jldoctest collocation
julia> p = Degree(2);

julia> U = KnotVector([0.0,1.0,1.5,2.5,3.0], [3,1,1,2,3]);

julia> x = IncreasingVector([0.5,1.5,2.75]);

julia> B = bspline_interpolation_matrix(p, U, x, p+1);
```
The interpolation matrix for quadratic B-splines is
```jldoctest collocation
julia> B[1]
3×7 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
 0.25  0.583333  0.166667   ⋅         ⋅     ⋅    ⋅ 
  ⋅     ⋅        0.666667  0.333333  0.0    ⋅    ⋅ 
  ⋅     ⋅         ⋅         ⋅        0.25  0.5  0.25
```
The interpolation matrix for linear and constant B-splines
is given by
```jldoctest collocation
julia> B[2]
3×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 0.5  0.5   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0  0.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   0.5  0.5

julia> B[3]
3×5 SparseArrays.SparseMatrixCSC{Float64, Int64} with 3 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0
```
It can be easily verified that the B-splines satisfy a partition of unity
```jldoctest collocation
julia> sum(B[1],dims=2)
3×1 Matrix{Float64}:
 1.0
 1.0
 1.0

julia> sum(B[2],dims=2)
3×1 Matrix{Float64}:
 1.0
 1.0
 1.0

julia> sum(B[3],dims=2)
3×1 Matrix{Float64}:
 1.0
 1.0
 1.0
```
The collocation matrices can be used to solve an interpolation problem or a
PDE using the collocation method. For example, the interpolation matrix
evaluated at the Greville points can be computed as follows
```jldoctest collocation
julia> x = grevillepoints(p, U);

julia> B = bspline_interpolation_matrix(p, U, x, p+1)[1]
7×7 SparseArrays.SparseMatrixCSC{Float64, Int64} with 21 stored entries:
 1.0   0.0        0.0        ⋅          ⋅     ⋅    ⋅ 
 0.25  0.583333   0.166667   ⋅          ⋅     ⋅    ⋅ 
  ⋅    0.0833333  0.833333  0.0833333   ⋅     ⋅    ⋅ 
  ⋅     ⋅         0.166667  0.583333   0.25   ⋅    ⋅ 
  ⋅     ⋅          ⋅         ⋅         1.0   0.0  0.0
  ⋅     ⋅          ⋅         ⋅         0.25  0.5  0.25
  ⋅     ⋅          ⋅         ⋅         0.0   0.0  1.0
```
The matrix ``\\mathsf{B}`` can then be used to interpolate a univariate function. For
example, up to quadratic polynomials are exactly reproduced
```jldoctest collocation
# sample grid to test polynomial reproduction
julia> y = global_insert(x, 4);

julia> C = bspline_interpolation_matrix(p, U, y, 1)[1];

julia> C * (B \\ x.^0) ≈ y.^0
true

julia> C * (B \\ x.^1) ≈ y.^1
true

julia> C * (B \\ x.^2) ≈ y.^2
true
```
Here matrix ``\\mathsf{C}`` evaluates the B-spline at a refined
partition ``y``.
"""
function bspline_interpolation_matrix(p::Degree, U::KnotVector, span::KnotSpanIndices, u::AbstractVector, nout::Int=1)
    @assert length(span)==length(u) "The number of span-indices are incorrect."
    n = dimsplinespace(p, U)
    Nu = bsplinebasisfuns(p, U, span, u, nout)
    return ntuple(i -> collocation_matrix(n+1-i, Nu[1:p+2-i,:,i], span.+(1-i)), nout)
end

function bspline_interpolation_matrix(p::Degree, U::KnotVector, u::AbstractVector, nout::Int=1)
    return bspline_interpolation_matrix(p, U, findspan(p, U, u), u, nout)
end

"""
    ders_bspline_interpolation_matrix(p, U, i, x, nout=1)

Given the polynomial degree ``p`` and knot-vector ``U`` compute the collocation
matrices of the first ``n_{out}`` derivatives of the B-spline basis functions
sampled at a set of points ``x``. The output is a set of sparse matrices whose
rows correspond to the evaluation points and whose columns are associated with
the B-spline degrees of freedom.

# Examples:
```jldoctest collocate_derivatives
julia> p = Degree(2);

julia> U = KnotVector([0.0,1.0,1.5,2.5,3.0], [3,1,1,2,3]);

julia> x = IncreasingVector([0.5,1.5,2.75]);

julia> B = ders_bspline_interpolation_matrix(p, U, x, p+1);
```
The output is a set of `SparseMatrixCSC` matrices. For example
the B-splines evaluate at points ``x`` are
```jldoctest collocate_derivatives
julia> B[1]
3×7 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:                       
 0.25  0.583333  0.166667   ⋅         ⋅     ⋅    ⋅                                            
  ⋅     ⋅        0.666667  0.333333  0.0    ⋅    ⋅                                            
  ⋅     ⋅         ⋅         ⋅        0.25  0.5  0.25                                          
```
Derivatives of quadratic B-splines evaluated at ``x``
```jldoctest collocate_derivatives
julia> B[2]
3×7 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
 -1.0  0.333333   0.666667   ⋅         ⋅    ⋅    ⋅ 
   ⋅    ⋅        -1.33333   1.33333   0.0   ⋅    ⋅ 
   ⋅    ⋅          ⋅         ⋅       -2.0  0.0  2.0
```
Second derivatives of quadratic B-splines evaluated at ``x``
```jldoctest collocate_derivatives
julia> B[3]
3×7 SparseArrays.SparseMatrixCSC{Float64, Int64} with 9 stored entries:
 2.0  -3.33333  1.33333    ⋅        ⋅      ⋅    ⋅ 
  ⋅     ⋅       1.33333  -3.33333  2.0     ⋅    ⋅ 
  ⋅     ⋅        ⋅         ⋅       8.0  -16.0  8.0
```
"""
function ders_bspline_interpolation_matrix(p::Degree, U::KnotVector, span, u, nout::Int=1)
    @assert length(span)==length(u) "The number of span-indices are incorrect."
    n = dimsplinespace(p, U)
    Nu = dersbsplinebasisfuns(p, U, span, u, nout)
    return ntuple(i -> collocation_matrix(n, Nu[:,:,i], span), nout)
end

function ders_bspline_interpolation_matrix(p::Degree, U::KnotVector, span::Int, u::Real, nout::Int=1)
    n = dimsplinespace(p, U)
    Nu = dersbsplinebasisfuns(p, U, span, u, nout)
    return ntuple(i -> collocation_matrix(n, Nu[:,i], span), nout)
end


function ders_bspline_interpolation_matrix(p::Degree, U::KnotVector, u, nout::Int=1)
    return ders_bspline_interpolation_matrix(p, U, findspan(p, U, u), u, nout)
end

# compute a collocation matrix using input from `bsplinebasisfuns` or
# `dersbsplinebasisfuns`
function collocation_matrix(n::Int, Nu, span)

    # initialize
    k = size(Nu,1)
    m = size(Nu,2)
    @assert length(span)==m "The number of span-indices are incorrect."

    # indices sparse matrix format
    I = repeat(collect(1:m)',k, 1)
    J = Int64[span[i]-k+j for j in 1:k, i in 1:m]

    # return matrixes
    return sparse(I[:], J[:], Nu[:], m, n)
end
