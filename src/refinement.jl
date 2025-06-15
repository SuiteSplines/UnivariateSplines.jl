export bezier_extraction_operator, h_refinement_operator!, h_refinement_operator
export refinement_operator

export refine, hRefinement, pRefinement, kRefinement, hpRefinement

import IgaBase: refine, hRefinement, pRefinement, kRefinement, hpRefinement

"""
    refine(p::Degree, R::AbstractRefinement)

Change in polynomial degree based on the refinement type.
"""
refine(p::Degree, method::hRefinement)  = p
refine(p::Degree, method::pRefinement)  = p + method.p
refine(p::Degree, method::kRefinement)  = p + method.p
refine(p::Degree, method::hpRefinement) = p + method.p

"""
    refine(p::KnotVector, R::AbstractRefinement)

Change in knot vector based on the refinement type.
"""
function refine(U::KnotVector, method::hRefinement)
    return global_insert(U, method.h)
end

function refine(U::KnotVector, method::pRefinement)
    u, m = deconstruct_vector(U)
    return KnotVector(u, m.+method.p)
end

function refine(U::KnotVector, method::hpRefinement)
    V = refine(U, hRefinement(method.h))
    return refine(V, pRefinement(method.p))
end

function refine(U::KnotVector, method::kRefinement)
    V = refine(U, pRefinement(method.p))
    return refine(V, hRefinement(method.h))
end

"""
    bezier_extraction_operator(p, U)

Compute the Bézier extraction operators corresponding to a B-spline basis of
polynomial degree ``p`` and knot vector ``U``. The output is a 3-dimensional
array ``\\mathsf{C} \\in \\mathbb{R}^{(p+1) \\times (p+1) \\times n_{el}}``.

The implementation is based on the following paper

Borden, Michael J., et al. "Isogeometric finite element data structures based on
Bézier extraction of NURBS." International Journal for Numerical Methods in
Engineering 87.1‐5 (2011): 15-47.

# Examples:
```jldoctest
julia> p = Degree(2); U = KnotVector([0.0,0.0,0.0,1.0,3.0,3.0,4.0,4.0,4.0]);

julia> bezier_extraction_operator(p, U)
3×3×3 Array{Float64, 3}:
[:, :, 1] =
 1.0  0.0       0.0
 0.0  1.0       0.0
 0.0  0.666667  0.333333

[:, :, 2] =
 0.666667  0.333333  0.0
 0.0       1.0       0.0
 0.0       0.0       1.0

[:, :, 3] =
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
"""
function bezier_extraction_operator(p::Degree, U::KnotVector{T}) where {T<:Real}

    # Initialization:
    a = p+1
    b = a+1
    nb = 1
    m = length(U)
    alphas = zeros(T,p-1)
    C = zeros(T, p+1,p+1,m-2*p)
    C[:,:,1] = Matrix{T}(I, p+1, p+1)

    while(b<m)
        C[:,:,nb+1] = Matrix{T}(I,p+1, p+1)  # Initialize the next extraction operator.
        i = b

        # compute knot multiplicity
        while ((b<m) && (U[b+1]==U[b])) b += 1; end
        mult = b-i+1

        if mult < p

            # numerator of alpa
            numer = U[b] - U[a]

            # compute and store the alfas
            for j in p:-1:mult+1
                alphas[j-mult] = numer / (U[a+j]-U[a])
            end

            # Update the matrix coefficients for r new knots
            r = p - mult
            for j in 1:r
                save = r-j+1
                s = mult+j

                for k in p+1:-1:s+1
                    alpha = alphas[k-s]
                    C[k,:,nb] = alpha * C[k,:,nb] + (1-alpha)*C[k-1,:,nb]
                end

                if b<m
                    C[save,save:j+save,nb+1] = C[p+1,p-j+1:p+1,nb]
                end
            end
        end

        # initialize next operator
        if b<m
            a = b
            b += 1
            nb+=1 # finished with current operator
        end
    end

    return C[:,:,1:nb]
end

"""
    h_refinement_operator!(p::Degree, U::KnotVector{T}, u::AbstractVector{T}) where {T<:Real}
    h_refinement_operator!(p::Degree, U::KnotVector{T}, u::T) where {T<:Real}
    h_refinement_operator!(p::Degree, U::KnotVector{T}, span::Integer, u::T) where {T<:Real}

Insert one or more knots ``u`` into knotvector ``U`` and output the transformation
operator from the coarse to the refined space. The knotvector is updated in-place.
The output is represented as a `SparseMatrixCSC` matrix.

**Examples:**
```jldoctest
julia> p = Degree(2)
2

julia> U = KnotVector([0.0,1.0,2.0,3.0,4], [3,1,1,2,3])
10-element NonDecreasingVector{Float64}:
 0.0
 0.0
 0.0
 1.0
 2.0
 3.0
 3.0
 4.0
 4.0
 4.0

julia> C = h_refinement_operator!(p, U, [0.5, 1.5, 2.5, 3.5]); # sparse matrix

julia> Matrix(C)
11×7 Matrix{Float64}:
 1.0  0.0   0.0   0.0   0.0  0.0  0.0
 0.5  0.5   0.0   0.0   0.0  0.0  0.0
 0.0  0.75  0.25  0.0   0.0  0.0  0.0
 0.0  0.25  0.75  0.0   0.0  0.0  0.0
 0.0  0.0   0.75  0.25  0.0  0.0  0.0
 0.0  0.0   0.25  0.75  0.0  0.0  0.0
 0.0  0.0   0.0   0.5   0.5  0.0  0.0
 0.0  0.0   0.0   0.0   1.0  0.0  0.0
 0.0  0.0   0.0   0.0   0.5  0.5  0.0
 0.0  0.0   0.0   0.0   0.0  0.5  0.5
 0.0  0.0   0.0   0.0   0.0  0.0  1.0

julia> @show U;
U = [0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.5, 4.0, 4.0, 4.0]
```
"""
function h_refinement_operator!(p::Degree, U::KnotVector{T}, span::Integer, u::T) where {T<:Real}

    # Initialize vectors for sparse refinement operator
    n = dimsplinespace(p, U)
    m = n+p+1
    rowindex = zeros(Int, m)
    colindex = zeros(Int, m)
    nzvals   = zeros(T, m)
    k = 1

    for i in 1:span-p
        rowindex[k] = i
        colindex[k] = i
        nzvals[k] = 1.0
        k+=1
    end

    for j in 1:p
        # determine factor of affine map
        alpha = (u - U[span-p+j]) / (U[span+j]-U[span-p+j])

        rowindex[k] = span-p+j
        colindex[k] = span-p+j-1
        nzvals[k] = 1.0 -alpha

        rowindex[k+1] = span-p+j
        colindex[k+1] = span-p+j
        nzvals[k+1] = alpha
        k+=2
    end

    for i in span+1:n+1
        rowindex[k] = i
        colindex[k] = i-1
        nzvals[k] = 1.0
        k+=1
    end

    # adapt knot vector
    insert!(U, span+1, u)

    return sparse(rowindex, colindex, nzvals)
end

function h_refinement_operator!(p::Degree, U::KnotVector{T}, u::T) where {T<:Real}
    return h_refinement_operator!(p, U, findspan(p, U, u), u)
end

function h_refinement_operator!(p::Degree, U::KnotVector{T}, u::AbstractVector{T}) where {T<:Real}

    # dimension spline space
    n = length(u)

    # initialize knot insertion matrix
    C = h_refinement_operator!(p, U, u[1])

    # loop over refinement vector u
    for i in 2:n
        # compute next refinement operator
        temp = h_refinement_operator!(p, U, u[i])

        # update knot refinement matrix
        C = temp * C
    end

    return C
end

"""
    refinement_operator(p::Degree, U::KnotVector, method::AbstractRefinement)

Compute the [`two_scale_operator`](@ref) given `p` and `U` and the applicable refinement strategy:
`IgaBase.hRefinement()`, `IgaBase.pRefinement()`, `IgaBase.kRefinement()`. See `IgaBase`
package for definition of refinement methods.
"""
function refinement_operator(p::Degree, U::KnotVector, method::AbstractRefinement)
    q, V = refine(p, method), refine(U, method)
    return two_scale_operator(p, U, q, V), q, V
end

"""
    two_scale_operator(p::Degree, U::KnotVector{T}, q::Degree, V::KnotVector{T}) where {T<:Real}

The `two_scale_operator` encodes the main logic for all spline refinemement operations.
    
The arguments `p` and `U` correspond to the current spline space and `q` and `V` are the "target" degree
knot vector.
"""
function two_scale_operator(p::Degree, U::KnotVector{T}, q::Degree, V::KnotVector{T}) where {T<:Real}

    # compute interpolation matrices of old and new basis
    n = dimsplinespace(p, U)
    x = grevillepoints(q, V)
    Bold = bspline_interpolation_matrix(p, U, x, 1)[1]
    Bnew = bspline_interpolation_matrix(q, V, x, 1)[1]

    # compute 2-scale relation and store result in the collocation matrix
    # which has the same structure
    for col in 1:n
        rows, vals = findnz(Bold[:,col])
        Bold[rows,col] = Matrix(Bnew[rows, rows]) \ vals
    end

    return droptol!(Bold, 1e2*eps(T))
end
