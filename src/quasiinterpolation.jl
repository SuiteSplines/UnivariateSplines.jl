using LinearAlgebra, SparseArrays

export approximate_collocation_inverse, approximate_l2_inverse

"""
    approximate_collocation_inverse(p, U)
    approximate_collocation_inverse(p, U, k1)

Compute an approximate inverse ``\\mathsf{A} ≈ \\mathsf{B}^{-1}``, where ``\\mathsf{B}``
is the consistent B-spline interpolation matrix. The approximation is of order
``k_1 \\leq p+1``. The quasi-interpolant is designed to reproduce polynomials
```math
    \\sum_{i=1}^n \\mu_i(x^k) B_{i,p}(x) = x^k  \\quad \\text{for } k=0,1,...,k_1-1
```
and is based on the following paper

T. Lyche and L. L. Schumaker. Local spline approximation methods. Journal of
Approximation Theory, 15(4):294-325, Dec. 1975.

# Examples:
```jldoctest quasi_interpolation
julia> p  = Degree(3);

julia> U  = KnotVector([0.0,1.0,1.5,2.5,3.0], [p+1,1,1,2,p+1]);

julia> y  = grevillepoints(p, U);
```
The approximate collocation inverse, using full approximation order ``k_1=p+1``
and the consistent collocation matrix are computed as follows
```jldoctest quasi_interpolation
julia> A = approximate_collocation_inverse(p, U, p+1);

julia> B = bspline_interpolation_matrix(p, U, y, 1)[1];
```
It can be verified that the quasi-interpolant reproduces constants, linears,
quadratic and cubic polynomials
```jldoctest quasi_interpolation
julia> B * A * (y.^0) ≈ y.^0
true

julia> B * A * (y.^1) ≈ y.^1
true

julia> B * A * (y.^2) ≈ y.^2
true

julia> B * A * (y.^3) ≈ y.^3
true
```
"""
function approximate_collocation_inverse(p::Int, U::KnotVector{T}) where {T<:Real}
    return approximate_collocation_inverse(p, U, p+1)
end

function approximate_collocation_inverse(p::Int, U::KnotVector{T}, k1::Int) where {T<:Real}

    # initialize
    y = grevillepoints(p, U)
    n = dimsplinespace(p, U)
    A = zeros(T,n,n); A[1,1] = 1.0; A[n,n] = 1.0

    for i in 2:n-1

        # determining the active interpolating sites
        J = ceil(Int,0.5*(k1-1))
        if i < J+1
            index1 = 1:k1
            index2 = 1:k1
        elseif i>n-J
            index1 = n-k1+1:n
            index2 = n-k1+1:n
        else
            if iseven(k1)
                index1 = i-J:i+J-1
                index2 = i-J+1:i+J
            else
                index1 = i-J:i+J
                index2 = i-J:i+J
            end
        end

        A[i,index1] =  0.5*bspline_dual_functional(U[i:i+p+1], y[index1])
        A[i,index2] += 0.5*bspline_dual_functional(U[i:i+p+1], y[index2])
    end
    return sparse(A)
end
# ToDo: implement generalized input and selection of evaluation sites

# Compute approximate dual functional to the B-spline basis function defined
# through the local knot vector kts
function bspline_dual_functional(kts::AbstractVector{T}, x::AbstractVector{T})  where {T<:Real}

    # local approximation order k1 ≦ p+1
    k1 = length(x)
    t = sum(kts[2:end-1]) / (length(kts)-2) # greville point
    ϕ = zeros(T,k1,k1); ϕ[1,1] = 1
    for p in 1:k1-1
        # Newton polynomial of degree p
        ϕ[p+1,1] = ϕ[p,1] * (t - kts[p+1])

        # ith derivative of degree p Newton polynomial
        for i in 1:p
            ϕ[p+1,i+1] =  ϕ[p,i] +  (t - kts[p+1]) * ϕ[p,i+1]
        end
    end

    # calculate temporary coefficients
    γ = zeros(k1)
    for m in k1:-1:1
        γ[m] = (-1)^(m-1) * (prod(1:m-1)/prod(k1-m+1:k1-1)) * ϕ[k1,k1-m+1]
    end

    # calculate devided difference table
    DD = divided_difference_table(k1,x)

    # calculate another set of temporary coefficients
    μ = zeros(T,k1)
    μ[1] = 1.0
    for m in 2:k1
        dd = DD * (x.-t).^(m-1)
        temp = γ[m]
        for j in 1:m-1
            temp = temp - dd[j] * μ[j]
        end
        μ[m] = temp / dd[m]
    end

    # fill local interpolation matrix
    return DD' * μ
end

# compute devided difference table
function divided_difference_table(k1::Int, x::AbstractVector{T}) where {T<:Real}
    # initialize the divided difference table
    DD = zeros(T,k1,k1)
    DD[1,1] = one(T)

    # Calculate rest of table
    for i in 2:k1
        for j in 1:i
            # calculate expanded form of devided differences
            temp = one(T)
            for k in 1:i
                if x[k]!=x[j]
                    temp = temp / (x[j]-x[k])
                end
            end

            # substitute in table
            DD[i,j] = temp
        end
    end
    return DD
end

# calculate Newton polynomials and derivatives in truncated table
function newton_triangular_table(kts, t::T) where T
    n = length(kts)-1
    ϕ = zeros(T,n,n); ϕ[1,1] = 1
    for p in 1:n-1
        # Newton polynomial of degree p
        ϕ[p+1,1] = ϕ[p,1] * (t - kts[p+1])

        # ith derivative of degree p Newton polynomial
        for i in 1:p
            ϕ[p+1,i+1] =  i * ϕ[p,i] + (t - kts[p+1]) * ϕ[p,i+1]
        end
    end
    return ϕ
end

# calculate Newton polynomials and derivatives in truncated table
function deboor_fix_dual_functional_coeffs(kts, t::T) where T
    
    # polynomial order
    k1 = length(kts)-1

    # calculate newton polynomials
    ϕ = newton_triangular_table(kts, t)

    # calculate temporary coefficients
    γ = zeros(k1)
    num = factorial(k1-1)
    for m in 1:k1
        γ[m] = (-1)^(m-1) * ϕ[k1,k1-m+1] / num
    end

    return γ
end

"""
    approximate_l2_inverse(p, U, k1)

Compute an approximate inverse ``\\mathsf{S} ≈ \\mathsf{M}^{-1}``, where ``\\mathsf{M}``
is the consisten B-spline mass matrix. The approximation is of order ``k_1 \\leq p+1``.
The ``L^2`` quasi-interpolant is designed to reproduce polynomials
```math
    \\sum_{i=1}^n \\mu_i(x^k) B_{i,p}(x) = x^k  \\quad \\text{for } k=0,1,...,k_1-1
```
and is based on the following paper

Chui, Charles K., Wenjie He, and Joachim Stöckler. "Nonstationary tight wavelet
frames, I: Bounded intervals." Applied and Computational Harmonic Analysis 17.2
(2004): 141-197.
"""
function approximate_l2_inverse(p::Int, U::AbstractVector{T}, k1::Int) where {T<:Real}

    # initialize
    n = length(U)-p-1
    D = sparse(I, n, n)

    # compute S recursively
    S = spdiagm(n, n, 0 => inverse_mass_scaling(p,0,U))
    for v=1:k1-1
        D = D * weighted_difference(U, p+v)
        S = S + D * spdiagm(n-v, n-v, 0 => inverse_mass_scaling(p,v,U)) * D'
    end
    return S
end
#ToDo: Add doc-examples

# construct n-1 x n sparse representation of scaled difference matrix
function weighted_difference(U::AbstractVector{T}, k::Int) where {T<:Real}
    n = length(U)-k
    A = k ./ (U[k+1:end] - U[1:end-k])
    return spdiagm(n, n-1, -1 => -A[2:end], 0 => A[1:end-1])
end

# compute factors U
function inverse_mass_scaling(p::Int, v::Int, U::AbstractVector{T}) where {T<:Real}

    # initialize
    V = Val(v)
    r = p+v
    n = length(U)-p-v-1
    c = factorial(p+1)*factorial(p-v) / (factorial(p+1+v)*factorial(p+v))
    u = zeros(n)

    # compute factors u depending on multivalued generalized blossoms
    for k=1:n
        u[k] = (c * (p+1+v)/(U[k+p+1+v]-U[k])) * blossom_label(V, r, moment(U[k+1:k+r], v))
    end
    return u
end

import Statistics.mean

# compute centered moments of degree
function moment(U::AbstractVector{T}, v::Int) where {T<:Real}
    x = mean(U)
    σ = zeros(2*v, 1)
    tau = U.-x
    for l=2:2*v
        σ[l] = mean(tau.^l)
    end
    return σ
end

# L2 multi-valued generalized blossom labels
blossom_label(::Val{0}, r::Int, σ) = 1.0
blossom_label(::Val{1}, r::Int, σ) = r^2 * σ[2]
blossom_label(::Val{2}, r::Int, σ) = 0.5*(r^2*(r^2-3*r+3)*σ[2]^2 - r^2*(r-1)*σ[4])
blossom_label(::Val{3}, r::Int, σ) = (1/6)*(r^3*(r-2)*(r^2-7*r+15)*σ[2]^3 -3*r^2*(r-2)*(r^2-5*r+10)*σ[4]*σ[2] -2*r^2*(3*r^2-15*r+20)*σ[3]^2 + 2*r^2*(r-1)*(r-2)*σ[6])
blossom_label(::Val{4}, r::Int, σ) = (1/24)*(r^4*(r^4 -18*r^3 +125*r^2 -384*r +441)*σ[2]^4 -6*r^3*(r^4 -16*r^3 +104*r^2 -305*r + 336)*σ[4]*σ[2]^2 +3*r^2*(r^4 -14*r^3 +95*r^2 -322*r + 420)*σ[4]^2 +8*r^2*(r-2)*(r-3)*(r^2-7*r+21)*σ[6]*σ[2] -8*r^3*(r-3)*(3*r^2-24*r+56)*σ[3]^2*σ[2] +48*r^2*(r-3)*(r^2-7*r+14)*σ[5]*σ[3] -6*r^2*(r-1)*(r-2)*(r-3)*σ[8])
blossom_label(::Val{5}, r::Int, σ) = (1/120)*(r^5*(r-4)*(r^4-26*r^3+261*r^2-1176*r+2025)*σ[2]^5 -10*r^4*(r-4)*(r^4-24*r^3+230*r^2-999*r+1674)*σ[4]*σ[2]^3 +20*r^3*(r-4)*(r^4-20*r^3+168*r^2-645*r+972)*σ[6]*σ[2]^2 +15*r^3*(r-4)*(r^4-22*r^3+211*r^2-942*r+1620)*σ[4]^2*σ[2] -20*r^4*(3*r^4-60*r^3+470*r^2-1665*r+2232)*σ[3]^2*σ[2]^2 -30*r^2*(r-2)*(r-3)*(r-4)*(r^2-9*r+36)*σ[8]*σ[2] -20*r^2*(r-4)*(r^4-18*r^3+173*r^2-828*r+1512)*σ[6]*σ[4] +240*r^3*(r^4-19*r^3+143*r^2-493*r+648)*σ[5]*σ[3]*σ[2] +20*r^4*(r-4)*(3*r^2-30*r+83)*σ[4]*σ[3]^2 -24*r^2*(5*r^4-90*r^3+655*r^2-2250*r+3024)*σ[5]^2 -240*r^2*(r-3)*(r-4)*(r^2-9*r+24)*σ[7]*σ[3] +24*r^2*(r-1)*(r-2)*(r-3)*(r-4)*σ[10])
