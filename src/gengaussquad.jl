export GeneralizedGaussrule

"""
    GeneralizedGaussrule(p, r, nel, a, b)

Compute a generalized Gaussian quadrature rule for a target spline-space
``\\mathbb{S}^p_{r}(a,b)`` defined in a uniform partition with ``n_{el}``
elements.

These rules are made available in the following paper

Hiemstra, René R., et al. "Optimal and reduced quadrature rules for tensor product
and hierarchically refined splines in isogeometric analysis." Computer Methods in
Applied Mechanics and Engineering 316 (2017): 966-1004.
"""
struct GeneralizedGaussrule{T<:Real} <: AbstractQuadrule{1}
    i::IncreasingVector{Int64}
    x::Vector{T}
    w::Vector{T}
end

function GeneralizedGaussrule(u::IncreasingRange; degree::Degree, regularity::Regularity)
    return GeneralizedGaussrule(degree, regularity, u.data.lendiv, u.data.start, u.data.stop)
end

function GeneralizedGaussrule(p::Degree, r::Regularity, e::Dimension, a, b)

    n = dimsplinespace(p, r, e)
    dim_quad_rule = ceil(Int, n/2)

    # construct rule on [0,e]
    Q = construct_reference_rule(select_even_or_odd(n), dim_quad_rule, e, r, p)

    # compute element index of each point
    indices = extract_element_indices(Q.x)
    indices = IncreasingVector([1,indices.+1...,indices[end]+2], false)

    # transform quadrature rule to interval (a,b)
    affine_transform!(Q, Interval(0,e), Interval(a,b))

    # add start and end-points to allow boundary evaluation
    x = [a, Q.x..., b]
    w = [0.0, Q.w..., 0.0]

    return GeneralizedGaussrule(indices, x, w)
end

# compute the element indices of the quadrature points
function extract_element_indices(x::AbstractVector)
    indices = Int64[1]
    value = 1
    while value < last(x)
        index = count_to_value(x, last(indices), value)
        push!(indices, index)
        value +=1
    end
    push!(indices, length(x)+1)

    return indices
end

function count_to_value(x::AbstractVector, start::Int, value::Int)
    index = start
    while x[index] < value
        index+=1
    end
    return index
end

dimsplinespace(p::Degree, r::Regularity, e::Dimension) = (p+1)*2 + (e-1)*(p-r) - p-1

choose_general_rule(::Even, m, m1, m2, m3) = (m>2*m1)
choose_general_rule(::Odd, m, m1, m2, m3) = (m>2*(m1+m3)-1)

function select_general_rule(p::Degree, r::Regularity, e::Dimension, m::Dimension)

    z = (p==4 && r==0) ? e : m

    if iseven(z)
        general_rule = string("/quadrule_even_r$r","_p$p",".mat")
    else
        general_rule = string("/quadrule_odd_r$r","_p$p",".mat")
    end

    return general_rule
end

function select_specific_rule(p::Degree, r::Regularity, e::Dimension, m::Dimension)
    specific_rule = string("/quadrule_e$e","_r$r","_p$p",".mat")
    return specific_rule
end

struct PrecomputedRule{S, T} <: AbstractQuadrule{1}
    x::S
    w::T
end

function read_general_rule(p::Degree, r::Regularity, general_rule)
    cd(joinpath(@__DIR__,"..","data","generalized_gaussian_quadrature")) do
        file = matopen(string("quadrule_p$p", "_r$r",  general_rule))
            bnodes = read(file, "bnodes"); bweights = read(file, "bweights");
            inodes = read(file, "inodes"); iweights = read(file, "iweights")
            mnodes = read(file, "mnodes"); mweights = read(file, "mweights")
        close(file)
        Qb = PrecomputedRule(bnodes, bweights)
        Qi = PrecomputedRule(inodes, iweights)
        Qm = PrecomputedRule(mnodes, mweights)

        return Qb, Qi, Qm
    end
end

function read_specific_rule(p::Degree, r::Regularity, specific_rule)
    cd(joinpath(@__DIR__,"..","data","generalized_gaussian_quadrature")) do
        file = matopen(string("quadrule_p$p", "_r$r", specific_rule))
            nodes = read(file, "nodes")
            weights = read(file, "weights")
        close(file)
    return PrecomputedRule(IncreasingVector(nodes, false), weights)
    end
end

function fill_boundary_rule!(::EvenOrOdd, nodes, weights, Qb, Qi, Qm)
    m, m1 = length(nodes), length(Qb)
    z = ceil(Int,0.5*m)
    nodes[1:m1], weights[1:m1] = Qb.x, Qb.w
end

function fill_interior_rule!(::EvenOrOdd, nodes, weights, Qb, Qi, Qm)

    m, m1, m2 = length(nodes), length(Qb), length(Qi)
    z = ceil(Int,0.5*m)

    left  = ceil(Int, Qb.x[end])
    right = ceil(Int, Qi.x[end])

    ii = m1
    while ii<z
        nodes[ii+1:ii+m2]   = Qi.x .+ left
        weights[ii+1:ii+m2] = Qi.w
        left                = left + right
        ii                  = ii+m2
    end
end

function fill_middle!(::Even, nodes, weights, Qb, Qi, Qm, e)
end

function fill_middle!(::Odd, nodes, weights, Qb, Qi, Qm, e)
    m, m3 = length(nodes), length(Qm)
    z = ceil(Int,0.5*m)

    nodes[z-m3+1:z]   = 0.5*e .+ Qm.x
    weights[z-m3+1:z] = Qm.w
end

function fill_opposite_side_using_symmetry!(::EvenOrOdd, nodes, weights, e)

    m = length(nodes)
    z = ceil(Int,0.5*m)

    nodes[m-z+1:m] = e .- reverse(nodes[1:z])
    weights[m-z+1:m] = reverse(weights[1:z])
end

function construct_specific_rule(p::Degree, r::Regularity, e::Dimension, m::Dimension)
    specific_rule = select_specific_rule(p, r, e, m)
    Q = read_specific_rule(p, r, specific_rule)
    return Q
end

function construct_general_rule(even_or_odd::EvenOrOdd, Qb, Qi, Qm, m, e)

    nodes, weights = zeros(m), zeros(m)
    fill_boundary_rule!(even_or_odd, nodes, weights, Qb, Qi, Qm)
    fill_interior_rule!(even_or_odd, nodes, weights, Qb, Qi, Qm)
    fill_middle!(even_or_odd, nodes, weights, Qb, Qi, Qm, e)
    fill_opposite_side_using_symmetry!(even_or_odd, nodes, weights, e)

    Q = PrecomputedRule(IncreasingVector(nodes, false), weights)
end

function construct_reference_rule(even_or_odd::EvenOrOdd, m::Dimension, e::Dimension, r::Regularity, p::Degree)

    # load data & initialize
    general_rule = select_general_rule(p, r, e, m)
    Qb, Qi, Qm = read_general_rule(p, r, general_rule)
    m1, m2, m3 = length(Qb), length(Qi), length(Qm)

    if choose_general_rule(even_or_odd, m, m1, m2, m3)
        Q = construct_general_rule(even_or_odd, Qb, Qi, Qm, m, e)
    else # load precomputed specific rule
        Q = construct_specific_rule(p, r, e, m)
    end

    return Q
end