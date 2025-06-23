module UnivariateSplines

using LinearAlgebra, SparseArrays, MAT, FastGaussQuadrature
using SortedSequences, IgaBase, AbstractMappings

import IgaBase: dimsplinespace, dimension, codimension

include("base.jl")
include("bsplinebasisfuns.jl")
include("dersbsplinebasisfuns.jl")
include("interpolation.jl")
include("quasiinterpolation.jl")
include("refinement.jl")
include("spaces.jl")
include("iterators.jl")
include("gaussquad.jl")
include("gengaussquad.jl")
include("weightedquad.jl")
include("bsplines.jl")
include("mappings.jl")
include("plotting.jl")

end # module
