export  dimension, codimension, ScalarFunction, GeometricMapping, Gradient, Hessian

import IgaBase: dimension, codimension
import AbstractMappings: ScalarFunction, GeometricMapping, Gradient, Hessian

AbstractMappings.n_input_args(bspline::Bspline) = 1