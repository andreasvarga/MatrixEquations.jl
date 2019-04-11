module MatrixEquations

using LinearAlgebra

export utqu, utqu!
export lyapc, glyapc, lyapd, glyapd, lyapds!, glyapds!, lyapcs!, glyapcs!
export arec, ared, garec, gared
export sylvc, sylvckr, sylvdkr

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
end
