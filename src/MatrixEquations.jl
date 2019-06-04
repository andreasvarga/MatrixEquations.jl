module MatrixEquations

using LinearAlgebra
using LapackUtil

export utqu, utqu!
export lyapc, lyapd, lyapcs!, lyapds!
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvds!, gsylvs!
export sylvsys, dsylvsys, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
include("lapackutil.jl")
include("sylvkr.jl")
end
