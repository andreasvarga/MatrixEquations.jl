module MatrixEquations

using LinearAlgebra

include("lapackutil.jl")
using .LapackUtil


export utqu, utqu!, qrupdate!, rqupdate!
export lanv2, ladiv, lag2
export lyapc, lyapd, lyapcs!, lyapds!
export plyapc, plyaps, plyapcs!, plyapd, plyapds!, plyap2, pglyap2
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvds!, gsylvs!
export sylvsys, dsylvsys, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sylvkr.jl")
include("plyapunov.jl")
end

