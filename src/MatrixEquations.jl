module MatrixEquations
# Release V1.0

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using LinearOperators
import LinearAlgebra: require_one_based_indexing


include("lapackutil.jl")
using .LapackUtil: tgsyl!, lanv2, ladiv, lag2, lacn2!, safemin

export utqu, utqu!, qrupdate!, rqupdate!, isschur, triu2vec, vec2triu
export lanv2, ladiv, lag2, lacn2!
export lyapc, lyapd, lyapcs!, lyapds!, lyapc2, lyapcsylv2, lyapds1!
export plyapc, plyaps, plyapcs!, plyapd, plyapds!, plyap2, pglyap2
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvcs!, sylvds!, gsylvs!
export sylvsys, dsylvsys, sylvsyss!, dsylvsyss!, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr
export opnorm1, opnorm1est, oprcondest, opsepest, trmatop
export lyapop, invlyapop, invlyapsop
export sylvop, invsylvop, invsylvsop
export sylvsysop, invsylvsysop, invsylvsyssop

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sylvkr.jl")
include("plyapunov.jl")
include("meoperators.jl")
include("condest.jl")
end
