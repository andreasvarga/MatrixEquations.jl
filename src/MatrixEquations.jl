module MatrixEquations

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using LinearAlgebra: require_one_based_indexing
import LinearAlgebra: mul!
using LinearMaps
#using JLD  


include("lapackutil.jl")
using .LapackUtil: tgsyl!, lanv2, ladiv, lag2, lacn2!, safemin, smlnum

export MatrixEquationsMaps
export utqu, utqu!, qrupdate!, rqupdate!, isschur, triu2vec, vec2triu, utnormalize!
export lanv2, ladiv, lag2, lacn2!
export _lanv2, _safemin, _lag2, _ladiv
export lyapc, lyapd, lyapcs!, lyapds!, tlyapc, tlyapcu!, hlyapc, hlyapcu! 
export plyapc, plyaps, plyapcs!, plyapd, plyapds!  
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvcs!, sylvcs1!, sylvcs2!, sylvds!, gsylvs!
export sylvsys, dsylvsys, sylvsyss!, dsylvsyss!, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr, 
       tsylvckr, hsylvckr, csylvckr, tsylvdkr, hsylvdkr, csylvdkr, tlyapckr, hlyapckr
export opnorm1, opnorm1est, oprcondest, opsepest
export lyapop, invlyapop, sylvop, invsylvop, sylvsysop, invsylvsysop, trmatop

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sylvkr.jl")
include("plyapunov.jl")
include("meoperators.jl")
include("condest.jl")
 
end
