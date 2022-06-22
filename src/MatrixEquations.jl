module MatrixEquations

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using LinearAlgebra: require_one_based_indexing
import LinearAlgebra: mul!
using LinearMaps


include("lapackutil.jl")
using .LapackUtil: tgsyl!, lanv2, ladiv, lag2, lacn2!, safemin, smlnum

export MatrixEquationsMaps
export utqu, utqu!, qrupdate!, rqupdate!, isschur, triu2vec, vec2triu, utnormalize!
export lanv2, ladiv, lag2, lacn2!
export lyapc, lyapd, lyapcs!, lyapds! 
export plyapc, plyaps, plyapcs!, plyapd, plyapds!  
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvcs!, sylvds!, gsylvs!
export sylvsys, dsylvsys, sylvsyss!, dsylvsyss!, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr
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
if !occursin(joinpath(".julia", "dev"), pathof(@__MODULE__))
    # Only perform extra precompilation for end users, not for developers
    include("precompilation.jl")
end
end
