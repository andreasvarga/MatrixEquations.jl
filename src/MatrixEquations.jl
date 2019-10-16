module MatrixEquations
# Release V1.0

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using LinearOperators


include("lapackutil.jl")
using .LapackUtil: tgsyl!, lanv2, ladiv, lag2, lacn2!

export utqu, utqu!, qrupdate!, rqupdate!, isschur
export lanv2, ladiv, lag2, lacn2!
export lyapc, lyapd, lyapcs!, lyapds!
export plyapc, plyaps, plyapcs!, plyapd, plyapds!, plyap2, pglyap2
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvds!, gsylvs!
export sylvsys, dsylvsys, tgsyl!
export sylvckr, sylvdkr, gsylvkr, sylvsyskr, dsylvsyskr
export lyapsepest, sylvsepest, sylvsyssepest
export opnorm1, opnorm1est, oprcondest, opsepest, trmat
export lyapcop, invlyapcop, invlyapcsop
export lyapdop, invlyapdop, invlyapdsop
export sylvcop, invsylvcop, invsylvcsop
export sylvdop, invsylvdop, invsylvdsop
export gsylvop, invgsylvop, invgsylvsop
export sylvsysop, invsylvsysop, invsylvsyssop

include("meutil.jl")
include("sylvester.jl")
include("lyapunov.jl")
include("riccati.jl")
include("sylvkr.jl")
include("plyapunov.jl")
include("meoperators.jl")
end
