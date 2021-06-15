module MatrixEquations
# Release V1.4

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using LinearOperators
#using StaticArrays
import LinearAlgebra: require_one_based_indexing


include("lapackutil.jl")
using .LapackUtil: tgsyl!, lanv2, ladiv, lag2, lacn2!, safemin, smlnum

export utqu, utqu!, qrupdate!, rqupdate!, isschur, triu2vec, vec2triu, utnormalize!
export lanv2, ladiv, lag2, lacn2!
export lyapc, lyapd, lyapcs!, lyapds! 
export plyapc, plyaps, plyapcs!, plyapd, plyapds!  
export arec, ared, garec, gared
export sylvc, sylvd, gsylv, sylvcs!, sylvds!, gsylvs!, gsylvs1!
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
# fallback for versions prior 1.3
# if VERSION < v"1.3.0" 
#     mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasReal} = 
#                             BLAS.gemm!('N', 'N', α, A, B, β, C)
#     mul!(C::StridedMatrix{T}, adjA::Transpose{T,<:StridedMatrix{T}}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasReal} = 
#                             BLAS.gemm!('T', 'N', α, parent(adjA), B, β, C)
#     mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, adjB::Transpose{T,<:StridedMatrix{T}}, α::T, β::T) where {T<:BlasReal} = 
#                             BLAS.gemm!('N', 'T', α, A, parent(adjB),  β, C)
#     mul!(C::StridedMatrix{T}, adjA::Transpose{T,<:StridedMatrix{T}}, adjB::Transpose{T,<:StridedMatrix{T}}, α::T, β::T) where {T<:BlasReal} = 
#                             BLAS.gemm!('T', 'T', α, parent(adjA), parent(adjB),  β, C)
#     # mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasComplex} = 
#     #                         BLAS.gemm!('N', 'N', α, A, B, β, C)
#     # mul!(C::StridedMatrix{T}, adjA::Adjoint{T,<:StridedMatrix{T}}, B::StridedMatrix{T}, α::T, β::T) where {T<:BlasComplex} = 
#     #                         BLAS.gemm!('C', 'N', α, parent(adjA), B, β, C)
#     # mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, adjB::Adjoint{T,<:StridedMatrix{T}}, α::T, β::T) where {T<:BlasComplex} = 
#     #                         BLAS.gemm!('N', 'C', α, A, parent(adjB),  β, C)
#     # mul!(C::StridedMatrix{T}, adjA::Adjoint{T,<:StridedMatrix{T}}, adjB::Adjoint{T,<:StridedMatrix{T}}, α::T, β::T) where {T<:BlasComplex} = 
#     #                         BLAS.gemm!('C', 'C', α, parent(adjA), parent(adjB),  β, C)
#     mul!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T<:BlasReal} = 
#          mul!(C,A,B,one(T),zero(T))
#  end
 
end
