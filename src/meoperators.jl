# TODO: These types aren't used anywhere for multiple dispatch, so why have them?
abstract type LyapunovMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
abstract type SylvesterMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
const MatrixEquationsMaps{T} = Union{LyapunovMatrixEquationsMaps{T},SylvesterMatrixEquationsMaps{T}}

abstract type ContinuousOrDiscrete end
struct Continuous <: ContinuousOrDiscrete end
struct Discrete <: ContinuousOrDiscrete end
abstract type TtypeOrHtype end
struct Ttype <: TtypeOrHtype end
struct Htype <: TtypeOrHtype end


struct trmatop{T} <: LinearMaps.LinearMap{T}
   size::Dims{2}
   function trmatop{T}(dims::Dims{2}) where {T}
      all(≥(0), dims) || throw(ArgumentError("dims must be non-negative"))
      return new{T}(dims)
   end
   function trmatop(m::Int, n::Int)
      (m ≥ (0) & n ≥ (0))  || throw(ArgumentError("dimensions must be non-negative"))
      return new{Bool}((m, n))
   end
end
trmatop(dims::Dims{2}) = trmatop{Bool}(dims)
trmatop(n::Int) = trmatop(n, n)
Base.size(A::trmatop) = (N = prod(A.size); return (N,N))
LinearAlgebra.issymmetric(A::trmatop) = true
LinearAlgebra.ishermitian(A::trmatop) = true
function LinearMaps._unsafe_mul!(y::AbstractVector, A::trmatop, x::AbstractVector)
   (m, n) = A.size
   k = 1
   for j = 1:n
      for i = 1:m
          y[(i-1)*n+j] = x[k]
          k += 1
      end
   end
   return y
end

"""
    M = trmatop(n, m)
    M = trmatop(A)

Define the transposition operator `M: X -> X'` for all `n x m` matrices or for all matrices of size of `A`. 
The corresponding commutation matrix (see [here](https://en.wikipedia.org/wiki/Commutation_matrix)) 
can be generated as `Matrix(M)`.
"""
function trmatop(A)
    trmatop{eltype(A)}(size(A))
end
"""
    M = eliminationop(n)
    M = eliminationop(A)

Define the elimination operator of all `n×n` matrices to select their upper triangular parts or of all square matrices of size of `A`.  
See [here](https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices) for the 
definition of an elimination matrix for the selection of lower triangular parts.
"""
struct eliminationop{T} <: LinearMaps.LinearMap{T}
   dim::Int
   function eliminationop{T}(dim::Int) where {T}
      dim >= 0 || throw(ArgumentError("dimension must be non-negative"))
      return new{T}(dim)
   end
   function eliminationop(n::Int)
      (n ≥ (0))  || throw(ArgumentError("dimension must be non-negative"))
      return new{Bool}(n)
   end
end
Base.size(A::eliminationop) = (n = A.dim; return (div(n*(n+1),2), n*n))
LinearAlgebra.adjoint(A::eliminationop{T}) where {T} = LinearAlgebra.transpose(A)


function LinearMaps._unsafe_mul!(y::AbstractVector, A::eliminationop, x::AbstractVector)
   # X = reshape(x, A.dim, A.dim)
   # y[:] = triu2vec(X)
   n = A.dim
   k = 1
   for j = 1:n
      for i = 1:j
          y[k] = x[(i-1)*n + j]
          k += 1
      end
   end
   return y
end
function LinearMaps._unsafe_mul!(x::AbstractVector, AT::LinearMaps.TransposeMap{T,<:eliminationop{T}}, y::AbstractVector) where {T}
   # X = vec2triu(y)
   n = AT.lmap.dim
   ZERO = zero(eltype(y))
   @inbounds begin
      k = 1
      for j = 1:n
          for i = 1:j
              x[(i-1)*n + j] = y[k]
              k += 1
          end
          for i = j+1:n
              x[(i-1)*n + j] = ZERO
          end
      end
    end
   return x
end
eliminationop(A) = eliminationop{eltype(A)}(LinearAlgebra.checksquare(A))

"""
    M = duplicationop(n)
    M = duplicationop(A)

Define the duplication operator of all `n×n` matrices to reconstruct a hermitian matrix from its upper triangular elements 
or of all square matrices of size of `A`.
See [here](https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices) for the 
definition of a duplication matrix from the lower triangular parts.

"""
struct duplicationop{T} <: LinearMaps.LinearMap{T}
   dim::Int
   function duplicationop{T}(dim::Int) where {T}
      dim >= 0 || throw(ArgumentError("dimension must be non-negative"))
      return new{T}(dim)
   end
   function duplicationop(n::Int)
      (n ≥ (0))  || throw(ArgumentError("dimension must be non-negative"))
      return new{Bool}(n)
   end
end
Base.size(A::duplicationop) = (n = A.dim; return (n*n, div(n*(n+1),2)))
LinearAlgebra.adjoint(A::duplicationop{T}) where {T} = LinearAlgebra.transpose(A)


function LinearMaps._unsafe_mul!(y::AbstractVector, A::duplicationop, x::AbstractVector)
   # y[:] = vec2triu(x, her = true)
   n = A.dim
   @inbounds begin
      k = 1
      for j = 1:n
          for i = 1:j
              y[(i-1)*n + j] = x[k]
              k += 1
          end
          for i = 1:j-1
              y[(j-1)*n + i] = conj(y[(i-1)*n + j])
          end
      end
    end
    return y
end

function LinearMaps._unsafe_mul!(x::AbstractVector, AT::LinearMaps.TransposeMap{T,<:duplicationop{T}}, y::AbstractVector) where {T}
   n = AT.lmap.dim
   # Y = reshape(y, n, n)
   # x[:] = triu2vec(Y+transpose(Y)-Diagonal(Y))
   @inbounds begin
      k = 1
      for j = 1:n
         for i = 1:j
            #x[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
            x[k] = i == j ? y[(j-1)*n + j] : y[(i-1)*n + j] + y[(j-1)*n + i]
            k += 1
         end
      end
   end
   return x
end
duplicationop(A) = duplicationop{eltype(A)}(LinearAlgebra.checksquare(A))



struct LyapLikeMap{T,TA <: AbstractMatrix,CD <: ContinuousOrDiscrete,TH<:TtypeOrHtype} <: LyapunovMatrixEquationsMaps{T}
   A::TA
   adj::Bool
   isig::Int
   function LyapLikeMap{T,TA,CD,TH}(A::TA; isig = 1, adj=false) where {T,TA<:AbstractMatrix{T},CD,TH}
      abs(isig) == 1 || throw(ArgumentError("only 1 or -1 values allowed for isig; got isig = $isig"))
      return new{T,TA,CD,TH}(A, adj, isig)
   end
end
LyapLikeMap(A::TA, ::CD = Continuous(), ::TH = Ttype(); adj::Bool = false, isig::Int = 1) where {T,TA<:AbstractMatrix{T},CD<:ContinuousOrDiscrete,TH<:TtypeOrHtype} =
   LyapLikeMap{T,TA,CD,TH}(A; adj, isig)

function Base.size(L::LyapLikeMap)
   m, n = size(L.A)
   return (m*m, m*n)
end

"""
    L = lyaplikeop(A; isig = 1, adj = false, htype = false)

For a matrix `A`, define for `adj = false` the continuous T-Lyapunov operator `L:X -> A*X+isig*transpose(X)*transpose(A)` if `htype = false`
or the continuous H-Lyapunov operator `L:X -> A*X+isig*X'*A'` if `htype = true`, or 
define for `adj = true` the continuous T-Lyapunov operator `L:X -> A*transpose(X)+X*transpose(A)` if `htype = false`,
or the continuous H-Lyapunov operator  `L:X -> A*X'+isig*X*A'` if `htype = true`. 
"""
function lyaplikeop(A::AbstractMatrix; disc=false, adj=false, isig = 1, htype = false)
    LyapLikeMap(A, ifelse(disc, Discrete(), Continuous()), ifelse(htype, Htype(), Ttype()); adj, isig)
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::LyapLikeMap{T,TA,Continuous,Ttype}, x::AbstractVector) where {T,TA}
   m, n = size(L.A)
   T1 = promote_type(T, eltype(x))
   X = L.adj ? reshape(convert(AbstractVector{T1}, x), m, n) : reshape(convert(AbstractVector{T1}, x), n, m)
   Y = L.adj ? L.A*transpose(X) : L.A*X
   y[:] .= L.isig == 1 ? (Y+transpose(Y))[:] : (Y-transpose(Y))[:]
   return y
end

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.TransposeMap{T,<:LyapLikeMap{T,<:Any,Continuous,Ttype}}, y::AbstractVector) where {T}
   m = size(LT.lmap.A,1)
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), m, m)
   temp = LT.lmap.isig ==1 ? Y+transpose(Y) : (LT.lmap.adj ? transpose(Y)-Y : Y-transpose(Y))
   x[:] = LT.lmap.adj ? (temp*LT.lmap.A)[:] : (transpose(LT.lmap.A)*temp)[:]
   return x
end

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.AdjointMap{T,<:LyapLikeMap{T,<:Any,Continuous,Ttype}}, y::AbstractVector) where {T}
   m = size(LT.lmap.A,1)
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), m, m)
   temp = LT.lmap.isig ==1 ? Y+transpose(Y) : (LT.lmap.adj ? transpose(Y)-Y : Y-transpose(Y))
   x[:] = LT.lmap.adj ? (temp*conj(LT.lmap.A))[:] : (LT.lmap.A'*temp)[:]
   return x
end

function LinearMaps._unsafe_mul!(y::AbstractVector, L::LyapLikeMap{T,TA,Continuous,Htype}, x::AbstractVector) where {T,TA}
   m, n = size(L.A)
   T1 = promote_type(T, eltype(x))
   X = L.adj ? reshape(convert(AbstractVector{T1}, x), m, n) : reshape(convert(AbstractVector{T1}, x), n, m)
   Y = L.adj ? L.A*X' : L.A*X
   y[:] .= L.isig == 1 ? (Y+Y')[:] : (Y-Y')[:]
   return y
end

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.TransposeMap{T,<:LyapLikeMap{T,<:Any,Continuous,Htype}}, y::AbstractVector) where {T}
   m = size(LT.lmap.A,1)
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), m, m)
   temp = LT.lmap.isig ==1 ? Y+transpose(Y) : (LT.lmap.adj ? transpose(Y)-Y : Y-transpose(Y))
   x[:] = LT.lmap.adj ? (temp*LT.lmap.A)[:] : (transpose(LT.lmap.A)*temp)[:]
   return x
end

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.AdjointMap{T,<:LyapLikeMap{T,<:Any,Continuous,Htype}}, y::AbstractVector) where {T}
   m = size(LT.lmap.A,1)
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), m, m)
   temp = LT.lmap.isig ==1 ? Y+Y' : (LT.lmap.adj ? Y'-Y : Y-Y')
   x[:] = LT.lmap.adj ? (temp*LT.lmap.A)[:] : (LT.lmap.A'*temp)[:]
   return x
end


struct UTLyapLikeMap{T,TU <: AbstractMatrix,CD <: ContinuousOrDiscrete,TH<:TtypeOrHtype} <: LyapunovMatrixEquationsMaps{T}
   U::TU
   adj::Bool
   function UTLyapLikeMap{T,TU,CD,TH}(U::TU, adj=false) where {T,TU<:AbstractMatrix{T},CD,TH}
      LinearAlgebra.checksquare(U)
      istriu(U) || error("U must be upper triangular")
      #(!adj && istriu(U.parent)) || (adj && istriu(U)) || error("U must be upper triangular")
      return new{T,TU,CD,TH}(U, adj)
   end
end

UTLyapLikeMap(U::TU, ::CD = Continuous(), ::TH = Ttype(); adj::Bool = false) where {T,TU<:AbstractMatrix{T},CD<:ContinuousOrDiscrete,TH<:TtypeOrHtype} =
   UTLyapLikeMap{T,TU,CD,TH}(U, adj)

function Base.size(L::UTLyapLikeMap)
   n = size(L.U, 1)
   N = Int(n * (n + 1) / 2)
   return (N, N)
end

"""
    L = tulyaplikeop(U; adj = false)

Define, for an upper triangular matrix `U`, the continuous T-Lyapunov operator `L:X -> transpose(U)*X+transpose(X)*U`, if `adj = false`,
or `L:X -> U*transpose(X)+X*transpose(U)` if `adj = true`.
"""
function tulyaplikeop(U::AbstractMatrix; disc=false, adj = false)
   UTLyapLikeMap(U, ifelse(disc, Discrete(), Continuous()), Ttype(); adj )
end
"""
    L = hulyaplikeop(U; adj = false)

Define, for an upper triangular matrix `U`, the continuous H-Lyapunov operator `L:X -> U'*X+X'*U`, if `adj = false`,
`L:X -> U*X'+X*U'` if `adj = true`. 
"""
function hulyaplikeop(U::AbstractMatrix; disc=false, adj = false)
   UTLyapLikeMap(U, ifelse(disc, Discrete(), Continuous()), Htype(); adj)
end

function LinearMaps._unsafe_mul!(y::AbstractVector, L::UTLyapLikeMap{T,TU,Continuous,Ttype}, x::AbstractVector) where {T,TU}
   T1 = promote_type(T, eltype(x))
   X = vec2triu(convert(AbstractVector{T1}, x), her=false)
   Y = L.adj ? L.U*transpose(X) : transpose(L.U)*X
   y[:] .= triu2vec(Y+transpose(Y))
   return y
end

#LinearAlgebra.adjoint(L::UTLyapLikeMap{T,TU,Continuous,Ttype}) where {T,TU}= LinearAlgebra.transpose(L)
#LinearAlgebra.transpose(L::UTLyapLikeMap{T,TU,Continuous,Ttype}) where {T,TU}= LinearAlgebra.adjoint(L)

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.TransposeMap{T,<:UTLyapLikeMap{T,<:Any,Continuous,Ttype}}, y::AbstractVector) where {T}
   n = size(LT.lmap.U,2)
   T1 = promote_type(T, eltype(y))
   Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
   x[:] = LT.lmap.adj ? triu2vec((Y+transpose(Y))*LT.lmap.U) : triu2vec(LT.lmap.U*(Y+transpose(Y)))
   #x[:] = LT.lmap.adj ? triu2vec(Y*LT.lmap.U+Y'*conj(LT.lmap.U)) : triu2vec(LT.lmap.U*Y+conj(LT.lmap.U)*Y')
   return x
end
function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.AdjointMap{T,<:UTLyapLikeMap{T,<:Any,Continuous,Ttype}}, y::AbstractVector) where {T}
   n = size(LT.lmap.U,2)
   T1 = promote_type(T, eltype(y))
   Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
   #x[:] = LT.lmap.adj ? triu2vec((Y+Y')*conj(LT.lmap.U)) : triu2vec(conj(LT.lmap.U)*(Y+Y'))
   x[:] = LT.lmap.adj ? triu2vec((Y+transpose(Y))*conj(LT.lmap.U)) : triu2vec(conj(LT.lmap.U)*(Y+transpose(Y)))
   return x
end

function LinearMaps._unsafe_mul!(y::AbstractVector, L::UTLyapLikeMap{T,TU,Continuous,Htype}, x::AbstractVector) where {T,TU}
   T1 = promote_type(T, eltype(x))
   X = vec2triu(convert(AbstractVector{T1}, x), her=false)
   Y = L.adj ? L.U*X' : L.U'*X
   y[:] .= triu2vec(Y+Y')
   return y
end

# function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.TransposeMap{T,<:UTLyapLikeMap{T,TU,Continuous,Htype}}, y::AbstractVector) where {T,TU}
#    n = size(LT.lmap.U,2)
#    T1 = promote_type(T, eltype(y))
#    Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
#    x[:] = LT.lmap.adj ? triu2vec(Y*LT.lmap.U+Y'*conj(LT.lmap.U)) : triu2vec(LT.lmap.U*Y+conj(LT.lmap.U)*Y')
#    return x
# end

function LinearMaps._unsafe_mul!(x::AbstractVector, LT::LinearMaps.AdjointMap{T,<:UTLyapLikeMap{T,TU,Continuous,Htype}}, y::AbstractVector) where {T,TU}
   n = size(LT.lmap.U,2)
   T1 = promote_type(T, eltype(y))
   Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
  # the following ensures Matrix(L') == Matrix(L)' 
   #x[:] = LT.lmap.adj ? triu2vec(Y*LT.lmap.U+Y'*conj(LT.lmap.U)) : triu2vec(LT.lmap.U*Y+conj(LT.lmap.U)*Y')
   x[:] = LT.lmap.adj ? triu2vec((Y+Y')*LT.lmap.U) : triu2vec(LT.lmap.U*(Y+Y'))
   return x
end
# function LinearMaps._unsafe_mul!(y::AbstractVector, LT::LinearMaps.AdjointMap{T,<:UTHLyapunovMap{T}}, x::AbstractVector) where {T}
#    n = size(LT.lmap.U,2)
#    T1 = promote_type(T, eltype(x))
#    X = vec2triu(convert(AbstractVector{T1}, x), her=false)
#    y[:] = LT.lmap.adj ? triu2vec((X+X')*LT.lmap.U) : triu2vec(LT.lmap.U'*(X+X'))
#    return y
# end



# struct UTLyapLikeMap{T,TU <: AbstractMatrix,CD <: ContinuousOrDiscrete} <: LyapunovMatrixEquationsMaps{T}
#    U::TU
#    adj::Bool
#    function UTLyapLikeMap{T,TU,CD}(U::TU, adj=false) where {T,TU<:AbstractMatrix{T},CD}
#       LinearAlgebra.checksquare(U)
#       (!adj && istriu(U.parent)) || (adj && istriu(U)) || error("U must be upper triangular")
#       return new{T,TU,CD}(U, adj)
#    end
# end

# UTLyapLikeMap(U::TU, ::CD = Continuous(), adj::Bool = false) where {T,TU<:AbstractMatrix{T},CD<:ContinuousOrDiscrete} =
#    UTLyapLikeMap{T,TU,CD}(U, adj)

# #LinearAlgebra.transpose(L::UTLyapLikeMap{<:Any,<:Any,CD}) where {CD} = LinearAlgebra.adjoint(L)

# function Base.size(L::UTLyapLikeMap)
#    n = size(L.U, 1)
#    N = Int(n * (n + 1) / 2)
#    return (N, N)
# end

# function LinearMaps._unsafe_mul!(y::AbstractVector, L::UTLyapLikeMap{T,<:Any,Continuous,Htype}, x::AbstractVector) where {T,TU}
#    T1 = promote_type(T, eltype(x))
#    X = vec2triu(convert(AbstractVector{T1}, x), her=false)
#    Y = L.adj ? L.U*X' : L.U*X
#    y[:] .= triu2vec(Y+Y')
#    return y
# end

# function LinearMaps._unsafe_mul!(y::AbstractVector, LT::LinearMaps.AdjointMap{T,<:UTLyapLikeMap{T}}, x::AbstractVector) where {T}
#    n = size(LT.lmap.U,2)
#    T1 = promote_type(T, eltype(x))
#    X = vec2triu(convert(AbstractVector{T1}, x), her=false)
#    y[:] = LT.lmap.adj ? triu2vec((X+X')*LT.lmap.U) : triu2vec(LT.lmap.U'*(X+X'))
#    return y
# end

struct LyapunovMap{T,TA <: AbstractMatrix,CD <: ContinuousOrDiscrete} <: LyapunovMatrixEquationsMaps{T}
   A::TA
   her::Bool
   function LyapunovMap{T,TA,CD}(A::TA, her=false) where {T,TA<:AbstractMatrix{T},CD}
      LinearAlgebra.checksquare(A)
      return new{T,TA,CD}(A, her)
   end
end

LyapunovMap(A::TA, ::CD = Continuous(), her::Bool = false) where {T,TA<:AbstractMatrix{T},CD<:ContinuousOrDiscrete} =
   LyapunovMap{T,TA,CD}(A, her)

# LinearAlgebra.adjoint(L::LyapunovMap{<:Any,<:Any,CD}) where {CD}   = LyapunovMap(L.A', CD(), L.her)
# LinearAlgebra.transpose(L::LyapunovMap{<:Any,<:Any,CD}) where {CD} = LyapunovMap(L.A', CD(), L.her)
# LinearAlgebra.transpose(L::LyapunovMap{<:Any,<:Any,CD}) where {CD} = LyapunovMap(L.A', CD(), L.her)

"""
    L = lyapop(A; disc = false, her = false)

Define, for an `n x n` matrix `A`, the continuous Lyapunov operator `L:X -> AX+XA'`
if `disc = false` or the discrete Lyapunov operator `L:X -> AXA'-X` if `disc = true`.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated matrix `M = Matrix(L)` is
``n^2 \\times n^2``.
If `her = true` the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated matrix `M = Matrix(L)` is
``n(n+1)/2 \\times n(n+1)/2``.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
lyapop(A::AbstractMatrix; disc=false, her=false) =
   LyapunovMap(A, ifelse(disc, Discrete(), Continuous()), her)
lyapop(A::Schur; disc=false, her=false) =
   LyapunovMap(A.T, ifelse(disc, Discrete(), Continuous()), her)
lyapop(A::Number; disc=false, her=false) =
   LyapunovMap(fill(A, 1, 1), ifelse(disc, Discrete(), Continuous()), her)
function Base.size(L::LyapunovMap)
   n = size(L.A, 1)
   N = L.her ? Int(n * (n + 1) / 2) : n * n
   return (N, N)
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::LyapunovMap{T,TA,Discrete}, x::AbstractVector) where {T,TA}
   require_one_based_indexing(y, x)
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   if L.her
      X = vec2triu(convert(AbstractVector{T1}, x), her=true)
      # (y .= triu2vec(utqu(X,L.A') - X))
      muldsym!(y, L.A, X)
   else
      X = reshape(convert(AbstractVector{T1}, x), n, n)
      # (y .= (L.A*X*L.A' - X)[:])
      mul!(y, -1, x)
      Y = reshape(y, n, n)
      mul!(Y, L.A, X * L.A', true, true)
   end
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, L::$ttype{T,LyapunovMap{T,TA,Discrete}}, y::AbstractVector) where  {T,TA}
      require_one_based_indexing(y, x)
      n = size(L.lmap.A, 1)
      T1 = promote_type(T, eltype(y))
      if L.lmap.her
         Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
         # x[:] = triu2vec(L.lmap.A'*Y*L.lmap.A - Y)
         muldsym!(x, L.lmap.A', Y, dual = true)
      else
         # (x .= (L.A'*Y*L.A - Y)[:])
         mul!(x, -1, y)
         Y = reshape(convert(AbstractVector{T1}, y), n, n)
         X = reshape(x, n, n)
         mul!(X, L.lmap.A'*Y, L.lmap.A, true, true)
      end
      return x
   end
end
   
function LinearMaps._unsafe_mul!(y::AbstractVector, L::LyapunovMap{T,TA,Continuous}, x::AbstractVector) where {T,TA}
   require_one_based_indexing(y, x)
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   if L.her
      X = vec2triu(convert(AbstractVector{T1}, x), her=true)
      #y[:] = triu2vec(L.A*X + X*L.A')
      mulcsym!(y, L.A, X)
   else
      X = reshape(convert(AbstractVector{T1}, x), n, n)
      # (y[:] = (L.A*X + X*L.A')[:])
      Y = reshape(y, n, n)
      mul!(Y, X, L.A')
      mul!(Y, L.A, X, true, true)
   end
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, L::$ttype{T,LyapunovMap{T,TA,Continuous}}, y::AbstractVector) where  {T,TA}
      require_one_based_indexing(y, x)
      n = size(L.lmap.A, 1)
      T1 = promote_type(T, eltype(y))
      if L.lmap.her
         Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
         #x[:] = triu2vec(L.lmap.A'*Y + Y*L.lmap.A)
         mulcsym!(x, L.lmap.A', Y, dual = true)
      else
         Y = reshape(convert(AbstractVector{T1}, y), n, n)
         # (x[:] = (L.A'*Y + Y*L.A)[:])
         X = reshape(x, n, n)
         mul!(X, Y, L.lmap.A)
         mul!(X, L.lmap.A', Y, true, true)
      end
      return x
   end
end
   
function mulcsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix; dual = false)
   require_one_based_indexing(y, A, X)
   # A*X + X*A'
   n = size(A, 1)
   if dual 
      #Y = A*X+X*A'
      Y = similar(X, n, n)
      mul!(Y, X, A')
      mul!(Y, A, X, true, true)
      # y[:] = triu2vec(Y+transpose(Y)-Diagonal(Y))
      @inbounds begin
         k = 1
         for j = 1:n
            for i = 1:j
               y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
               k += 1
            end
         end
      end
      return y
   end
   ZERO = zero(eltype(y))
   @inbounds begin
      k = 1
      for j = 1:n
         for i = 1:j
            temp = ZERO
            for l = 1:n
               temp += A[i,l] * X[l,j] + X[i,l] * conj(A[j,l])
            end
            y[k] = temp
            k += 1
         end
      end
   end
   return y
end
function muldsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix; dual = false)
   require_one_based_indexing(y, X)
   # A*X*A' - X
   n = size(A, 1)
   if dual 
      #Y = A*X*A' - X
      Y = similar(X, n, n)
      mul!(Y, -1, X)
      mul!(Y, A*X, A', true, true)
      # y[:] = triu2vec(Y+transpose(Y)-Diagonal(Y))
      @inbounds begin
         k = 1
         for j = 1:n
            for i = 1:j
               y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
               k += 1
            end
         end
      end
      return y
   end
   # t = triu(X)-diag(X)/2
   t = UpperTriangular(X) - Diagonal(X[diagind(X)] ./ 2)
   Y = similar(t, n, n)
   # Y = A*t*A'
   mul!(Y, A * t, A')
   # Y + Y' - X
   @inbounds begin
      k = 1
      for j = 1:n
         for i = 1:j
            y[k] = Y[i,j] + conj(Y[j,i]) - X[i,j]
            k += 1
         end
      end
   end
   return y
end

struct GeneralizedLyapunovMap{T,TA <: AbstractMatrix{T},TE <: AbstractMatrix{T},CD<:ContinuousOrDiscrete} <: LyapunovMatrixEquationsMaps{T}
   A::TA
   E::TE
   her::Bool
   function GeneralizedLyapunovMap{T,TA,TE,CD}(A::TA, E::TE, her=false) where {T,TA<:AbstractMatrix{T},TE<:AbstractMatrix{T},CD}
      n = LinearAlgebra.checksquare(A)
      n == LinearAlgebra.checksquare(E) ||
               throw(DimensionMismatch("E must be a square matrix of dimension $n"))
      return new{T,TA,TE,CD}(A, E, her)
   end
end
function GeneralizedLyapunovMap(A::AbstractMatrix, E::AbstractMatrix, ::CD = Continuous(), her::Bool = false) where {CD<:ContinuousOrDiscrete}
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) ||
            throw(DimensionMismatch("E must be a square matrix of dimension $n"))
   T = promote_type(eltype(A), eltype(E))
   A = convert(AbstractMatrix{T}, A)
   E = convert(AbstractMatrix{T}, E)
   return GeneralizedLyapunovMap{T,typeof(A),typeof(E),CD}(A, E, her)
end

# LinearAlgebra.adjoint(L::GeneralizedLyapunovMap{<:Any,<:Any,<:Any,CD}) where {CD} =
#    GeneralizedLyapunovMap(L.A', L.E', CD(), L.her)
# LinearAlgebra.transpose(L::GeneralizedLyapunovMap{<:Any,<:Any,<:Any,CD}) where {CD} =
#    GeneralizedLyapunovMap(L.A', L.E', CD(), L.her)

"""
    L = lyapop(A, E; disc = false, her = false)

Define, for a pair `(A,E)` of `n x n` matrices, the continuous Lyapunov operator `L:X -> AXE'+EXA'`
if `disc = false` or the discrete Lyapunov operator `L:X -> AXA'-EXE'` if `disc = true`.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated matrix `M = Matrix(L)` is
``n^2 \\times n^2``.
If `her = true` the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated `M = Matrix(L)` is a
``n(n+1)/2 \\times n(n+1)/2``.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
lyapop(A::AbstractMatrix, E::AbstractMatrix; disc=false, her=false) =
   GeneralizedLyapunovMap(A, E, ifelse(disc, Discrete(), Continuous()), her)
lyapop(F::GeneralizedSchur; disc=false, her=false) =
   GeneralizedLyapunovMap(F.S, F.T, ifelse(disc, Discrete(), Continuous()), her)
function lyapop(A::Number, E::Number; disc=false, her=false)
   A, E = promote(A, E)
   GeneralizedLyapunovMap(fill(A, 1, 1), fill(E, 1, 1), ifelse(disc, Discrete(), Continuous()), her)
end
function Base.size(L::GeneralizedLyapunovMap)
   n = size(L.A, 1)
   N = L.her ? Int(n * (n + 1) / 2) : n * n
   return (N, N)
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::GeneralizedLyapunovMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where {T}
   require_one_based_indexing(y, x)
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   if L.her
      X = vec2triu(convert(AbstractVector{T1}, x), her=true)
      # (y .= triu2vec(utqu(X,L.A') - utqu(X,L.E')))
      muldsym!(y, L.A, L.E, X)
   else
      X = reshape(convert(AbstractVector{T1}, x), n, n)
      Y = reshape(y, n, n)
      temp = similar(Y, (n, n))
      # (y .= (L.A*X*L.A' - L.E*X*L.E')[:])
      # Y .= X
      mul!(temp, X, L.A')
      mul!(Y, L.A, temp)
      mul!(temp, X, L.E')
      mul!(Y, L.E, temp, -1, 1)
   end
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, L::$ttype{T,GeneralizedLyapunovMap{T,TA,TE,Discrete}}, y::AbstractVector) where  {T,TA,TE}
      require_one_based_indexing(y, x)
      n = size(L.lmap.A, 1)
      T1 = promote_type(T, eltype(y))
      if L.lmap.her
         Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
         # x[:] = triu2vec(L.lmap.A'*Y*L.lmap.A - Y)
         muldsym!(x, L.lmap.A', L.lmap.E', Y, dual = true)
      else
         # (x .= (L.A'*Y*L.A - L.E'*X*L.E)[:])
         X = reshape(x, n, n)
         Y = reshape(convert(AbstractVector{T1}, y), n, n)
         temp = similar(Y, (n, n))
         mul!(temp, Y, L.lmap.A)
         mul!(X, L.lmap.A', temp)
         mul!(temp, Y, L.lmap.E)
         mul!(X, L.lmap.E', temp, -1, 1)
      end
      return x
   end
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::GeneralizedLyapunovMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where {T}
   require_one_based_indexing(y, x)
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   if L.her
      X = vec2triu(convert(AbstractVector{T1}, x), her=true)
      mulcsym!(y, L.A, L.E, X)
   else
      X = reshape(convert(AbstractVector{T1}, x), n, n)
      Y = reshape(y, n, n)
      temp = similar(Y, (n, n))
      # (y[:] = (L.A*X*L.E' + L.E*X*L.A')[:])
      mul!(temp, L.E, X)
      mul!(Y, temp, L.A')
      mul!(temp, X, L.E')
      mul!(Y, L.A, temp, 1, 1)
   end
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, L::$ttype{T,GeneralizedLyapunovMap{T,TA,TE,Continuous}}, y::AbstractVector) where  {T,TA,TE}
      require_one_based_indexing(y, x)
      n = size(L.lmap.A, 1)
      T1 = promote_type(T, eltype(y))
      if L.lmap.her
         Y = vec2triu(convert(AbstractVector{T1}, y), her=false)
         #x[:] = triu2vec(L.lmap.A'*Y + Y*L.lmap.A)
         mulcsym!(x, L.lmap.A', L.lmap.E',  Y, dual = true)
      else
         X = reshape(x, n, n)
         Y = reshape(convert(AbstractVector{T1}, y), n, n)
         temp = similar(Y, (n, n))
         # (x[:] = (L.A'*Y*L.E + L.E'*Y*L.A)[:])
         mul!(temp, L.lmap.E', Y)
         mul!(X, temp, L.lmap.A)
         mul!(temp, Y, L.lmap.E)
         mul!(X, L.lmap.A', temp, 1, 1)
      end
      return x
   end
end
function mulcsym!(y::AbstractVector, A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix; dual = false)
   require_one_based_indexing(y, A)
   # AXE' + EXA'
   n = size(A, 1)
   Y = similar(X, n, n)
   if dual 
      #Y = AXE' + EXA'
      Y = similar(X, n, n)
      temp = similar(Y, (n, n))
      mul!(temp, E, X)
      mul!(Y, temp, A')
      mul!(temp, X, E')
      mul!(Y, A, temp, 1, 1)
      # y[:] = triu2vec(Y+transpose(Y)-Diagonal(Y))
      @inbounds begin
         k = 1
         for j = 1:n
            for i = 1:j
               y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
               k += 1
            end
         end
      end
   else
      ZERO = zero(eltype(y))
      # Y = XE'
      mul!(Y, X, E')
      # AY + Y'A'
      @inbounds  begin
         k = 1
         for j = 1:n
            for i = 1:j
               temp = ZERO
               for l = 1:n
                  temp += A[i,l] * Y[l,j] + conj(Y[l,i] * A[j,l])
               end
               y[k] = temp
               k += 1
            end
         end
      end
   end
   return y
end
function muldsym!(y::AbstractVector, A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix; dual = false)
   require_one_based_indexing(y)
   # AXA' - EXE'
   n = size(A, 1)
   Y = similar(X, n, n)
   if dual 
      #Y = AXA' - EXE'
      temp = similar(Y, (n, n))
      mul!(temp, A, X)
      mul!(Y, temp, A')
      mul!(temp, X, E')
      mul!(Y, E, temp, -1, 1)
      # y[:] = triu2vec(Y+transpose(Y)-Diagonal(Y))
      @inbounds begin
         k = 1
         for j = 1:n
            for i = 1:j
               y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
               k += 1
            end
         end
      end
   else
      # t = triu(X)-diag(X)/2
      t = UpperTriangular(X) - Diagonal(X[diagind(X)] ./ 2)
      # Y = A*t*A' - E*t*E'
      mul!(Y, A * t, A')
      mul!(Y, E * t, E', -1, 1)
      # Y + Y'
      @inbounds  begin
         k = 1
         for j = 1:n
            for i = 1:j
               y[k] = Y[i,j] + conj(Y[j,i])
               k += 1
            end
         end
      end
   end
   return y
end

struct InverseLyapunovMap{T,TA <: AbstractMatrix,adj,CD} <: LyapunovMatrixEquationsMaps{T}
   A::TA
   her::Bool
   sf::Bool
   function InverseLyapunovMap{T,TA,adj,CD}(A::TA, her::Bool, sf::Bool) where {T <: BlasFloat, TA <: AbstractMatrix{T}, adj, CD}
      LinearAlgebra.checksquare(A)
      return new{T,TA,adj,CD}(A, her, sf)
   end
end
function InverseLyapunovMap(A::AbstractMatrix{T}, ::CD = Continuous(), her::Bool = false) where {T <: BlasFloat,CD <: ContinuousOrDiscrete}
   schur_flag = isschur(A)
   return InverseLyapunovMap{T,typeof(A),false,CD}(A, her, schur_flag)
end
function InverseLyapunovMap(A::Adjoint{T,<:AbstractMatrix{T}}, ::CD = Continuous(), her::Bool = false) where {T <: BlasFloat,CD <: ContinuousOrDiscrete}
   Ap = A.parent
   schur_flag = isschur(Ap)
   return InverseLyapunovMap{T,typeof(Ap),true,CD}(Ap, her, schur_flag)
end

LinearAlgebra.adjoint(L::InverseLyapunovMap{<:Any,<:Any,adj,CD}) where {adj,CD} = 
   eltype(L) <: Complex ? InverseLyapunovMap(adj ? L.A : L.A', CD(), L.her) : LinearAlgebra.transpose(L)
# LinearAlgebra.adjoint(L::InverseLyapunovMap{<:Any,<:Any,adj,CD}) where {adj,CD} =
#    InverseLyapunovMap(adj ? L.A : L.A', CD(), L.her)
# LinearAlgebra.transpose(L::InverseLyapunovMap{<:Any,<:Any,adj,CD}) where {adj,CD} =
#    InverseLyapunovMap(adj ? L.A : L.A', CD(), L.her)
LinearAlgebra.inv(L::LyapunovMap{T,TA,CD}) where {T,TA,CD} =
   InverseLyapunovMap(L.A, CD(), L.her)
LinearAlgebra.inv(L::InverseLyapunovMap{<:Any,<:Any,adj,CD}) where {adj,CD} =
   LyapunovMap(adj ? L.A' : L.A, CD(), L.her)

"""
    LINV = invlyapop(A; disc = false, her = false)

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AX+XA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-X` for `disc = true`, where
`A` is an `n x n` matrix.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated matrix `M = Matrix(LINV)` is
``n^2 \\times n^2``.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated matrix `M = Matrix(LINV)` is
``n(n+1)/2 \\times n(n+1)/2``.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
invlyapop(A::AbstractMatrix{<:BlasFloat}; disc=false, her=false) =
   InverseLyapunovMap(A, ifelse(disc, Discrete(), Continuous()), her)
function invlyapop(A::AbstractMatrix; disc=false, her=false)
   A = convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A)
   InverseLyapunovMap(A, ifelse(disc, Discrete(), Continuous()), her)
end
invlyapop(A::Schur{<:BlasFloat}; disc=false, her=false) =
   InverseLyapunovMap(A.T, ifelse(disc, Discrete(), Continuous()), her)
function Base.size(L::InverseLyapunovMap)
   n = size(L.A, 1)
   N = L.her ? Int(n * (n + 1) / 2) : n * n
   return (N, N)
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseLyapunovMap{T,<:Any,adj,Discrete}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf && !(T <: Real && T1 <: Complex) 
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            lyapds!(L.A, Y, adj=adj)
            copyto!(y, triu2vec(Y))
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            adj ? sylvds!(-L.A, L.A, Y, adjA=true) : sylvds!(-L.A, L.A, Y, adjB=true)
            copyto!(y, Y)
         end
      else
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            y .= triu2vec(lyapd(adj ? L.A' : L.A, Y))
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            copyto!(y, lyapd(adj ? L.A' : L.A, Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::LinearMaps.TransposeMap{<:Any,<:InverseLyapunovMap{T,<:Any,adj,Discrete}}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.lmap.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.lmap.sf && !(T <: Real && T1 <: Complex) 
         if L.lmap.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=false)
            adj ? sylvds!(-L.lmap.A, L.lmap.A, Y, adjB=true) : sylvds!(-L.lmap.A, L.lmap.A, Y, adjA=true)
            # lyapds!(L.A, Y, adj=adj)
            # copyto!(y, triu2vec(Y))
            @inbounds begin
               k = 1
               for j = 1:n
                  for i = 1:j
                     y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
                     k += 1
                  end
               end
            end
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            adj ? sylvds!(-L.lmap.A, L.lmap.A, Y, adjB=true) : sylvds!(-L.lmap.A, L.lmap.A, Y, adjA=true)
            copyto!(y, Y)
         end
      else
         if L.lmap.her
            Y1 = vec2triu(-convert(AbstractVector{T1}, x), her=false)
            #y .= triu2vec(lyapd(adj ? L.A' : L.A, Y))
            adj ? (Y = sylvd(-L.lmap.A, L.lmap.A', Y1)) : (Y = sylvd(-L.lmap.A', L.lmap.A, Y1))
            @inbounds begin
               k = 1
               for j = 1:n
                  for i = 1:j
                     y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
                     k += 1
                  end
               end
            end
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            copyto!(y, lyapd(adj ? L.lmap.A : L.lmap.A', Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end

function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseLyapunovMap{T,<:Any,adj,Continuous}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf && !(T <: Real && T1 <: Complex) 
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            lyapcs!(L.A, Y; adj=adj)
            copyto!(y, triu2vec(Y))
         else
            Y = copy(reshape(convert(AbstractVector{T1}, x), n, n))
            adj ? (sylvcs!(L.A, L.A, Y, adjA=true)) : (sylvcs!(L.A, L.A, Y, adjB=true))
            copyto!(y, Y)
         end
      else
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            y .= triu2vec(lyapc(adj ? L.A' : L.A, Y))
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            copyto!(y, lyapc(adj ? L.A' : L.A, Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::LinearMaps.TransposeMap{<:Any,<:InverseLyapunovMap{T,<:Any,adj,Continuous}}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.lmap.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.lmap.sf && !(T <: Real && T1 <: Complex) 
         if L.lmap.her
            Y = vec2triu(convert(AbstractVector{T1}, x), her=false)
            adj ? (sylvcs!(L.lmap.A, L.lmap.A, Y, adjB=true)) : (sylvcs!(L.lmap.A, L.lmap.A, Y, adjA=true))
            @inbounds begin
               k = 1
               for j = 1:n
                  for i = 1:j
                     y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
                     k += 1
                  end
               end
            end
         else
            Y = copy(reshape(convert(AbstractVector{T1}, x), n, n))
            adj ? (sylvcs!(L.lmap.A, L.lmap.A, Y, adjB=true)) : (sylvcs!(L.lmap.A, L.lmap.A, Y, adjA=true))
            copyto!(y, Y)
         end
      else
         if L.lmap.her
            Y1 = vec2triu(convert(AbstractVector{T1}, x), her=false)
            #y .= triu2vec(lyapc(adj ? L.A' : L.A, Y))
            adj ? (Y = sylvc(L.lmap.A, L.lmap.A', Y1)) : (Y = sylvc(L.lmap.A', L.lmap.A, Y1))
            @inbounds begin
               k = 1
               for j = 1:n
                  for i = 1:j
                     y[k] = i == j ? Y[j,j] : Y[i,j] + Y[j,i]
                     k += 1
                  end
               end
            end
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            copyto!(y, lyapc(adj ? L.lmap.A : L.lmap.A', Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end


struct InverseGeneralizedLyapunovMap{T,TA <: AbstractMatrix,TE <: AbstractMatrix,adj,CD <: ContinuousOrDiscrete} <: LyapunovMatrixEquationsMaps{T}
  A::TA
  E::TE
  her::Bool
  sf::Bool
  function InverseGeneralizedLyapunovMap{T,TA,TE,adj,CD}(A::TA, E::TE, her::Bool, schur_flag::Bool) where {T<:BlasFloat,TA<:AbstractMatrix{T},TE<:AbstractMatrix{T},adj,CD}
      LinearAlgebra.checksquare(A)
      return new{T,TA,TE,adj,CD}(A, E, her, schur_flag)
   end
end

function InverseGeneralizedLyapunovMap(A::AbstractMatrix, E::AbstractMatrix, ::CD = Continuous(), her::Bool = false) where {CD <: ContinuousOrDiscrete}
   T = promote_type(eltype(A), eltype(E), Float64)
   A = convert(AbstractMatrix{T}, A)
   E = convert(AbstractMatrix{T}, E)
   schur_flag = isschur(A, E)
   return InverseGeneralizedLyapunovMap{T,typeof(A),typeof(E),false,CD}(A, E, her, schur_flag)
end
function InverseGeneralizedLyapunovMap(A::Adjoint{<:Any,<:AbstractMatrix}, E::Adjoint{<:Any,<:AbstractMatrix}, ::CD = Continuous(), her::Bool = false) where {CD <: ContinuousOrDiscrete}
   A = A.parent
   E = E.parent
   T = promote_type(eltype(A), eltype(E), Float64)
   A = convert(AbstractMatrix{T}, A)
   E = convert(AbstractMatrix{T}, E)
   schur_flag = isschur(A, E)
   return InverseGeneralizedLyapunovMap{T,typeof(A),typeof(E),true,CD}(A, E, her, schur_flag)
end

LinearAlgebra.adjoint(L::InverseGeneralizedLyapunovMap{<:Any,<:Any,<:Any,adj,CD}) where {adj,CD} =
   InverseGeneralizedLyapunovMap(adj ? L.A : L.A', adj ? L.E : L.E', CD(), L.her)
LinearAlgebra.transpose(L::InverseGeneralizedLyapunovMap{<:Any,<:Any,<:Any,adj,CD}) where {adj,CD} =
   InverseGeneralizedLyapunovMap(adj ? L.A : L.A', adj ? L.E : L.E', CD(), L.her)
LinearAlgebra.inv(L::GeneralizedLyapunovMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   InverseGeneralizedLyapunovMap(L.A, L.E, CD(), L.her)
LinearAlgebra.inv(L::InverseGeneralizedLyapunovMap{<:Any,<:Any,<:Any,adj,CD}) where {adj,CD} =
   GeneralizedLyapunovMap(adj ? L.A' : L.A, adj ? L.E' : L.E, CD(), L.her)

"""
    LINV = invlyapop(A, E; disc = false, her = false)

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AXE'+EXA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-EXE'` for `disc = true`, where
`(A,E)` is a pair of `n x n` matrices.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated matrix `M = Matrix(LINV)` is
``n^2 \\times n^2``.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated matrix `M = Matrix(LINV)` is
``n(n+1)/2 \\times n(n+1)/2``.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
invlyapop(A::AbstractMatrix, E::AbstractMatrix; disc=false, her=false) =
   InverseGeneralizedLyapunovMap(A, E, ifelse(disc, Discrete(), Continuous()), her)
invlyapop(F::GeneralizedSchur; disc=false, her=false) =
   InverseGeneralizedLyapunovMap(F.S, F.T, ifelse(disc, Discrete(), Continuous()), her)
function Base.size(L::InverseGeneralizedLyapunovMap)
   n = size(L.A, 1)
   N = L.her ? Int(n * (n + 1) / 2) : n * n
   return (N, N)
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseGeneralizedLyapunovMap{T,<:Any,<:Any,adj,Discrete}, x::AbstractVector) where {T,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf && !(T <: Real && T1 <: Complex) 
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            lyapds!(L.A, L.E, Y; adj=adj)
            y .= triu2vec(Y)
         else
            copyto!(y, x)
            Y = reshape(y, (n, n))
            adj ? gsylvs!(L.A, L.A, -L.E, L.E, Y, adjAC=true) : gsylvs!(L.A, L.A, -L.E, L.E, Y, adjBD=true)
         end
      else
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            y .= triu2vec(lyapd(adj ? L.A' : L.A, adj ? L.E' : L.E, Y))
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            # adj ? (y .= gsylv(-L.A',L.A,L.E',L.E,Y)[:]) : (y .= gsylv(-L.A,L.A',L.E,L.E',Y)[:])
            copyto!(y, lyapd(adj ? L.A' : L.A, adj ? L.E' : L.E, Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseGeneralizedLyapunovMap{T,<:Any,<:Any,adj,Continuous}, x::AbstractVector) where {T,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf && !(T <: Real && T1 <: Complex) 
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            lyapcs!(L.A, L.E, Y; adj=adj)
            y .= triu2vec(Y)
         else
            copyto!(y, x)
            Y = reshape(y, (n, n))
            adj ? (gsylvs!(L.A, L.E, L.E, L.A, Y, adjAC=true, DBSchur=true)) : (gsylvs!(L.A, L.E, L.E, L.A, Y, adjBD=true, DBSchur=true))
         end
      else
         if L.her
            Y = vec2triu(-convert(AbstractVector{T1}, x), her=true)
            y .= triu2vec(lyapc(adj ? L.A' : L.A, adj ? L.E' : L.E, Y))
         else
            Y = reshape(-convert(AbstractVector{T1}, x), n, n)
            # adj ? (y .= gsylv(L.A',L.E,L.E',L.A,Y)[:]) : (y .= gsylv(L.A,L.E',L.E,L.A',Y)[:])
            copyto!(y, lyapc(adj ? L.A' : L.A, adj ? L.E' : L.E, Y))
         end
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end

struct SylvesterMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,CD <: ContinuousOrDiscrete} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   function SylvesterMap{T,TA,TB,CD}(A::TA, B::TB) where {T,TA<:AbstractMatrix{T},TB<:AbstractMatrix{T},CD}
      LinearAlgebra.checksquare(A, B)
      return new{T,TA,TB,CD}(A, B)
   end
end
function SylvesterMap(A::AbstractMatrix, B::AbstractMatrix, ::CD = Continuous()) where {CD}
   T = promote_type(eltype(A), eltype(B))
   A = convert(AbstractMatrix{T}, A)
   B = convert(AbstractMatrix{T}, B)
   return SylvesterMap{T,typeof(A),typeof(B),CD}(A, B)
end

# LinearAlgebra.adjoint(L::SylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
#    SylvesterMap(L.A', L.B', CD())
# LinearAlgebra.transpose(L::SylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
#    SylvesterMap(L.A', L.B', CD())

"""
    M = sylvop(A, B; disc = false)

Define the continuous Sylvester operator `M: X -> AX+XB` if `disc = false`
or the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`, where `A` and `B` are square matrices.
"""
sylvop(A::AbstractMatrix, B::AbstractMatrix; disc=false) =
   SylvesterMap(A, B, ifelse(disc, Discrete(), Continuous()))
sylvop(A::Schur, B::Schur; disc=false) = SylvesterMap(A.T, B.T, ifelse(disc, Discrete(), Continuous()))
function sylvop(A::Number, B::Number; disc=false)
   A, B = promote(A, B)
   SylvesterMap(fill(A, 1, 1), fill(B, 1, 1), ifelse(disc, Discrete(), Continuous()))
end
Base.size(L::SylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::SylvesterMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where T
   require_one_based_indexing(y, x)
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   X = reshape(convert(AbstractVector{T1}, x), (m, n))
   Y = reshape(y, (m, n))
   # Y = A * X * B + X
   copyto!(y, x)
   mul!(Y, L.A * X, L.B, true, true)
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, LT::$ttype{T,SylvesterMap{T,TA,TB,Discrete}}, y::AbstractVector) where {T,TA,TB}
      require_one_based_indexing(y, x)
      m = size(LT.lmap.A, 1)
      n = size(LT.lmap.B, 1)
      T1 = promote_type(T, eltype(y))
      Y = reshape(convert(AbstractVector{T1}, y), (m, n))
      X = reshape(x, (m, n))
      # X = A' * Y * B' + X
      copyto!(x, y)
      mul!(X, LT.lmap.A' * Y, LT.lmap.B', true, true)
      return x
   end
end

function LinearMaps._unsafe_mul!(y::AbstractVector, L::SylvesterMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where T
   require_one_based_indexing(y, x)
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   X = reshape(convert(AbstractVector{T1}, x), (m, n))
   Y = reshape(y, (m, n))
   # Y = A * X + X * B
   mul!(Y, L.A, X)
   mul!(Y, X, L.B, true, true)
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, LT::$ttype{T,SylvesterMap{T,TA,TB,Continuous}}, y::AbstractVector) where {T,TA,TB}
      require_one_based_indexing(y, x)
      m = size(LT.lmap.A, 1)
      n = size(LT.lmap.B, 1)
      T1 = promote_type(T, eltype(y))
      Y = reshape(convert(AbstractVector{T1}, y), (m, n))
      X = reshape(x, (m, n))
      # X = A' * Y + Y * B'
      mul!(X, LT.lmap.A', Y)
      mul!(X, Y, LT.lmap.B', true, true)
      return x
   end
end

struct InverseSylvesterMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,CD} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   adjA::Bool
   adjB::Bool
   sf::Bool
   function InverseSylvesterMap{T,TA,TB,CD}(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T,TA,TB,CD}
      LinearAlgebra.checksquare(A, B)
      adjA = isa(A, Adjoint)
      adjB = isa(B, Adjoint)
      A = adjA ? A.parent : A
      B = adjB ? B.parent : B
      schur_flag = isschur(A) && isschur(B)
      return new{T,typeof(A),typeof(B),CD}(A, B, adjA, adjB, schur_flag)
   end
end
function InverseSylvesterMap(A::AbstractMatrix, B::AbstractMatrix, ::CD = Continuous) where {CD}
   T = promote_type(eltype(A), eltype(B))
   A = convert(AbstractMatrix{T}, A)
   B = convert(AbstractMatrix{T}, B)
   return InverseSylvesterMap{T,typeof(A),typeof(B),CD}(A, B)
end

LinearAlgebra.adjoint(L::InverseSylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   InverseSylvesterMap(L.adjA ? L.A : L.A', L.adjB ? L.B : L.B', CD())
LinearAlgebra.transpose(L::InverseSylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   InverseSylvesterMap(L.adjA ? L.A : L.A', L.adjB ? L.B : L.B', CD())
LinearAlgebra.inv(L::SylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   InverseSylvesterMap(L.A, L.B, CD())
LinearAlgebra.inv(L::InverseSylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   SylvesterMap(L.adjA ? L.A' : L.A, L.adjB ? L.B' : L.B, CD())

"""
    MINV = invsylvop(A, B; disc = false)

Define `MINV`, the inverse of the continuous Sylvester operator  `M: X -> AX+XB` if `disc = false`
or of the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`, where `A` and `B` are square matrices.
"""
function invsylvop(A::AbstractMatrix, B::AbstractMatrix; disc=false)
   InverseSylvesterMap(A, B, ifelse(disc, Discrete(), Continuous()))
end
invsylvop(A::Schur, B::Schur; disc=false) =
   InverseSylvesterMap(A.T, B.T, ifelse(disc, Discrete(), Continuous()))
function invsylvop(A::Number, B::Number; disc=false)
   A, B = promote(A, B)
   InverseSylvesterMap(fill(A, 1, 1), fill(B, 1, 1), ifelse(disc, Discrete(), Continuous()))
end
Base.size(L::InverseSylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseSylvesterMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where {T}
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
         copyto!(y, x)
         Y = reshape(y, (m, n))
         sylvds!(L.A, L.B, Y, adjA=L.adjA, adjB=L.adjB)
      else
         Y = reshape(convert(AbstractVector{T1}, x), m, n)
         copyto!(y, sylvd(L.adjA ? L.A' : L.A, L.adjB ? L.B' : L.B, Y))
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseSylvesterMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where T
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
         copyto!(y, x)
         Y = reshape(y, (m, n))
         sylvcs!(L.A, L.B, Y, adjA=L.adjA, adjB=L.adjB)
      else
         Y = reshape(convert(AbstractVector{T1}, x), m, n)
         copyto!(y, sylvc(L.adjA ? L.A' : L.A, L.adjB ? L.B' : L.B, Y))
      end
      return y
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
end

struct GeneralizedSylvesterMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,TC <: AbstractMatrix,TD <: AbstractMatrix} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   C::TC
   D::TD
   function GeneralizedSylvesterMap(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
      m, n = LinearAlgebra.checksquare(A, B)
      [m, n] == LinearAlgebra.checksquare(C, D) ||
               throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      T = promote_type(map(eltype, (A, B, C, D))...)
      A = convert(AbstractMatrix{T}, A)
      B = convert(AbstractMatrix{T}, B)
      C = convert(AbstractMatrix{T}, C)
      D = convert(AbstractMatrix{T}, D)
      return new{T,typeof(A),typeof(B),typeof(C),typeof(D)}(A, B, C, D)
   end
end

"""
    M = sylvop(A, B, C, D)

Define the generalized Sylvester operator `M: X -> AXB+CXD`, where `(A,C)` and `(B,D)` are pairs of square matrices.
"""
sylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   GeneralizedSylvesterMap(A, B, C, D)
Base.size(L::GeneralizedSylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::GeneralizedSylvesterMap{T}, x::AbstractVector) where T
   require_one_based_indexing(y, x)
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   X = reshape(convert(AbstractVector{T1}, x), (m, n))
   Y = reshape(y, (m, n))
   # Y = A * X * B + C * X * D
   temp = similar(Y, (m, n))
   mul!(temp, L.A, X)
   mul!(Y, temp, L.B, 1, 0)
   mul!(temp, L.C, X)
   mul!(Y, temp, L.D, 1, 1)
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(x::AbstractVector, LT::$ttype{T,GeneralizedSylvesterMap{T,TA,TB,TC,TD}}, y::AbstractVector) where {T,TA,TB,TC,TD}
      require_one_based_indexing(y, x)
      m = size(LT.lmap.A, 1)
      n = size(LT.lmap.B, 1)
      T1 = promote_type(T, eltype(y))
      Y = reshape(convert(AbstractVector{T1}, y), (m, n))
      X = reshape(x, (m, n))
      temp = similar(Y, (m, n))
      # X = A' * Y * B' + C' * Y * D'
      mul!(temp, LT.lmap.A', Y)
      mul!(X, temp, LT.lmap.B', true, false)
      mul!(temp, LT.lmap.C', Y)
      mul!(X, temp, LT.lmap.D', true, true)
      return x
   end
end

struct InverseGeneralizedSylvesterMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,TC <: AbstractMatrix,TD <: AbstractMatrix} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   C::TC
   D::TD
   adjAC::Bool
   adjBD::Bool
   sf::Bool
   BDSchur::Bool
   DBSchur::Bool
   function InverseGeneralizedSylvesterMap(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
      m, n = LinearAlgebra.checksquare(A, B)
      [m, n] == LinearAlgebra.checksquare(C, D) ||
               throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      adjAC = isa(A, Adjoint) && isa(C, Adjoint)
      adjBD = isa(B, Adjoint) && isa(D, Adjoint)
      ACSchur = (adjAC ? isschur(A.parent, C.parent) : isschur(A, C))
      BDSchur = (adjBD ? isschur(B.parent, D.parent) : isschur(B, D))
      DBSchur = (adjBD ? isschur(D.parent, B.parent) : isschur(D, B))
      schur_flag = ACSchur && (BDSchur || DBSchur)
      A = adjAC ? A.parent : A
      B = adjBD ? B.parent : B
      C = adjAC ? C.parent : C
      D = adjBD ? D.parent : D
      T = promote_type(map(eltype, (A, B, C, D))...)
      A = convert(AbstractMatrix{T}, A)
      B = convert(AbstractMatrix{T}, B)
      C = convert(AbstractMatrix{T}, C)
      D = convert(AbstractMatrix{T}, D)
    return new{T,typeof(A),typeof(B),typeof(C),typeof(D)}(A, B, C, D, adjAC, adjBD, schur_flag, BDSchur, DBSchur)
   end
end

LinearAlgebra.adjoint(L::InverseGeneralizedSylvesterMap) =
   InverseGeneralizedSylvesterMap(L.adjAC ? L.A : L.A', L.adjBD ? L.B : L.B',
                                    L.adjAC ? L.C : L.C', L.adjBD ? L.D : L.D')
LinearAlgebra.transpose(L::InverseGeneralizedSylvesterMap) =
   InverseGeneralizedSylvesterMap(L.adjAC ? L.A : L.A', L.adjBD ? L.B : L.B',
                                    L.adjAC ? L.C : L.C', L.adjBD ? L.D : L.D')
LinearAlgebra.inv(L::GeneralizedSylvesterMap) =
   InverseGeneralizedSylvesterMap(L.A, L.B, L.C, L.D)
LinearAlgebra.inv(L::InverseGeneralizedSylvesterMap) =
   GeneralizedSylvesterMap(L.adjAC ? L.A' : L.A, L.adjBD ? L.B' : L.B,
                           L.adjAC ? L.C' : L.C, L.adjBD ? L.D' : L.D)

"""
    MINV = invsylvop(A, B, C, D)

Define `MINV`, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`,
where (A,C) and (B,D) are pairs of square matrices.
"""
invsylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   InverseGeneralizedSylvesterMap(A, B, C, D)
invsylvop(F::GeneralizedSchur, G::GeneralizedSchur) = invsylvop(F.S, G.S, F.T, G.T)
Base.size(L::InverseGeneralizedSylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseGeneralizedSylvesterMap{T}, x::AbstractVector) where T
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
         copyto!(y, x)
         Y = reshape(y, m, n)
         gsylvs!(L.A, L.B, L.C, L.D, Y, adjAC=L.adjAC, adjBD=L.adjBD, DBSchur=L.DBSchur && !L.BDSchur)
      else
         X = reshape(convert(AbstractVector{T1}, x), m, n)
         copyto!(y, gsylv(L.adjAC ? L.A' : L.A, L.adjBD ? L.B' : L.B,
                           L.adjAC ? L.C' : L.C, L.adjBD ? L.D' : L.D, X))
      end
   catch err
      findfirst("SingularException", string(err)) === nothing &&
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end

struct SylvesterSystemMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,TC <: AbstractMatrix,TD <: AbstractMatrix} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   C::TC
   D::TD
   function SylvesterSystemMap(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
      m, n = LinearAlgebra.checksquare(A, B)
      [m, n] == LinearAlgebra.checksquare(C, D) ||
         throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      T = promote_type(map(eltype, (A, B, C, D))...)
      A = convert(AbstractMatrix{T}, A)
      B = convert(AbstractMatrix{T}, B)
      C = convert(AbstractMatrix{T}, C)
      D = convert(AbstractMatrix{T}, D)
      return new{T,typeof(A),typeof(B),typeof(C),typeof(D)}(A, B, C, D)
   end
end

"""
    M = sylvsysop(A, B, C, D)

Define the operator `M: (X,Y) -> (AX+YB, CX+YD)`, where `(A,C)` and `(B,D)` are pairs of
square matrices.
"""
sylvsysop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   SylvesterSystemMap(A, B, C, D)
Base.size(L::SylvesterSystemMap) = (N = size(L.A, 1) * size(L.B, 1) * 2; return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::SylvesterSystemMap{T}, x::AbstractVector) where T
   require_one_based_indexing(y, x)
   m = size(L.A, 1)
   n = size(L.B, 1)
   mn = m * n
   T1 = promote_type(T, eltype(x))
   X1 = reshape(convert(AbstractVector{T1}, view(x, 1:mn)), m, n)
   X2 = reshape(convert(AbstractVector{T1}, view(x, mn + 1:2 * mn)), m, n)
   U = reshape(view(y, 1:m * n), (m, n))
   V = reshape(view(y, mn + 1:2 * mn), (m, n))
   # U = A * X + Y * B
   # V = C * X + Y * D
   mul!(U, L.A, X1)
   mul!(U, X2, L.B, 1, 1)
   mul!(V, L.C, X1)
   mul!(V, X2, L.D, 1, 1)
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(y::AbstractVector, LT::$ttype{T,<:SylvesterSystemMap{T}}, x::AbstractVector) where T
      require_one_based_indexing(y, x)
      m = size(LT.lmap.A, 1)
      n = size(LT.lmap.B, 1)
      mn = m * n
      T1 = promote_type(T, eltype(x))
      X1 = reshape(convert(AbstractVector{T1}, view(x, 1:mn)), m, n)
      X2 = reshape(convert(AbstractVector{T1}, view(x, mn + 1:2 * mn)), m, n)
      U = reshape(view(y, 1:m * n), (m, n))
      V = reshape(view(y, mn + 1:2 * mn), (m, n))
      # U = A' * X + C' * Y
      # V = X * B' + Y * D'
      mul!(U, LT.lmap.A', X1)
      mul!(U, LT.lmap.C', X2, 1, 1)
      mul!(V, X1, LT.lmap.B')
      mul!(V, X2, LT.lmap.D', 1, 1)
      return y
   end
end

struct InverseSylvesterSystemMap{T,TA <: AbstractMatrix,TB <: AbstractMatrix,TC <: AbstractMatrix,TD <: AbstractMatrix} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   C::TC
   D::TD
   sf::Bool
   function InverseSylvesterSystemMap(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
      m, n = LinearAlgebra.checksquare(A, B)
      [m, n] == LinearAlgebra.checksquare(C, D) ||
               throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      T = promote_type(map(eltype, (A, B, C, D))...)
      A = convert(AbstractMatrix{T}, A)
      B = convert(AbstractMatrix{T}, B)
      C = convert(AbstractMatrix{T}, C)
      D = convert(AbstractMatrix{T}, D)
      schur_flag = isschur(A, C) && isschur(B, D)
      return new{T,typeof(A),typeof(B),typeof(C),typeof(D)}(A, B, C, D, schur_flag)
   end
end

LinearAlgebra.inv(L::SylvesterSystemMap) = InverseSylvesterSystemMap(L.A, L.B, L.C, L.D)
LinearAlgebra.inv(L::InverseSylvesterSystemMap) = SylvesterSystemMap(L.A, L.B, L.C, L.D)

"""
    MINV = invsylvsysop(A, B, C, D)

Define `MINV`, the inverse of the linear operator `M: (X,Y) -> (AX+YB, CX+YD )`,
where `(A,C)` and `(B,D)` a pairs of square matrices.
"""
invsylvsysop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   InverseSylvesterSystemMap(A, B, C, D)
invsylvsysop(AC::GeneralizedSchur, BD::GeneralizedSchur) =
   InverseSylvesterSystemMap(AC.S, BD.S, AC.T, BD.T)
Base.size(L::InverseSylvesterSystemMap) = (N = size(L.A, 1) * size(L.B, 1) * 2; return (N, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::InverseSylvesterSystemMap{T}, x::AbstractVector) where T
   require_one_based_indexing(y, x)
   m = size(L.A, 1)
   n = size(L.B, 1)
   T1 = promote_type(T, eltype(x))
   mn = m * n
   E = reshape(eltype(x) == T1 ? x[1:mn] : convert(Vector{T1}, view(x, 1:m * n)), m, n)
   F = reshape(eltype(x) == T1 ? x[mn + 1:2 * mn] : convert(Vector{T1}, view(x, mn + 1:2 * mn)), m, n)
   Y1 = view(y, 1:mn)
   Y2 = view(y, mn + 1:2mn)
   try
      if L.sf
         X1, X2 = sylvsyss!(L.A, L.B, E, L.C, L.D, F)
         copyto!(Y1, X1)
         copyto!(Y2, X2)
      else
         X1, X2  = sylvsys(L.A, L.B, E, L.C, L.D, F)
         copyto!(Y1, X1)
         copyto!(Y2, X2)
      end
   catch err
      findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end
for ttype in (LinearMaps.TransposeMap, LinearMaps.AdjointMap)
   @eval function LinearMaps._unsafe_mul!(y::AbstractVector, L::$ttype{T,<:InverseSylvesterSystemMap}, x::AbstractVector) where T
      require_one_based_indexing(y, x)
      m = size(L.lmap.A, 1)
      n = size(L.lmap.B, 1)
      mn = m * n
      T1 = promote_type(T, eltype(x))
      E = reshape(eltype(x) == T1 ? x[1:mn] : convert(Vector{T1}, view(x, 1:mn)), m, n)
      F = reshape(eltype(x) == T1 ? x[mn + 1:2 * mn] : convert(Vector{T1}, view(x, mn + 1:2 * mn)), m, n)
      Y1 = view(y, 1:mn)
      Y2 = view(y, mn + 1:2mn)
      try
         if L.lmap.sf
            X1, X2 = dsylvsyss!(true,L.lmap.A, L.lmap.B, E, L.lmap.C, L.lmap.D, F)
            copyto!(Y1, X1)
            copyto!(Y2, X2)
         else
            X1, X2 = dsylvsys(L.lmap.A', L.lmap.B', E, L.lmap.C', L.lmap.D', F)
            copyto!(Y1, X1)
            copyto!(Y2, X2)
         end
      catch err
         findfirst("LAPACKException", string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
      end
      return y
   end
end
struct GeneralizedTSylvesterMap{T,TA,TB,TC,TD,TH<:TtypeOrHtype} <: SylvesterMatrixEquationsMaps{T}
   A::TA
   B::TB
   C::TC
   D::TD
   m::Int
   n::Int
   mx::Int
   nx::Int
   function GeneralizedTSylvesterMap{T,TA,TB,TC,TD,TH}(A, B, C, D; mx = -1, nx = -1) where {T,TA,TB,TC,TD,TH}
      la = length(A)
      la == length(B) ||  throw(DimensionMismatch("A and B must have the same lengths"))
      lc = length(C)
      lc == length(D) ||  throw(DimensionMismatch("C and D must have the same lengths"))
      la+lc > 0 || throw(DimensionMismatch("either A or C must be non-empty"))
      m = mx; n = nx; 
      squareA = any(isa.(A,UniformScaling)) 
      squareC = any(isa.(C,UniformScaling))
      squareB = any(isa.(B,UniformScaling)) 
      squareD = any(isa.(D,UniformScaling))
      for i = 1:la
          isa(A[i],UniformScaling) && continue
          if m < 0 
             m = size(A[i],1)
             mx < 0 ? mx = size(A[i],2) : (mx == size(A[i],2) || throw(DimensionMismatch("all matrices in vector A must have column dimension $mx but A[$i] has column dimension $(size(A[i],2))")) )
             (squareA && m != mx) && throw(DimensionMismatch("all matrices in vector A must be square")) 
             continue
          else
             mx < 0 && (mx = size(A[i],2))
          end
          (m == size(A[i],1) && mx == size(A[i],2)) || throw(DimensionMismatch("all matrices in vector A must have the same dimensions"))
      end
      for i = 1:la
         isa(B[i],UniformScaling) && continue
         if n < 0 
            n = size(B[i],2)
            nx < 0 ? nx = size(B[i],1) : (nx == size(B[i],1) || throw(DimensionMismatch("all matrices in vector B must have row dimension $nx but B[$i] has row dimension $(size(B[i],1))")) )
            (squareB && n != nx) && throw(DimensionMismatch("all matrices in vector B must be square")) 
            continue
         else
            nx < 0 && (nx = size(B[i],1)) 
         end
         (n == size(B[i],2) && nx == size(B[i],1)) || throw(DimensionMismatch("all matrices in vector B must have the same dimensions"))
      end
      for i = 1:lc
          isa(C[i],UniformScaling) && continue
          if m < 0 
             m = size(C[i],1)
             nx < 0 ? nx = size(C[i],2) : (nx == size(C[i],2) || throw(DimensionMismatch("all matrices in vector C must have column dimension $nx but C[$i] has column dimension $(size(C[i],2))")) )
             (squareC && m != nx) && throw(DimensionMismatch("all matrices in vector C must be square")) 
             continue
          else
             nx < 0 && (nx = size(C[i],2))
          end
          (m == size(C[i],1) && nx == size(C[i],2)) || throw(DimensionMismatch("all matrices in vector C must have the same dimensions"))
      end
      for i = 1:lc
          isa(D[i],UniformScaling) && continue
          if n < 0 
             n = size(D[i],2)
             mx < 0 ? mx = size(D[i],1) : (mx == size(D[i],1) || throw(DimensionMismatch("all matrices in vector D must have row dimension $mx but D[$i] has row dimension $(size(D[i],1))")) )
             (squareD && n != mx) && throw(DimensionMismatch("all matrices in vector D must be square")) 
             continue
          else 
             mx < 0 && (mx = size(D[i],1))             
          end
          (n == size(D[i],2) && mx == size(D[i],1)) || throw(DimensionMismatch("all matrices in vector D must have the same dimensions"))
      end
      (m >= 0 && mx >= 0) || throw(DimensionMismatch("at least one of elements of A or D must be a matrix; use mx to specify the row dimension of X"))
      (n >= 0 && nx >= 0) || throw(DimensionMismatch("at least one of elements of B or C must be a matrix; use nx to specify the column dimension of X"))
          
      T1 = promote_type(eltype.(A)...)       
      T1 = promote_type(T1, eltype.(B)...)       
      T1 = promote_type(T1, eltype.(C)...)       
      T1 = promote_type(T1, eltype.(D)...)      
      T1 == Bool && (T1 = Float64) 
      A1 = [ isa(A[i],UniformScaling) ? convert(UniformScaling{T}, A[i]) : convert(AbstractMatrix{T}, A[i]) for i in 1:la ]
      B1 = [ isa(B[i],UniformScaling) ? convert(UniformScaling{T}, B[i]) : convert(AbstractMatrix{T}, B[i]) for i in 1:la ]
      C1 = [ isa(C[i],UniformScaling) ? convert(UniformScaling{T}, C[i]) : convert(AbstractMatrix{T}, C[i]) for i in 1:lc ]
      D1 = [ isa(D[i],UniformScaling) ? convert(UniformScaling{T}, D[i]) : convert(AbstractMatrix{T}, D[i]) for i in 1:lc ]
      return new{T1,typeof(A1),typeof(B1),typeof(C1),typeof(D1),TH}(A1, B1, C1, D1, m, n, mx, nx)
   end
end
function GeneralizedTSylvesterMap(A, B, C, D, ::TH = Ttype(); mx = -1, nx = -1) where {TH}
   la = length(A)
   la == length(B) ||  throw(DimensionMismatch("A and B must have the same lengths"))
   lc = length(C)
   lc == length(D) ||  throw(DimensionMismatch("C and D must have the same lengths"))
   la+lc > 0 || throw(DimensionMismatch("either A or C must be non-empty"))
   T = promote_type(eltype.(A)...)       
   T = promote_type(T, eltype.(B)...)       
   T = promote_type(T, eltype.(C)...)       
   T = promote_type(T, eltype.(D)...)      
   T == Bool && (T = Float64) 
   A1 = [ isa(A[i],UniformScaling) ? convert(UniformScaling{T}, A[i]) : convert(AbstractMatrix{T}, A[i]) for i in 1:la ]
   B1 = [ isa(B[i],UniformScaling) ? convert(UniformScaling{T}, B[i]) : convert(AbstractMatrix{T}, B[i]) for i in 1:la ]
   C1 = [ isa(C[i],UniformScaling) ? convert(UniformScaling{T}, C[i]) : convert(AbstractMatrix{T}, C[i]) for i in 1:lc ]
   D1 = [ isa(D[i],UniformScaling) ? convert(UniformScaling{T}, D[i]) : convert(AbstractMatrix{T}, D[i]) for i in 1:lc ]
   return GeneralizedTSylvesterMap{T,typeof(A1),typeof(B1),typeof(C1),typeof(D1),TH}(A1, B1, C1, D1; mx, nx)
end
"""
    M = gsylvop(A, B, C, D; mx, nx, htype = false)

Define the generalized T-Sylvester operator `M: X -> ∑ A_i*X*B_i + ∑ C_j'*transpose(X)*D_j`, if `htype = false` or
the generalized H-Sylvester operator `M: X -> ∑ A_i*X*B_i + ∑ C_j'*X'*D_j`, if `htype = true`.
`A_i` and `C_j` are matrices having the same row dimension and `B_i` and `D_j` are matrices having the same column dimension. 
`A_i` and `B_i` are contained in the `k`-vectors of matrices `A` and `B`, respectively, and 
`C_j` and `D_j` are contained in the `l`-vectors of matrices `C` and `D`, respectively. Any of the component matrices can be given as an `UniformScaling`. 
The keyword parameters `mx` and `nx` can be used to specify the row and column dimensions of `X`, if they cannot be inferred from the data.   
"""
function gsylvop(A, B, C, D; mx = -1, nx = -1, htype = false)
    GeneralizedTSylvesterMap(A, B, C, D, ifelse(htype, Htype(), Ttype()); mx, nx)
end
Base.size(L::GeneralizedTSylvesterMap) = (M = L.m*L.n; N = L.mx * L.nx; return (M, N))
function LinearMaps._unsafe_mul!(y::AbstractVector, L::GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Ttype}, x::AbstractVector) where T
   T1 = promote_type(T, eltype(x))
   X = reshape(convert(AbstractVector{T1}, x), (L.mx, L.nx))
   Y = reshape(y, (L.m, L.n))
   # Y = ∑ A_i * X * B_i + ∑ C_j * transpose(X) * D_j
   la = length(L.A)
   lc = length(L.C)
   temp = similar(Y, (L.m, L.nx))
   if la > 0
      mul!(temp, L.A[1], X)
      mul!(Y, temp, L.B[1], 1, 0)
      for i = 2:la
          mul!(temp, L.A[i], X)
          mul!(Y, temp, L.B[i], 1, 1)
      end
   end
   temp = similar(Y, (L.m, L.mx))
   if lc > 0
      mul!(temp, L.C[1], transpose(X))
      mul!(Y, temp, L.D[1], 1, la> 0 ? 1 : 0)
      for i = 2:lc
          mul!(temp, L.C[i], transpose(X))
          mul!(Y, temp, L.D[i], 1, 1)
      end
   end
   return y
end
function LinearMaps._unsafe_mul!(x::AbstractVector, L::LinearMaps.TransposeMap{T,<:GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Ttype}}, y::AbstractVector) where T
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), (L.lmap.m, L.lmap.n))
   X = reshape(x, (L.lmap.mx, L.lmap.nx))
   # X = ∑ transpose(A_i) * Y * transpose(B_i) + ∑ D_j * transpose(Y) * C_j
   la = length(L.lmap.A)
   lc = length(L.lmap.C)
   temp = similar(Y, (L.lmap.mx, L.lmap.n))
   if la > 0
      mul!(temp, transpose(L.lmap.A[1]), Y)
      mul!(X, temp, transpose(L.lmap.B[1]), 1, 0)
      for i = 2:la
          mul!(temp, transpose(L.lmap.A[i]), Y)
          mul!(X, temp, transpose(L.lmap.B[i]), 1, 1)
      end
   end
   temp = similar(Y, (L.lmap.mx, L.lmap.m))
   if lc > 0
      mul!(temp, L.lmap.D[1], transpose(Y))
      mul!(X, temp, L.lmap.C[1], 1, la> 0 ? 1 : 0)
      for i = 2:lc
          mul!(temp, L.lmap.D[i], transpose(Y))
          mul!(X, temp, L.lmap.C[i], 1, 1)
      end
   end
   return x
end
function LinearMaps._unsafe_mul!(x::AbstractVector, L::LinearMaps.AdjointMap{T,<:GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Ttype}}, y::AbstractVector) where T
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), (L.lmap.m, L.lmap.n))
   X = reshape(x, (L.lmap.mx, L.lmap.nx))
   # X = ∑ transpose(A_i) * Y * transpose(B_i) + ∑ D_j * transpose(Y) * C_j
   la = length(L.lmap.A)
   lc = length(L.lmap.C)
   temp = similar(Y, (L.lmap.mx, L.lmap.n))
   if la > 0
      mul!(temp, L.lmap.A[1]', Y)
      mul!(X, temp, L.lmap.B[1]', 1, 0)
      for i = 2:la
          mul!(temp, L.lmap.A[i]', Y)
          mul!(X, temp, L.lmap.B[i]', 1, 1)
      end
   end
   temp = similar(Y, (L.lmap.mx, L.lmap.m))
   if lc > 0
      mul!(temp, conj(L.lmap.D[1]), transpose(Y))
      mul!(X, temp, conj(L.lmap.C[1]), 1, la> 0 ? 1 : 0)
      for i = 2:lc
          mul!(temp, conj(L.lmap.D[i]), transpose(Y))
          mul!(X, temp, conj(L.lmap.C[i]), 1, 1)
      end
   end
   return x
end
function LinearMaps._unsafe_mul!(y::AbstractVector, L::GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Htype}, x::AbstractVector) where T
   T1 = promote_type(T, eltype(x))
   X = reshape(convert(AbstractVector{T1}, x), (L.mx, L.nx))
   Y = reshape(y, (L.m, L.n))
   # Y = ∑ A_i * X * B_i + ∑ C_j * X' * D_j
   la = length(L.A)
   lc = length(L.C)
   temp = similar(Y, (L.m, L.nx))
   if la > 0
      mul!(temp, L.A[1], X)
      mul!(Y, temp, L.B[1], 1, 0)
      for i = 2:la
          mul!(temp, L.A[i], X)
          mul!(Y, temp, L.B[i], 1, 1)
      end
   end
   temp = similar(Y, (L.m, L.mx))
   if lc > 0
      mul!(temp, L.C[1], X')
      mul!(Y, temp, L.D[1], 1, la> 0 ? 1 : 0)
      for i = 2:lc
          mul!(temp, L.C[i], X')
          mul!(Y, temp, L.D[i], 1, 1)
      end
   end
   return y
end
# function LinearMaps._unsafe_mul!(x::AbstractVector, L::LinearMaps.TransposeMap{T,<:GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Htype}}, y::AbstractVector) where T
#    T1 = promote_type(T, eltype(y))
#    Y = reshape(convert(AbstractVector{T1}, y), (L.lmap.m, L.lmap.n))
#    X = reshape(x, (L.lmap.mx, L.lmap.nx))
#    # X = ∑ transpose(A_i) * Y * transpose(B_i) + ∑ D_j * transpose(Y) * C_j
#    la = length(L.lmap.A)
#    lc = length(L.lmap.C)
#    temp = similar(Y, (L.lmap.mx, L.lmap.n))
#    if la > 0
#       mul!(temp, transpose(L.lmap.A[1]), Y)
#       mul!(X, temp, transpose(L.lmap.B[1]), 1, 0)
#       for i = 2:la
#           mul!(temp, transpose(L.lmap.A[i]), Y)
#           mul!(X, temp, transpose(L.lmap.B[i]), 1, 1)
#       end
#    end
#    temp = similar(Y, (L.lmap.mx, L.lmap.m))
#    if lc > 0
#       mul!(temp, L.lmap.D[1], transpose(Y))
#       mul!(X, temp, L.lmap.C[1], 1, la> 0 ? 1 : 0)
#       for i = 2:lc
#           mul!(temp, L.lmap.D[i], transpose(Y))
#           mul!(X, temp, L.lmap.C[i], 1, 1)
#       end
#    end
#    return x
# end
function LinearMaps._unsafe_mul!(x::AbstractVector, L::LinearMaps.AdjointMap{T,<:GeneralizedTSylvesterMap{T,<:Any,<:Any,<:Any,<:Any,Htype}}, y::AbstractVector) where T
   T1 = promote_type(T, eltype(y))
   Y = reshape(convert(AbstractVector{T1}, y), (L.lmap.m, L.lmap.n))
   X = reshape(x, (L.lmap.mx, L.lmap.nx))
   # X = ∑ A_i' * Y * B_i' + ∑ D_j * Y' * C_j
   la = length(L.lmap.A)
   lc = length(L.lmap.C)
   temp = similar(Y, (L.lmap.mx, L.lmap.n))
   if la > 0
      mul!(temp, L.lmap.A[1]', Y)
      mul!(X, temp, L.lmap.B[1]', 1, 0)
      for i = 2:la
          mul!(temp, L.lmap.A[i]', Y)
          mul!(X, temp, L.lmap.B[i]', 1, 1)
      end
   end
   temp = similar(Y, (L.lmap.mx, L.lmap.m))
   if lc > 0
      mul!(temp, L.lmap.D[1], Y')
      mul!(X, temp, L.lmap.C[1], 1, la> 0 ? 1 : 0)
      for i = 2:lc
          mul!(temp, L.lmap.D[i], Y')
          mul!(X, temp, L.lmap.C[i], 1, 1)
      end
   end
   return x
end


