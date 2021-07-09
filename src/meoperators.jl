abstract type LyapunovMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
abstract type SylvesterMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
const MatrixEquationsMaps{T} = Union{LyapunovMatrixEquationsMaps{T}, SylvesterMatrixEquationsMaps{T} }

"""
    M = trmatop(n, m) 

Define the transposition operator `M: X -> X'` for all `n x m` matrices.
"""
struct trmatop{Int} <: LinearMaps.LinearMap{Int}
    size::Dims{2}
    function trmatop(dims::Dims{2}) where {T}
        all(≥(0), dims) || throw(ArgumentError("dims must be non-negative"))
        return new{Int}(dims)
    end
    function trmatop(m::Int,n::Int) where {T}
      (m ≥(0) & n ≥(0))  || throw(ArgumentError("dimensions must be non-negative"))
      return new{Int}((m,n))
    end
end
trmatop(n::Int) = trmatop(n,n)
Base.size(A::trmatop) = (prod(A.size),prod(A.size))
LinearAlgebra.issymmetric(A::trmatop) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::trmatop) = A.size[1] == A.size[2]
function LinearAlgebra.mul!(y::AbstractVector, A::trmatop, x::AbstractVector)
    X = reshape(x, A.size...)
    LinearMaps.check_dim_mul(y, A, x)
    y[:] = adjoint(X)[:]
    return y[:]
end
"""
    M = trmatop(A) 

Define the transposition operator `M: X -> X'` of all matrices of the size of `A`.
"""
trmatop(A) = trmatop(size(A))
struct LyapunovMap{T} <: LyapunovMatrixEquationsMaps{T}
  A::AbstractMatrix
  disc::Bool
  her::Bool
  adj::Bool
  function LyapunovMap(A::AbstractMatrix{T}; disc = false, her = false) where {T}
      LinearAlgebra.checksquare(A)
      return new{T}(A, disc, her, isa(A,Adjoint))
  end
end
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
function lyapop(A::AbstractMatrix{T}; disc = false, her = false) where T
    LyapunovMap(A; disc = disc, her = her)
end
lyapop(A::Schur{T,Matrix{T}}; kwargs...) where T = LyapunovMap(A.T; kwargs...)
lyapop(A::Number; kwargs...) = LyapunovMap(A*ones(eltype(A),1,1); kwargs...) 
function Base.size(L::LyapunovMap)
    n = size(L.A,1)
    N = L.her ? Int(n*(n+1)/2) : n*n
    return (N,N)
end
Base.size(L::Adjoint{<:Any, <: LyapunovMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::LyapunovMap{T}, x::AbstractVector) where T
  n = size(L.A,1)
  T1 = promote_type(T, eltype(x))
  if L.her
     T == T1 ?  X = vec2triu(x, her = true) :  X = vec2triu(convert(Vector{T1}, x), her = true)
     if L.disc
        # (y .= triu2vec(utqu(X,L.A') - X))
        muldsym!(y, L.A, X)
     else
        mulcsym!(y, L.A, X)
     end
     return y
   else
     T == T1 ? X = reshape(x, n, n) : X = reshape(convert(Vector{T1}, x), n, n)
     if L.disc
        # (y .= (L.A*X*L.A' - X)[:])
        Y = reshape(y, (n, n))  
        Y .= X 
        temp = similar(Y, (n, n))
        mul!(temp, X, L.A')
        mul!(Y, L.A, temp, 1, -1)
        return y
     else
        # (y[:] = (L.A*X + X*L.A')[:])
        Y = reshape(y, (n, n))   
        mul!(Y, X, L.A')
        mul!(Y, L.A, X, 1, 1)
        return y
     end
  end
end 
function  mulcsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix) 
   # A*X + X*A'
   n = size(A,1)
   ZERO = zero(eltype(y))
   @inbounds  begin
      k = 1
      for j = 1:n
          for i = 1:j
             temp = ZERO
             for l = 1:n
                 temp += A[i,l]*X[l,j] + X[i,l]*conj(A[j,l])
             end
             y[k] = temp
             k += 1
          end
      end
   end
   return y
end
function  muldsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix) 
   # A*X*A' - X
   n = size(A,1)
   # t = triu(X)-diag(X)/2
   t = UpperTriangular(X)-Diagonal(X[diagind(X)]./2)
   Y = similar(X, n, n)
   # Y = A*t*A'
   mul!(Y, A*t, A') 
   # Y + Y' - X
   @inbounds  begin
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

function Base.Matrix{T}(A::LyapunovMatrixEquationsMaps) where {T}
   A.her && A.adj && (return Matrix(A')')
   M, N = size(A)
   mat = Matrix{T}(undef, (M, N))
   v = fill(zero(T), N)
   @inbounds for i in 1:N
       v[i] = one(T)
       # need mul!, e.g., for TransposeMap{<:CustomMap}
       mul!(view(mat, :, i), A, v)
       v[i] = zero(T)
   end
   return mat
end

LinearAlgebra.adjoint(L::LyapunovMap{T}) where T = lyapop(L.A'; disc = L.disc, her = L.her)
LinearAlgebra.transpose(L::LyapunovMap{T}) where T = lyapop(L.A'; disc = L.disc, her = L.her)
struct GeneralizedLyapunovMap{T} <: LyapunovMatrixEquationsMaps{T}
   A::AbstractMatrix
   E::AbstractMatrix
   disc::Bool
   her::Bool
   adj::Bool
   function GeneralizedLyapunovMap(A::AbstractMatrix{T}, E::AbstractMatrix{T}; disc = false, her = false) where {T}
       n = LinearAlgebra.checksquare(A)
       n == LinearAlgebra.checksquare(E) ||
            throw(DimensionMismatch("E must be a square matrix of dimension $n"))
       adj = isa(A,Adjoint) && isa(E,Adjoint)
       return new{T}(A, E, disc, her, adj)
   end
end
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
function lyapop(A::AbstractMatrix{T}, E::AbstractMatrix{T}; disc = false, her = false) where T
    GeneralizedLyapunovMap(A, E; disc = disc, her = her)
end
lyapop(F::GeneralizedSchur{T,Matrix{T}}; kwargs...) where T = GeneralizedLyapunovMap(F.S, F.T; kwargs...)
function lyapop(A::Number, E::Number; kwargs...)
    T = promote_type(eltype(A),eltype(E))
    GeneralizedLyapunovMap(A*ones(T,1,1), E*ones(T,1,1); kwargs...)
end
function lyapop(A::AbstractMatrix{T1}, E::AbstractMatrix{T2}; kwargs...) where {T1,T2}
   T = promote_type(eltype(A),eltype(E)) 
   GeneralizedLyapunovMap(convert(Matrix{T},A), convert(Matrix{T},E); kwargs...)
end
function Base.size(L::GeneralizedLyapunovMap)
     n = size(L.A,1)
     N = L.her ? Int(n*(n+1)/2) : n*n
     return (N,N)
end
Base.size(L::Adjoint{<:Any, <: GeneralizedLyapunovMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::GeneralizedLyapunovMap{T}, x::AbstractVector) where T
   n = size(L.A,1)
   T1 = promote_type(T, eltype(x))
   if L.her
      T == T1 ?  X = vec2triu(x, her = true) :  X = vec2triu(convert(Vector{T1}, x), her = true)
      if L.disc
         # (y .= triu2vec(utqu(X,L.A') - utqu(X,L.E')))
         muldsym!(y, L.A, L.E, X)
      else
         mulcsym!(y, L.A, L.E, X) 
      end
      return y
    else
      T == T1 ? X = reshape(x, n, n) : X = reshape(convert(Vector{T1}, x), n, n)
      if L.disc
         # (y .= (L.A*X*L.A' - L.E*X*L.E')[:])
         Y = reshape(y, (n, n))  
         #Y .= X 
         temp = similar(Y, (n, n))
         mul!(temp, X, L.A')
         mul!(Y, L.A, temp)
         mul!(temp, X, L.E')
         mul!(Y, L.E, temp, -1, 1)
         return y
      else
         # (y[:] = (L.A*X*L.E' + L.E*X*L.A')[:])
         Y = reshape(y, (n, n))   
         temp = similar(Y, (n, n))
         mul!(temp, L.E, X)
         mul!(Y, temp, L.A')
         mul!(temp, X, L.E')
         mul!(Y, L.A, temp, 1, 1)
         return y
      end
   end
end 
function  mulcsym!(y::AbstractVector, A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix) 
   # AXE' + EXA' 
   n = size(A,1)
   Y = similar(X, n, n) 
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
                 temp += A[i,l]*Y[l,j] + conj(Y[l,i]*A[j,l])
             end
             y[k] = temp
             k += 1
          end
      end
   end
   return y
end
function  muldsym!(y::AbstractVector,A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix) 
   # AXA' - EXE' 
   n = size(A,1)
   # t = triu(X)-diag(X)/2
   t = UpperTriangular(X)-Diagonal(X[diagind(X)]./2)
   Y = similar(X, n, n)
   # Y = A*t*A' - E*t*E'
   mul!(Y, A*t, A') 
   mul!(Y, E*t, E', -1, 1)
   # Y + Y' 
   @inbounds  begin
      k = 1
      for j = 1:n
         for i = 1:j
             y[k] = Y[i,j]+conj(Y[j,i])
             k += 1
          end
      end
   end
   return y
end
LinearAlgebra.adjoint(L::GeneralizedLyapunovMap{T}) where T = lyapop(L.A', L.E'; disc = L.disc, her = L.her)
LinearAlgebra.transpose(L::GeneralizedLyapunovMap{T}) where T = lyapop(L.A', L.E'; disc = L.disc, her = L.her)

struct InverseLyapunovMap{T} <: LyapunovMatrixEquationsMaps{T}
  A::AbstractMatrix
  disc::Bool
  her::Bool
  adj::Bool
  sf::Bool
  function InverseLyapunovMap(A::AbstractMatrix{T}; disc = false, her = false) where {T <: BlasFloat}
      LinearAlgebra.checksquare(A)
      adj = isa(A,Adjoint)
      schur_flag = adj ? isschur(A.parent) : isschur(A)
      return new{T}(adj ? A.parent : A, disc, her, adj, schur_flag)
  end
end
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
function invlyapop(A::AbstractMatrix{T}; disc = false, her = false) where {T <: BlasFloat}
    InverseLyapunovMap(A; disc = disc, her = her)
end
function invlyapop(A::AbstractMatrix; disc = false, her = false) 
   T = eltype(A)
   T <: BlasFloat || (T = promote_type(T, Float64) )
   InverseLyapunovMap(T.(A); disc = disc, her = her)     
end
invlyapop(A::Schur{T,Matrix{T}}; kwargs...) where {T <: BlasFloat} = InverseLyapunovMap(A.T; kwargs...)
function Base.size(L::InverseLyapunovMap)
    n = size(L.A,1)
    N = L.her ? Int(n*(n+1)/2) : n*n
    return (N,N)
end
Base.size(L::Adjoint{<:Any, <: InverseLyapunovMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::InverseLyapunovMap{T}, x::AbstractVector) where {T <: BlasFloat}
  n = size(L.A,1)
  T1 = promote_type(T, eltype(x))
  y = try
    if L.sf
       if L.her
          T1 == eltype(x) ? Y = vec2triu(-x, her = true) : Y = vec2triu(-convert(Vector{T1},x), her = true)
          L.disc ? lyapds!(L.A, Y; adj = L.adj) : lyapcs!(L.A, Y; adj = L.adj)
          y .= triu2vec(Y)
       else
          if L.disc
            T1 == eltype(x) ? Y = reshape(-x, n, n) : Y = reshape(convert(Vector{T1}, -x), n, n)
            L.adj ? sylvds!(-L.A, L.A, Y, adjA = true) : sylvds!(-L.A, L.A, Y, adjB = true)
          else
            T1 == eltype(x) ? Y = copy(reshape(x, n, n)) : Y = reshape(convert(Vector{T1}, x), n, n)
           L.adj ? (sylvcs!(L.A, L.A, Y, adjA = true)) : (sylvcs!(L.A, L.A, Y, adjB = true))
          end
          y .= Y[:]
       end
    else
       if L.her
          T1 == eltype(x) ? Y = vec2triu(-x, her = true) : Y = vec2triu(-convert(Vector{T1},x), her = true)
          if L.disc
             L.adj ? (y .= triu2vec(lyapd(L.A',Y))) : (y .= triu2vec(lyapd(L.A,Y)))
          else
             L.adj ? (y .= triu2vec(lyapc(L.A',Y))) : (y .= triu2vec(lyapc(L.A,Y)))
          end
       else
          T1 == eltype(x) ? Y = reshape(-x, n, n) : Y = reshape(convert(Vector{T1}, -x), n, n)
          if L.disc
             L.adj ? (y .= lyapd(L.A',Y)[:]) : (y .= lyapd(L.A,Y)[:])
          else
             L.adj ? (y .= lyapc(L.A',Y)[:]) : (y .= lyapc(L.A,Y)[:])
         end
       end
    end
  catch err
    findfirst("SingularException",string(err)) === nothing &&
    findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
  end
  return y
end
LinearAlgebra.adjoint(L::InverseLyapunovMap{T}) where T = invlyapop(L.adj ? L.A : L.A'; disc = L.disc, her = L.her)
function LinearAlgebra.transpose(L::InverseLyapunovMap{T}) where T
    invlyapop(L.adj ? L.A : L.A'; disc = L.disc, her = L.her)
end
LinearAlgebra.inv(L::LyapunovMap{T}) where T = invlyapop(L.A; disc = L.disc, her = L.her)
LinearAlgebra.inv(L::InverseLyapunovMap{T}) where T = lyapop(L.A; disc = L.disc, her = L.her)

struct InverseGeneralizedLyapunovMap{T} <: LyapunovMatrixEquationsMaps{T}
  A::AbstractMatrix
  E::AbstractMatrix
  disc::Bool
  her::Bool
  adj::Bool
  sf::Bool
  function InverseGeneralizedLyapunovMap(A::AbstractMatrix{T}, E::AbstractMatrix{T}; disc = false, her = false) where {T}
      LinearAlgebra.checksquare(A)
      adj = isa(A,Adjoint) && isa(E,Adjoint)
      schur_flag = adj ? isschur(A.parent, A.parent) : isschur(A, E)
      return new{T}(adj ? A.parent : A, adj ? E.parent : E, disc, her, adj, schur_flag)
  end
end
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
function invlyapop(A::AbstractMatrix{T}, E::AbstractMatrix{T}; disc = false, her = false) where {T <: BlasFloat}
    InverseGeneralizedLyapunovMap(A, E; disc = disc, her = her)
end
invlyapop(F::GeneralizedSchur{T,Matrix{T}}; kwargs...) where T = InverseGeneralizedLyapunovMap(F.S, F.T; kwargs...)
function invlyapop(A::AbstractMatrix, E::AbstractMatrix; kwargs...) 
    T = promote_type(eltype(A), eltype(E))
    T <: BlasFloat || (T = promote_type(T, Float64) )
    invlyapop(convert(Matrix{T},A), convert(Matrix{T},E); kwargs...)
end
function Base.size(L::InverseGeneralizedLyapunovMap)
    n = size(L.A,1)
    N = L.her ? Int(n*(n+1)/2) : n*n
    return (N,N)
end
Base.size(L::Adjoint{<:Any, <: InverseGeneralizedLyapunovMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::InverseGeneralizedLyapunovMap{T}, x::AbstractVector) where T
  n = size(L.A,1)
  T1 = promote_type(T, eltype(x))
  y = try
    if L.sf
       if L.her
          T1 == eltype(x) ? Y = vec2triu(-x, her = true) : Y = vec2triu(-convert(Vector{T1},x), her = true)
          L.disc ? lyapds!(L.A, L.E, Y; adj = L.adj) : lyapcs!(L.A, L.E, Y; adj = L.adj)
          y .= triu2vec(Y)
       else
         T1 == eltype(x) ? Y = copy(reshape(x, n, n)) : Y = copy(reshape(convert(Vector{T1},x), n, n)) 
         if L.disc
            L.adj ? gsylvs!(L.A, L.A, -L.E, L.E, Y, adjAC = true) : gsylvs!(L.A, L.A, -L.E, L.E, Y, adjBD = true)
         else
            L.adj ? (gsylvs!(L.A,L.E,L.E,L.A,Y,adjAC = true,DBSchur = true)) : (gsylvs!(L.A,L.E,L.E,L.A,Y,adjBD = true,DBSchur = true))
         end
         y .= Y[:]
       end
    else
       if L.her
          T1 == eltype(x) ? Y = vec2triu(-x, her = true) : Y = vec2triu(-convert(Vector{T1}, x), her = true) 
          if L.disc
             L.adj ? (y .= triu2vec(lyapd(L.A', L.E', Y))) : (y .= triu2vec(lyapd(L.A, L.E, Y)))
          else
             L.adj ? (y .= triu2vec(lyapc(L.A', L.E', Y))) : (y .= triu2vec(lyapc(L.A, L.E, Y)))
          end
       else
          T1 == eltype(x) ? Y = reshape(-x, n, n) : Y = reshape(-convert(Vector{T1}, x), n, n)
          if L.disc
             # L.adj ? (y .= gsylv(-L.A',L.A,L.E',L.E,Y)[:]) : (y .= gsylv(-L.A,L.A',L.E,L.E',Y)[:])
             L.adj ? (y .= lyapd(L.A', L.E', Y)[:]) : (y .= lyapd(L.A, L.E, Y)[:])
          else
             # L.adj ? (y .= gsylv(L.A',L.E,L.E',L.A,Y)[:]) : (y .= gsylv(L.A,L.E',L.E,L.A',Y)[:])
             L.adj ? (y .= lyapc(L.A', L.E', Y)[:]) : (y .= lyapc(L.A, L.E, Y)[:])
          end
          return y
       end
    end
  catch err
    findfirst("SingularException",string(err)) === nothing &&
    findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
  end
  return y
end
LinearAlgebra.adjoint(L::InverseGeneralizedLyapunovMap{T}) where T = invlyapop(L.adj ? L.A : L.A', L.adj ? L.E : L.E'; disc = L.disc, her = L.her)
LinearAlgebra.transpose(L::InverseGeneralizedLyapunovMap{T}) where T = invlyapop(L.adj ? L.A : L.A', L.adj ? L.E : L.E'; disc = L.disc, her = L.her)
LinearAlgebra.inv(L::GeneralizedLyapunovMap{T}) where T = invlyapop(L.A, L.E; disc = L.disc, her = L.her)
LinearAlgebra.inv(L::InverseGeneralizedLyapunovMap{T}) where T = lyapop(L.A, L.E; disc = L.disc, her = L.her)

struct SylvesterMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   disc::Bool
   function SylvesterMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}; disc = false) where {T}
      LinearAlgebra.checksquare(A, B)
      return new{T}(A, B, disc)
   end
end
"""
    M = sylvop(A, B; disc = false) 

Define the continuous Sylvester operator `M: X -> AX+XB` if `disc = false`
or the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`, where `A` and `B` are square matrices.
"""
function sylvop(A::AbstractMatrix{T}, B::AbstractMatrix{T}; disc = false) where T
    SylvesterMap(A, B; disc = disc)
end
sylvop(A::Schur{T,Matrix{T}}, B::Schur{T,Matrix{T}}; kwargs...) where T = SylvesterMap(A.T, B.T; kwargs...)
function sylvop(A::Number, B::Number; kwargs...)
   T = promote_type(eltype(A),eltype(B))
   SylvesterMap(A*ones(T,1,1), B*ones(T,1,1); kwargs...)
end
function sylvop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}; kwargs...) where {T1,T2}
   T = promote_type(eltype(A),eltype(B)) 
   SylvesterMap(convert(Matrix{T},A), convert(Matrix{T},B); kwargs...)
end
function Base.size(L::SylvesterMap)
   N = size(L.A,1)*size(L.B,1)
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: SylvesterMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::SylvesterMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   T1 = promote_type(T, eltype(x))
   T == T1 ? X = reshape(x, m, n) : X = reshape(convert(Vector{T1}, x), m, n)
   Y = reshape(y, (m, n))  
   if L.disc
      # Y = A * X * B + X
      Y .= X
      temp = similar(Y, (m, n))
      mul!(temp, L.A, X) 
      mul!(Y, temp, L.B, 1, 1) 
   else
      # Y = A * X + X * B
      mul!(Y, L.A, X) 
      mul!(Y, X, L.B, 1, 1) 
   end
   return y
end 
LinearAlgebra.adjoint(L::SylvesterMap{T}) where T = SylvesterMap(L.A', L.B'; disc = L.disc)
LinearAlgebra.transpose(L::SylvesterMap{T}) where T = SylvesterMap(L.A', L.B'; disc = L.disc)
struct InverseSylvesterMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   disc::Bool
   adjA::Bool
   adjB::Bool
   sf::Bool
   function InverseSylvesterMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}; disc = false) where {T}
      LinearAlgebra.checksquare(A, B)
      adjA = isa(A,Adjoint)
      adjB = isa(B,Adjoint)
      schur_flag = (adjA ? isschur(A.parent) : isschur(A)) && (adjB ? isschur(B.parent) : isschur(B)) 
      return new{T}(adjA ? A.parent : A, adjB ? B.parent : B, disc, adjA, adjB, schur_flag)
   end
end
"""
    MINV = invsylvop(A, B; disc = false) 

Define `MINV`, the inverse of the continuous Sylvester operator  `M: X -> AX+XB` if `disc = false`
or of the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`, where `A` and `B` are square matrices.
"""
function invsylvop(A::AbstractMatrix{T}, B::AbstractMatrix{T}; disc = false) where T
    InverseSylvesterMap(A, B; disc = disc)
end
invsylvop(A::Schur{T,Matrix{T}}, B::Schur{T,Matrix{T}}; kwargs...) where T = InverseSylvesterMap(A.T, B.T; kwargs...)
function invsylvop(A::Number, B::Number; kwargs...)
   T = promote_type(eltype(A),eltype(B))
   InverseSylvesterMap(A*ones(T,1,1), B*ones(T,1,1); kwargs...)
end
function invsylvop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}; kwargs...) where {T1,T2}
   T = promote_type(eltype(A),eltype(B)) 
   InverseSylvesterMap(convert(Matrix{T},A), convert(Matrix{T},B); kwargs...)
end
function Base.size(L::InverseSylvesterMap)
   m = size(L.A,1)
   N = size(L.A,1)*size(L.B,1)
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: InverseSylvesterMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::InverseSylvesterMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   T1 = promote_type(T, eltype(x))
   y = try
      if L.sf
         eltype(x) == T1 ? Y = reshape(copy(x), m, n) : Y = reshape(convert(Vector{T1}, x), m, n)
         if L.disc
            sylvds!(L.A, L.B, Y, adjA = L.adjA, adjB = L.adjB)
         else
            sylvcs!(L.A, L.B, Y, adjA = L.adjA, adjB = L.adjB)
         end
         y .= Y[:]
      else
         eltype(x) == T1 ? Y = reshape(x, m, n) : Y = reshape(convert(Vector{T1}, x), m, n)
         if L.disc
            y .= sylvd(L.adjA ? L.A' : L.A, L.adjB ? L.B' : L.B, Y)[:]
         else
            y .= sylvc(L.adjA ? L.A' : L.A, L.adjB ? L.B' : L.B, Y)[:]
         end
      end
   catch err
       findfirst("SingularException",string(err)) === nothing &&
       findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end 
LinearAlgebra.adjoint(L::InverseSylvesterMap{T}) where T = InverseSylvesterMap(L.adjA ? L.A : L.A', L.adjB ? L.B : L.B'; disc = L.disc)
LinearAlgebra.transpose(L::InverseSylvesterMap{T}) where T = InverseSylvesterMap(L.adjA ? L.A : L.A', L.adjB ? L.B : L.B'; disc = L.disc)
LinearAlgebra.inv(L::SylvesterMap{T}) where T = InverseSylvesterMap(L.A, L.B; disc = L.disc)
LinearAlgebra.inv(L::InverseSylvesterMap{T}) where T = SylvesterMap(L.A, L.B; disc = L.disc)
struct GeneralizedSylvesterMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   C::AbstractMatrix
   D::AbstractMatrix
   function GeneralizedSylvesterMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
      m, n = LinearAlgebra.checksquare(A, B)
      [m; n] == LinearAlgebra.checksquare(C,D) ||
             throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      return new{T}(A, B, C, D)
   end
end
"""
    M = sylvop(A, B, C, D) 

Define the generalized Sylvester operator `M: X -> AXB+CXD`, where `(A,C)` and `(B,D)` a pairs of square matrices.
"""
function sylvop(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
    GeneralizedSylvesterMap(A, B, C, D)
end
function sylvop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}, C::AbstractMatrix{T3}, D::AbstractMatrix{T4}) where {T1,T2,T3,T4}
   T = promote_type(T1, T2, T3, T4) 
   GeneralizedSylvesterMap(convert(Matrix{T},A), convert(Matrix{T},B), convert(Matrix{T},C), convert(Matrix{T},D))
end
function Base.size(L::GeneralizedSylvesterMap)
   N = size(L.A,1)*size(L.B,1)
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: GeneralizedSylvesterMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::GeneralizedSylvesterMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   T1 = promote_type(T, eltype(x))
   eltype(x) == T1 ? X = reshape(x, m, n) : X = reshape(convert(Vector{T1}, x), m, n)
   Y = reshape(y, (m, n))  
   # Y = A * X * B + C * X * D
   temp = similar(Y, (m, n))
   mul!(temp, L.A, X) 
   mul!(Y, temp, L.B, 1, 0) 
   mul!(temp, L.C, X) 
   mul!(Y, temp, L.D, 1, 1) 
   return y
end 
LinearAlgebra.adjoint(L::GeneralizedSylvesterMap{T}) where T = 
   GeneralizedSylvesterMap(L.A', L.B', L.C', L.D')
LinearAlgebra.transpose(L::GeneralizedSylvesterMap{T}) where T = 
   GeneralizedSylvesterMap(L.A', L.B', L.C', L.D')


struct InverseGeneralizedSylvesterMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   C::AbstractMatrix
   D::AbstractMatrix
   adjAC::Bool
   adjBD::Bool
   sf::Bool
   BDSchur::Bool
   DBSchur::Bool
   function InverseGeneralizedSylvesterMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
      m, n = LinearAlgebra.checksquare(A, B)
      [m; n] == LinearAlgebra.checksquare(C,D) ||
             throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      adjAC = isa(A,Adjoint) && isa(C,Adjoint) 
      adjBD = isa(B,Adjoint) && isa(D,Adjoint) 
      ACSchur = (adjAC ? isschur(A.parent,C.parent) : isschur(A,C))
      BDSchur = (adjBD ? isschur(B.parent,D.parent) : isschur(B,D))
      DBSchur = (adjBD ? isschur(D.parent,B.parent) : isschur(D,B))
      schur_flag = ACSchur && (BDSchur || DBSchur) 
      return new{T}(adjAC ? A.parent : A, adjBD ? B.parent : B, adjAC ? C.parent : C, adjBD ? D.parent : D, adjAC, adjBD, schur_flag, BDSchur, DBSchur)
   end
end

"""
    MINV = invsylvop(A, B, C, D) 

Define `MINV`, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`, 
where (A,C) and (B,D) a pairs of square matrices.
"""
function invsylvop(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
   InverseGeneralizedSylvesterMap(A, B, C, D)
end
function invsylvop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}, C::AbstractMatrix{T3}, D::AbstractMatrix{T4}) where {T1,T2,T3,T4}
   T = promote_type(T1, T2, T3, T4) 
   InverseGeneralizedSylvesterMap(convert(Matrix{T},A), convert(Matrix{T},B), convert(Matrix{T},C), convert(Matrix{T},D))
end
function Base.size(L::InverseGeneralizedSylvesterMap)
   N = size(L.A,1)*size(L.B,1)
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: InverseGeneralizedSylvesterMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::InverseGeneralizedSylvesterMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   T1 = promote_type(T, eltype(x))
   y = try
      if L.sf
         eltype(x) == T1 ? Y = reshape(copy(x), m, n) : Y = reshape(convert(Vector{T1}, x), m, n)
         gsylvs!(L.A, L.B, L.C, L.D, Y, adjAC = L.adjAC, adjBD = L.adjBD, DBSchur = L.DBSchur && !L.BDSchur)
         y .= Y[:]
      else
         eltype(x) == T1 ? Y = reshape(x, m, n) : Y = reshape(convert(Vector{T1}, x), m, n)
         y .= gsylv(L.adjAC ? L.A' : L.A, L.adjBD ? L.B' : L.B, 
                    L.adjAC ? L.C' : L.C, L.adjBD ? L.D' : L.D, Y)[:]
      end
   catch err
       findfirst("SingularException",string(err)) === nothing &&
       findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end 
invsylvop(F::GeneralizedSchur{T,Matrix{T}}, G::GeneralizedSchur{T,Matrix{T}}) where T = invsylvop(F.S, G.S, F.T, G.T)
LinearAlgebra.adjoint(L::InverseGeneralizedSylvesterMap{T}) where T = 
      InverseGeneralizedSylvesterMap(L.adjAC ? L.A : L.A', L.adjBD ? L.B : L.B', 
                                     L.adjAC ? L.C : L.C', L.adjBD ? L.D : L.D')
LinearAlgebra.transpose(L::InverseGeneralizedSylvesterMap{T}) where T = 
      InverseGeneralizedSylvesterMap(L.adjAC ? L.A : L.A', L.adjBD ? L.B : L.B', 
                                     L.adjAC ? L.C : L.C', L.adjBD ? L.D : L.D')
LinearAlgebra.inv(L::GeneralizedSylvesterMap{T}) where T = 
      InverseGeneralizedSylvesterMap(L.A, L.B, L.C, L.D)
LinearAlgebra.inv(L::InverseGeneralizedSylvesterMap{T}) where T = 
      GeneralizedSylvesterMap(L.A, L.B, L.C, L.D)

struct SylvesterSystemMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   C::AbstractMatrix
   D::AbstractMatrix
   function SylvesterSystemMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
      m, n = LinearAlgebra.checksquare(A, B)
      [m; n] == LinearAlgebra.checksquare(C,D) ||
             throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      return new{T}(A, B, C, D)
   end
end
"""
    M = sylvsysop(A, B, C, D) 

Define the operator `M: (X,Y) -> (AX+YB, CX+YD )`, 
where `(A,C)` and `(B,D)` a pairs of square matrices.
"""
function sylvsysop(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
    SylvesterSystemMap(A, B, C, D)
end
function sylvsysop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}, C::AbstractMatrix{T3}, D::AbstractMatrix{T4}) where {T1,T2,T3,T4}
   T = promote_type(T1, T2, T3, T4) 
   SylvesterSystemMap(convert(Matrix{T},A), convert(Matrix{T},B), convert(Matrix{T},C), convert(Matrix{T},D))
end
function Base.size(L::SylvesterSystemMap)
   N = size(L.A,1)*size(L.B,1)*2
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: SylvesterSystemMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::SylvesterSystemMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   mn = m*n
   T1 = promote_type(T, eltype(x))
   eltype(x) == T1 ? X = reshape(view(x,1:mn), m, n) : X = reshape(convert(Vector{T1}, view(x,1:m*n)), m, n)
   eltype(x) == T1 ? Y = reshape(view(x,mn+1:2*mn), m, n) : Y = reshape(convert(Vector{T1}, view(x,mn+1:2*mn)), m, n)
   U = reshape(view(y,1:m*n), (m, n))  
   V = reshape(view(y,mn+1:2*mn), (m, n))  
   # U = A * X + Y * B 
   # V = C * X + Y * D
   mul!(U, L.A, X) 
   mul!(U, Y, L.B, 1, 1) 
   mul!(V, L.C, X) 
   mul!(V, Y, L.D, 1, 1) 
   return y
end 
function LinearAlgebra.mul!(y::AbstractVector, LT::Union{LinearMaps.TransposeMap{T, SylvesterSystemMap{T}},LinearMaps.AdjointMap{T, SylvesterSystemMap{T}}}, x::AbstractVector) where T
   m = size(LT.lmap.A,1)
   n = size(LT.lmap.B,1)
   mn = m*n
   T1 = promote_type(T, eltype(x))
   eltype(x) == T1 ? X = reshape(view(x,1:mn), m, n) : X = reshape(convert(Vector{T1}, view(x,1:m*n)), m, n)
   eltype(x) == T1 ? Y = reshape(view(x,mn+1:2*mn), m, n) : X = reshape(convert(Vector{T1}, view(x,mn+1:2*mn)), m, n)
   U = reshape(view(y,1:m*n), (m, n))  
   V = reshape(view(y,mn+1:2*mn), (m, n))  
   # U = A' * X + C' * Y 
   # V = X * B' + Y * D'
   mul!(U, LT.lmap.A', X) 
   mul!(U, LT.lmap.C', Y, 1, 1) 
   mul!(V, X, LT.lmap.B') 
   mul!(V, Y, LT.lmap.D', 1, 1) 
   return y
end 
struct InverseSylvesterSystemMap{T} <: SylvesterMatrixEquationsMaps{T}
   A::AbstractMatrix
   B::AbstractMatrix
   C::AbstractMatrix
   D::AbstractMatrix
   sf::Bool
   function InverseSylvesterSystemMap(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where {T}
      m, n = LinearAlgebra.checksquare(A, B)
      [m; n] == LinearAlgebra.checksquare(C,D) ||
             throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
      schur_flag = isschur(A,C) && isschur(B,D)
      return new{T}(A, B, C, D, schur_flag)
   end
end
"""
    MINV = invsylvsysop(A, B, C, D) 

Define `MINV`, the inverse of the linear operator `M: (X,Y) -> (AX+YB, CX+YD )`, 
where `(A,C)` and `(B,D)` a pairs of square matrices.
"""
function invsylvsysop(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}) where T
    InverseSylvesterSystemMap(A, B, C, D)
end
function invsylvsysop(A::AbstractMatrix{T1}, B::AbstractMatrix{T2}, C::AbstractMatrix{T3}, D::AbstractMatrix{T4}) where {T1,T2,T3,T4}
   T = promote_type(T1, T2, T3, T4) 
   InverseSylvesterSystemMap(convert(Matrix{T},A), convert(Matrix{T},B), convert(Matrix{T},C), convert(Matrix{T},D))
end
function Base.size(L::InverseSylvesterSystemMap)
   N = size(L.A,1)*size(L.B,1)*2
   return (N,N)
end
Base.size(L::Adjoint{<:Any, <: InverseSylvesterSystemMap}) = size(L.parent)
function LinearAlgebra.mul!(y::AbstractVector, L::InverseSylvesterSystemMap{T}, x::AbstractVector) where T
   m = size(L.A,1)
   n = size(L.B,1)
   T1 = promote_type(T, eltype(x))
   mn = m*n
   eltype(x) == T1 ? E = reshape(x[1:mn], m, n) : E = reshape(convert(Vector{T1}, view(x,1:m*n)), m, n)
   eltype(x) == T1 ? F = reshape(x[mn+1:2*mn], m, n) : F = reshape(convert(Vector{T1}, view(x,mn+1:2*mn)), m, n)
   X = reshape(view(y,1:m*n), (m, n))  
   Y = reshape(view(y,mn+1:2*mn), (m, n))  
   y = try
      if L.sf
         X, Y = sylvsyss!(L.A,L.B,E,L.C,L.D,F) 
         y .= [X Y][:]
      else
         X, Y = sylvsys(L.A,L.B,E,L.C,L.D,F)
         y .= [X Y][:]
      end
   catch err
       findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end 
function LinearAlgebra.mul!(y::AbstractVector, L::Union{LinearMaps.TransposeMap{T, InverseSylvesterSystemMap{T}},LinearMaps.AdjointMap{T, InverseSylvesterSystemMap{T}}}, x::AbstractVector) where T
   m = size(L.lmap.A,1)
   n = size(L.lmap.B,1)
   mn = m*n
   T1 = promote_type(T, eltype(x))
   mn = m*n
   eltype(x) == T1 ? E = reshape(x[1:mn], m, n) : E = reshape(convert(Vector{T1}, view(x,1:m*n)), m, n)
   eltype(x) == T1 ? F = reshape(x[mn+1:2*mn], m, n) : F = reshape(convert(Vector{T1}, view(x,mn+1:2*mn)), m, n)
   X = reshape(view(y,1:m*n), (m, n))  
   Y = reshape(view(y,mn+1:2*mn), (m, n))  
   y = try
      if L.lmap.sf
         X, Y = dsylvsyss!(L.lmap.A,L.lmap.B,E,L.lmap.C,L.lmap.D,F) 
         y .= [X Y][:]
      else
         X, Y = dsylvsys(L.lmap.A',L.lmap.B',E,L.lmap.C',L.lmap.D',F)
         y .= [X Y][:]
      end
   catch err
       findfirst("LAPACKException",string(err)) === nothing ? rethrow() : throw("ME:SingularException: Singular operator")
   end
   return y
end 
invsylvsysop(AC :: GeneralizedSchur, BD :: GeneralizedSchur) = invsylvsysop(AC.S, BD.S, AC.T, BD.T)
LinearAlgebra.inv(L::SylvesterSystemMap{T}) where T = InverseSylvesterSystemMap(L.A, L.B, L.C, L.D)
LinearAlgebra.inv(L::InverseSylvesterSystemMap{T}) where T = SylvesterSystemMap(L.A, L.B, L.C, L.D)

