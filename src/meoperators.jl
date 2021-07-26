# TODO: These types aren't used anywhere for multiple dispatch, so why have them?
abstract type LyapunovMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
abstract type SylvesterMatrixEquationsMaps{T} <: LinearMaps.LinearMap{T} end
const MatrixEquationsMaps{T} = Union{LyapunovMatrixEquationsMaps{T},SylvesterMatrixEquationsMaps{T}}

abstract type ContinuousOrDiscrete end
struct Continuous <: ContinuousOrDiscrete end
struct Discrete <: ContinuousOrDiscrete end

"""
    M = trmatop(n, m)

Define the transposition operator `M: X -> X'` for all `n x m` matrices.
"""
struct trmatop{Int} <: LinearMaps.LinearMap{Int}
   size::Dims{2}
   function trmatop(dims::Dims{2})
      all(≥(0), dims) || throw(ArgumentError("dims must be non-negative"))
      return new{Int}(dims)
   end
   function trmatop(m::Int, n::Int)
      (m ≥ (0) & n ≥ (0))  || throw(ArgumentError("dimensions must be non-negative"))
      return new{Int}((m, n))
   end
end
trmatop(n::Int) = trmatop(n, n)
Base.size(A::trmatop) = (prod(A.size), prod(A.size))
LinearAlgebra.issymmetric(A::trmatop) = A.size[1] == A.size[2]
LinearAlgebra.ishermitian(A::trmatop) = A.size[1] == A.size[2]
function mul!(y::AbstractVector, A::trmatop, x::AbstractVector)
   X = reshape(x, A.size...)
   LinearMaps.check_dim_mul(y, A, x)
   copyto!(y, adjoint(X))
   return y
end

"""
    M = trmatop(A)

Define the transposition operator `M: X -> X'` of all matrices of the size of `A`.
"""
trmatop(A) = trmatop(size(A))

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

LinearAlgebra.adjoint(L::LyapunovMap{<:Any,<:Any,CD}) where {CD}   = LyapunovMap(L.A', CD(), L.her)
LinearAlgebra.transpose(L::LyapunovMap{<:Any,<:Any,CD}) where {CD} = LyapunovMap(L.A', CD(), L.her)

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
function mul!(y::AbstractVector, L::LyapunovMap{T,TA,Discrete}, x::AbstractVector) where {T,TA}
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
function mul!(y::AbstractVector, L::LyapunovMap{T,TA,Continuous}, x::AbstractVector) where {T,TA}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   if L.her
      X = vec2triu(convert(AbstractVector{T1}, x), her=true)
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
function mulcsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix)
   require_one_based_indexing(y, A, X)
   # A*X + X*A'
   n = size(A, 1)
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
function muldsym!(y::AbstractVector, A::AbstractMatrix, X::AbstractMatrix)
   require_one_based_indexing(y, X)
   # A*X*A' - X
   n = size(A, 1)
   # t = triu(X)-diag(X)/2
   t = UpperTriangular(X) - Diagonal(X[diagind(X)] ./ 2)
   Y = similar(X, n, n)
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

LinearAlgebra.adjoint(L::GeneralizedLyapunovMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   GeneralizedLyapunovMap(L.A', L.E', CD(), L.her)
LinearAlgebra.transpose(L::GeneralizedLyapunovMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   GeneralizedLyapunovMap(L.A', L.E', CD(), L.her)

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
function mul!(y::AbstractVector, L::GeneralizedLyapunovMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where {T}
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
function mul!(y::AbstractVector, L::GeneralizedLyapunovMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where {T}
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
function mulcsym!(y::AbstractVector, A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix)
   require_one_based_indexing(y, A)
   # AXE' + EXA'
   n = size(A, 1)
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
               temp += A[i,l] * Y[l,j] + conj(Y[l,i] * A[j,l])
            end
            y[k] = temp
            k += 1
         end
      end
   end
   return y
end
function muldsym!(y::AbstractVector, A::AbstractMatrix, E::AbstractMatrix, X::AbstractMatrix)
   require_one_based_indexing(y)
   # AXA' - EXE'
   n = size(A, 1)
   # t = triu(X)-diag(X)/2
   t = UpperTriangular(X) - Diagonal(X[diagind(X)] ./ 2)
   Y = similar(X, n, n)
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
   InverseLyapunovMap(adj ? L.A : L.A', CD(), L.her)
LinearAlgebra.transpose(L::InverseLyapunovMap{<:Any,<:Any,adj,CD}) where {adj,CD} =
   InverseLyapunovMap(adj ? L.A : L.A', CD(), L.her)
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
function mul!(y::AbstractVector, L::InverseLyapunovMap{T,<:Any,adj,Discrete}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
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
function mul!(y::AbstractVector, L::InverseLyapunovMap{T,<:Any,adj,Continuous}, x::AbstractVector) where {T <: BlasFloat,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
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
function mul!(y::AbstractVector, L::InverseGeneralizedLyapunovMap{T,<:Any,<:Any,adj,Discrete}, x::AbstractVector) where {T,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
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
function mul!(y::AbstractVector, L::InverseGeneralizedLyapunovMap{T,<:Any,<:Any,adj,Continuous}, x::AbstractVector) where {T,adj}
   n = size(L.A, 1)
   T1 = promote_type(T, eltype(x))
   try
      if L.sf
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

LinearAlgebra.adjoint(L::SylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   SylvesterMap(L.A', L.B', CD())
LinearAlgebra.transpose(L::SylvesterMap{<:Any,<:Any,<:Any,CD}) where {CD} =
   SylvesterMap(L.A', L.B', CD())

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
function mul!(y::AbstractVector, L::SylvesterMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where T
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
function mul!(y::AbstractVector, L::SylvesterMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where T
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
function mul!(y::AbstractVector, L::InverseSylvesterMap{T,<:Any,<:Any,Discrete}, x::AbstractVector) where {T}
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
function mul!(y::AbstractVector, L::InverseSylvesterMap{T,<:Any,<:Any,Continuous}, x::AbstractVector) where T
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

LinearAlgebra.adjoint(L::GeneralizedSylvesterMap{T}) where T =
   GeneralizedSylvesterMap(L.A', L.B', L.C', L.D')
LinearAlgebra.transpose(L::GeneralizedSylvesterMap{T}) where T =
   GeneralizedSylvesterMap(L.A', L.B', L.C', L.D')

"""
    M = sylvop(A, B, C, D)

Define the generalized Sylvester operator `M: X -> AXB+CXD`, where `(A,C)` and `(B,D)` a pairs of square matrices.
"""
sylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   GeneralizedSylvesterMap(A, B, C, D)
Base.size(L::GeneralizedSylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function mul!(y::AbstractVector, L::GeneralizedSylvesterMap{T}, x::AbstractVector) where T
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
where (A,C) and (B,D) a pairs of square matrices.
"""
invsylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) =
   InverseGeneralizedSylvesterMap(A, B, C, D)
invsylvop(F::GeneralizedSchur, G::GeneralizedSchur) = invsylvop(F.S, G.S, F.T, G.T)
Base.size(L::InverseGeneralizedSylvesterMap) = (N = size(L.A, 1) * size(L.B, 1); return (N, N))
function mul!(y::AbstractVector, L::InverseGeneralizedSylvesterMap{T}, x::AbstractVector) where T
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
function mul!(y::AbstractVector, L::SylvesterSystemMap{T}, x::AbstractVector) where T
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
   @eval function mul!(y::AbstractVector, LT::$ttype{T,<:SylvesterSystemMap{T}}, x::AbstractVector) where T
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
function mul!(y::AbstractVector, L::InverseSylvesterSystemMap{T}, x::AbstractVector) where T
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
   @eval function mul!(y::AbstractVector, L::$ttype{T,<:InverseSylvesterSystemMap}, x::AbstractVector) where T
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
            X1, X2 = dsylvsyss!(L.lmap.A, L.lmap.B, E, L.lmap.C, L.lmap.D, F)
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
