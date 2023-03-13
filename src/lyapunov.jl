# Continuous Lyapunov equations
"""
    X = lyapc(A, C)

Compute `X`, the solution of the continuous Lyapunov equation

      AX + XA' + C = 0,

where `A` is a square real or complex matrix and `C` is a square matrix.
`A` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
The solution `X` is symmetric or hermitian if `C` is a symmetric or hermitian.

The following particular cases are also adressed:

    X = lyapc(α*I,C)  or  X = lyapc(α,C)

Solve the matrix equation `(α+α')X + C = 0`.

    x = lyapc(α,γ)

Solve the equation `(α+α')x + γ = 0`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> C = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> X = lyapc(A, C)
2×2 Array{Float64,2}:
  0.5  -0.5
 -0.5   0.25

julia> A*X + X*A' + C
2×2 Array{Float64,2}:
 -8.88178e-16   2.22045e-16
  2.22045e-16  -4.44089e-16
```
"""
function lyapc(A::AbstractMatrix, C::AbstractMatrix)
   """
   The Bartels-Steward Schur form based method is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(C) == n ||
      throw(DimensionMismatch("C must be a square matrix of dimension $n"))

   adj = isa(A,Adjoint)
   her = ishermitian(C)

   T2 = promote_type(eltype(A), eltype(C))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))

   # Reduce A to Schur form and transform C
   if adj
      AS, Q = schur(A.parent)
   else
      AS, Q = schur(A)
   end

   #X = Q'*C*Q
   if her
      X = utqu(C,Q)
      lyapcs!(AS, X, adj = adj)
      #X <- Q*X*Q'
      utqu!(X,Q')
      return X
   else
      X = Q' * C * Q
      adj ? (sylvcs!(AS, AS, X, adjA = true)) : (sylvcs!(AS, AS, X, adjB = true))
      return rmul!(Q * X * Q',-1)
   end
end
# (α+α')X  + C = 0
lyapc(A::UniformScaling, C::AbstractMatrix) = -C/(A+A')
lyapc(A::Union{Real,Complex}, C::AbstractMatrix) = real(A) == 0 ? throw(SingularException(1)) : -C/(A+A')
# (α+α')x + γ = 0
lyapc(A::Union{Real,Complex}, C::Union{Real,Complex})  = real(A) == 0 ? throw(SingularException(1)) : -C/(A+A')
"""
    X = lyapc(A, E, C)

Compute `X`, the solution of the generalized continuous Lyapunov equation

     AXE' + EXA' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a square matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
The solution `X` is symmetric or hermitian if `C` is a symmetric or hermitian.

The following particular cases are also adressed:

    X = lyapc(A,β*I,C)  or  X = lyapc(A,β,C)

Solve the matrix equation `AXβ' + βXA' + C = 0`.

    X = lyapc(α*I,E,C)  or  X = lyapc(α,E,C)

Solve the matrix equation `αXE' + EXα' + C = 0`.

    X = lyapc(α*I,β*I,C)  or  X = lyapc(α,β,C)

Solve the matrix equation `(αβ'+α'β)X + C = 0`.

    x = lyapc(α,β,γ)

Solve the equation `(αβ'+α'β)x + γ = 0`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> E = [ 1. 2.; 0. 1.]
2×2 Array{Float64,2}:
 1.0  2.0
 0.0  1.0

julia> C = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> X = lyapc(A, E, C)
2×2 Array{Float64,2}:
 -2.5   2.5
  2.5  -2.25

julia> A*X*E' + E*X*A' + C
2×2 Array{Float64,2}:
 -5.32907e-15  -2.66454e-15
 -4.44089e-15   0.0
```
"""
function lyapc(A::AbstractMatrix, E::AbstractMatrix, C::AbstractMatrix)
   """
   The extension of the Bartels-Steward method based on the generalized Schur form
   is employed.

   Reference:
   T. Penzl. Numerical solution of generalized Lyapunov equations.
   Adv. Comput. Math., 8:33–48, 1998.
   """
   T2 = promote_type(eltype(A), eltype(E), eltype(C))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   
   # generalized Schur form decomposition available only for complex data 
   T2 <: BlasFloat || T2 <: Complex || (return real(lyapc(complex(A),complex(E),complex(C))))
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(C) == n ||
      throw(DimensionMismatch("C must be a square matrix of dimension $n"))
   isequal(E,I) && size(E,1) == n && (return lyapc(A, C))
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a square matrix of dimension $n"))

   adjA = isa(A,Adjoint)
   adjE = isa(E,Adjoint)
   if adjA && !adjE
      A = copy(A)
      adjA = false
   elseif !adjA && adjE
      E = copy(E)
      adjE = false
   end

   adj = adjA & adjE
   her = ishermitian(C)

   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(E) == T2 || (adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   if adj
      as, es, z, q = schur(A.parent,E.parent)
   else
      as, es, q, z = schur(A,E)
   end
   if her
      #x = q'*C*q
      x = utqu(C,q)
      lyapcs!(as,es,x, adj = adj)
      #x = z*x*z'
      utqu!(x,z')
      return x
   else
      x = q' * C * q
      adj ? (gsylvs!(as,es,es,as,x,adjAC = true,DBSchur = true)) : (gsylvs!(as,es,es,as,x,adjBD = true,DBSchur = true))
      return rmul!(z*x*z', -1)
   end
end
# Aβ'X + XA'β + C = 0
lyapc(A::AbstractMatrix, E::Union{Real,Complex,UniformScaling}, C::AbstractMatrix) =
      isa(A,Adjoint) ? lyapc((A.parent*E')',C) : lyapc(A*E',C)
# αXE' + EXα' + C = 0
lyapc(A::Union{Real,Complex,UniformScaling}, E::AbstractMatrix, C::AbstractMatrix) =
      isa(E,Adjoint) ? lyapc((E.parent*A')',C) : lyapc(A'*E,C)
# (αβ'+α'β)X + C = 0
lyapc(A::UniformScaling, E::UniformScaling, C::AbstractMatrix) = -(A*E'+A'*E)\C
lyapc(A::Union{Real,Complex}, E::Union{Real,Complex}, C::AbstractMatrix) = real(A*E') == 0 ? throw(SingularException(1)) : -C/(A*E'+A'*E)
# (αβ'+α'β)X  + γ = 0
lyapc(A::Union{Real,Complex}, E::Union{Real,Complex}, C::Union{Real,Complex}) = real(A*E') == 0 ? throw(SingularException(1)) : -C/(A*E'+A'*E)
# Discrete Lyapunov equations
"""
    X = lyapd(A, C)

Compute `X`, the solution of the discrete Lyapunov equation

       AXA' - X + C = 0,

where `A` is a square real or complex matrix and `C` is a square matrix.
`A` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The solution `X` is symmetric or hermitian if `C` is a symmetric or hermitian.

The following particular cases are also adressed:

    X = lyapd(α*I,C)  or  X = lyapd(α,C)

Solve the matrix equation `(αα'-1)X + C = 0`.

    x = lyapd(α,γ)

Solve the equation `(αα'-1)x + γ = 0`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> C = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> X = lyapd(A, C)
2×2 Array{Float64,2}:
  0.2375  -0.2125
 -0.2125   0.1375

julia> A*X*A' - X + C
2×2 Array{Float64,2}:
 5.55112e-16  6.66134e-16
 2.22045e-16  4.44089e-16
```
"""
function lyapd(A::AbstractMatrix, C::AbstractMatrix)
   """
   The discrete analog of the Bartels-Steward method based on the Schur form
   is employed.

   Reference:
   G. Kitagawa. An Algorithm for solving the matrix equation X = F X F' + S,
   International Journal of Control, 25:745-753, 1977.
   """
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(C) == n ||
       throw(DimensionMismatch("C must be a square matrix of dimension $n"))

   adj = isa(A,Adjoint)
   her = ishermitian(C)

   T2 = promote_type(eltype(A), eltype(C))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))

   # Reduce A to Schur form and transform C
   if adj
      AS, Q = schur(A.parent)
   else
      AS, Q = schur(A)
   end
   #X = Q'*C*Q
   if her
      X = utqu(C,Q)
      lyapds!(AS, X, adj = adj)
      #X <- Q*X*Q'
      utqu!(X,Q')
      return X
   else
      #X = rmul!( Q' * C * Q, -1)
      X = Q' * C * Q
      adj ? (sylvds!(-AS, AS, X, adjA = true)) : (sylvds!(-AS, AS, X, adjB = true))
      return Q * X * Q'
   end
end
# (αα'-1)X + C = 0
lyapd(A::UniformScaling, C::AbstractMatrix) = (I-A'*A)\C
lyapd(A::Union{Real,Complex}, C::AbstractMatrix) = A*A' == 1 ? throw(SingularException(1)) : C/(1-A'*A)
# (αα'-1)x + γ = 0
lyapd(A::Union{Real,Complex}, C::Union{Real,Complex}) = A*A' == 1 ? throw(SingularException(1)) : C/(one(C)-A'*A)
"""
    X = lyapd(A, E, C)

Compute `X`, the solution of the generalized discrete Lyapunov equation

         AXA' - EXE' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a square matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The solution `X` is symmetric or hermitian if `C` is a symmetric or hermitian.

The following particular cases are also adressed:

    X = lyapd(A,β*I,C)  or  X = lyapd(A,β,C)

Solve the matrix equation `AXA' - βXβ' + C = 0`.

    X = lyapd(α*I,E,C)  or  X = lyapd(α,E,C)

Solve the matrix equation `αXα' - EXE' + C = 0`.

    X = lyapd(α*I,β*I,C)  or  X = lyapd(α,β,C)

Solve the matrix equation `(αα'-ββ')X + C = 0`.

    x = lyapd(α,β,γ)

Solve the equation `(αα'-ββ')x + γ = 0`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> E = [ 1. 2.; 0. -1.]
2×2 Array{Float64,2}:
 1.0   2.0
 0.0  -1.0

julia> C = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> X = lyapd(A, E, C)
2×2 Array{Float64,2}:
  1.775  -1.225
 -1.225   0.775

julia> A*X*A' - E*X*E' + C
2×2 Array{Float64,2}:
 -2.22045e-16  -4.44089e-16
 -1.33227e-15   1.11022e-15
```
"""
function lyapd(A::AbstractMatrix, E::AbstractMatrix, C::AbstractMatrix)
   """
   The extension of the Bartels-Steward method based on the generalized Schur form
   is employed.

   Reference:
   T. Penzl. Numerical solution of generalized Lyapunov equations.
   Adv. Comput. Math., 8:33–48, 1998.
   """
   T2 = promote_type(eltype(A), eltype(E), eltype(C))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   # use complex solver until real generalized Schur form will be available 
   T2 <: BlasFloat || T2 <: Complex || (return real(lyapd(complex(A),complex(E),complex(C))))   
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(C) == n ||
      throw(DimensionMismatch("C must be a square matrix of dimension $n"))
   isequal(E,I) && size(E,1) == n && (return lyapd(A, C))
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a square matrix of dimension $n"))

   adjA = isa(A,Adjoint)
   adjE = isa(E,Adjoint)
   if adjA && !adjE
      A = copy(A)
      adjA = false
   elseif !adjA && adjE
      E = copy(E)
      adjE = false
   end

   adj = adjA & adjE
   her = ishermitian(C)

   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(E) == T2 || (adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   if adj
      as, es, z, q = schur(A.parent,E.parent)
   else
      as, es, q, z = schur(A,E)
   end
   if her
      #x = q'*C*q
      x = utqu(C,q)
      lyapds!(as,es,x, adj = adj)
      #x = z*x*z'
      utqu!(x,z')
      return x
   else
      x = q' * C * q
      gsylvs!(as, as, -es, es, x, adjAC = adj, adjBD = !adj)
      return rmul!(z*x*z', -1)
   end
end
# AXA' - Xββ' + C = 0
lyapd(A::AbstractMatrix, E::Union{Real,Complex,UniformScaling}, C::AbstractMatrix) =
      isa(A,Adjoint) ? lyapd((A.parent/E)',C/(E*E')) : lyapd(A/E,C/(E*E'))
# αXα' - EXE' + C = 0
lyapd(A::Union{Real,Complex,UniformScaling}, E::AbstractMatrix, C::AbstractMatrix) =
      isa(E,Adjoint) ? lyapd((E.parent/A)',C/(-A*A')) : lyapd(E/A,C/(-A*A'))
# (α'α-β'β)X + C = 0
lyapd(A::UniformScaling, E::UniformScaling, C::AbstractMatrix) = C/(E'*E-A'*A)
lyapd(A::Union{Real,Complex}, E::Union{Real,Complex}, C::AbstractMatrix) = A*A'== E*E' ? throw(SingularException(1)) : C/(E'*E-A'*A)
# (α'α-β'β)X + γ  = 0
lyapd(A::Union{Real,Complex}, E::Union{Real,Complex}, C::Union{Real,Complex}) = A*A'== E*E' ? throw(SingularException(1)) : C/(E'*E-A'*A)
"""
    lyapcs!(A,C;adj = false)

Solve the continuous Lyapunov matrix equation

                op(A)X + Xop(A)' + C = 0,

where `op(A) = A` if `adj = false` and `op(A) = A'` if `adj = true`.
`A` is a square real matrix in a real Schur form, or a square complex matrix in a
complex Schur form and `C` is a symmetric or hermitian matrix.
`A` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
`C` contains on output the solution `X`.
"""
function lyapcs!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:Real}
#function lyapcs!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:BlasReal}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && issymmetric(C)) ||
      throw(DimensionMismatch("C must be a $n x $n symmetric matrix"))

   ONE = one(T1)
   ZERO = zero(T1)

   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   #W = Array{T1,2}(I,2,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if adj
      # """
      # The (K,L)th block of X is determined starting from
      # upper-left corner column by column by

      # A(K,K)'*X(K,L) + X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      # where
      #          K-1                    L-1
      # R(K,L) = SUM [A(I,K)'*X(I,L)] + SUM [X(K,J)*A(J,L)].
      #          I=1                    J=1
      # """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          i = j
          for kk = ll:p
              dk = ba[kk]
              k = i:i+dk-1
              y = view(C,k,l)
              if kk > ll
                 ia = j:i-1
                 # y += A[ia,k]'*C[ia,l]
                 mul!(y,transpose(view(A,ia,k)),view(C,ia,l),ONE,ONE)
               end
               if i == j
                  lyapc2!(adj,y,dk,view(A,k,k),Xw,Yw)
               else
                  lyapcsylv2!(adj,y,dk,dl,view(A,k,k),view(A,l,l),Xw,Yw)
                  transpose!(view(C,l,k),y)
               end
               i += dk
          end
          j += dl
          if j <= n
             for jr = j:n
                for ir = jr:n
                    for lll = l
                        C[ir,jr] += C[ir,lll]*A[lll,jr] + A[lll,ir]*C[jr,lll]
                    end
                    C[jr,ir] = C[ir,jr]
                end
             end
          end
       end
   else
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-right corner column by column by

      # A(K,K)*X(K,L) + X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      # where
      #           N                     N
      # R(K,L) = SUM [A(K,I)*X(I,L)] + SUM [X(K,J)*A(L,J)'].
      #         I=K+1                 J=L+1
      # """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = j
          for kk = ll:-1:1
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              y = view(C,k,l)
              if kk < ll
                 ia = i+1:j
                 #  y += A[k,ia]*C[ia,l]
                 mul!(y,view(A,k,ia),view(C,ia,l),ONE,ONE)
              end
              if i == j
                 lyapc2!(adj,y,dk,view(A,k,k),Xw,Yw)
              else
                 lyapcsylv2!(adj,y,dk,dl,view(A,k,k),view(A,l,l),Xw,Yw)
                 transpose!(view(C,l,k),y)
              end
              i -= dk
           end
           j -= dl
           if j >= 0
              for jr = 1:j
                 for ir = 1:jr
                     for lll = l
                         C[ir,jr] += C[ir,lll]*A[jr,lll] + A[ir,lll]*C[lll,jr]
                      end
                      C[jr,ir] = C[ir,jr]
                 end
              end
           end
       end
   end
end
function lyapc2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
#function lyapc2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
# speed and reduced allocation oriented implementation of a solver for 1x1 or 2x2 continuous Lyapunov equations
#      A'*X + X*A = -C if adj = true  -> R = kron(I,A')+kron(A',I) = (kron(I,A)+kron(A,I))'
#      A*X + X*A' = -C if adj = false -> R = kron(I,A)+kron(A,I)
   if na == 1
      rmul!(C,inv(-2*A[1,1]))
      any(!isfinite, C) && throw("ME:SingularException: A has eigenvalue(s) ≈ 0")
      return C
   end
   ZERO = zero(T)
   i1 = 1:3
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   @inbounds  Y[1] = -C[1,1]/2
   @inbounds  Y[2] = -C[2,1]
   @inbounds  Y[3] = -C[2,2]/2
   if adj
      # @inbounds R = [ A[1,1]    A[2,1]         0;
      #                 A[1,2]    A[1,1]+A[2,2]  A[2,1];
      #                 0         A[1,2]         A[2,2] ]
      @inbounds  R[1,1] = A[1,1]
      @inbounds  R[1,2] = A[2,1]
      @inbounds  R[1,3] = ZERO
      @inbounds  R[2,1] = A[1,2]
      @inbounds  R[2,2] = A[1,1]+A[2,2]
      @inbounds  R[2,3] = A[2,1]
      @inbounds  R[3,1] = ZERO
      @inbounds  R[3,2] = A[1,2]
      @inbounds  R[3,3] = A[2,2]
   else
      # @inbounds R = [ A[1,1]    A[1,2]         0;
      #                 A[2,1]    A[1,1]+A[2,2]  A[1,2];
      #                 0         A[2,1]         A[2,2] ]
      @inbounds  R[1,1] = A[1,1]
      @inbounds  R[1,2] = A[1,2]
      @inbounds  R[1,3] = ZERO
      @inbounds  R[2,1] = A[2,1]
      @inbounds  R[2,2] = A[1,1]+A[2,2]
      @inbounds  R[2,3] = A[1,2]
      @inbounds  R[3,1] = ZERO
      @inbounds  R[3,2] = A[2,1]
      @inbounds  R[3,3] = A[2,2]
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalues α and β such that α+β ≈ 0")
   @inbounds C[1,1] = Y[1]
   @inbounds C[1,2] = Y[2]
   @inbounds C[2,1] = Y[2]
   @inbounds C[2,2] = Y[3]
   return C
end
function lyapcsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
#function lyapcsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
# speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations
# encountered in solving continuous Lyapunov equations:
#      A'*X + X*B = -C if adj = true  -> R = kron(I,A')+kron(B',I) = (kron(I,A)+kron(B,I))'
#      A*X + X*B' = -C if adj = false -> R = kron(I,A)+kron(B,I)
   if na == 1 && nb == 1
      temp = A[1,1] + B[1,1]
      rmul!(C,inv(-temp))
      any(!isfinite, C) && throw("ME:SingularException: A has eigenvalues α and β such that α+β ≈ 0")
      return C
   end
   ZERO = zero(T)
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if adj
      if na == 1
         # @inbounds R = [ A[1,1]+B[1,1]    B[2,1];
         #                      B[1,2]      A[1,1]+B[2,2]]
         @inbounds  R[1,1] = A[1,1]+B[1,1]
         @inbounds  R[1,2] = B[2,1]
         @inbounds  R[2,1] = B[1,2]
         @inbounds  R[2,2] = A[1,1]+B[2,2]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # @inbounds R = [ A[1,1]+B[1,1]       A[2,1];
            #                  A[1,2]          A[2,2]+B[1,1] ]
            @inbounds  R[1,1] = A[1,1]+B[1,1]
            @inbounds  R[1,2] = A[2,1]
            @inbounds  R[2,1] = A[1,2]
            @inbounds  R[2,2] = A[2,2]+B[1,1]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # @inbounds R = [ A[1,1]+B[1,1]       A[2,1]       B[2,1]           0;
            #                 A[1,2]        A[2,2]+B[1,1]      0             B[2,1];
            #                 B[1,2]              0       A[1,1]+B[2,2]     A[2,1];
            #                   0               B[1,2]       A[1,2]       A[2,2]+B[2,2]]
            @inbounds  R[1,1] = A[1,1]+B[1,1]
            @inbounds  R[1,2] = A[2,1]
            @inbounds  R[1,3] = B[2,1]
            @inbounds  R[1,4] = ZERO
            @inbounds  R[2,1] = A[1,2]
            @inbounds  R[2,2] = A[2,2]+B[1,1]
            @inbounds  R[2,3] = ZERO
            @inbounds  R[2,4] = B[2,1]
            @inbounds  R[3,1] = B[1,2]
            @inbounds  R[3,2] = ZERO
            @inbounds  R[3,3] = A[1,1]+B[2,2]
            @inbounds  R[3,4] = A[2,1]
            @inbounds  R[4,1] = ZERO
            @inbounds  R[4,2] = B[1,2]
            @inbounds  R[4,3] = A[1,2]
            @inbounds  R[4,4] = A[2,2]+B[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   else
      if na == 1
         # @inbounds R = [ A[1,1]+B[1,1]       B[1,2];
         #                    B[2,1]       A[1,1] + B[2,2]]
         @inbounds  R[1,1] = A[1,1]+B[1,1]
         @inbounds  R[1,2] = B[1,2]
         @inbounds  R[2,1] = B[2,1]
         @inbounds  R[2,2] = A[1,1]+B[2,2]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # @inbounds R = [ A[1,1]+B[1,1]       A[1,2];
            #                 A[2,1]        A[2,2]+B[1,1]]
            @inbounds  R[1,1] = A[1,1]+B[1,1]
            @inbounds  R[1,2] = A[1,2]
            @inbounds  R[2,1] = A[2,1]
            @inbounds  R[2,2] = A[2,2]+B[1,1]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # @inbounds R = [ A[1,1]+B[1,1]       A[1,2]       B[1,2]         0;
            #                 A[2,1]      A[2,2]+B[1,1]       0         B[1,2];
            #                  B[2,1]            0      A[1,1]+B[2,2]     A[1,2];
            #                  0             B[2,1]        A[2,1]    A[2,2]+B[2,2]]
            @inbounds  R[1,1] = A[1,1]+B[1,1]
            @inbounds  R[1,2] = A[1,2]
            @inbounds  R[1,3] = B[1,2]
            @inbounds  R[1,4] = ZERO
            @inbounds  R[2,1] = A[2,1]
            @inbounds  R[2,2] = A[2,2]+B[1,1]
            @inbounds  R[2,3] = ZERO
            @inbounds  R[2,4] = B[1,2]
            @inbounds  R[3,1] = B[2,1]
            @inbounds  R[3,2] = ZERO
            @inbounds  R[3,3] = A[1,1]+B[2,2]
            @inbounds  R[3,4] = A[1,2]
            @inbounds  R[4,1] = ZERO
            @inbounds  R[4,2] = B[2,1]
            @inbounds  R[4,3] = A[2,1]
            @inbounds  R[4,4] = A[2,2]+B[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalues α and β such that α+β ≈ 0")
   C[:,:] = Y
   return C
end
function lyapcs!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:Complex}
#function lyapcs!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:BlasComplex}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && ishermitian(C)) ||
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))

   if adj
      # """
      # The (K,L)th element of X is determined starting from
      # upper-left corner column by column by

      # A(K,K)'*X(K,L) + X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      # where
      #          K-1                    L-1
      # R(K,L) = SUM [A(I,K)'*X(I,L)] + SUM [X(K,J)*A(J,L)].
      #          I=1                    J=1
      # """
      for l = 1:n
          for k = l:n
              y = C[k,l]
              for ia = l:k-1
                  y += A[ia,k]'*C[ia,l]
              end
              C[k,l] = -y/(A[k,k]'+A[l,l])
              isfinite(C[k,l]) || throw("ME:SingularException: A has eigenvalues α and β such that α+β ≈ 0")
              k == l ? (C[k,k] = real(C[k,k])) : (C[l,k] = C[k,l]')
           end
           for jr = l+1:n
               for ir = jr:n
                   C[ir,jr] += C[ir,l]*A[l,jr] + (A[l,ir]*C[jr,l])'
                   if jr != ir
                      C[jr,ir] = C[ir,jr]'
                   end
               end
           end
      end
   else
      # """
      # The (K,L)th element of X is determined starting from
      # bottom-right corner column by column by

      # A(K,K)*X(K,L) + X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      # where
      #           N                     N
      # R(K,L) = SUM [A(K,I)*X(I,L)] + SUM [X(K,J)*A(L,J)'].
      #         I=K+1                 J=L+1
      # """
      for l = n:-1:1
          for k = l:-1:1
              y = C[k,l]
              for ia = k+1:l
                 y += A[k,ia]*C[ia,l]
              end
              C[k,l] = -y/(A[k,k]+A[l,l]')
              isfinite(C[k,l]) || throw("ME:SingularException: A has eigenvalues α and β such that α+β ≈ 0")
              k == l ? (C[k,k] = real(C[k,k])) : (C[l,k] = C[k,l]')
           end
           for jr = 1:l-1
               for ir = 1:jr
                  C[ir,jr] += C[ir,l]*A[jr,l]' + A[ir,l]*C[l,jr]
                  if ir != jr
                     C[jr,ir] = C[ir,jr]'
                  end
               end
           end
       end
   end
end
"""
    lyapcs!(A, E, C; adj = false)

Solve the generalized continuous Lyapunov matrix equation

                op(A)Xop(E)' + op(E)Xop(A)' + C = 0,

where `op(A) = A` and `op(E) = E` if `adj = false` and `op(A) = A'` and
`op(E) = E'` if `adj = true`. The pair `(A,E)` is in a generalized real or
complex Schur form and `C` is a symmetric or hermitian matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapcs!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:Real}
# function lyapcs!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:BlasReal}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && issymmetric(C)) ||
      throw(DimensionMismatch("C must be a $n x $n symmetric matrix"))
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (lyapcs!(A, C, adj = adj); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   ONE = one(T1)
   ZERO = zero(T1)

   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   G = Matrix{T1}(undef,2,2)
   W = Array{T1,2}(undef,n,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if adj
      # """
      # The (K,L)th block of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*E(L,L) + E(K,K)'*X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      # where
      #           K           L-1
      # R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*E(J,L)]} +
      #          I=1          J=1

      #           K           L-1
      #          SUM {E(I,K)'*SUM [X(I,J)*A(J,L)]} +
      #          I=1          J=1

      #          K-1
      #         {SUM [A(I,K)'*X(I,L)]}*E(L,L) +
      #          I=1

      #          K-1
      #         {SUM [E(I,K)'*X(I,L)]}*A(L,L).
      #          I=1
      # """
      i = 1
      for kk = 1:p
          dk = ba[kk]
          dkk = 1:dk
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
             dl = ba[ll]
             j1 = j+dl-1
             l = j:j1
             y = view(G,1:dk,1:dl)
             Ckl = view(C,k,l)
             copyto!(y,Ckl)
             if kk > 1
                 #   C[l,k] = C[l,ir]*A[ir,k]
                 #   W[l,dkk] = C[l,ir]*E[ir,k]
                 mul!(view(C,l,k),view(C,l,ir),view(A,ir,k))
                 mul!(view(W,l,dkk),view(C,l,ir),view(E,ir,k))
                 ic = 1:j1
                 #   y += C[ic,k]'*E[ic,l] + W[ic,dkk]'*A[ic,l]
                 mul!(y,transpose(view(C,ic,k)),view(E,ic,l),ONE,ONE)
                 mul!(y,transpose(view(W,ic,dkk)),view(A,ic,l),ONE,ONE)
             end
             if i == j
                lyapc2!(adj,y,dk,view(A,k,k),view(E,k,k),Xw,Yw)
                copyto!(Ckl,y)
               else
                lyapcsylv2!(adj,y,dk,dl,view(A,k,k),view(E,k,k),view(A,l,l),view(E,l,l),Xw,Yw)
                copyto!(Ckl,y)
             end
             j += dl
             if j <= i
                #  C[l,k] += C[k,l]'*A[k,k]
                #  W[l,dkk] += C[k,l]'*E[k,k]
                mul!(view(C,l,k),transpose(Ckl),view(A,k,k),ONE,ONE)
                mul!(view(W,l,dkk),transpose(Ckl),view(E,k,k),ONE,ONE)
             end
          end
          if kk > 1
             ir = 1:i-1
             #C[ir,k] = C[k,ir]'
             transpose!(view(C,ir,k),view(C,k,ir))
          end
          i += dk
      end
   else
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-right corner column by column by

      # A(K,K)*X(K,L)*E(L,L)' + E(K,K)*X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      # where

      #           N            N
      # R(K,L) = SUM {A(K,I)* SUM [X(I,J)*E(L,J)']} +
      #          I=K         J=L+1

      #           N            N
      #          SUM {E(K,I)* SUM [X(I,J)*A(L,J)']} +
      #          I=K         J=L+1

      #             N
      #          { SUM [A(K,J)*X(J,L)]}*E(L,L)' +
      #           J=K+1

      #             N
      #          { SUM [E(K,J)*X(J,L)]}*A(L,L)'
      #           J=K+1
      # """
      j = n
      for ll = p:-1:1
        dl = ba[ll]
        l = j-dl+1:j
        dll = 1:dl
        i = n
        ir = j+1:n
        for kk = p:-1:ll
            dk = ba[kk]
            i1 = i-dk+1
            k = i1:i
            Clk = view(C,l,k)
            y = view(G,1:dl,1:dk)
            copyto!(y,Clk)
            if ll < p
               # C[k,l] = C[k,ir]*A[l,ir]'
               # W[k,dll] = C[k,ir]*E[l,ir]'
               mul!(view(C,k,l),view(C,k,ir),transpose(view(A,l,ir)))
               mul!(view(W,k,dll),view(C,k,ir),transpose(view(E,l,ir)))
               ic = i1:n
               # y += (E[k,ic]*C[ic,l] + A[k,ic]*W[ic,dll])'
               mul!(y,transpose(view(C,ic,l)),transpose(view(E,k,ic)),ONE,ONE)
               mul!(y,transpose(view(W,ic,dll)),transpose(view(A,k,ic)),ONE,ONE)
            end
            if i == j
               lyapc2!(adj,y,dk,view(A,k,k),view(E,k,k),Xw,Yw)
               copyto!(Clk,y)
            else
               lyapcsylv2!(adj,y,dl,dk,view(A,l,l),view(E,l,l),view(A,k,k),view(E,k,k),Xw,Yw)
               copyto!(Clk,y)
            end
            i -= dk
            if i >= j
               # C[k,l] += (A[l,l]*C[l,k])'
               # W[k,dll] += (E[l,l]*C[l,k])'
               mul!(view(C,k,l),transpose(Clk),transpose(view(A,l,l)),ONE,ONE)
               mul!(view(W,k,dll),transpose(Clk),transpose(view(E,l,l)),ONE,ONE)
            else
               break
            end
        end
        if ll < p
           ir = j+1:n
           # C[ir,l] = C[l,ir]'
           transpose!(view(C,ir,l),view(C,l,ir))
         end
        j -= dl
      end
   end
end
@inline function lyapc2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
#  @inline function lyapc2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
      # speed and reduced allocation oriented implementation of a solver for 1x1 or 2x2 generalized continuous Lyapunov equations
# LAPACK generated diagonal structure of E is exploited when possible
#      A'*X*E + E'*X*A = -C if adj = true  -> R = kron(E',A')+kron(A',E') = (kron(E,A)+kron(A,E))'
#      A*X*E' + E*X*A' = -C if adj = false -> R = kron(E,A)+kron(A,E)
   if na == 1
      temp = E[1,1]*A[1,1]
      rmul!(C,inv(-2*temp))
      any(!isfinite, C) &&  throw("ME:SingularException: A-λE has zero or infinite eigenvalue(s)")
      return C
   end
   ZERO = zero(T)
   i1 = 1:3
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   @inbounds  Y[1] = -C[1,1]/2
   @inbounds  Y[2] = -C[2,1]
   @inbounds  Y[3] = -C[2,2]/2
   if adj
      # Rt =
      #  [        a11*e11,                   a21*e11,         0
      #  a11*e12 + a12*e11, a11*e22 + a21*e12 + a22*e11,   a21*e22
      #          a12*e12,       a12*e22 + a22*e12, a22*e22]
      # iszero(E[1,2]) ?
      #    (@inbounds R = [  A[1,1]*E[1,1]             A[2,1]*E[1,1]               0 ;
      #                      A[1,2]*E[1,1]    A[1,1]*E[2,2]+A[2,2]*E[1,1]    A[2,1]*E[2,2];
      #                           0                   A[1,2]*E[2,2]          A[2,2]*E[2,2]]) :
      #    (@inbounds R = [  A[1,1]*E[1,1]                    A[2,1]*E[1,1]                              0 ;
      #          A[1,1]*E[1,2]+A[1,2]*E[1,1]    A[1,1]*E[2,2]+A[2,1]*E[1,2]+A[2,2]*E[1,1]    A[2,1]*E[2,2];
      #          A[1,2]*E[1,2]                    A[1,2]*E[2,2]+A[2,2]*E[1,2]                A[2,2]*E[2,2]] )
      @inbounds  R[1,1] = A[1,1]*E[1,1]
      @inbounds  R[1,2] = A[2,1]*E[1,1]
      @inbounds  R[1,3] = ZERO
      @inbounds  R[2,1] = A[1,1]*E[1,2]+A[1,2]*E[1,1]
      @inbounds  R[2,2] = A[1,1]*E[2,2]+A[2,1]*E[1,2]+A[2,2]*E[1,1]
      @inbounds  R[2,3] = A[2,1]*E[2,2]
      @inbounds  R[3,1] = A[1,2]*E[1,2]
      @inbounds  R[3,2] = A[1,2]*E[2,2]+A[2,2]*E[1,2]
      @inbounds  R[3,3] = A[2,2]*E[2,2]
   else
      # R =
      # [ a11*e11,       a11*e12 + a12*e11,         a12*e12
      # a21*e11, a11*e22 + a21*e12 + a22*e11, a12*e22 + a22*e12
      #       0,                   a21*e22,         a22*e22]
      # iszero(E[1,2]) ?
      #    (@inbounds  R = [  A[1,1]*E[1,1]            A[1,2]*E[1,1]                    0;
      #                       A[2,1]*E[1,1]    A[1,1]*E[2,2]+A[2,2]*E[1,1]     A[1,2]*E[2,2];
      #                             0                  A[2,1]*E[2,2]           A[2,2]*E[2,2]] ) :
      #    (@inbounds  R = [  A[1,1]*E[1,1]    A[1,1]*E[1,2]+A[1,2]*E[1,1]             A[1,2]*E[1,2];
      #           A[2,1]*E[1,1]    A[1,1]*E[2,2]+A[2,1]*E[1,2]+A[2,2]*E[1,1]     A[1,2]*E[2,2]+A[2,2]*E[1,2];
      #                 0                   A[2,1]*E[2,2]                              A[2,2]*E[2,2]] )
      @inbounds  R[1,1] = A[1,1]*E[1,1]
      @inbounds  R[1,2] = A[1,1]*E[1,2]+A[1,2]*E[1,1]
      @inbounds  R[1,3] = A[1,2]*E[1,2]
      @inbounds  R[2,1] = A[2,1]*E[1,1]
      @inbounds  R[2,2] = A[1,1]*E[2,2]+A[2,1]*E[1,2]+A[2,2]*E[1,1]
      @inbounds  R[2,3] = A[1,2]*E[2,2]+A[2,2]*E[1,2]
      @inbounds  R[3,1] = ZERO
      @inbounds  R[3,2] = A[2,1]*E[2,2]
      @inbounds  R[3,3] = A[2,2]*E[2,2]
   end
   luslv!(R,Y) && throw("ME:SingularException: A-λE has eigenvalues α and β such that α+β ≈ 0")
   @inbounds C[1,1] = Y[1]
   @inbounds C[1,2] = Y[2]
   @inbounds C[2,1] = Y[2]
   @inbounds C[2,2] = Y[3]
   return C
end
@inline function lyapcsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},F::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
#  @inline function lyapcsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},F::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
# speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations
# encountered in solving generalized continuous Lyapunov equations:
#      A'*X*F + E'*X*B = -C if adj = true  -> R = kron(F',A')+kron(B',E') = (kron(F,A)+kron(B,E))'
#      A*X*F' + E*X*B' = -C if adj = false -> R = kron(F,A)+kron(B,E)
   if na == 1 && nb == 1
      temp = A[1,1]*F[1,1] + E[1,1]*B[1,1]
      rmul!(C,inv(-temp))
      any(!isfinite, C) && throw("ME:SingularException: A-λE has eigenvalues α and β such that α+β ≈ 0")
      return C
   end
   ZERO = zero(T)
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if adj
      if na == 1
         # R12t =
         # [ a11*f11+b11*e11           b21*e11]
         # [ a11*f12+b12*e11 a11*f22+b22*e11]
         # @inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1]           B[2,1]*E[1,1];
         #                 A[1,1]*F[1,2]+B[1,2]*E[1,1] A[1,1]*F[2,2]+B[2,2]*E[1,1]]
         @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
         @inbounds  R[1,2] = B[2,1]*E[1,1]
         @inbounds  R[2,1] = A[1,1]*F[1,2]+B[1,2]*E[1,1]
         @inbounds  R[2,2] = A[1,1]*F[2,2]+B[2,2]*E[1,1]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # R21t =
            # [ a11*f11+b11*e11           a21*f11]
            # [ a12*f11+b11*e12 a22*f11+b11*e22]
            # @inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1]           A[2,1]*F[1,1];
            #                 A[1,2]*F[1,1]+B[1,1]*E[1,2] A[2,2]*F[1,1]+B[1,1]*E[2,2] ]
            @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
            @inbounds  R[1,2] = A[2,1]*F[1,1]
            @inbounds  R[2,1] = A[1,2]*F[1,1]+B[1,1]*E[1,2]
            @inbounds  R[2,2] = A[2,2]*F[1,1]+B[1,1]*E[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # Rt =
            # [ a11*f11+b11*e11           a21*f11           b21*e11                 0]
            # [ a12*f11+b11*e12 a22*f11+b11*e22           b21*e12           b21*e22]
            # [ a11*f12+b12*e11           a21*f12 a11*f22+b22*e11           a21*f22]
            # [ a12*f12+b12*e12 a22*f12+b12*e22 a12*f22+b22*e12 a22*f22+b22*e22]
            # (iszero(E[1,2]) && iszero(F[1,2])) ?
            #  (@inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1]           A[2,1]*F[1,1]           B[2,1]*E[1,1]                 0;
            #                A[1,2]*F[1,1]  A[2,2]*F[1,1]+B[1,1]*E[2,2]           0           B[2,1]*E[2,2];
            #                B[1,2]*E[1,1]        0  A[1,1]*F[2,2]+B[2,2]*E[1,1]           A[2,1]*F[2,2];
            #                0  B[1,2]*E[2,2] A[1,2]*F[2,2]+B[2,2]*E[1,2] A[2,2]*F[2,2]+B[2,2]*E[2,2]]) :
            #  (@inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1]           A[2,1]*F[1,1]           B[2,1]*E[1,1]                 0;
            #                A[1,2]*F[1,1]+B[1,1]*E[1,2] A[2,2]*F[1,1]+B[1,1]*E[2,2]           B[2,1]*E[1,2]           B[2,1]*E[2,2];
            #                A[1,1]*F[1,2]+B[1,2]*E[1,1]           A[2,1]*F[1,2] A[1,1]*F[2,2]+B[2,2]*E[1,1]           A[2,1]*F[2,2];
            #                A[1,2]*F[1,2]+B[1,2]*E[1,2] A[2,2]*F[1,2]+B[1,2]*E[2,2] A[1,2]*F[2,2]+B[2,2]*E[1,2] A[2,2]*F[2,2]+B[2,2]*E[2,2]])
            @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
            @inbounds  R[1,2] = A[2,1]*F[1,1]
            @inbounds  R[1,3] = B[2,1]*E[1,1]
            @inbounds  R[1,4] = ZERO
            @inbounds  R[2,1] = A[1,2]*F[1,1]+B[1,1]*E[1,2]
            @inbounds  R[2,2] = A[2,2]*F[1,1]+B[1,1]*E[2,2]
            @inbounds  R[2,3] = B[2,1]*E[1,2]
            @inbounds  R[2,4] = B[2,1]*E[2,2]
            @inbounds  R[3,1] = A[1,1]*F[1,2]+B[1,2]*E[1,1]
            @inbounds  R[3,2] = A[2,1]*F[1,2]
            @inbounds  R[3,3] = A[1,1]*F[2,2]+B[2,2]*E[1,1]
            @inbounds  R[3,4] = A[2,1]*F[2,2]
            @inbounds  R[4,1] = A[1,2]*F[1,2]+B[1,2]*E[1,2]
            @inbounds  R[4,2] = A[2,2]*F[1,2]+B[1,2]*E[2,2]
            @inbounds  R[4,3] = A[1,2]*F[2,2]+B[2,2]*E[1,2]
            @inbounds  R[4,4] = A[2,2]*F[2,2]+B[2,2]*E[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   else
      if na == 1
         # R12 =
         # [ a11*f11+b11*e11 a11*f12+b12*e11]
         # [           b21*e11 a11*f22+b22*e11]
         # @inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1] A[1,1]*F[1,2]+B[1,2]*E[1,1];
         #                    B[2,1]*E[1,1] A[1,1]*F[2,2]+B[2,2]*E[1,1]]
         @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
         @inbounds  R[1,2] = A[1,1]*F[1,2]+B[1,2]*E[1,1]
         @inbounds  R[2,1] = B[2,1]*E[1,1]
         @inbounds  R[2,2] = A[1,1]*F[2,2]+B[2,2]*E[1,1]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # R21 =
            # [ a11*f11+b11*e11 a12*f11+b11*e12]
            # [           a21*f11 a22*f11+b11*e22]
            # @inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1] A[1,2]*F[1,1]+B[1,1]*E[1,2];
            #                  A[2,1]*F[1,1] A[2,2]*F[1,1]+B[1,1]*E[2,2]]
            @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
            @inbounds  R[1,2] = A[1,2]*F[1,1]+B[1,1]*E[1,2]
            @inbounds  R[2,1] = A[2,1]*F[1,1]
            @inbounds  R[2,2] = A[2,2]*F[1,1]+B[1,1]*E[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # R =
            # [ a11*f11+b11*e11 a12*f11+b11*e12 a11*f12+b12*e11 a12*f12+b12*e12]
            # [           a21*f11 a22*f11+b11*e22           a21*f12 a22*f12+b12*e22]
            # [           b21*e11           b21*e12 a11*f22+b22*e11 a12*f22+b22*e12]
            # [                 0           b21*e22           a21*f22 a22*f22+b22*e22]
            # (iszero(E[1,2]) && iszero(F[1,2])) ?
            # (@inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1] A[1,2]*F[1,1] B[1,2]*E[1,1] 0;
            #                A[2,1]*F[1,1] A[2,2]*F[1,1]+B[1,1]*E[2,2]           0   B[1,2]*E[2,2];
            #                B[2,1]*E[1,1]           0   A[1,1]*F[2,2]+B[2,2]*E[1,1] A[1,2]*F[2,2];
            #                0           B[2,1]*E[2,2]           A[2,1]*F[2,2] A[2,2]*F[2,2]+B[2,2]*E[2,2]]) :
            # (@inbounds R = [ A[1,1]*F[1,1]+B[1,1]*E[1,1] A[1,2]*F[1,1]+B[1,1]*E[1,2] A[1,1]*F[1,2]+B[1,2]*E[1,1] A[1,2]*F[1,2]+B[1,2]*E[1,2];
            #                A[2,1]*F[1,1] A[2,2]*F[1,1]+B[1,1]*E[2,2]           A[2,1]*F[1,2] A[2,2]*F[1,2]+B[1,2]*E[2,2];
            #                B[2,1]*E[1,1]           B[2,1]*E[1,2] A[1,1]*F[2,2]+B[2,2]*E[1,1] A[1,2]*F[2,2]+B[2,2]*E[1,2];
            #                0           B[2,1]*E[2,2]           A[2,1]*F[2,2] A[2,2]*F[2,2]+B[2,2]*E[2,2]])
            @inbounds  R[1,1] = A[1,1]*F[1,1]+B[1,1]*E[1,1]
            @inbounds  R[1,2] = A[1,2]*F[1,1]+B[1,1]*E[1,2]
            @inbounds  R[1,3] = A[1,1]*F[1,2]+B[1,2]*E[1,1]
            @inbounds  R[1,4] = A[1,2]*F[1,2]+B[1,2]*E[1,2]
            @inbounds  R[2,1] = A[2,1]*F[1,1]
            @inbounds  R[2,2] = A[2,2]*F[1,1]+B[1,1]*E[2,2]
            @inbounds  R[2,3] = A[2,1]*F[1,2]
            @inbounds  R[2,4] = A[2,2]*F[1,2]+B[1,2]*E[2,2]
            @inbounds  R[3,1] = B[2,1]*E[1,1]
            @inbounds  R[3,2] = B[2,1]*E[1,2]
            @inbounds  R[3,3] = A[1,1]*F[2,2]+B[2,2]*E[1,1]
            @inbounds  R[3,4] = A[1,2]*F[2,2]+B[2,2]*E[1,2]
            @inbounds  R[4,1] = ZERO
            @inbounds  R[4,2] = B[2,1]*E[2,2]
            @inbounds  R[4,3] = A[2,1]*F[2,2]
            @inbounds  R[4,4] = A[2,2]*F[2,2]+B[2,2]*E[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   end
   luslv!(R,Y) && throw("ME:SingularException: A-λE has eigenvalues α and β such that α+β ≈ 0")
   C[:,:] = Y
   return C
end
function lyapcs!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:Complex}
#function lyapcs!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:BlasComplex}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && ishermitian(C)) ||
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (lyapcs!(A, C, adj = adj); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   W = Array{T1,1}(undef,n)
   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               W[l] = C[l,1]*E[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
                  W[l] += C[l,ir]*E[ir,k]
               end
               for ic = 1:l
                   y += C[ic,k]'*E[ic,l] + W[ic]'*A[ic,l]
                end
            end
            C[k,l] = -y/(A[k,k]'*E[l,l]+E[k,k]'*A[l,l])
            isfinite(C[k,l]) || throw("ME:SingularException: A-λE has eigenvalues α and β such that α+β ≈ 0")
            k == l && (C[k,k] = real(C[k,k]))
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
               W[l] += C[k,l]'*E[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               W[k] = C[k,l+1]*E[l,l+1]'
               for ir = l+2:n
                   C[k,l] += C[k,ir]*A[l,ir]'
                   W[k] += C[k,ir]*E[l,ir]'
               end
               for ic = k:n
                   y += (E[k,ic]*C[ic,l] + A[k,ic]*W[ic])'
               end
            end
            C[l,k] = -y/(A[k,k]'*E[l,l]+E[k,k]'*A[l,l])
            isfinite(C[l,k]) ||  throw("ME:SingularException: A-λE has eigenvalues α and β such that α+β ≈ 0")
            k == l && (C[k,k] = real(C[k,k]))
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
               W[k] += (E[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end

"""
    lyapds!(A, C; adj = false)

Solve the discrete Lyapunov matrix equation

                op(A)Xop(A)' - X + C = 0,

where `op(A) = A` if `adj = false` and `op(A) = A'` if `adj = true`.
`A` is in a real or complex Schur form and `C` is a symmetric or hermitian matrix.
`A` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapds!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:Real}
# function lyapds!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:BlasReal}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && issymmetric(C)) ||
      throw(DimensionMismatch("C must be a $n x $n symmetric matrix"))

   ONE = one(T1)

   # determine the dimensions of the diagonal blocks of real Schur form

   G = Matrix{T1}(undef,2,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   ba, p = sfstruct(A)
   if adj
      # """
      # The (K,L)th block of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*A(L,L) - X(K,L) = -C(K,L) - R(K,L),

      # where
      #           K           L-1
      # R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*A(J,L)]} +
      #          I=1          J=1

      #           K-1
      #          {SUM [A(I,K)'*X(I,L)]}*A(L,L).
      #           I=1
      # """
      i = 1
      @inbounds  for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
              dl = ba[ll]
              j1 = j+dl-1
              l = j:j1
              Ckl = view(C,k,l)
              y = view(G,1:dk,1:dl)
              copyto!(y,Ckl)
              if kk > 1
                # C[l,k] = C[l,ir]*A[ir,k]
                 mul!(view(C,l,k),view(C,l,ir),view(A,ir,k))
                 ic = 1:j1
                 #y += C[ic,k]'*A[ic,l]
                 mul!(y,transpose(view(C,ic,k)),view(A,ic,l),ONE,ONE)
              end
              if kk == ll
                 lyapd2!(adj,y,dk,view(A,k,k),Xw,Yw)
                 copyto!(Ckl,y)
              else
                 lyapdsylv2!(adj,y,dk,dl,view(A,k,k),view(A,l,l),Xw,Yw)
                 copyto!(Ckl,y)
              end
              j += dl
              if ll < kk
                 # C[l,k] += C[k,l]'*A[k,k]
                 mul!(view(C,l,k),transpose(Ckl),view(A,k,k),ONE,ONE)
               end
          end
          if kk > 1
             # C[ir,k] = C[k,ir]'
             transpose!(view(C,ir,k),view(C,k,ir))
          end
          i += dk
      end
   else
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-right corner column by column by

      # A(K,K)*X(K,L)*A(L,L)' - X(K,L) = -C(K,L) - R(K,L),

      # where

      #           N            N
      # R(K,L) = SUM {A(K,I)* SUM [X(I,J)*A(L,J)']} +
      #          I=K         J=L+1

      #             N
      #          { SUM [A(K,J)*X(J,L)]}*A(L,L)'
      #           J=K+1
      # """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          ir = j+1:n
          for kk = p:-1:ll
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              Clk = view(C,l,k)
              y = view(G,1:dl,1:dk)
              copyto!(y,Clk)
              if ll < p
                 # C[k,l] = C[k,ir]*A[l,ir]'
                 mul!(view(C,k,l),view(C,k,ir),transpose(view(A,l,ir)))
                 ic = i1:n
                 # y += (A[k,ic]*C[ic,l])'
                 mul!(y,transpose(view(C,ic,l)),transpose(view(A,k,ic)),ONE,ONE)
              end
              if i == j
                 lyapd2!(adj,y,dk,view(A,k,k),Xw,Yw)
                 copyto!(Clk,y)
              else
                 lyapdsylv2!(adj,y,dl,dk,view(A,l,l),view(A,k,k),Xw,Yw)
                 copyto!(Clk,y)
              end
              i -= dk
              if i >= j
                 #C[k,l] += (A[l,l]*C[l,k])'
                 mul!(view(C,k,l),transpose(Clk),transpose(view(A,l,l)),ONE,ONE)
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             # C[ir,l] = C[l,ir]'
             transpose!(view(C,ir,l),view(C,l,ir))
          end
          j -= dl
      end
   end
end

function lyapd2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
# function lyapd2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 or 2x2 continuous Lyapunov equations
   #      A'*X*A - X = -C if adj = true  -> R = kron(A',A')-I = (kron(A,A)-I)'
   #      A*X*A' - X = -C if adj = false -> R = kron(A,A)-I
   MONE = -one(T)
   if na == 1
      temp = A[1,1]^2+MONE
      rmul!(C,inv(-temp))
      any(!isfinite, C) && throw("ME:SingularException: A has eigenvalue(s) with moduli ≈ 1")
      return C
   end
   TWO = 2*one(T)
   i1 = 1:3
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   @inbounds  Y[1] = -C[1,1]
   @inbounds  Y[2] = -C[2,1]
   @inbounds  Y[3] = -C[2,2]
   if adj
      # Rt =
      # [ a11^2-1              2*a11*a21      a21^2]
      # [   a11*a12  a11*a22+a12*a21-1    a21*a22]
      # [     a12^2              2*a12*a22  a22^2-1]
      # @inbounds R = [ A[1,1]^2+MONE    TWO*A[1,1]*A[2,1]         A[2,1]^2;
      #                 A[1,1]*A[1,2]    A[1,1]*A[2,2]+A[1,2]*A[2,1]+MONE  A[2,1]*A[2,2];
      #                 A[1,2]^2         TWO*A[1,2]*A[2,2]         A[2,2]^2+MONE ]
      @inbounds  R[1,1] = A[1,1]^2+MONE
      @inbounds  R[1,2] = TWO*A[1,1]*A[2,1]
      @inbounds  R[1,3] = A[2,1]^2
      @inbounds  R[2,1] = A[1,1]*A[1,2]
      @inbounds  R[2,2] = A[1,1]*A[2,2]+A[1,2]*A[2,1]+MONE
      @inbounds  R[2,3] = A[2,1]*A[2,2]
      @inbounds  R[3,1] = A[1,2]^2
      @inbounds  R[3,2] = TWO*A[1,2]*A[2,2]
      @inbounds  R[3,3] = A[2,2]^2+MONE
   else
      # R =
      # [ a11^2-1              2*a11*a12      a12^2]
      # [   a11*a21  a11*a22+a12*a21-1    a12*a22]
      # [     a21^2              2*a21*a22  a22^2-1]

      # @inbounds R = [ A[1,1]^2+MONE    TWO*A[1,1]*A[1,2]         A[1,2]^2;
      #                 A[1,1]*A[2,1]    A[1,1]*A[2,2]+A[1,2]*A[2,1]+MONE  A[1,2]*A[2,2];
      #                 A[2,1]^2         TWO*A[2,1]*A[2,2]         A[2,2]^2+MONE ]
      @inbounds  R[1,1] = A[1,1]^2+MONE
      @inbounds  R[1,2] = TWO*A[1,1]*A[1,2]
      @inbounds  R[1,3] = A[1,2]^2
      @inbounds  R[2,1] = A[1,1]*A[2,1]
      @inbounds  R[2,2] = A[1,1]*A[2,2]+A[1,2]*A[2,1]+MONE
      @inbounds  R[2,3] = A[1,2]*A[2,2]
      @inbounds  R[3,1] = A[2,1]^2
      @inbounds  R[3,2] = TWO*A[2,1]*A[2,2]
      @inbounds  R[3,3] = A[2,2]^2+MONE
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalues α and β such that αβ ≈ 1")
   @inbounds C[1,1] = Y[1]
   @inbounds C[1,2] = Y[2]
   @inbounds C[2,1] = Y[2]
   @inbounds C[2,2] = Y[3]
   return C
end
function lyapdsylv2!(adj::Bool,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::AbstractVector{T}) where T <: Real
# function lyapdsylv2!(adj::Bool,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::AbstractVector{T}) where T <:BlasReal
#    # speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations
#    # encountered in solving discrete Lyapunov equations:
#    #      A'*X*B - X = -C if adj = true  -> R = kron(B',A') - I = (kron(B,A)-I)'
#    #      A*X*B' - X = -C if adj = false -> R = kron(B,A)-I

   MONE = -one(T)
   if na == 1 && nb == 1
      temp = A[1,1]*B[1,1] + MONE
      rmul!(C,inv(-temp))
      any(!isfinite, C) && throw("ME:SingularException: A has eigenvalues α and β such that αβ ≈ 1")
      return C
   end
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if adj
      # @inbounds  R = kron(transpose(B),transpose(A)) - I
      if na == 1
         # R12t =
         # [ a11*b11-1      a11*b21]
         # [     a11*b12  a11*b22-1]
         # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[1,1]*B[2,1];
         #                 A[1,1]*B[1,2]  A[1,1]*B[2,2]+MONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
         @inbounds  R[1,2] = A[1,1]*B[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+MONE
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
   else
         if nb == 1
            # R21t =
            # [ a11*b11-1      a21*b11]
            # [     a12*b11  a22*b11-1]
            # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[2,1]*B[1,1];
            #                 A[1,2]*B[1,1]  A[2,2]*B[1,1]+MONE ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+MONE
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # Rt =
            # [ a11*b11-1      a21*b11      a11*b21      a21*b21]
            # [     a12*b11  a22*b11-1      a12*b21      a22*b21]
            # [     a11*b12      a21*b12  a11*b22-1      a21*b22]
            # [     a12*b12      a22*b12      a12*b22  a22*b22-1]
            # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]+MONE      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]+MONE      A[2,1]*B[2,2];
            # A[1,2]*B[1,2]      A[2,2]*B[1,2]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+MONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]
            @inbounds  R[1,4] = A[2,1]*B[2,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+MONE
            @inbounds  R[2,3] = A[1,2]*B[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]
            @inbounds  R[3,2] = A[2,1]*B[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+MONE
            @inbounds  R[3,4] = A[2,1]*B[2,2]
            @inbounds  R[4,1] = A[1,2]*B[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]
            @inbounds  R[4,3] = A[1,2]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+MONE
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   else
      # @inbounds  R = kron(B,A) - I
      if na == 1
         # R12 =
         # [ a11*b11-1      a11*b12]
         # [     a11*b21  a11*b22-1]
         # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[1,1]*B[1,2];
         #                 A[1,1]*B[2,1]  A[1,1]*B[2,2]+MONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
         @inbounds  R[1,2] = A[1,1]*B[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+MONE
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
   else
         if nb == 1
            # R21 =
            #    [ a11*b11-1      a12*b11]
            #    [     a21*b11  a22*b11-1]
            # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[1,2]*B[1,1];
            #                 A[2,1]*B[1,1]  A[2,2]*B[1,1]+MONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+MONE
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # R =
            # [ a11*b11-1      a12*b11      a11*b12      a12*b12]
            # [     a21*b11  a22*b11-1      a21*b12      a22*b12]
            # [     a11*b21      a12*b21  a11*b22-1      a12*b22]
            # [     a21*b21      a22*b21      a21*b22  a22*b22-1]
            # @inbounds R = [ A[1,1]*B[1,1]+MONE      A[1,2]*B[1,1]      A[1,1]*B[1,2]      A[1,2]*B[1,2];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]+MONE      A[2,1]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]+MONE      A[1,2]*B[2,2];
            # A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+MONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+MONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]
            @inbounds  R[1,4] = A[1,2]*B[1,2]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+MONE
            @inbounds  R[2,3] = A[2,1]*B[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]
            @inbounds  R[3,2] = A[1,2]*B[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+MONE
            @inbounds  R[3,4] = A[1,2]*B[2,2]
            @inbounds  R[4,1] = A[2,1]*B[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]
            @inbounds  R[4,3] = A[2,1]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+MONE
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalues α and β such that αβ ≈ 1")
   C[:,:] = Y
   return C
end
function lyapds!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:Complex}
# function lyapds!(A::AbstractMatrix{T1},C::AbstractMatrix{T1}; adj = false) where {T1<:BlasComplex}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && ishermitian(C)) ||
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))

   ONE = one(T1)

   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
               end
               for ic = 1:l
                  y += C[ic,k]'*A[ic,l]
               end
            end
            temp = ONE-A[k,k]'*A[l,l]
            C[k,l] = y/temp
            isfinite(C[k,l]) || throw("ME:SingularException: A has eigenvalues α and β such that αβ ≈ 1")
            k == l && (C[k,k] = real(C[k,k]))
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               for ir = l+2:n
                  C[k,l] += C[k,ir]*A[l,ir]'
               end
               for ic = k:n
                   y += (A[k,ic]*C[ic,l])'
               end
            end
            temp = ONE-A[k,k]'*A[l,l]
            C[l,k] = y/temp
            isfinite(C[l,k]) || throw("ME:SingularException: A has eigenvalues α and β such that αβ ≈ 1")
            k == l && (C[k,k] = real(C[k,k]))
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end
"""
    lyapds!(A, E, C; adj = false)

Solve the generalized discrete Lyapunov matrix equation

                op(A)Xop(A)' - op(E)Xop(E)' + C = 0,

where `op(A) = A` and `op(E) = E` if `adj = false` and `op(A) = A'` and
`op(E) = E'` if `adj = true`. The pair `(A,E)` is in a generalized real or
complex Schur form and `C` is a symmetric or hermitian matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapds!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:Real}
# function lyapds!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:BlasReal}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && issymmetric(C)) ||
      throw(DimensionMismatch("C must be a $n x $n symmetric matrix"))
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (lyapds!(A, C, adj = adj); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   ONE = one(T1)
   MONE = -ONE

   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   G = Matrix{T1}(undef,2,2)
   W = Array{T1,2}(undef,n,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if adj
      # """
      # The (K,L)th block of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*A(L,L) - E(K,K)'*X(K,L)*E(L,L) = -C(K,L) - R(K,L),

      # where
      #           K           L-1
      # R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*A(J,L)]} -
      #          I=1          J=1

      #           K           L-1
      #          SUM {E(I,K)'*SUM [X(I,J)*E(J,L)]} +
      #          I=1          J=1

      #           K-1
      #          {SUM [A(I,K)'*X(I,L)]}*A(L,L) -
      #           I=1

      #           K-1
      #          {SUM [E(I,K)'*X(I,L)]}*E(L,L).
      #           I=1
      # """
      i = 1
      for kk = 1:p
          dk = ba[kk]
          dkk = 1:dk
          k = i:i+dk-1
          j = 1
          ir = 1:i-1
          for ll = 1:kk
             dl = ba[ll]
             j1 = j+dl-1
             l = j:j1
             #y = C[k,l]
             y = view(G,1:dk,1:dl)
             Ckl = view(C,k,l)
             copyto!(y,Ckl)
             if kk > 1
                 # C[l,k] = C[l,ir]*A[ir,k]
                 # W[l,dkk] = C[l,ir]*E[ir,k]
                 mul!(view(C,l,k),view(C,l,ir),view(A,ir,k))
                 mul!(view(W,l,dkk),view(C,l,ir),view(E,ir,k))
                 ic = 1:j1
                 # y += C[ic,k]'*A[ic,l] -W[ic,dkk]'*E[ic,l]
                 mul!(y,transpose(view(C,ic,k)),view(A,ic,l),ONE,ONE)
                 mul!(y,transpose(view(W,ic,dkk)),view(E,ic,l),MONE,ONE)
             end
             if i == j
                lyapd2!(adj,y,dk,view(A,k,k),view(E,k,k),Xw,Yw)
                copyto!(Ckl,y)
             else
                lyapdsylv2!(adj,y,dk,dl,view(A,k,k),view(E,k,k),view(A,l,l),view(E,l,l),Xw,Yw)
                copyto!(Ckl,y)
             end
             j += dl
             if j <= i
                 # C[l,k] += C[k,l]'*A[k,k]
                 # W[l,dkk] += C[k,l]'*E[k,k]
                 mul!(view(C,l,k),transpose(view(C,k,l)),view(A,k,k),ONE,ONE)
                 mul!(view(W,l,dkk),transpose(view(C,k,l)),view(E,k,k),ONE,ONE)
              end
          end
          if kk > 1
             # C[ir,k] = C[k,ir]'
             transpose!(view(C,ir,k),view(C,k,ir))
          end
          i += dk
      end
   else
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-right corner column by column by

      # A(K,K)*X(K,L)*A(L,L)' - E(K,K)*X(K,L)*E(L,L)' = -C(K,L) - R(K,L),

      # where

      #           N            N
      # R(K,L) = SUM {A(K,I)* SUM [X(I,J)*A(L,J)']} -
      #          I=K         J=L+1

      #           N            N
      #          SUM {E(K,I)* SUM [X(I,J)*E(L,J)']} +
      #          I=K         J=L+1

      #             N
      #          { SUM [A(K,J)*X(J,L)]}*A(L,L)' -
      #           J=K+1

      #             N
      #          { SUM [E(K,J)*X(J,L)]}*E(L,L)'
      #           J=K+1
      # """
      j = n
      for ll = p:-1:1
        dl = ba[ll]
        l = j-dl+1:j
        dll = 1:dl
        i = n
        ir = j+1:n
        for kk = p:-1:ll
            dk = ba[kk]
            i1 = i-dk+1
            k = i1:i
            Clk = view(C,l,k)
            y = view(G,1:dl,1:dk)
            copyto!(y,Clk)
            if ll < p
               # C[k,l] = C[k,ir]*A[l,ir]'
               # W[k,dll] = C[k,ir]*E[l,ir]'
               mul!(view(C,k,l),view(C,k,ir),transpose(view(A,l,ir)))
               mul!(view(W,k,dll),view(C,k,ir),transpose(view(E,l,ir)))
               ic = i1:n
               # y += (A[k,ic]*C[ic,l] - E[k,ic]*W[ic,dll])'
               mul!(y,transpose(view(C,ic,l)),transpose(view(A,k,ic)),ONE,ONE)
               mul!(y,transpose(view(W,ic,dll)),transpose(view(E,k,ic)),MONE,ONE)
            end
            if i == j
               lyapd2!(adj,y,dk,view(A,k,k),view(E,k,k),Xw,Yw)
               copyto!(Clk,y)
            else
               lyapdsylv2!(adj,y,dl,dk,view(A,l,l),view(E,l,l),view(A,k,k),view(E,k,k),Xw,Yw)
               copyto!(Clk,y)
            end
            i -= dk
            if i >= j
               # C[k,l] += (A[l,l]*C[l,k])'
               # W[k,dll] += (E[l,l]*C[l,k])'
               mul!(view(C,k,l),transpose(view(C,l,k)),transpose(view(A,l,l)),ONE,ONE)
               mul!(view(W,k,dll),transpose(view(C,l,k)),transpose(view(E,l,l)),ONE,ONE)
            else
               break
            end
        end
        if ll < p
           ir = i+2:n
           # C[ir,l] = C[l,ir]'
           transpose!(view(C,ir,l),view(C,l,ir))
         end
        j -= dl
      end
   end
end
@inline function lyapd2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
# @inline function lyapd2!(adj,C::AbstractMatrix{T},na::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 or 2x2 continuous Lyapunov equations
   #      A'*X*A - E'*X*E = -C if adj = true  -> R = kron(A',A')-kron(E',E') = (kron(A,A)-kron(E,E))'
   #      A*X*A' - E*X*E' = -C if adj = false -> R = kron(A,A)-kron(E,E)
   MONE = -one(T)
   if na == 1
      temp = E[1,1]^2-A[1,1]^2
      rmul!(C,inv(temp))
      any(!isfinite, C) &&  throw("ME:SingularException: A-λE has eigenvalue(s) with moduli equal to one")
      return C
   end
   TWO = 2*one(T)
   i1 = 1:3
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   @inbounds  Y[1] = -C[1,1]
   @inbounds  Y[2] = -C[2,1]
   @inbounds  Y[3] = -C[2,2]
   if adj
      # Rt =
      # [     a11^2 - e11^2,                   2*a11*a21,         a21^2]
      # [ a11*a12 - e11*e12, a11*a22 + a12*a21 - e11*e22,       a21*a22]
      # [     a12^2 - e12^2,       2*a12*a22 - 2*e12*e22, a22^2 - e22^2]
      # iszero(E[1,2]) ?
      #    (@inbounds R = [ A[1,1]^2-E[1,1]^2    TWO*A[1,1]*A[2,1]         A[2,1]^2;
      #                 A[1,1]*A[1,2]    A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]  A[2,1]*A[2,2];
      #                 A[1,2]^2         TWO*A[1,2]*A[2,2]         A[2,2]^2-E[2,2]^2 ] ) :
         # (@inbounds R = [ A[1,1]^2-E[1,1]^2    TWO*A[1,1]*A[2,1]         A[2,1]^2;
         #              A[1,1]*A[1,2]-E[1,1]*E[1,2]    A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]  A[2,1]*A[2,2];
         #              A[1,2]^2-E[1,2]^2         TWO*(A[1,2]*A[2,2]-E[1,2]*E[2,2])         A[2,2]^2-E[2,2]^2 ])
      @inbounds  R[1,1] = A[1,1]^2-E[1,1]^2
      @inbounds  R[1,2] = TWO*A[1,1]*A[2,1]
      @inbounds  R[1,3] = A[2,1]^2
      @inbounds  R[2,1] = A[1,1]*A[1,2]-E[1,1]*E[1,2]
      @inbounds  R[2,2] = A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]
      @inbounds  R[2,3] = A[2,1]*A[2,2]
      @inbounds  R[3,1] = A[1,2]^2-E[1,2]^2
      @inbounds  R[3,2] = TWO*(A[1,2]*A[2,2]-E[1,2]*E[2,2])
      @inbounds  R[3,3] = A[2,2]^2-E[2,2]^2
   else
      # R =
      # [ a11^2 - e11^2,       2*a11*a12 - 2*e11*e12,     a12^2 - e12^2]
      # [       a11*a21, a11*a22 + a12*a21 - e11*e22, a12*a22 - e12*e22]
      # [         a21^2,                   2*a21*a22,     a22^2 - e22^2]
      # iszero(E[1,2]) ?
      #    (@inbounds R = [ A[1,1]^2-E[1,1]^2    TWO*A[1,1]*A[1,2]         A[1,2]^2;
      #                 A[1,1]*A[2,1]    A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]  A[1,2]*A[2,2];
      #                 A[2,1]^2         TWO*A[2,1]*A[2,2]         A[2,2]^2-E[2,2]^2 ] ) :

      #    (@inbounds R = [ A[1,1]^2-E[1,1]^2    TWO*(A[1,1]*A[1,2]-E[1,1]*E[1,2])         A[1,2]^2-E[1,2]^2;
      #                 A[1,1]*A[2,1]    A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]  A[1,2]*A[2,2]-E[1,2]*E[2,2];
      #                 A[2,1]^2         TWO*A[2,1]*A[2,2]         A[2,2]^2-E[2,2]^2 ] )
      @inbounds  R[1,1] = A[1,1]^2-E[1,1]^2
      @inbounds  R[1,2] = TWO*(A[1,1]*A[1,2]-E[1,1]*E[1,2])
      @inbounds  R[1,3] = A[1,2]^2-E[1,2]^2
      @inbounds  R[2,1] = A[1,1]*A[2,1]
      @inbounds  R[2,2] = A[1,1]*A[2,2]+A[1,2]*A[2,1]-E[1,1]*E[2,2]
      @inbounds  R[2,3] = A[1,2]*A[2,2]-E[1,2]*E[2,2]
      @inbounds  R[3,1] = A[2,1]^2
      @inbounds  R[3,2] = TWO*A[2,1]*A[2,2]
      @inbounds  R[3,3] = A[2,2]^2-E[2,2]^2
   end
   luslv!(R,Y) && throw("ME:SingularException: A-λE has eigenvalues α and β such that αβ ≈ 1")
   @inbounds C[1,1] = Y[1]
   @inbounds C[1,2] = Y[2]
   @inbounds C[2,1] = Y[2]
   @inbounds C[2,2] = Y[3]
   return C
end
@inline function lyapdsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},F::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:Real
# @inline function lyapdsylv2!(adj,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},F::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations
   # encountered in solving discrete Lyapunov equations:
   #      A'*X*B - E'*X*F = -C if adj = true  -> R = kron(B',A') - kron(F',E') = (kron(B,A)-kron(F,E))'
   #      A*X*B' - E*X*F' = -C if adj = false -> R = kron(B,A)-kron(F,E)
   if na == 1 && nb == 1
      temp = E[1,1]*F[1,1] - A[1,1]*B[1,1]
      rmul!(C,inv(temp))
      any(!isfinite, C) && throw("ME:SingularException: A-λE has eigenvalues α and β such that αβ ≈ 1")
      return C
   end
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if adj
      if na == 1
         # R12t =
         # [ a11*b11 - e11*f11,           a11*b21]
         # [ a11*b12 - e11*f12, a11*b22 - e11*f22]
         # @inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[1,1]*B[2,1];
         #                 A[1,1]*B[1,2]-E[1,1]*F[1,2]  A[1,1]*B[2,2]-E[1,1]*F[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
         @inbounds  R[1,2] = A[1,1]*B[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]-E[1,1]*F[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]-E[1,1]*F[2,2]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # R21t =
            # [ a11*b11 - e11*f11,           a21*b11]
            # [ a12*b11 - e12*f11, a22*b11 - e22*f11]
            # @inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[2,1]*B[1,1];
            #                 A[1,2]*B[1,1]-E[1,2]*F[1,1]  A[2,2]*B[1,1]-E[2,2]*F[1,1] ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]-E[1,2]*F[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]-E[2,2]*F[1,1]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # Rt =
            # [ a11*b11 - e11*f11,           a21*b11,           a11*b21,           a21*b21]
            # [ a12*b11 - e12*f11, a22*b11 - e22*f11,           a12*b21,           a22*b21]
            # [ a11*b12 - e11*f12,           a21*b12, a11*b22 - e11*f22,           a21*b22]
            # [ a12*b12 - e12*f12, a22*b12 - e22*f12, a12*b22 - e12*f22, a22*b22 - e22*f22]
            # (iszero(E[1,2]) && iszero(F[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]-E[2,2]*F[1,1]      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]-E[1,1]*F[2,2]      A[2,1]*B[2,2];
            # A[1,2]*B[1,2]      A[2,2]*B[1,2]      A[1,2]*B[2,2]  A[2,2]*B[2,2]-E[2,2]*F[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            # A[1,2]*B[1,1]-E[1,1]*F[1,2]  A[2,2]*B[1,1]-E[2,2]*F[1,1]      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]-E[1,1]*F[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]-E[1,1]*F[2,2]      A[2,1]*B[2,2];
            # A[1,2]*B[1,2]-E[1,2]*F[1,2]      A[2,2]*B[1,2]-E[2,2]*F[1,2]      A[1,2]*B[2,2]-E[1,2]*F[2,2]  A[2,2]*B[2,2]-E[2,2]*F[2,2]])
            @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]
            @inbounds  R[1,4] = A[2,1]*B[2,1]
            #@inbounds  R[2,1] = A[1,2]*B[1,1]-E[1,1]*F[1,2]
            @inbounds  R[2,1] = A[1,2]*B[1,1]-E[1,2]*F[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]-E[2,2]*F[1,1]
            @inbounds  R[2,3] = A[1,2]*B[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]-E[1,1]*F[1,2]
            @inbounds  R[3,2] = A[2,1]*B[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]-E[1,1]*F[2,2]
            @inbounds  R[3,4] = A[2,1]*B[2,2]
            @inbounds  R[4,1] = A[1,2]*B[1,2]-E[1,2]*F[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]-E[2,2]*F[1,2]
            @inbounds  R[4,3] = A[1,2]*B[2,2]-E[1,2]*F[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]-E[2,2]*F[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   else
      if na == 1
         # R12 =
         # [ a11*b11 - e11*f11, a11*b12 - e11*f12]
         # [           a11*b21, a11*b22 - e11*f22]
         # @inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[1,1]*B[1,2]-E[1,1]*F[1,2];
         #                 A[1,1]*B[2,1]  A[1,1]*B[2,2]-E[1,1]*F[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
         @inbounds  R[1,2] = A[1,1]*B[1,2]-E[1,1]*F[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]-E[1,1]*F[2,2]
         @inbounds  Y[1] = -C[1,1]
         @inbounds  Y[2] = -C[1,2]
      else
         if nb == 1
            # R21 =
            # [ a11*b11 - e11*f11, a12*b11 - e12*f11]
            # [           a21*b11, a22*b11 - e22*f11]
            # @inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[1,2]*B[1,1]-E[1,2]*F[1,1];
            #                 A[2,1]*B[1,1]  A[2,2]*B[1,1]-E[2,2]*F[1,1]]
            @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]-E[1,2]*F[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]-E[2,2]*F[1,1]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
         else
            # R =
            # [ a11*b11 - e11*f11, a12*b11 - e12*f11, a11*b12 - e11*f12, a12*b12 - e12*f12]
            # [           a21*b11, a22*b11 - e22*f11,           a21*b12, a22*b12 - e22*f12]
            # [           a11*b21,           a12*b21, a11*b22 - e11*f22, a12*b22 - e12*f22]
            # [           a21*b21,           a22*b21,           a21*b22, a22*b22 - e22*f22]
            # (iszero(E[1,2]) && iszero(F[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[1,2]*B[1,1]      A[1,1]*B[1,2]      A[1,2]*B[1,2];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]-E[2,2]*F[1,1]      A[2,1]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]-E[1,1]*F[2,2]      A[1,2]*B[2,2];
            # A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]-E[2,2]*F[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]-E[1,1]*F[1,1]      A[1,2]*B[1,1]-E[1,2]*F[1,1]      A[1,1]*B[1,2]-E[1,1]*F[1,2]      A[1,2]*B[1,2]-E[1,2]*F[1,2];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]-E[2,2]*F[1,1]      A[2,1]*B[1,2]      A[2,2]*B[1,2]-E[2,2]*F[1,2];
            # A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]-E[1,1]*F[2,2]      A[1,2]*B[2,2]-E[1,2]*F[2,2];
            # A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]-E[2,2]*F[2,2]])
            @inbounds  R[1,1] = A[1,1]*B[1,1]-E[1,1]*F[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]-E[1,2]*F[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]-E[1,1]*F[1,2]
            @inbounds  R[1,4] = A[1,2]*B[1,2]-E[1,2]*F[1,2]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]-E[2,2]*F[1,1]
            @inbounds  R[2,3] = A[2,1]*B[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]-E[2,2]*F[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]
            @inbounds  R[3,2] = A[1,2]*B[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]-E[1,1]*F[2,2]
            @inbounds  R[3,4] = A[1,2]*B[2,2]-E[1,2]*F[2,2]
            @inbounds  R[4,1] = A[2,1]*B[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]
            @inbounds  R[4,3] = A[2,1]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]-E[2,2]*F[2,2]
            @inbounds  Y[1] = -C[1,1]
            @inbounds  Y[2] = -C[2,1]
            @inbounds  Y[3] = -C[1,2]
            @inbounds  Y[4] = -C[2,2]
         end
      end
   end
   luslv!(R,Y) && throw("ME:SingularException: A-λE has eigenvalues α and β such that αβ ≈ 1")
   C[:,:] = Y
   return C
end
function lyapds!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:Complex}
# function lyapds!(A::AbstractMatrix{T1},E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, C::AbstractMatrix{T1}; adj = false) where {T1<:BlasComplex}
   n = LinearAlgebra.checksquare(A)
   (LinearAlgebra.checksquare(C) == n && ishermitian(C)) ||
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (lyapds!(A, C, adj = adj); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   W = Array{T1,1}(undef,n)
   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               W[l] = C[l,1]*E[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
                  W[l] += C[l,ir]*E[ir,k]
               end
               for ic = 1:l
                   y += C[ic,k]'*A[ic,l] - W[ic]'*E[ic,l]
                end
            end
            C[k,l] = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            isfinite(C[k,l]) || throw("ME:SingularException: A-λE has eigenvalues α and β such that αβ ≈ 1")
            k == l && (C[k,k] = real(C[k,k]))
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
               W[l] += C[k,l]'*E[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               W[k] = C[k,l+1]*E[l,l+1]'
               for ir = l+2:n
                  C[k,l] += C[k,ir]*A[l,ir]'
                  W[k] += C[k,ir]*E[l,ir]'
               end
               for ic = k:n
                   y += (A[k,ic]*C[ic,l] - E[k,ic]*W[ic])'
               end
            end
            C[l,k] = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            isfinite(C[l,k]) || throw("ME:SingularException: A-λE has eigenvalues α and β such that αβ ≈ 1")
            k == l && (C[k,k] = real(C[k,k]))
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
               W[k] += (E[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end
# Lyapunov-like equations 
"""
    X = tlyapc(A,C, isig = 1; fast = true, atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute for `isig = ±1` a solution of the continuous T-Lyapunov matrix equation

                A*X + isig*transpose(X)*transpose(A) + C = 0

using explicit formulas based on full rank factorizations of `A`  (see [1] and [2]). 
`A` and `C` are `m×n` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
The matrix `C` must be symmetric if `isig = 1` and skew-symmetric if `isig = -1`.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`, where `N = min(m,n)` and ϵ is the machine precision of 
the element type of `A`.

The underlying rank revealing factorization of `A` is the QR-factorization with column pivoting, 
if `fast = true` (default), or the more reliable SVD-decomposition, if `fast = false`.

[1] H. W. Braden. The equations A^T X ± X^T A = B. SIAM J. Matrix Anal. Appl., 20(2):295–302, 1998.

[2] C.-Y. Chiang, E. K.-W. Chu, W.-W. Lin, On the ★-Sylvester equation AX ± X^★ B^★ = C.
    Applied Mathematics and Computation, 218:8393–8407, 2012.
"""
function tlyapc(A, C, isig = 1; fast::Bool = true, atol::Real = 0.0, rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    abs(isig) == 1 || error(" isig must be either 1 or -1")
    if isig == 1
       issymmetric(C) || error("C must be symmetric for isig = 1")
       # temporary fix to avoid false results for DoubleFloats 
       # C == transpose(C) || error("C must be symmetric for isig = 1")
    else
       iszero(C+transpose(C)) || error("C must be skew-symmetric for isig = -1")
    end
    T2 = promote_type(eltype(A), eltype(C))
    T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
    eltype(A) == T2 || (A = convert(Matrix{T2},A))
    eltype(C) == T2 || (C = convert(Matrix{T2},C))
    if fast 
       @static VERSION < v"1.7.0" ? F = qr(A, Val(true)) : F = qr(A, ColumnNorm())
       tol = max(atol, rtol*norm(F.R,Inf))
       r = count(x -> abs(x) > tol, diag(F.R))
       Ct = F.Q'*C*conj(Matrix(F.Q))
       i1 = 1:r; j2 = r+1:m
       norm(view(Ct,j2,j2),Inf) <= tol || @warn "Incompatible equation: least-squares solution computed"
       return ([ldiv!(UpperTriangular(2*F.R[i1,i1]),view(Ct,i1,i1)) ldiv!(UpperTriangular(F.R[i1,i1]),view(Ct,i1,j2));
                zeros(T2,n-r,m)]*transpose(F.Q))[invperm(F.p),:]
   else
       m > n ? F = svd(A,full = true) : F = svd(A)
       tol = max(atol, rtol*F.S[1])
       r = count(x -> x > tol, F.S)
       Ct = F.U'*C*conj(F.U)
       i1 = 1:r; j2 = r+1:m
       norm(view(Ct,j2,j2),Inf) <= tol || @warn "Incompatible equation: least-squares solution computed"
       return view(F.Vt,i1,:)'*[Diagonal(F.S[i1]*2)\Ct[i1,i1] Diagonal(F.S)\Ct[i1,j2]]*transpose(F.U)
    end
end 
"""
    X = hlyapc(A,C, isig = 1; atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute for `isig = ±1` a solution of the continuous H-Lyapunov matrix equation

                A*X + isig*adjoint(X)*adjoint(A) + C = 0

using explicit formulas based on full rank factorizations of `A`  (see [1] and [2]). 
`A` and `C` are `m×n` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
The matrix `C` must be hermitian if `isig = 1` and skew-hermitian if `isig = -1`.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`, where `N = min(m,n)` and ϵ is the machine precision 
of the element type of `A`.

The underlying rank revealing factorization of `Ae` (or of `A` if real) is the QR-factorization with column pivoting, 
if `fast = true` (default), or the more reliable SVD-decomposition, if `fast = false`.

[1]  H. W. Braden. The equations A^T X ± X^T A = B. SIAM J. Matrix Anal. Appl., 20(2):295–302, 1998.

[2] C.-Y. Chiang, E. K.-W. Chu, W.-W. Lin, On the ★-Sylvester equation AX ± X^★ B^★ = C.
    Applied Mathematics and Computation, 218:8393–8407, 2012.
"""
function hlyapc(A, C, isig = 1; fast::Bool = true, atol::Real = 0.0, rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
   promote_type(eltype(A),eltype(C)) <: Real && (return tlyapc(A, C; fast, atol, rtol))
   m = LinearAlgebra.checksquare(C)
   ma, n = size(A)
   ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
   abs(isig) == 1 || error(" isig must be either 1 or -1")
   if isig == 1
      ishermitian(C) || error("C must be hermitian for isig = 1")
      # temporary fix to avoid false results for DoubleFloats 
      # C == adjoint(C) || error("C must be hermitian for isig = 1")
   else
      iszero(C+adjoint(C)) || error("C must be skew-hermitian for isig = -1")
   end
   T2 = promote_type(eltype(A), eltype(C))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   eltype(A) == T2 || (A = convert(Matrix{T2},A))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))
   if fast 
      @static VERSION < v"1.7.0" ? F = qr(A, Val(true)) : F = qr(A, ColumnNorm())
      tol = max(atol, rtol*norm(F.R,Inf))
      r = count(x -> abs(x) > tol, diag(F.R))
      Ct = F.Q'*C*F.Q
      i1 = 1:r; j2 = r+1:m
      norm(view(Ct,j2,j2),Inf) <= tol || @warn "Incompatible equation: least-squares solution computed"
      return ([ldiv!(UpperTriangular(2*F.R[i1,i1]),view(Ct,i1,i1)) ldiv!(UpperTriangular(F.R[i1,i1]),view(Ct,i1,j2));
               zeros(T2,n-r,m)]*F.Q')[invperm(F.p),:]
  else
      m > n ? F = svd(A,full = true) : F = svd(A)
      tol = max(atol, rtol*F.S[1])
      r = count(x -> x > tol, F.S)
      Ct = F.U'*C*F.U
      i1 = 1:r; j2 = r+1:m
      norm(view(Ct,j2,j2),Inf) <= tol || @warn "Incompatible equation: least-squares solution computed"
      return view(F.Vt,i1,:)'*[Diagonal(F.S[i1]*2)\Ct[i1,i1] Diagonal(F.S)\Ct[i1,j2]]*F.U'
   end
end
