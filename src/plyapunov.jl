# Positive-definite continuous Lyapunov equations
"""
    U = plyapc(A, B)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous Lyapunov equation

      AX + XA' + BB' = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of rows as `A`. `A` must have only eigenvalues with negative real parts.

    U = plyapc(A', B')

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the continuous Lyapunov equation

      A'X + XA + B'B = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of columns as `A`. `A` must have only eigenvalues with negative real parts.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-2. 1.;-1. -2.]
2×2 Array{Float64,2}:
 -2.0   1.0
 -1.0  -2.0

julia> B = [1. 1. ;1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> U = plyapc(A,B)
2×2 UpperTriangular{Float64,Array{Float64,2}}:
 0.481812  0.801784
  ⋅        0.935414

julia> A*U*U'+U*U'*A'+B*B'
2×2 Array{Float64,2}:
 0.0          8.88178e-16
 8.88178e-16  3.55271e-15
```
"""
function plyapc(A::AbstractMatrix, B::AbstractMatrix)
   """
   Method

   The Bartels-Steward Schur form based method is employed [1], with the
   modifications proposed by Hammarling [2].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(B,Adjoint))
      error("Only calls with A and B or with A' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end

   T2 = promote_type(eltype(A), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))
   ONE = one(real(T2))

   # Reduce A to Schur form and transform B
   if adj
      AS, Q, EV = schur(A.parent)
   else
      AS, Q, EV = schur(A)
   end

   if maximum(real(EV)) >= ZERO
      error("A must have only eigenvalues with negative real part")
   end

   if adj
      #U'U = Q'*B'*B*Q
      u = B.parent*Q
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
   else
      #UU' = Q'*B*B'*Q
      u = Q'*B
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
   end
   plyapcs!(AS, U, adj = adj)
   tau = similar(U,n)
   if adj
      #X = Q*U'*U*Q'
      U, tau = LinearAlgebra.LAPACK.geqrf!(U*Q',tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- Q*U*U'*Q'
      U, tau = LinearAlgebra.LAPACK.gerqf!(Q*U,tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
plyapc(A::Union{Real,Complex}, B::Union{Real,Complex}) =
      real(A) < 0 ? abs(B)/sqrt( -2 * real(A) ) :
      error("A must be a negative number or must have negative real part")
"""
    U = plyapc(A, E, B)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous generalized Lyapunov equation

      AXE' + EXA' + BB' = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λ E` must have only
eigenvalues with negative real parts.

    U = plyapc(A', E', B')

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the continuous generalized Lyapunov equation

      A'XE + E'XA + B'B = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λ E` must have only
eigenvalues with negative real parts.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-2. 1.;-1. -2.]
2×2 Array{Float64,2}:
 -2.0   1.0
 -1.0  -2.0

julia> E = [1. 0.; 1. 1.]
2×2 Array{Float64,2}:
 1.0  0.0
 1.0  1.0

julia> B = [1. 1. ;1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> U = plyapc(A,E,B)
2×2 UpperTriangular{Float64,Array{Float64,2}}:
 0.408248  0.730297
  ⋅        0.547723

julia> A*U*U'*E'+E*U*U'*A'+B*B'
2×2 Array{Float64,2}:
  0.0          -8.88178e-16
 -1.33227e-15  -2.66454e-15
```
"""
function plyapc(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix)
   """
   Method

   A generalization of Bartels-Steward Schur form based method is employed [1],
   with the modifications proposed by Hammarling [2] and Penzl [3].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [3] Penzl, T.
       Numerical solution of generalized Lyapunov equations.
       Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(E,Adjoint)) && xor(adj,isa(B,Adjoint))
      error("Only calls with A, E and B or with A', E' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)
      return plyapc(A, B)
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
   end

   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end

   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(E) !== T2
     adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAEB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))
   ONE = one(real(T2))

   # Reduce (A,E) to generalized Schur form and transform C
   # (AS,ES) = (Q'*A*Z, Q'*E*Z)
   if adj
      AS, ES, Q, Z, α, β = schur(A.parent,E.parent)
   else
      AS, ES, Q, Z, α, β = schur(A,E)
   end

   if maximum(real(α./β)) >= ZERO
      error("A-λE must have only eigenvalues with negative real parts")
   end

   if adj
      #U'*U = Z'*B'*B*Z
      u = B.parent*Z
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
   else
      #U*U' = Q'*B*B'*Q
      u = Q'*B
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
   end
   plyapcs!(AS, ES, U, adj = adj)
   tau = similar(U,n)
   if adj
      #X = Q*U'*U*Q'
      U, tau = LinearAlgebra.LAPACK.geqrf!(U*Q',tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAEB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- Z*U*U'*Z'
      U, tau = LinearAlgebra.LAPACK.gerqf!(Z*U,tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAEB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
plyapc(A::Union{Real,Complex}, E::Union{Real,Complex}, B::Union{Real,Complex}) =
      real(A*E') < 0 ? abs(B)/sqrt( -2 * real(A*E') ) :
      error("A*E' must be a negative number or must have negative real part")
"""
    U = plyapd(A, B)

Compute `U`, the upper triangular Cholesky factor of the solution `X = UU'` of
the discrete Lyapunov equation

      AXA' - X + BB' = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of rows as `A`. `A` must have only eigenvalues with moduli less than one.

    U = plyapd(A', B')

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the discrete Lyapunov equation

      A'XA - X + B'B = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of columns as `A`. `A` must have only eigenvalues with moduli less than one.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-0.5 .1;-0.1 -0.5]
2×2 Array{Float64,2}:
 -0.5   0.1
 -0.1  -0.5

julia> B = [1. 1. ;1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> U = plyapd(A,B)
2×2 UpperTriangular{Float64,Array{Float64,2}}:
 0.670145  1.35277
  ⋅        2.67962

julia> A*U*U'*A'-U*U'+B*B'
2×2 Array{Float64,2}:
 -4.44089e-16  4.44089e-16
  4.44089e-16  1.77636e-15
```
"""
function plyapd(A::AbstractMatrix, B::AbstractMatrix)
   """
   Method

   The Bartels-Steward Schur form based method is employed [1], with the
   modifications proposed by Hammarling in [2] and [3].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [3] Hammarling, S.J. Numerical solution of the discrete-time, convergent,
       non-negative definite Lyapunov equation.
       Systems & Control Letters 17 (1991) 137-139.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(B,Adjoint))
      error("Only calls with A and B or with A' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end

   T2 = promote_type(eltype(A), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))
   ONE = one(real(T2))

   # Reduce A to Schur form and transform B
   if adj
      AS, Q, EV = schur(A.parent)
   else
      AS, Q, EV = schur(A)
   end
   if maximum(abs.(EV)) >= ONE
      error("A must have only eigenvalues with moduli less than one")
   end

   if adj
      #U'U = Q'*B'*B*Q
      u = B.parent*Q
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
   else
      #UU' = Q'*B*B'*Q
      u = Q'*B
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
   end
   plyapds!(AS, U, adj = adj)
   tau = similar(U,n)
   if adj
      #X = Q*U'*U*Q'
      U, tau = LinearAlgebra.LAPACK.geqrf!(U*Q',tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- Q*U*U'*Q'
      U, tau = LinearAlgebra.LAPACK.gerqf!(Q*U,tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
plyapd(A::Union{Real,Complex}, B::Union{Real,Complex}) =
      abs(A) < real(one(A)) ? real(abs(B)/sqrt( (one(A)-abs(A))*(one(A)+abs(A)) )) :
      error("A must be a subunitary number")
"""
    U = plyapd(A, E, B)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
discrete generalized Lyapunov equation

      AXA' - EXE' + BB' = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λ E` must have only
eigenvalues with moduli less than one.

    U = plyapd(A', E', B')

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the discrete generalized Lyapunov equation

      A'XA - E'XE + B'B = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λ E` must have only
eigenvalues with moduli less than one.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-0.5 .1;-0.1 -0.5]
2×2 Array{Float64,2}:
 -0.5   0.1
 -0.1  -0.5

julia> E = [1. 0.; 1. 1.]
2×2 Array{Float64,2}:
 1.0  0.0
 1.0  1.0

julia> B = [1. 1. ;1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> U = plyapd(A,E,B)
2×2 UpperTriangular{Float64,Array{Float64,2}}:
 1.56276  0.416976
  ⋅       1.34062

julia> A*U*U'*A'-E*U*U'*E'+B*B'
2×2 Array{Float64,2}:
 1.77636e-15  2.22045e-15
 2.22045e-15  2.66454e-15
```
"""
function plyapd(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix)
   """
   Method

   The Bartels-Steward Schur form based method is employed [1], with the
   modifications proposed by Hammarling in [2] and Penzl in [3].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [3] Penzl, T.
       Numerical solution of generalized Lyapunov equations.
       Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(E,Adjoint)) && xor(adj,isa(B,Adjoint))
      error("Only calls with A, E and B or with A', E' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)
      return plyapd(A, B)
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
   end

   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end

   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(E) !== T2
     adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAEB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))
   ONE = one(real(T2))

   # Reduce (A,E) to generalized Schur form and transform C
   # (AS,ES) = (Q'*A*Z, Q'*E*Z)
   if adj
      AS, ES, Q, Z, α, β = schur(A.parent,E.parent)
   else
      AS, ES, Q, Z, α, β = schur(A,E)
   end

   if maximum(abs.(α./β)) >= ONE
      error("A-λE must have only eigenvalues with moduli less than one")
   end

   if adj
      #U'*U = Z'*B'*B*Z
      u = B.parent*Z
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
   else
      #U*U' = Q'*B*B'*Q
      u = Q'*B
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
   end
   plyapds!(AS, ES, U, adj = adj)
   tau = similar(U,n)
   if adj
      #X = Q*U'*U*Q'
      U, tau = LinearAlgebra.LAPACK.geqrf!(U*Q',tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAEB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- Z*U*U'*Z'
      U, tau = LinearAlgebra.LAPACK.gerqf!(Z*U,tau)
      U = UpperTriangular(U)
      # Make the diagonal elements of U non-negative.
      if realAEB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
plyapd(A::Union{Real,Complex}, E::Union{Real,Complex}, B::Union{Real,Complex}) =
     abs(A) < abs(E) ? real(abs(B)/sqrt( (abs(E)-abs(A))*(abs(E)+abs(A)) )) :
      error("A/E must be a subunitary number")

"""
    U = plyaps(A, B; disc = false)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous Lyapunov equation

      AX + XA' + BB' = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of rows as `A`.
`A` must have only eigenvalues with negative real parts. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A', B'; disc = false)

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the continuous Lyapunov equation

      A'X + XA + B'B = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of columns as `A`.
`A` must have only eigenvalues with negative real parts. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A, B, disc = true)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
discrete Lyapunov equation

      AXA' - X + BB' = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of rows as `A`.
`A` must have only eigenvalues with moduli less than one. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A', B', disc = true)

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the discrete Lyapunov equation

      A'XA - X + B'B = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of columns as `A`.
`A` must have only eigenvalues with moduli less than one. Only the upper
Hessenberg part of `A` is referenced.
"""
function plyaps(A::AbstractMatrix, B::AbstractMatrix; disc = false)
   """
   Method

   The Bartels-Steward Schur form based method is employed [1], with the
   modifications proposed by Hammarling in [2] and [3].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [3] Hammarling, S.J. Numerical solution of the discrete-time, convergent,
       non-negative definite Lyapunov equation.
       Systems & Control Letters 17 (1991) 137-139.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(B,Adjoint))
      error("Only calls with A and B or with A' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end

   T2 = promote_type(eltype(A), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))

   if adj
      #U'U = B'*B
      u = copy(B.parent)
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
      if disc
         plyapds!(A.parent, U, adj = adj)
      else
         plyapcs!(A.parent, U, adj = adj)
      end
   else
      #UU' = B*B'
      u = copy(B)
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
      if disc
         plyapds!(A, U, adj = adj)
      else
         plyapcs!(A, U, adj = adj)
      end
   end
   if adj
      #X = U'*U
      # Make the diagonal elements of U non-negative.
      if realAB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- U*U'
      # Make the diagonal elements of U non-negative.
      if realAB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
"""
    U = plyaps(A, E, B; disc = false)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous generalized Lyapunov equation

      AXE' + EXA' + BB' = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λ E` must have only
eigenvalues with negative real parts.

    U = plyaps(A', E', B'; disc = false)

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the continuous generalized Lyapunov equation

      A'XE + E'XA + B'B = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λ E` must have only
eigenvalues with negative real parts.

    U = plyaps(A, E, B, disc = true)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
discrete generalized Lyapunov equation

      AXA' - EXE' + BB' = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λ E` must have only
eigenvalues with moduli less than one.

    U = plyaps(A', E', B', disc = true)

Compute `U`, the upper triangular Cholesky factor of the solution `X = U'U` of
the discrete generalized Lyapunov equation

      A'XA - E'XE + B'B = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λ E` must have only
eigenvalues with moduli less than one.
"""
function plyaps(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; disc = false)
   """
   Method

   Generalizations of Bartels-Steward Schur form based method is employed [1],
   with the modifications proposed by Hammarling [2] and Penzl [3].

   Reference:

   [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
       equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
       Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [3] Penzl, T.
       Numerical solution of generalized Lyapunov equations.
       Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   """
   adj = isa(A,Adjoint)
   if xor(adj,isa(E,Adjoint)) && xor(adj,isa(B,Adjoint))
      error("Only calls with A, E and B or with A', E' and B' allowed")
   end

   n = LinearAlgebra.checksquare(A)
   if typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)
      return plyaps(A, B)
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
   end

   if adj
      nb, mb = size(B)
      if nb != n
         throw(DimensionMismatch("B must be a matrix of column dimension $n"))
      end
   else
      mb, nb = size(B)
      if mb != n
         throw(DimensionMismatch("B must be a matrix of row dimension $n"))
      end
   end


   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
   end
   if eltype(A) !== T2
     adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
   end
   if eltype(E) !== T2
     adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E)
   end
   if eltype(B) !== T2
      adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B)
   end

   realAEB = eltype(A) <: AbstractFloat
   ZERO = zero(real(T2))

   if adj
      #U'*U = B'*B
      u = copy(B.parent)
      tau = similar(u,min(n,mb))
      u, tau = LinearAlgebra.LAPACK.geqrf!(u,tau)
      if mb < n
         U = UpperTriangular([u; zeros(T2,n-mb,n)])
      else
         U = UpperTriangular(u[1:n,:])
      end
      if disc
         plyapds!(A.parent, E.parent, U, adj = adj)
      else
         plyapcs!(A.parent, E.parent, U, adj = adj)
      end
   else
      #U*U' = B*B
      u = copy(B)
      tau = similar(u,min(n,nb))
      u, tau = LinearAlgebra.LAPACK.gerqf!(u,tau)
      if nb < n
         U = UpperTriangular([zeros(T2,n,n-nb) u])
      else
         U = UpperTriangular(u[:,nb-n+1:end])
      end
      if disc
         plyapds!(A, E, U, adj = adj)
      else
         plyapcs!(A, E, U, adj = adj)
      end
   end
   if adj
      #X = U'*U
      # Make the diagonal elements of U non-negative.
      if realAEB
         for i = 1:n
             if U[i,i] < ZERO
               for j = i:n
                   U[i,j] = -U[i,j]
               end
             end
          end
       else
          for i = 1:n
              d = abs(U[i,i])
              if d != ZERO
                tmp = conj(U[i,i])/d
                for j = i:n
                    U[i,j] *= tmp
                end
              end
           end
       end
   else
      #X <- U*U'
      # Make the diagonal elements of U non-negative.
      if realAEB
         for j = 1:n
             if U[j,j] < ZERO
                for i = 1:j
                    U[i,j] = -U[i,j]
                end
             end
          end
       else
          for j = 1:n
              d = abs(U[j,j])
              if d != ZERO
                 tmp = conj(U[j,j])/d
                 for i = 1:j
                     U[i,j] *= tmp
                 end
              end
           end
       end
   end
   return U
end
"""
    plyapcs!(A,R;adj = false)

Solve the positive continuous Lyapunov matrix equation

                op(A)X + Xop(A)' + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
`A` is a square real matrix in a real Schur form , or a square complex matrix in a
complex Schur form and `R` is an upper triangular matrix.
`A` must have all eigenvalues with negative real parts.
`R` contains on output the solution `U`.
"""
function plyapcs!(A::T1, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Float32},Matrix{Float64}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end

   T = eltype(A)
   ZERO = zero(T)
   ONE = one(T)
   TWO = 2*ONE
   EPS = eps(ONE)*TWO
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))

   # determine the structure of the real Schur form
   ba = fill(1,n)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   Wr = Array{eltype(A),2}(undef,n,2)
   Wu = Array{eltype(A),2}(undef,n,2)
   Wy = Array{eltype(A),2}(undef,n,2)
   Wz = Array{eltype(A),2}(undef,n,2)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),

      """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = A[j,j]
             if λ >= 0.
                error("A is not stable")
             end
             TEMP = sqrt( abs( TWO*λ ) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular Lyapunov equation")
             end
             α = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/α
             β = A[l,l]
          else
             R[l,l], scale, β, α = plyap2(A[l,l], R[l,l], adj = true)
             if scale != ONE
                error("Singular Lyapunov equation")
             end
          end
          if ll < p
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             rbar = view(Wr,ir1,1:dl)
             ubar = view(Wu,ir1,1:dl)
             y = view(Wy,ir1,1:dl)
             #z = view(Wz,ir1,1:dl)
             rbar = R[l,j1]'
             z = rbar*α + A[l,j1]'*R[l,l]'
             # Solve S1'*ubar+ubar*β + z = 0
             S1 = view(A,j1,j1)
             ubar, scale = LAPACK.trsyl!('T','N', S1, β, z)
             rmul!(ubar, -inv(scale))
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             y = rbar - ubar * α'
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',

      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = A[j,j]
             if λ >= 0.
                error("A is not stable")
             end
             TEMP = sqrt( abs( TWO*λ ) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular Lyapunov equation")
             end
             α = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/α
             β = A[l,l]
          else
             U, scale, β, α = plyap2(A[l,l], R[l,l], adj = false)
             if scale != ONE
                error("Singular Lyapunov equation")
             end
             R[l,l] = UpperTriangular(U)
          end
          if ll > 1
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:dl)
             ubar = view(Wu,j1,1:dl)
             y = view(Wy,j1,1:dl)
             z = view(Wz,j1,1:dl)
             rbar = R[j1,l]
             z = rbar*α' + A[j1,l]*R[l,l]
             # Solve S1*ubar+ubar*β' + z = 0
             S1 = view(A,j1,j1)
             ubar, scale = LAPACK.trsyl!('N','T', S1, β, z)
             rmul!(ubar, -inv(scale))
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
function plyapcs!(A::T1, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Complex{Float64}},Matrix{Complex{Float32}}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end

   T = real(eltype(A))
   ONE = one(T)
   ZERO = zero(T)
   TWO = 2*ONE
   EPS = eps(ONE)*2
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))

   Wr = Array{eltype(A),2}(undef,n,1)
   Wu = Array{eltype(A),2}(undef,n,1)
   Wy = Array{eltype(A),2}(undef,n,1)
   Wz = Array{eltype(A),2}(undef,n,1)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),

      """
      for j = 1:n
          λ = real(A[j,j])
          if λ >= ZERO
             error("A is not stable")
          end
          TEMP = sqrt( -TWO*λ )
          if TEMP < SMIN
             TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular Lyapunov equation")
          end
          α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j < n
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             rbar = view(Wr,ir1,1:1)
             ubar = view(Wu,ir1,1:1)
             y = view(Wy,ir1,1:1)
             #z = view(Wz,ir1,1:1)
             rbar = R[l,j1]'
             #z = rbar*α + A[l,j1]'*R[l,l]
             z = rbar*α + (R[l,l]'*A[l,j1])'
             # Solve S1'*ubar+ubar*β + z = 0
             S1 = view(A,j1,j1)
             ubar, scale = LAPACK.trsyl!('C','N', S1, β, z)
             rmul!(ubar, -inv(scale))
             #ubar = -(LowerTriangular(S1'+β[1,1]*I))\z
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             y = conj(rbar - ubar * α')
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',

      """
      for j = n:-1:1
          λ = real(A[j,j])
          if λ >= ZERO
             error("A is not stable")
          end
          TEMP = sqrt( -TWO*λ  )
          if TEMP < SMIN
             TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular Lyapunov equation")
          end
          α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j > 1
             j -= 1
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:1)
             ubar = view(Wu,j1,1:1)
             y = view(Wy,j1,1:1)
             z = view(Wz,j1,1:1)
             rbar = R[j1,l]
             z = rbar*α' + A[j1,l]*R[l,l]
             # Solve S1*ubar+ubar*β' + z = 0
             S1 = view(A,j1,j1)
             ubar, scale = LAPACK.trsyl!('N','C', S1, β, z)
             rmul!(ubar, -inv(scale))
             #ubar = -(UpperTriangular(S1+β[1,1]'*I))\z
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
"""
    plyapcs!(A,E,R;adj = false)

Solve the positive continuous generalized Lyapunov matrix equation

                op(A)Xop(E)' + op(E)*Xop(A)' + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
The pair `(A,E)` is in a generalized real/complex Schur form and `R` is an upper
triangular matrix. The pencil `A-λE` must have all eigenvalues with negative
real parts. `R` contains on output the solution `U`.
"""
function plyapcs!(A::T1, E::Union{T1,UniformScaling{Bool}}, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Float32},Matrix{Float64}}
   n = LinearAlgebra.checksquare(A)
   if typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)
      plyapcs!(A, R, adj = adj)
      return
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
   end
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end

   T = eltype(A)
   ZERO = zero(T)
   ONE = one(T)
   TWO = 2*ONE
   EPS = eps(ONE)*TWO
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))

   # determine the structure of the generalized real Schur form
   ba = fill(1,n)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   Wr = Array{eltype(A),2}(undef,n,2)
   Wu = Array{eltype(A),2}(undef,n,2)
   Wy = Array{eltype(A),2}(undef,n,2)
   Wz = Array{eltype(A),2}(undef,n,2)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L)*E(L,L) + E(L,L)'*X(L,L)*A(L,L) = -R(L,L)'*R(L,L),

      """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = A[j,j]*E[j,j]
             if λ >= ZERO
                error("A-λE has eigenvalues with non-negative real parts")
             end
             TEMP = sqrt( -TWO*λ )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular generalized Lyapunov equation")
             end
             TEMP = sign(R[j,j])*TEMP
             R[j,j] = R[j,j]/TEMP
             β = A[l,l]/E[j,j]
             α = TEMP/E[j,j]
          else
             R[l,l], scale, β, α = pglyap2(A[l,l], E[l,l], R[l,l], adj = true)
             if scale != ONE
                error("Singular generalized Lyapunov equation")
             end
          end
          if ll < p
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             rbar = view(Wr,ir1,1:dl)
             ubar = view(Wu,ir1,1:dl)
             y = view(Wy,ir1,1:dl)
             #z = view(Wz,ir1,1:dl)
             rbar = R[l,j1]'
             v = (R[l,l]*E[l,j1])'
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             z = rbar*α + (R[l,l]*A[l,j1])'+v*β
             # Solve A[j1,j1]'*ubar+E[j1,j1]'*ubar*β + z = 0
             η = one(β)
             if dl == 2; η[2,1] = eps(ONE)^2; end
             ubar = gsylvs!(A[j1,j1],η,E[j1,j1],β,-z;adjAC=true,adjBD=false)
             R[l,j1] = ubar'
             # update the Cholesky factor R2'*R2 <- R2'*R2 + y'*y
             #y = rbar - (E[j1,j1]'*ubar+E[l,j1]'*R[l,l]') * α'
             y = rbar - (E[j1,j1]'*ubar+v) * α'
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(L,L)*X(L,L)*E(L,L)' + E(L,L)*X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',

      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = A[j,j]*E[j,j]
             if λ >= ZERO
                error("A-λE has eigenvalues with non-negative real parts")
             end
             TEMP = sqrt( -TWO*λ )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular generalized Lyapunov equation")
             end
             TEMP = sign(R[j,j])*TEMP
             R[j,j] = R[j,j]/TEMP
             β = A[l,l]/E[j,j]
             α = TEMP/E[j,j]
          else
             R[l,l], scale, β, α = pglyap2(A[l,l], E[l,l], R[l,l], adj = false)
             if scale != ONE
                error("Singular generalized Lyapunov equation")
             end
          end
          if ll > 1
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:dl)
             ubar = view(Wu,j1,1:dl)
             y = view(Wy,j1,1:dl)
             z = view(Wz,j1,1:dl)
             rbar = R[j1,l]
             v = E[j1,l]*R[l,l]
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             z = rbar*α' + A[j1,l]*R[l,l]+v*β'
             # Solve S1*ubar+ubar*β' + z = 0
             η = one(β)
             if dl == 2; η[2,1] = eps(ONE)^2; end
             ubar = gsylvs!(A[j1,j1],η,E[j1,j1],β,-z;adjAC=false,adjBD=true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar - (E[j1,j1]*ubar+v) * α
             #y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
function plyapcs!(A::T1, E::Union{T1,UniformScaling{Bool}}, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Complex{Float64}},Matrix{Complex{Float32}}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end
   if (typeof(E) == UniformScaling{Bool}) || isempty(E) || (isequal(E,I) && size(E,1) == n)
      plyapcs!(A, R, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   T = real(eltype(A))
   ZERO = zero(T)
   ONE = one(T)
   TWO = 2*ONE
   EPS = eps(ONE)*2
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(max(maximum(abs.(A)),maximum(abs.(E))))

   Wr = Array{eltype(A),2}(undef,n,1)
   Wu = Array{eltype(A),2}(undef,n,1)
   Wy = Array{eltype(A),2}(undef,n,1)
   Wz = Array{eltype(A),2}(undef,n,1)
   η  = complex(fill(ONE,(1,1)))
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner row by row by

      A(L,L)'*X(L,L)*E(L,L) + E(L,L)'*X(L,L)*A(L,L) = -R(L,L)'*R(L,L),

      """
      for j = 1:n
          δ = -TWO*real(A[j,j]'*E[j,j])
          if δ <= ZERO
             error("A-λE has unstable eigenvalues")
          end
          TEMP = sqrt( δ )
          if TEMP < SMIN
             TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular generalized Lyapunov equation")
          end
          TEMP = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          # Form the right-hand side corresponding to (11.6)
          # A = [ A[l,l]   A[l,j1]  ]     E = [ E[l,l]   E[l,j1]  ]
          #     [ 0        A[j1,j1] ]         [ 0        E[j1,j1] ]
          if j < n
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             # Form the right-hand side of (11.12)
             # z = rbar'*α + s'*u11'
             rbar = view(Wr,ir1,1:1)
             ubar = view(Wu,ir1,1:1)
             y = view(Wy,ir1,1:1)
             #z = view(Wz,ir1,1:1)
             rbar = R[l,j1]'
             v = (R[l,l]*E[l,j1])'
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             z = rbar*α + (R[l,l]*A[l,j1])'+v*β
             # Solve A[j1,j1]'*ubar+E[j1,j1]'*ubar*β + z = 0
             ubar = gsylvs!(A[j1,j1],η,E[j1,j1],β,-z;adjAC=true,adjBD=false)
             R[l,j1] = ubar'
             # update the Cholesky factor R2'*R2 <- R2'*R2 + y'*y
             #y = conj(rbar - (E[j1,j1]'*ubar+E[l,j1]'*R[l,l]') * α')
             y = conj(rbar - (E[j1,j1]'*ubar+v) * α')
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
      return R
   else
      """
      The (L,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(L,L)*X(L,L)*E(L,L)' + E(L,L)*X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',

      """
      for j = n:-1:1
          δ = -TWO*real(A[j,j]'*E[j,j])
          if δ <= ZERO
            error("A-λE has unstable eigenvalues")
          end
          TEMP = sqrt( δ )
          if TEMP < SMIN
            TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular generalized Lyapunov equation")
          end
          TEMP = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          # Form the right-hand side corresponding to the dual of (11.6)
          # A = [ A[j1,j1]  A[j1,l]  ]     E = [ E[j1,j1]  E[j1,l]  ]
          #     [ 0         A[l,l]   ]         [ 0         E[l,l]   ]
          if j > 1
             j -= 1
             j1 = 1:j
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:1)
             ubar = view(Wu,j1,1:1)
             y = view(Wy,j1,1:1)
             z = view(Wz,j1,1:1)
             rbar = R[j1,l]
             v = E[j1,l]*R[l,l]
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             z = rbar*α' + A[j1,l]*R[l,l]+v*β'
             # Solve S1*ubar+ubar*β' + z = 0
             ubar = gsylvs!(A[j1,j1],η,E[j1,j1],β,-z;adjAC=false,adjBD=true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar - (E[j1,j1]*ubar+v) * α
             #y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
"""
    plyapds!(A,R;adj = false)

Solve the positive discrete Lyapunov matrix equation

                op(A)Xop(A)' - X + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
`A` is a square real matrix in a real Schur form , or a square complex matrix in a
complex Schur form and `R` is an upper triangular matrix.
`A` must have all eigenvalues with moduli less than one.
`R` contains on output the upper triangular solution `U`.
"""
function plyapds!(A::T1, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Float32},Matrix{Float64}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end

   T = eltype(A)
   ZERO = zero(T)
   ONE = one(T)
   TWO = 2*ONE
   EPS = eps(ONE)*TWO
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))

   # determine the structure of the real Schur form
   ba = fill(1,n)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   Wr = Array{eltype(A),2}(undef,n,2)
   Wu = Array{eltype(A),2}(undef,n,2)
   Wy = Array{eltype(A),2}(undef,n,2)
   Wz = Array{eltype(A),2}(undef,n,2)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L)*A(L,L) - X(L,L) = -R(L,L)'*R(L,L),
      """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = abs(A[j,j])
             if λ >= 1.
                error("A is not convergent")
             end
             TEMP = sqrt( (ONE - λ)*(ONE + λ) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular Lyapunov equation")
             end
             α = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/α
             β = A[l,l]
          else
             u11, scale, β, α = plyap2(A[l,l], R[l,l], adj = true, disc = true)
             if scale != ONE
                error("Singular Lyapunov equation")
             end
             R[l,l] = UpperTriangular(u11)
          end
          if ll < p
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:dl)
             ubar = view(Wu,ir1,1:dl)
             y = view(Wy,ir1,1:dl)
             #z = view(Wz,ir1,1:dl)
             # Form the right-hand side of (10.16)
             # z = rbar*α + s'*u11*β
             rbar = R[l,j1]'
             v = (R[l,l]*A[l,j1])'
             #z = rbar*α + A[l,j1]'*R[l,l]*β
             z = rbar*α + v*β
             # Solve S1'*ubar*β+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = sylvds!(A[j1,j1], -β, z, adjA = true, adjB = false)
             #ubar = sylvds!(S1, -β, z, adjA = true, adjB = false)
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             v += (ubar'*S1)'
             if dl == 1
                y = rbar*β - v*α
             else
                F = qr([α; β])
                vy = [rbar v]*F.Q
                y = vy[:,dl+1:end]
                # alternative formula of Varga
                #y = rbar - (A[l,j1]'*R[l,l]'+S1'*ubar+ubar)*inv(I+β')*α'
             end
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)*X(L,L)*A(L,L)' - X(L,L) = -R(L,L)*R(L,L)',
      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = abs(A[j,j])
             if λ >= 1.
                error("A is not convergent")
             end
             TEMP = sqrt( (ONE - λ)*(ONE + λ) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular Lyapunov equation")
             end
             α = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/α
             β = A[l,l]
          else
             u11, scale, β, α = plyap2(A[l,l], R[l,l], adj = false, disc = true)
             if scale != ONE
                error("Singular Lyapunov equation")
             end
             R[l,l] = UpperTriangular(u11)
          end
          if ll > 1
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:dl)
             ubar = view(Wu,j1,1:dl)
             y = view(Wy,j1,1:dl)
             z = view(Wz,j1,1:dl)
             rbar = R[j1,l]
             v = A[j1,l]*R[l,l]
             #z = rbar*α + A[j1,l]*R[l,l]*β'
             z = rbar*α' + v*β'
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             #ubar = sylvds!(S1, -β, z, adjA = false, adjB = true)
             ubar = sylvds!(A[j1,j1], -β, z, adjA = false, adjB = true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             v += S1*ubar
             if dl == 1
                y = rbar*β - v*α
             else
                F = qr([α'; β'])
                vy = [rbar v]*F.Q
                y = vy[:,dl+1:end]
             end
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
   return UpperTriangular(R)
end
function plyapds!(A::T1, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Complex{Float64}},Matrix{Complex{Float32}}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end

   T = real(eltype(A))
   ONE = one(T)
   EPS = eps(ONE)*2
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))

   Wr = Array{eltype(A),2}(undef,n,1)
   Wu = Array{eltype(A),2}(undef,n,1)
   Wy = Array{eltype(A),2}(undef,n,1)
   Wz = Array{eltype(A),2}(undef,n,1)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L)*A(L,L) - X(L,L) = -R(L,L)'*R(L,L),

      """
      for j = 1:n
          λ = abs(A[j,j])
          if λ >= ONE
             error("A is not convergent")
          end
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          if TEMP < SMIN
             TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular Lyapunov equation")
          end
          α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j < n
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:1)
             ubar = view(Wu,ir1,1:1)
             y = view(Wy,ir1,1:1)
             #z = view(Wz,ir1,1:1)
             # Form the right-hand side of (10.16)
             # z = rbar*α + s'*u11*β
             rbar = R[l,j1]'
             ust = R[l,l]'*A[l,j1]
             #z = rbar*α + A[l,j1]'*R[l,l]*β
             z = rbar*α + ust'*β
             # Solve S1'*ubar*β+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = sylvds!(A[j1,j1], -β, z, adjA = true, adjB = false)
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             y = conj(rbar*β' - (ubar'*UpperTriangular(S1)+ust)' * α')
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)*X(L,L)*A(L,L)' - X(L,L) = -R(L,L)*R(L,L)',
      """
      for j = n:-1:1
          λ = abs(A[j,j])
          if λ >= ONE
            error("A is not convergent")
          end
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          if TEMP < SMIN
            TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular Lyapunov equation")
          end
          α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j > 1
             j -= 1
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             rbar = view(Wr,j1,1:1)
             ubar = view(Wu,j1,1:1)
             y = view(Wy,j1,1:1)
             z = view(Wz,j1,1:1)
             rbar = R[j1,l]
             ust = A[j1,l]*R[l,l]
             #z = rbar*α + A[j1,l]*R[l,l]*β'
             z = rbar*α' + ust*β'
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = sylvds!(A[j1,j1], -β, z, adjA = false, adjB = true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar*β - (UpperTriangular(S1)*ubar+ust) * α
             #y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
"""
    plyapds!(A,E,R;adj = false)

Solve the positive discrete generalized Lyapunov matrix equation

                op(A)Xop(A)' - op(E)Xop(E) + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
The pair `(A,E)` of square real or complex matrices is in a generalized Schur form
and `R` is an upper triangular matrix. `A-λE` must have all eigenvalues with
moduli less than one. `R` contains on output the upper triangular solution `U`.
"""
function plyapds!(A::T1, E::Union{T1,UniformScaling{Bool}}, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Float32},Matrix{Float64}}
   """
   The method of [1] for the discrete case is implemented.

   [1] Penzl, T.
       Numerical solution of generalized Lyapunov equations.
       Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end
   if typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)
      plyapds!(A, R, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   T = eltype(A)
   ONE = one(T)
   EPS = eps(ONE)*2
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(max(maximum(abs.(A)),maximum(abs.(E))))

   # determine the structure of the real Schur form
   ba = fill(1,n)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   Wr = Array{eltype(A),2}(undef,n,2)
   Wu = Array{eltype(A),2}(undef,n,2)
   Wy = Array{eltype(A),2}(undef,n,2)
   Wz = Array{eltype(A),2}(undef,n,2)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L)*A(L,L) - E(L,L)'*X(L,L)*E(L,L) = -R(L,L)'*R(L,L),
      """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = abs(A[j,j])
             if abs(A[j,j]) >= abs(E[j,j])
                error("A-λE must have only eigenvalues with moduli less than one")
             end
             TEMP = sqrt( real((E[j,j] - A[j,j])*(E[j,j] + A[j,j])) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular generalized discrete Lyapunov equation")
             end
             TEMP = sign(R[j,j])*TEMP
             R[j,j] = R[j,j]/TEMP
             l = j:j
             β = A[l,l]/E[j,j]
             α = TEMP/E[j,j]
          else
             u11, scale, β, α = pglyap2(A[l,l], E[l,l], R[l,l], adj = true, disc = true)
             if scale != ONE
                error("Singular generalized discrete Lyapunov equation")
             end
             R[l,l] = UpperTriangular(u11)
          end
          if ll < p
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:dl)
             ubar = view(Wu,ir1,1:dl)
             y = view(Wy,ir1,1:dl)
             #z = view(Wz,ir1,1:dl)
             # Form the right-hand side of (22) in [1]
             #z = rbar*α + A[l,j1]'*R[l,l]*β
             rbar = R[l,j1]'
             v = (R[l,l]*A[l,j1])'
             vst = (R[l,l]*E[l,j1])'
             #z = rbar*α + A[l,j1]'*R[l,l]'*β - E[l,j1]'*R[l,l]'
             z = rbar*α + v*β - vst
             # Solve S1'*ubar*β+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = gsylvs!(A[j1,j1], β, E[j1,j1], -one(β),-z, adjAC = true, adjBD = false)
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             v += (ubar'*S1)'
             if dl == 1
                y = rbar*β - v*α
             else
                F = qr([α; β])
                vy = [rbar v]*F.Q
                y = vy[:,dl+1:end]
                # alternative formula of Varga
                #y = rbar - (A[l,j1]'*R[l,l]'+S1'*ubar+ubar)*inv(I+β')*α'
             end
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)*X(L,L)*A(L,L)' - E(L,L)*X(L,L)*E(L,L)' = -R(L,L)*R(L,L)',
      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = abs(A[j,j])
             if abs(A[j,j]) >= abs(E[j,j])
                error("A-λE must have only eigenvalues with moduli less than one")
             end
             TEMP = sqrt( real((E[j,j] - A[j,j])*(E[j,j] + A[j,j])) )
             if TEMP < SMIN
                TEMP  = SMIN
             end
             DR = abs( R[j,j] )
             if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
                error("Singular generalized discrete Lyapunov equation")
             end
             TEMP = sign(R[j,j])*TEMP
             R[j,j] = R[j,j]/TEMP
             l = j:j
             β = A[l,l]/E[j,j]
             α = TEMP/E[j,j]
          else
             u11, scale, β, α = pglyap2(A[l,l], E[l,l], R[l,l], adj = false, disc = true)
             if scale != ONE
                error("Singular generalized discrete Lyapunov equation")
             end
             R[l,l] = UpperTriangular(u11)
          end
          if ll > 1
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,1:dl)
             ubar = view(Wu,j1,1:dl)
             y = view(Wy,j1,1:dl)
             z = view(Wz,j1,1:dl)
             rbar = R[j1,l]
             v = A[j1,l]*R[l,l]
             vst = E[j1,l]*R[l,l]
             #z = rbar*α + A[j1,l]*R[l,l]*β'
             z = rbar*α' + v*β' - vst
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = gsylvs!(A[j1,j1], β, E[j1,j1], -one(β),-z, adjAC = false, adjBD = true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             v += S1*ubar
             if dl == 1
                y = rbar*β - v*α
             else
                F = qr([α'; β'])
                vy = [rbar v]*F.Q
                y = vy[:,dl+1:end]
             end
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
   return UpperTriangular(R)
end
function plyapds!(A::T1, E::Union{T1,UniformScaling{Bool}}, R::UpperTriangular; adj = false)  where T1<:Union{Matrix{Complex{Float64}},Matrix{Complex{Float32}}}
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(R) != n
      throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   end
   if (typeof(E) == UniformScaling{Bool}) || isempty(E) || (isone(E) && size(E,1) == n)
      plyapds!(A, R, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   T = real(eltype(A))
   ONE = one(T)
   EPS = eps(ONE)*2
   T == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*n*n / EPS
   BIGNUM = ONE / small
   SMIN = eps(max(maximum(abs.(A)),maximum(abs.(E))))

   Wr = Array{eltype(A),2}(undef,n,1)
   Wu = Array{eltype(A),2}(undef,n,1)
   Wy = Array{eltype(A),2}(undef,n,1)
   Wz = Array{eltype(A),2}(undef,n,1)
   if adj
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)'*X(L,L)*A(L,L) - E(L,L)'*X(L,L)*E(L,L) = -R(L,L)'*R(L,L),

      """
      for j = 1:n
          λ = abs(A[j,j])
          if abs(A[j,j]) >= abs(E[j,j])
             error("A-λE must have only eigenvalues with moduli less than one")
          end
          TEMP = sqrt( real((E[j,j]' - A[j,j]')*(E[j,j] + A[j,j])) )
          if TEMP < SMIN
             TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
             error("Singular generalized discrete Lyapunov equation")
          end
          TEMP = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          if j < n
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:1)
             ubar = view(Wu,ir1,1:1)
             y = view(Wy,ir1,1:1)
             #z = view(Wz,ir1,1:1)
             # Form the right-hand side of (10.16)
             # z = rbar*α + s'*u11*β
             rbar = R[l,j1]'
             ust = R[l,l]'*A[l,j1]
             vst = R[l,l]'*E[l,j1]
             #z = rbar*α + A[l,j1]'*R[l,l]*β
             z = rbar*α + ust'*β - vst'
             # Solve S1'*ubar*β+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = gsylvs!(A[j1,j1], β, E[j1,j1], -one(β),-z, adjAC = true, adjBD = false)
             R[l,j1] = ubar'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             y = conj(rbar*β' - (ubar'*UpperTriangular(S1)+ust)' * α')
             RR = view(R,j1,j1)
             qrupdate!(RR, y)
          end
      end
   else
      """
      The (L,L)th block of X is determined starting from
      upper-left corner column by column by

      A(L,L)*X(L,L)*A(L,L)' - E(L,L)*X(L,L)*E(L,L)' = -R(L,L)*R(L,L)',
      """
      for j = n:-1:1
          λ = abs(A[j,j])
          if abs(A[j,j]) >= abs(E[j,j])
            error("A-λE must have only eigenvalues with moduli less than one")
          end
          TEMP = sqrt( real((E[j,j]' - A[j,j]')*(E[j,j] + A[j,j])) )
          if TEMP < SMIN
            TEMP  = SMIN
          end
          DR = abs( R[j,j] )
          if TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP
            error("Singular generalized discrete Lyapunov equation")
          end
          TEMP = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          if j > 1
             j -= 1
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             rbar = view(Wr,j1,1:1)
             ubar = view(Wu,j1,1:1)
             y = view(Wy,j1,1:1)
             z = view(Wz,j1,1:1)
             rbar = R[j1,l]
             ust = A[j1,l]*R[l,l]
             vst = E[j1,l]*R[l,l]
             #z = rbar*α + A[j1,l]*R[l,l]*β'
             z = rbar*α' + ust*β'-vst
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             ubar = gsylvs!(A[j1,j1], β, E[j1,j1], -one(β),-z, adjAC = false, adjBD = true)
             R[j1,l] = ubar
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             y = rbar*β - (UpperTriangular(S1)*ubar+ust) * α
             #y = rbar - ubar*α
             RR = view(R,j1,j1)
             rqupdate!(RR, y)
          end
       end
   end
end
"""
    plyap2(A, R; adj = false, disc = false) -> (U, scale, β, α)

Solve for the Cholesky factor  `U`  of  `X`,

     op(U)*op(U)' = X,

where  `U`  is a two-by-two upper triangular matrix, either the
continuous-time two-by-two Lyapunov equation

      op(A)*X + X*op(A)' = -scale^2*op(R)*op(R)',

when disc = false, or the discrete-time two-by-two Lyapunov equation

      op(A)*X*op(A)' - X = -scale^2*op(R)*op(R)',

when `disc = true`, where `op(K) = K` if `adj = false` or `op(K) = K'`
if `adj = true`,  `A`  is a two-by-two matrix with complex conjugate eigenvalues,
`R`  is a two-by-two upper triangular matrix, and `scale` is
an output scale factor, set less than or equal to `1` to avoid overflow in  `X`.
The routine also computes two matrices, `β` and `α`, so that

      U*A = β*U  and  U*α = scale^2 *R,  if  adj = false, or

      β*U = U*A  and  α*U = scale^2 *R,  if  adj = true,

which are used by the general Lyapunov solver.

In the continuous-time case  `A`  must be stable, so that its
eigenvalues must have strictly negative real parts.
In the discrete-time case  `A`  must be convergent, that is, its eigenvalues
must have moduli less than one. These conditions are checked and
an error message is issued if not fulfilled.

If the Lyapunov equation is (nearly) singular, then perturbed values are
used to solve the equation. If `disc = false`, this means that
the matrix `A` has computed eigenvalues with negative real parts,
it is only just stable in the sense that small perturbations in `A` can make
one or more of the eigenvalues have a non-negative real part.
If `disc = true`, this means that while the  matrix `A` has computed eigenvalues
inside the unit circle, it is nevertheless only just convergent, in
the sense that small perturbations in `A` can make one or more of the
eigenvalues lie outside the unit circle.
"""
function plyap2(A::T, R::T; adj = false, disc = false) where T<:Union{Matrix{Float64},Matrix{Float32}}
   """
   This function is based on the SLICOT routine SB03OY, which implements the
   the LAPACK scheme for solving 2-by-2 Sylvester equations, adapted in [1]
   for 2-by-2 Lyapunov equations, but directly computing the Cholesky factor
   of the solution.

   [1] Hammarling S. J.
       Numerical solution of the stable, non-negative definite Lyapunov equation.
       IMA J. Num. Anal., 2, pp. 303-325, 1982.
   """
   T1 = eltype(A)
   ZERO = zero(T1)
   ONE = one(T1)
   TWO = 2*ONE
   FOUR = 4*ONE
   EPS = eps(ONE)*TWO
   T1 == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*FOUR / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))
   scale = ONE
   noadj = !adj
   U = similar(R)
   U[2,1] = ZERO
   β = similar(A)
   α = similar(R)
   S11 = A[1,1]
   S12 = A[1,2]
   S21 = A[2,1]
   S22 = A[2,2]
   TEMPR, TEMPI, E1, E2, CSP, CSQ = LapackUtil.lanv2( S11, S12, S21, S22)
   if TEMPI == ZERO
      error("A has real eigenvalues")
   end
   ABSB = hypot(E1,E2)
   if disc
      if ABSB  >= ONE
         error("A is not convergent")
      end
   else
      if E1 >= ZERO
         error("A is not stable")
      end
   end
#     Compute the cos and sine that define  Qhat.  The sine is real.
   TEMP1 = S11 - E1
   if noadj
      TEMP2 = -E2
   else
      TEMP2 =  E2
   end
   CSQR, CSQI, SNQ = cgivens2( TEMP1, TEMP2, S21, small )

   #     beta in (6.9) is given by  beta = E1 + i*E2,  compute  t.
   TEMP1 = CSQR*S12 - SNQ*S11
   TEMP2 = CSQI*S12
   TEMPR   = CSQR*S22 - SNQ*S21
   TEMPI   = CSQI*S22
   T1      = CSQR*TEMP1 - CSQI*TEMP2 + SNQ*TEMPR
   T2      = CSQR*TEMP2 + CSQI*TEMP1 + SNQ*TEMPI
   if noadj
#                                                         (     -- )
#        Case op(M) = M.  Note that the modified  R  is  ( p3  p2 ).
#                                                         ( 0   p1 )
#
#        Compute the cos and sine that define  Phat.
#
      TEMP1 =  CSQR*R[2,2] - SNQ*R[1,2]
      TEMP2 = -CSQI*R[2,2]
      CSPR, CSPI, SNP, P1 = cgivens2( TEMP1, TEMP2, -SNQ*R[1,1], small )
#    Compute p1, p2 and p3 of the relation corresponding to (6.11).
      TEMP1 =  CSQR*R[1,2] + SNQ*R[2,2]
      TEMP2 = -CSQI*R[1,2]
      TEMPR   =  CSQR*R[1,1]
      TEMPI   = -CSQI*R[1,1]
      P2R     =  CSPR*TEMP1 - CSPI*TEMP2 + SNP*TEMPR
      P2I     = -CSPR*TEMP2 - CSPI*TEMP1 - SNP*TEMPI
      P3R     =  CSPR*TEMPR   + CSPI*TEMPI   - SNP*TEMP1
      P3I     =  CSPR*TEMPI   - CSPI*TEMPR   - SNP*TEMP2
   else
#     Case op(M) = M'.
#     Compute the cos and sine that define  Phat.
      TEMP1 = CSQR*R[1,1] + SNQ*R[1,2]
      TEMP2 = CSQI*R[1,1]
      CSPR, CSPI, SNP, P1 = cgivens2( TEMP1, TEMP2, SNQ*R[2,2], small  )
#     Compute p1, p2 and p3 of (6.11).
      TEMP1 = CSQR*R[1,2] - SNQ*R[1,1]
      TEMP2 = CSQI*R[1,2]
      TEMPR   = CSQR*R[2,2]
      TEMPI   = CSQI*R[2,2]
      P2R     = CSPR*TEMP1 - CSPI*TEMP2 + SNP*TEMPR
      P2I     = CSPR*TEMP2 + CSPI*TEMP1 + SNP*TEMPI
      P3R     = CSPR*TEMPR   + CSPI*TEMPI   - SNP*TEMP1
      P3I     = CSPI*TEMPR   - CSPR*TEMPI   + SNP*TEMP2
   end
#  Make  p3  real by multiplying by  conjg ( p3 )/abs( p3 )  to give
#  p3 := abs( p3 ).
   if P3I == ZERO
      P3  = abs( P3R )
      DP1 = copysign( ONE, P3R )
      DP2 = ZERO
   else
      P3  =  hypot(P3R,P3I)
      DP1 =  P3R/P3
      DP2 = -P3I/P3
   end
#  Now compute the quantities v1, v2, v3 and y in (6.13) - (6.15),
#  or (10.23) - (10.25). Care is taken to avoid overflows.
   if disc
      ALPHA = sqrt( abs( ONE - ABSB )*( ONE + ABSB ) )
   else
      ALPHA = sqrt( abs( TWO*E1 ) )
   end

   SCALOC = ONE
   if ALPHA < SMIN
      ALPHA = SMIN
      # INFO = 1
   end
   ABST = abs( P1 )
   if ALPHA < ONE && ABST > ONE && ABST > BIGNUM*ALPHA
      SCALOC = ONE / ABST
   end
   if SCALOC !== ONE
      P1    = SCALOC*P1
      P2R = SCALOC*P2R
      P2I = SCALOC*P2I
      P3    = SCALOC*P3
      scale = SCALOC*scale
   end
   V1 = P1/ALPHA

   if disc
      G1 = (ONE - E1 )*( ONE + E1 ) + E2*E2
      G2 = -TWO*E1*E2
      ABSG =  hypot(G1,G2)
      SCALOC = ONE
      if ABSG < SMIN
         ABSG = SMIN
         #INFO = 1
      end
      TEMP1 = ALPHA*P2R + V1*( E1*T1 - E2*T2 )
      TEMP2 = ALPHA*P2I + V1*( E1*T2 + E2*T1 )
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      if ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG
            SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         V1      = SCALOC*V1
         TEMP1   = SCALOC*TEMP1
         TEMP2   = SCALOC*TEMP2
         P1      = SCALOC*P1
         P2R     = SCALOC*P2R
         P2I     = SCALOC*P2I
         P3      = SCALOC*P3
         scale   = SCALOC*scale
      end
      TEMP1 = TEMP1/ABSG
      TEMP2 = TEMP2/ABSG

      SCALOC = ONE
      V2R    = G1*TEMP1 + G2*TEMP2
      V2I    = G1*TEMP2 - G2*TEMP1
      ABST   = max( abs( V2R ), abs( V2I ) )
      if ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG
         SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         V1    = SCALOC*V1
         V2R   = SCALOC*V2R
         V2I   = SCALOC*V2I
         P1    = SCALOC*P1
         P2R   = SCALOC*P2R
         P2I   = SCALOC*P2I
         P3    = SCALOC*P3
         scale = SCALOC*scale
      end
      V2R = V2R/ABSG
      V2I = V2I/ABSG

      SCALOC  = ONE
      TEMP1 = P1*T1 - TWO*E2*P2I
      TEMP2 = P1*T2 + TWO*E2*P2R
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      if ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG
         SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         TEMP1 = SCALOC*TEMP1
         TEMP2 = SCALOC*TEMP2
         V1      = SCALOC*V1
         V2R   = SCALOC*V2R
         V2I   = SCALOC*V2I
         P3      = SCALOC*P3
         scale   = SCALOC*scale
      end
      TEMP1 = TEMP1/ABSG
      TEMP2 = TEMP2/ABSG

      SCALOC  = ONE
      YR  = -( G1*TEMP1 + G2*TEMP2 )
      YI  = -( G1*TEMP2 - G2*TEMP1 )
      ABST    = max( abs( YR ), abs( YI ) )
      if ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG
         SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         YR  = SCALOC*YR
         YI  = SCALOC*YI
         V1    = SCALOC*V1
         V2R = SCALOC*V2R
         V2I = SCALOC*V2I
         P3    = SCALOC*P3
         scale = SCALOC*scale
      end
      YR = YR/ABSG
      YI = YI/ABSG
   else

      SCALOC = ONE
      if ABSB < SMIN
         ABSB = SMIN
         #INFO = 1
      end
      TEMP1 = ALPHA*P2R + V1*T1
      TEMP2 = ALPHA*P2I + V1*T2
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      if ABSB < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSB
         SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         V1      = SCALOC*V1
         TEMP1   = SCALOC*TEMP1
         TEMP2   = SCALOC*TEMP2
         P2R     = SCALOC*P2R
         P2I     = SCALOC*P2I
         P3      = SCALOC*P3
         scale   = SCALOC*scale
      end
      TEMP1 = TEMP1/( TWO*ABSB )
      TEMP2 = TEMP2/( TWO*ABSB )
      SCALOC  = ONE
      V2R     =  -(E1*TEMP1 + E2*TEMP2)
      V2I     =  -(E1*TEMP2 - E2*TEMP1)
      ABST = max( abs( V2R ), abs( V2I ) )
      if ABSB < ONE  &&  ABST > ONE &&  ABST > BIGNUM*ABSB
         SCALOC = ONE / ABST
      end
      if SCALOC !== ONE
         V1    = SCALOC*V1
         V2R   = SCALOC*V2R
         V2I   = SCALOC*V2I
         P2R   = SCALOC*P2R
         P2I   = SCALOC*P2I
         P3    = SCALOC*P3
         scale = SCALOC*scale
      end
      V2R = V2R/ABSB
      V2I = V2I/ABSB
      YR  = P2R - ALPHA*V2R
      YI  = P2I - ALPHA*V2I
   end

   SCALOC = ONE
   V3     = hypot3(P3,YR,YI)
   if ALPHA < ONE  &&  V3 > ONE && V3 > BIGNUM*ALPHA
      SCALOC = ONE / V3
   end
   if SCALOC !== ONE
      V1    = SCALOC*V1
      V2R = SCALOC*V2R
      V2I = SCALOC*V2I
      V3    = SCALOC*V3
      P3    = SCALOC*P3
      scale = SCALOC*scale
   end
   V3 = V3/ALPHA

   if noadj
#     Case op(M) = M.
#     Form  X = conjg( Qhat' )*v11.
      X11R   =  CSQR*V3
      X11I   =  CSQI*V3
      X21R   =  SNQ*V3
      X21I   =  ZERO
      X12R   =  CSQR*V2R+CSQI*V2I-SNQ*V1
      X12I   = -CSQR*V2I+CSQI*V2R
      X22R   =  CSQR*V1 + SNQ*V2R
      X22I   = -CSQI*V1 - SNQ*V2I
#     Obtain u11 from the RQ-factorization of X. The conjugate of
#     X22 should be taken.
      X22I = -X22I
      CSTR, CSTI, SNT, TMP = cgivens2( X22R, X22I, X21R, small )
      U[2,2] = TMP
      U[1,2] = CSTR*X12R - CSTI*X12I + SNT*X11R
      TEMPR  = CSTR*X11R + CSTI*X11I - SNT*X12R
      TEMPI  = CSTR*X11I - CSTI*X11R - SNT*X12I
      if TEMPI == ZERO
         U[1,1] = abs( TEMPR )
         DT1    = copysign( ONE, TEMPR )
         DT2    = ZERO
      else
         U[1,1] =  hypot(TEMPR,TEMPI)
         DT1    =  TEMPR/U[1,1]
         DT2    = -TEMPI/U[1,1]
      end
   else
#     Case op(M) = M'.
#     Now form  X = v11*conjg( Qhat' ).
      X11R   =  CSQR*V1 - SNQ*V2R
      X11I   = -CSQI*V1 + SNQ*V2I
      X21R   = -SNQ*V3
      X21I   =  ZERO
      X12R   =  CSQR*V2R + CSQI*V2I + SNQ*V1
      X12I   = -CSQR*V2I + CSQI*V2R
      X22R   =  CSQR*V3
      X22I   =  CSQI*V3
#     Obtain u11 from the QR-factorization of X.
      CSTR, CSTI, SNT, TMP = cgivens2( X11R, X11I, X21R, small  )
      U[1,1] = TMP
      U[1,2] = CSTR*X12R + CSTI*X12I + SNT*X22R
      TEMPR  = CSTR*X22R - CSTI*X22I - SNT*X12R
      TEMPI  = CSTR*X22I + CSTI*X22R - SNT*X12I
      if TEMPI == ZERO
         U[2,2] = abs( TEMPR )
         DT1    = copysign( ONE, TEMPR )
         DT2    = ZERO
      else
         U[2,2] =  hypot(TEMPR,TEMPI)
         DT1    =  TEMPR/U[2,2]
         DT2    = -TEMPI/U[2,2]
      end
   end
#  The computations below are not needed when β and α are not
#  useful. Compute delta, eta and gamma as in (6.21) or (10.26).
   if abs( YR ) < small  && abs( YI ) <= small
      DELTA1 = ZERO
      DELTA2 = ZERO
      GAMMA1 = ZERO
      GAMMA2 = ZERO
      ETA = ALPHA
   else
      DELTA1 =  YR/V3
      DELTA2 =  YI/V3
      GAMMA1 =  -ALPHA*DELTA1
      GAMMA2 =  -ALPHA*DELTA2
      ETA = P3/V3
      if disc
         TEMPR  = E1*DELTA1 - E2*DELTA2
         DELTA2 = E1*DELTA2 + E2*DELTA1
         DELTA1 = TEMPR
      end
   end
   if noadj
#     Case op(M) = M.
#     Find  X = conjg( That' )*( inv( v11 )*s11hat*v11 ).
#     ( Defer the scaling.)
      X11R =  CSTR*E1 + CSTI*E2
      X11I = -CSTR*E2 + CSTI*E1
      X21R =  SNT*E1
      X21I = -SNT*E2
      X12R =  CSTR*GAMMA1 + CSTI*GAMMA2 - SNT*E1
      X12I = -CSTR*GAMMA2 + CSTI*GAMMA1 - SNT*E2
      X22R =  CSTR*E1 + CSTI*E2 + SNT*GAMMA1
      X22I =  CSTR*E2 - CSTI*E1 - SNT*GAMMA2
#     Now find  B = X*That. ( Include the scaling here.)
      β[1,1] = CSTR*X11R + CSTI*X11I - SNT*X12R
      TEMPR  = CSTR*X21R + CSTI*X21I - SNT*X22R
      TEMPI  = CSTR*X21I - CSTI*X21R - SNT*X22I
      β[2,1] = DT1*TEMPR   - DT2*TEMPI
      TEMPR  = CSTR*X12R - CSTI*X12I + SNT*X11R
      TEMPI  = CSTR*X12I + CSTI*X12R + SNT*X11I
      β[1,2] = DT1*TEMPR   + DT2*TEMPI
      β[2,2] = CSTR*X22R - CSTI*X22I + SNT*X21R
#     Form  X = ( inv( v11 )*p11 )*conjg( Phat' ).
      TEMPR  =  DP1*ETA
      TEMPI  = -DP2*ETA
      X11R =  CSPR*TEMPR - CSPI*TEMPI + SNP*DELTA1
      X11I =  CSPR*TEMPI + CSPI*TEMPR - SNP*DELTA2
      X21R =  SNP*ALPHA
      X12R = -SNP*TEMPR + CSPR*DELTA1 - CSPI*DELTA2
      X12I = -SNP*TEMPI - CSPR*DELTA2 - CSPI*DELTA1
      X22R =  CSPR*ALPHA
      X22I = -CSPI*ALPHA
#     Finally form  A = conjg( That' )*X.
      TEMPR  = CSTR*X11R - CSTI*X11I - SNT*X21R
      TEMPI  = CSTR*X22I + CSTI*X22R
      α[1,1] = DT1*TEMPR   + DT2*TEMPI
      TEMPR  = CSTR*X12R - CSTI*X12I - SNT*X22R
      TEMPI  = CSTR*X12I + CSTI*X12R - SNT*X22R
      α[1,2] = DT1*TEMPR   + DT2*TEMPI
      α[2,1] = ZERO
      α[2,2] = CSTR*X22R + CSTI*X22I + SNT*X12R
   else
#     Case op(M) = M'.
#     Find  X = That*( v11*s11hat*inv( v11 ) ). ( Defer the scaling.)
      X11R =  CSTR*E1 + CSTI*E2
      X11I =  CSTR*E2 - CSTI*E1
      X21R = -SNT*E1
      X21I = -SNT*E2
      X12R =  CSTR*GAMMA1 - CSTI*GAMMA2 + SNT*E1
      X12I = -CSTR*GAMMA2 - CSTI*GAMMA1 - SNT*E2
      X22R =  CSTR*E1 + CSTI*E2 - SNT*GAMMA1
      X22I = -CSTR*E2 + CSTI*E1 + SNT*GAMMA2
#     Now find  B = X*conjg( That' ). ( Include the scaling here.)
      β[1,1] = CSTR*X11R - CSTI*X11I + SNT*X12R
      TEMPR  = CSTR*X21R - CSTI*X21I + SNT*X22R
      TEMPI  = CSTR*X21I + CSTI*X21R + SNT*X22I
      β[2,1] = DT1*TEMPR   - DT2*TEMPI
      TEMPR  = CSTR*X12R + CSTI*X12I - SNT*X11R
      TEMPI  = CSTR*X12I - CSTI*X12R - SNT*X11I
      β[1,2] = DT1*TEMPR   + DT2*TEMPI
      β[2,2] = CSTR*X22R + CSTI*X22I - SNT*X21R
#     Form  X = Phat*( p11*inv( v11 ) ).
      TEMPR  =  DP1*ETA
      TEMPI  = -DP2*ETA
      X11R =  CSPR*ALPHA
      X11I =  CSPI*ALPHA
      X21R =  SNP*ALPHA
      X12R =  CSPR*DELTA1 + CSPI*DELTA2 - SNP*TEMPR
      X12I = -CSPR*DELTA2 + CSPI*DELTA1 - SNP*TEMPI
      X22R =  CSPR*TEMPR + CSPI*TEMPI + SNP*DELTA1
      X22I =  CSPR*TEMPI - CSPI*TEMPR - SNP*DELTA2
#     Finally form  A = X*conjg( That' ).
      α[1,1] = CSTR*X11R - CSTI*X11I + SNT*X12R
      α[2,1] = ZERO
      α[1,2] = CSTR*X12R + CSTI*X12I - SNT*X11R
      TEMPR  = CSTR*X22R + CSTI*X22I - SNT*X21R
      TEMPI  = CSTR*X22I - CSTI*X22R
      α[2,2] = DT1*TEMPR   + DT2*TEMPI
   end

   if scale !== ONE
      α[1,1] = scale*α[1,1]
      α[1,2] = scale*α[1,2]
      α[2,2] = scale*α[2,2]
   end
   return U, scale, β, α
end
"""
    pglyap2(A, E, R; adj = false, disc = false) -> (U, scale, β, α)

Solve for the Cholesky factor  `U`  of  `X`,

     op(U)*op(U)' = X,

where  `U`  is a two-by-two upper triangular matrix, either the
continuous-time two-by-two generalized Lyapunov equation

      op(A)*X*op(E)' + op(E)*X*op(A)' = -scale^2*op(R)*op(R)',

when disc = false, or the discrete-time two-by-two Lyapunov equation

      op(A)*X*op(A)' - op(E)*X*op(E)' = -scale^2*op(R)*op(R)',

when `disc = true`, where `op(K) = K` if `adj = false` or `op(K) = K'`
if `adj = true`,  `A` and `E` are two-by-two matrices such that the pencil `A-λE`
has complex conjugate eigenvalues, `R`  is a two-by-two upper triangular matrix,
and `scale` is an output scale factor, set less than or equal to one
to avoid overflow in  `X`.
The routine also computes two matrices, `β` and `α`, so that,
for `adj = true`:

      β*U*E = U*A
      α*U*E/scale = scale*R

for `adj = false`:

      E*U*β = A*U
      E*U*α/scale = scale*R
      α = R'*inv(E')*inv(U')

which are used by the general Lyapunov solver.

The pencil `A-λE` must have a pair of complex conjugate eigenvalues.
In the continuous-time case the eigenvalues must have strictly negative
real parts, while n the discrete-time case the eigenvalues must have
moduli less than unity. These conditions are checked and
an error message is issued if not fulfilled.

If the Lyapunov equation is (nearly) singular, then perturbed values are
used to solve the equation. If `disc = false`, this means that
the pencil `(A-λE)` has computed eigenvalues with negative real parts,
it is only just stable in the sense that small perturbations in `A` or `E`
can make one or both of the eigenvalues have a non-negative real part.
If `disc = true`, this means that while the pencil `(A-λE)` has computed
eigenvalues inside the unit circle, it is nevertheless only just convergent, in
the sense that small perturbations in `A` or `E`  can make one or more of the
eigenvalues lie outside the unit circle.
"""
function pglyap2(A::TT, E::TT, R::TT; adj = false, disc = false) where TT<:Union{Matrix{Float64},Matrix{Float32}}
   """
   This function is based on the SLICOT routine SG03BX, which implements the
   generalization of the method due to Hammarling ([1], section 6) for Lyapunov
   equations of order 2. A more detailed description is given in [2].

   [1] Hammarling S. J.
       Numerical solution of the stable, non-negative definite Lyapunov equation.
       IMA J. Num. Anal., 2, pp. 303-325, 1982.
   [2] Penzl, T.
       Numerical solution of generalized Lyapunov equations.
       Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   """
   T1 = eltype(A)
   ZERO = zero(T1)
   ONE = one(T1)
   TWO = 2*ONE
   FOUR = 4*ONE
   EPS = eps(ONE)*TWO
   T1 == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   small = SMLNUM*FOUR / EPS
   BIGNUM = ONE / small
   SMIN = eps(maximum(abs.(A)))
   scale = ONE
   noadj = !adj
   ISCONT = !disc
   U = similar(R)
   U[2,1] = ZERO
   β = similar(A)
   AA  = similar(A)
   EE  = similar(E)
   RR  = similar(R)
   QBR = similar(A)
   QBI = similar(A)
   QUR = similar(A)
   QUI = similar(A)
   UR = similar(A)
   UI = similar(A)
   M1R = similar(A)
   M1I = similar(A)
   M2R = similar(A)
   M2I = similar(A)
   β = similar(A)
   α = similar(R)
   # Make copies of A, E, and B.
   AA = copy(A)
   EE = copy(E)
   RR = copy(R)
   if noadj
      V = AA[1,1]
      AA[1,1] = AA[2,2]
      AA[2,2] = V
      V = EE[1,1]
      EE[1,1] = EE[2,2]
      EE[2,2] = V
      V = RR[1,1]
      RR[1,1] = RR[2,2]
      RR[2,2] = V
   end
   scale1, scale2, LAMR, W, LAMI = LapackUtil.lag2(AA,EE,small)
   if LAMI == ZERO
      error("The pair (A,E) has real generalized eigenvalues")
   end
   # Compute right orthogonal transformation matrix Q.
   @inbounds CR, CI, SR, SI, L =  cgivensc2( scale1*AA[1,1] - EE[1,1]*LAMR,
                                   -EE[1,1]*LAMI, scale1*AA[2,1], ZERO, small )
   QR = [ CR SR; -SR CR ]
   QI = [ -CI  -SI; -SI CI ]
   #  A := Q * A
   AR = QR*AA
   AI = QI*AA
   #  E := Q * E
   ER = QR*EE
   EI = QI*EE
   # Compute left orthogonal transformation matrix Z.
   @inbounds CR, CI, SR, SI, L =  cgivensc2( ER[2,2], EI[2,2], ER[2,1], EI[2,1], small )
   ZR =  [ CR SR; -SR CR ]
   ZI =  [ CI -SI; -SI -CI ]
   # E := E * Z
   @inbounds TR = ER[1:1,:]*ZR - EI[1:1,:]*ZI
   @inbounds TI = ER[1:1,:]*ZI + EI[1:1,:]*ZR
   @inbounds ER[1:1,:] = TR
   @inbounds EI[1:1,:] = TI
   ER[2,1] = ZERO
   ER[2,2] = L
   EI[2,1] = ZERO
   EI[2,2] = ZERO

   # Make main diagonal entries of E real and positive.
   @inbounds V = hypot( ER[1,1], EI[1,1] )
   @inbounds XR, XI = LapackUtil.ladiv(V, ZERO, ER[1,1], EI[1,1])
   ER[1,1] = V
   EI[1,1] = ZERO
   YR = ZR[1,1]
   YI = ZI[1,1]
   ZR[1,1] = XR*YR - XI*YI
   ZI[1,1] = XR*YI + XI*YR
   YR = ZR[2,1]
   YI = ZI[2,1]
   ZR[2,1] = XR*YR - XI*YI
   ZI[2,1] = XR*YI + XI*YR
   # A := A * Z
   @inbounds TR = AR*ZR - AI*ZI
   @inbounds AI = AI*ZR + AR*ZI
   @inbounds AR = TR
   # End of QZ-step.
   @inbounds BR = RR*ZR
   @inbounds BI = RR*ZI
   # Overwrite B with the upper triangular matrix of its
   # QR-factorization. The elements on the main diagonal are real
   # and non-negative.
   @inbounds CR, CI, SR, SI, L = cgivensc2( BR[1,1], BI[1,1], BR[2,1], BI[2,1], small )
   QBR[1,1] =  CR
   QBR[1,2] =  SR
   QBR[2,1] = -SR
   QBR[2,2] =  CR
   QBI[1,1] = -CI
   QBI[1,2] = -SI
   QBI[2,1] = -SI
   QBI[2,2] =  CI
   @inbounds TR = QBR*BR[:,2] - QBI*BI[:,2]
   @inbounds TI = QBI*BR[:,2] + QBR*BI[:,2]
   @inbounds BR[:,2] = TR
   @inbounds BI[:,2] = TI
   BR[1,1] = L
   BR[2,1] = ZERO
   BI[1,1] = ZERO
   BI[2,1] = ZERO
   V = hypot( BR[2,2], BI[2,2] )
   if V >= max( EPS*max( BR[1,1], hypot( BR[1,2], BI[1,2] ) ), SMLNUM )
      XR, XI = LapackUtil.ladiv( V, ZERO, BR[2,2], BI[2,2] )
      BR[2,2] = V
      YR = QBR[2,1]
      YI = QBI[2,1]
      QBR[2,1] = XR*YR - XI*YI
      QBI[2,1] = XR*YI + XI*YR
      YR = QBR[2,2]
      YI = QBI[2,2]
      QBR[2,2] = XR*YR - XI*YI
      QBI[2,2] = XR*YI + XI*YR
   else
      BR[2,2] = ZERO
   end
   BI[2,2] = ZERO

   # Compute the Cholesky factor of the solution of the reduced
   # equation. The solution may be scaled to avoid overflow.

   if ISCONT

      # Continuous-time equation.

      # Step I:  Compute U[1,1]. Set U[2,1] = 0.

      V = -TWO*( AR[1,1]*ER[1,1] + AI[1,1]*EI[1,1] )
      if V <= ZERO
         error("The eigenvalues of the pencil A - λ E  are not in the open right half plane")
      end
      V = sqrt( V )
      T = TWO*abs( BR[1,1] )*SMLNUM
      if T > V
         scale1  = V/T
         scale   = scale1*scale
         BR[1,1] = scale1*BR[1,1]
         BR[1,2] = scale1*BR[1,2]
         BI[1,2] = scale1*BI[1,2]
         BR[2,2] = scale1*BR[2,2]
      end
      UR[1,1] = BR[1,1]/V
      UI[1,1] = ZERO
      UR[2,1] = ZERO
      UI[2,1] = ZERO

      # Step II:  Compute U[1,2].

      T = max( EPS*max( BR[2,2], hypot( BR[1,2], BI[1,2] ) ), SMLNUM )
      if abs( BR[1,1] ) < T
         UR[1,2] = ZERO
         UI[1,2] = ZERO
      else
         XR = AR[1,1]*ER[1,2] + AI[1,1]*EI[1,2]
         XI = AI[1,1]*ER[1,2] - AR[1,1]*EI[1,2]
         XR = XR + AR[1,2]*ER[1,1] + AI[1,2]*EI[1,1]
         XI = XI - AI[1,2]*ER[1,1] + AR[1,2]*EI[1,1]
         XR = -BR[1,2]*V - XR*UR[1,1]
         XI =  BI[1,2]*V - XI*UR[1,1]
         YR =  AR[2,2]*ER[1,1] + AI[2,2]*EI[1,1]
         YI = -AI[2,2]*ER[1,1] + AR[2,2]*EI[1,1]
         YR = YR + ER[2,2]*AR[1,1] + EI[2,2]*AI[1,1]
         YI = YI - EI[2,2]*AR[1,1] + ER[2,2]*AI[1,1]
         T  = TWO*hypot( XR, XI )*SMLNUM
         if T > hypot( YR, YI )
            scale1  = hypot( YR, YI )/T
            scale   = scale1*scale
            BR[1,1] = scale1*BR[1,1]
            BR[1,2] = scale1*BR[1,2]
            BI[1,2] = scale1*BI[1,2]
            BR[2,2] = scale1*BR[2,2]
            UR[1,1] = scale1*UR[1,1]
            XR = scale1*XR
            XI = scale1*XI
         end
         UR[1,2], UI[1,2] = LapackUtil.ladiv( XR, XI, YR, YI )
         UI[1,2] = -UI[1,2]
      end

      # Step III:  Compute U[2,2].

      XR = ( ER[1,2]*UR[1,1] + ER[2,2]*UR[1,2] - EI[2,2]*UI[1,2] )*V
      XI = (-EI[1,2]*UR[1,1] - ER[2,2]*UI[1,2] - EI[2,2]*UR[1,2] )*V
      T  = TWO*hypot( XR, XI )*SMLNUM
      if T > hypot( ER[1,1], EI[1,1] )
         scale1  = hypot( ER[1,1], EI[1,1] )/T
         scale   = scale1*scale
         UR[1,1] = scale1*UR[1,1]
         UR[1,2] = scale1*UR[1,2]
         UI[1,2] = scale1*UI[1,2]
         BR[1,1] = scale1*BR[1,1]
         BR[1,2] = scale1*BR[1,2]
         BI[1,2] = scale1*BI[1,2]
         BR[2,2] = scale1*BR[2,2]
         XR = scale1*XR
         XI = scale1*XI
      end
      YR, YI = LapackUtil.ladiv( XR, XI, ER[1,1], -EI[1,1] )
      YR =  BR[1,2] - YR
      YI = -BI[1,2] - YI
      V  = -TWO*( AR[2,2]*ER[2,2] + AI[2,2]*EI[2,2] )
      if V <= ZERO
         INFO = 3
         RETURN
      end
      V = sqrt( V )
      W = hypot4( BR[2,2], BI[2,2], YR, YI )
      T = TWO*W*SMLNUM
      if T > V
         scale1  = V/T
         scale   = scale1*scale
         UR[1,1] = scale1*UR[1,1]
         UR[1,2] = scale1*UR[1,2]
         UI[1,2] = scale1*UI[1,2]
         BR[1,1] = scale1*BR[1,1]
         BR[1,2] = scale1*BR[1,2]
         BI[1,2] = scale1*BI[1,2]
         BR[2,2] = scale1*BR[2,2]
         W = scale1*W
      end
      UR[2,2] = W/V
      UI[2,2] = ZERO

      # Compute matrices M1 and M2 for the reduced equation.

      M1R[2,1] = ZERO
      M1I[2,1] = ZERO
      M2R[2,1] = ZERO
      M2I[2,1] = ZERO
      BETAR, BETAI = LapackUtil.ladiv( AR[1,1], AI[1,1], ER[1,1], EI[1,1] )
      M1R[1,1] =  BETAR
      M1I[1,1] =  BETAI
      M1R[2,2] =  BETAR
      M1I[2,2] = -BETAI
      ALPHA = sqrt( -TWO*BETAR )
      M2R[1,1] = ALPHA
      M2I[1,1] = ZERO
      V  = ER[1,1]*ER[2,2]
      XR = ( -BR[1,1]*ER[1,2] + ER[1,1]*BR[1,2] )/V
      XI = ( -BR[1,1]*EI[1,2] + ER[1,1]*BI[1,2] )/V
      YR =  XR - ALPHA*UR[1,2]
      YI = -XI + ALPHA*UI[1,2]
      if ( YR != ZERO ) || ( YI != ZERO )
         M2R[1,2] =  YR/UR[2,2]
         M2I[1,2] = -YI/UR[2,2]
         M2R[2,2] =  BR[2,2]/( ER[2,2]*UR[2,2] )
         M2I[2,2] =  ZERO
         M1R[1,2] = -ALPHA*M2R[1,2]
         M1I[1,2] = -ALPHA*M2I[1,2]
      else
         M2R[1,2] = ZERO
         M2I[1,2] = ZERO
         M2R[2,2] = ALPHA
         M2I[2,2] = ZERO
         M1R[1,2] = ZERO
         M1I[1,2] = ZERO
      end
   else

      # Discrete-time equation.

      # Step I:  Compute U[1,1]. Set U[2,1] = 0.
      T = max(abs(AR[1,1]),abs(AI[1,1]),abs(ER[1,1]),abs(EI[1,1]))
      #V = ER[1,1]^2 + EI[1,1]^2 - AR[1,1]^2 - AI[1,1]^2
      V = (ER[1,1]/T)^2 + (EI[1,1]/T)^2 - (AR[1,1]/T)^2 - (AI[1,1]/T)^2
      if V <= ZERO
         error("The eigenvalues of the pencil A - λ E  are not inside the unit circle")
      end
      V = T*sqrt( V )
      T = TWO*abs( BR[1,1] )*SMLNUM
      if T > V
         scale1  = V/T
         scale   = scale1*scale
         BR[1,1] = scale1*BR[1,1]
         BR[1,2] = scale1*BR[1,2]
         BI[1,2] = scale1*BI[1,2]
         BR[2,2] = scale1*BR[2,2]
      end
      UR[1,1] = BR[1,1]/V
      UI[1,1] = ZERO
      UR[2,1] = ZERO
      UI[2,1] = ZERO
      # Step II:  Compute U[1,2].

      T = max( EPS*max( BR[2,2], hypot( BR[1,2], BI[1,2] ) ), SMLNUM )
      if abs( BR[1,1] ) < T
         UR[1,2] = ZERO
         UI[1,2] = ZERO
      else
         XR =  AR[1,1]*AR[1,2] + AI[1,1]*AI[1,2]
         XI =  AI[1,1]*AR[1,2] - AR[1,1]*AI[1,2]
         XR =  XR - ER[1,2]*ER[1,1] - EI[1,2]*EI[1,1]
         XI =  XI + EI[1,2]*ER[1,1] - ER[1,2]*EI[1,1]
         XR = -BR[1,2]*V - XR*UR[1,1]
         XI =  BI[1,2]*V - XI*UR[1,1]
         YR =  AR[2,2]*AR[1,1] + AI[2,2]*AI[1,1]
         YI = -AI[2,2]*AR[1,1] + AR[2,2]*AI[1,1]
         YR = YR - ER[2,2]*ER[1,1] - EI[2,2]*EI[1,1]
         YI = YI + EI[2,2]*ER[1,1] - ER[2,2]*EI[1,1]
         T  = TWO*hypot( XR, XI )*SMLNUM
         if T > hypot( YR, YI )
            scale1  = hypot( YR, YI )/T
            scale   = scale1*scale
            BR[1,1] = scale1*BR[1,1]
            BR[1,2] = scale1*BR[1,2]
            BI[1,2] = scale1*BI[1,2]
            BR[2,2] = scale1*BR[2,2]
            UR[1,1] = scale1*UR[1,1]
            XR = scale1*XR
            XI = scale1*XI
         end
         t1, t2 = LapackUtil.ladiv( XR, XI, YR, YI )
         UR[1,2] = t1
         UI[1,2] = -t2
      end
      # Step III:  Compute U[2,2].

      XR =  ER[1,2]*UR[1,1] + ER[2,2]*UR[1,2] - EI[2,2]*UI[1,2]
      XI = -EI[1,2]*UR[1,1] - ER[2,2]*UI[1,2] - EI[2,2]*UR[1,2]
      YR =  AR[1,2]*UR[1,1] + AR[2,2]*UR[1,2] - AI[2,2]*UI[1,2]
      YI = -AI[1,2]*UR[1,1] - AR[2,2]*UI[1,2] - AI[2,2]*UR[1,2]
      V  = ER[2,2]^2 + EI[2,2]^2 - AR[2,2]^2 - AI[2,2]^2
      if V <= ZERO
         error("The eigenvalues of the pencil A - λ E  are not inside the unit circle")
      end
      V = sqrt( V )
      T = max( abs( BR[2,2] ), abs( BR[1,2] ), abs( BI[1,2] ),
               abs( XR ), abs( XI ), abs( YR ), abs( YI) )
      if T <= SMLNUM
         W = ZERO
      else
         W = ( BR[2,2]/T )^2 + ( BR[1,2]/T )^2 + ( BI[1,2]/T )^2 -
             ( XR/T )^2 - ( XI/T )^2 + ( YR/T )^2 + ( YI/T )^2
         if W < ZERO
            #  this condition usually does not occur -> we simply set W = 0
            #  and thus U22 = 0
            W = ZERO
         else
            W = T*sqrt( W )
         end
      end
      T = TWO*W*SMLNUM
      if T > V
         scale1  = V/T
         scale   = scale1*scale
         UR[1,1] = scale1*UR[1,1]
         UR[1,2] = scale1*UR[1,2]
         UI[1,2] = scale1*UI[1,2]
         BR[1,1] = scale1*BR[1,1]
         BR[1,2] = scale1*BR[1,2]
         BI[1,2] = scale1*BI[1,2]
         BR[2,2] = scale1*BR[2,2]
         W = scale1*W
      end
      UR[2,2] = W/V
      UI[2,2] = ZERO

      #Compute matrices M1 and M2 for the reduced equation.

      B11  = BR[1,1]/ER[1,1]
      T    = ER[1,1]*ER[2,2]
      B12R = ( ER[1,1]*BR[1,2] - BR[1,1]*ER[1,2] )/T
      B12I = ( ER[1,1]*BI[1,2] - BR[1,1]*EI[1,2] )/T
      B22  = BR[2,2]/ER[2,2]
      M1R[2,1] = ZERO
      M1I[2,1] = ZERO
      M2R[2,1] = ZERO
      M2I[2,1] = ZERO
      BETAR, BETAI = LapackUtil.ladiv( AR[1,1], AI[1,1], ER[1,1], EI[1,1] )
      M1R[1,1] =  BETAR
      M1I[1,1] =  BETAI
      M1R[2,2] =  BETAR
      M1I[2,2] = -BETAI
      V = hypot( BETAR, BETAI )
      ALPHA = sqrt( ( ONE - V )*( ONE + V ) )
      M2R[1,1] = ALPHA
      M2I[1,1] = ZERO
      XR = ( AI[1,1]*EI[1,2] - AR[1,1]*ER[1,2] )/T + AR[1,2]/ER[2,2]
      XI = ( AR[1,1]*EI[1,2] + AI[1,1]*ER[1,2] )/T - AI[1,2]/ER[2,2]
      XR = -TWO*BETAI*B12I - B11*XR
      XI = -TWO*BETAI*B12R - B11*XI
      V  =  ONE + ( BETAI - BETAR )*( BETAI + BETAR )
      W  = -TWO*BETAI*BETAR
      YR, YI = LapackUtil.ladiv( XR, XI, V, W )
      #if ( YR != ZERO ) || ( YI != ZERO )
      # - to avoid NaNs, the above has been changed to:
      if ( abs(YR) > SMLNUM ) || ( abs(YI) > SMLNUM )
         M2R[1,2] =  ( YR*BETAR - YI*BETAI )/UR[2,2]
         M2I[1,2] = -( YI*BETAR + YR*BETAI )/UR[2,2]
         M2R[2,2] =  B22/UR[2,2]
         M2I[2,2] =  ZERO
         M1R[1,2] = -ALPHA*YR/UR[2,2]
         M1I[1,2] =  ALPHA*YI/UR[2,2]
      else
         M2R[1,2] = ZERO
         M2I[1,2] = ZERO
         M2R[2,2] = ALPHA
         M2I[2,2] = ZERO
         M1R[1,2] = ZERO
         M1I[1,2] = ZERO
      end
   end

   # Transform U back:  U := U * Q.
   # (Note:  Z is used as workspace.)
   @inbounds ZR = UR*QR - UI*QI
   @inbounds ZI = UR*QI + UI*QR

   # Overwrite U with the upper triangular matrix of its
   # QR-factorization. The elements on the main diagonal are real
   # and non-negative.

   CR, CI, SR, SI, L = cgivensc2( ZR[1,1], ZI[1,1], ZR[2,1], ZI[2,1], small )
   QUR[1,1] =  CR
   QUR[1,2] =  SR
   QUR[2,1] = -SR
   QUR[2,2] =  CR
   QUI[1,1] = -CI
   QUI[1,2] = -SI
   QUI[2,1] = -SI
   QUI[2,2] =  CI
   U[:,2] = QUR*ZR[:,2] - QUI*ZI[:,2]
   UI[:,2] = QUI*ZR[:,2] + QUR*ZI[:,2]
   U[1,1] = L
   U[2,1] = ZERO
   V = hypot( U[2,2], UI[2,2] )
   if V  !=  ZERO
      XR, XI = LapackUtil.ladiv( V, ZERO, U[2,2], UI[2,2] )
      YR = QUR[2,1]
      YI = QUI[2,1]
      QUR[2,1] = XR*YR - XI*YI
      QUI[2,1] = XR*YI + XI*YR
      YR = QUR[2,2]
      YI = QUI[2,2]
      QUR[2,2] = XR*YR - XI*YI
      QUI[2,2] = XR*YI + XI*YR
   end
   U[2,2] = V

   # Transform the matrices M1 and M2 back.

   # M1 := QU * M1 * QU^H
   @inbounds TR = M1R*QUR' + M1I*QUI'
   @inbounds TI = -M1R*QUI' + M1I*QUR'
   @inbounds β = QUR*TR - QUI*TI
   # M2 := QB^H * M2 * QU^H
   @inbounds TR = M2R*QUR' + M2I*QUI'
   @inbounds TI = -M2R*QUI' + M2I*QUR'
   @inbounds α = QBR'*TR + QBI'*TI

   # If the transposed equation (op(K)=K^T, K=A,B,E,U) is to be
   # solved, transpose the matrix U with respect to the
   # anti-diagonal and the matrices M1, M2 with respect to the diagonal
   # and the anti-diagonal.

   if noadj
      V = U[1,1]
      U[1,1] = U[2,2]
      U[2,2] = V
      V = β[1,1]
      β[1,1] = β[2,2]
      β[2,2] = V
      V = α[1,1]
      α[1,1] = α[2,2]
      α[2,2] = V
   end
   return U, scale, β, α
end
"""
    cgivens2(ar, ai, b, small) -> (cr, ci, s, d)

Construct a complex Givens plane rotation such that, for a complex number  `a`
and a real number  `b`,

        ( conj( c )  s )*( a ) = ( d ),
        (      -s    c ) ( b )   ( 0 )

where  `d`  is always real. `a` and `b` are unaltered.
On entry, `ar` and `ai` must contain the real and imaginary part,
respectively, of the complex number `a` and `b` contains a real number.
On exit, `cr` and `ci` contain the real and imaginary part, respectively,
of the complex number c, the cosines of the plane rotation and
`s` contains the real number `s`, the sines of the plane rotation.
`small` is a small real number. If the norm `d` of `[ a; b ]` is smaller
than `small`, then the rotation is taken as a unit matrix.

This function is based on the SLICOT routine SB03OV.
"""
@inline function cgivens2(ar, ai, b, small)
   d = max(abs(ar), abs(ai), abs(b))
   ZERO = 0.0
   ONE = 1.0
   if d < small
      cr = ONE
      ci = ZERO
      s = ZERO
   else
      d = d*hypot3(ar/d, ai/d, b/d)
      cr = ar/d
      ci = ai/d
      s = b/d
   end
   return cr, ci, s, d
end
"""
    hypot3(X, Y, Z)

Compute the hypotenuse `sqrt(X^2+Y^2+Z^2)` avoiding overflow and underflow.
Based on the LAPACK function DLAPY3.
`Note:` The Julia function `hypot` is not reliable for more than two arguments.
"""
@inline function hypot3(X, Y, Z)
   XABS = abs( X )
   YABS = abs( Y )
   ZABS = abs( Z )
   W = max( XABS, YABS, ZABS )
   R = W*sqrt( ( XABS / W )^2+( YABS / W )^2+ ( ZABS / W )^2 )
end
"""
    cgivensc2(xr, xi, yr, yi, small) -> (cr, ci, sr, si, d)

Construct a complex Givens plane rotation such that, for the complex numbers
`x`  and  `y`,

        ( conj( c )  conj( s) )*( x ) = ( d ),
        (      -s          c  ) ( y )   ( 0 )

where  `d`  is always real. `x` and `y` are unaltered.
On entry, `xr` and `xi` must contain the real and imaginary part,
respectively, of the complex number `x` and `yr` and `yi` must contain
the real and imaginary part, respectively, of the complex number `y`.
On exit, `cr` and `ci` contain the real and imaginary part, respectively,
of the complex number c, the cosines of the plane rotation and
`sr` and `si` contain the real and imaginary part, respectively,
of the complex number s, the sines of the plane rotation.
s contains the real number s, the sines of the plane rotation.
`small` is a small real number. If the norm `d` of `[ x; y ]` is smaller
than `small`, then the rotation is taken as a unit matrix.

This function is based on the SLICOT routine SG03BY.
"""
@inline function cgivensc2(xr, xi, yr, yi, small)
   d = max(abs(xr), abs(xi), abs(yr), abs(yi))
   ZERO = 0.0
   ONE = 1.0
   if d < small
      cr = ONE
      ci = ZERO
      sr = ZERO
      si = ZERO
   else
      d = d*hypot4(xr/d, xi/d, yr/d, yi/d)
      cr = xr/d
      ci = xi/d
      sr = yr/d
      si = yi/d
   end
   return cr, ci, sr, si, d
end
"""
    hypot4(X, Y, Z, T)

Compute the hypotenuse `sqrt(X^2+Y^2+Z^2+T^2)` avoiding overflow and underflow.
Based on the LAPACK function DLAPY3.
`Note:` The Julia function `hypot` is not reliable for more than two arguments.
"""
@inline function hypot4(X, Y, Z, T)
   XABS = abs( X )
   YABS = abs( Y )
   ZABS = abs( Z )
   TABS = abs( T )
   W = max( XABS, YABS, ZABS, TABS )
   R = W*sqrt( ( XABS / W )^2+( YABS / W )^2+ ( ZABS / W )^2+ ( TABS / W )^2  )
end
