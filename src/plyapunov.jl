rev!(t) = reverse!(reverse!(t,dims=1),dims=2)
"""
    U = plyapc(A, B; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous Lyapunov equation

      AX + XA' + BB' = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of rows as `A`. `A` must have only eigenvalues with negative real parts.

    U = plyapc(A', B'; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the continuous Lyapunov equation

      A'X + XA + B'B = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of columns as `A`. `A` must have only eigenvalues with negative real parts.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 

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
function plyapc(A::AbstractMatrix, B::AbstractMatrix; blocksize = 64)
   # Method

   # The Bartels-Steward Schur form based method is employed [1], with the
   # modifications proposed by Hammarling [2]. For `BlasFloat` type data, the Sylvester equations 
   # are solved using the recursive blocking based algorithm of [3].

   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems—
   #     Part I: One-sided and coupled Sylvester-type matrix equations, ACM Trans. Math. Software, 
   #     28 (2002), pp. 392–415.

   T2 = promote_type(eltype(A), eltype(B))
   adiag = isdiag(A)
   aDiag = isa(A, Diagonal)
   aHerm = isa(A,Hermitian)
   aSym = isa(A,Symmetric)
   adj = isa(B,Adjoint)
   if aDiag
      eltype(A) <: Complex &&
      (xor(adj,isa(parent(parent(A)),Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   elseif !aHerm && !aSym
      (xor(adj,isa(A,Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   if eltype(A) != T2 
      if adiag
         adj ? A = convert(Diagonal{T2},Diagonal(parent(parent(A))))' : A = convert(Diagonal{T2},A)
      elseif aHerm || aSym
         A = LinearAlgebra.copy_oftype(A,T2)  
      else       
         adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
      end   
   end 
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   adiag && (return plyapcs!(Diagonal(A),utriuB(B); adj))
   
   # Reduce A to Schur form and transform B
   if ishermitian(A)
      # Reduce A to diagonal form and transform C
      AS, Q, EV = schur(Hermitian(A))
   else
      # Reduce A to Schur form and transform C
      if adj
         AS, Q, EV = schur(A.parent)
      else
         AS, Q, EV = schur(A)
      end
   end
   maximum(real(EV)) >= zero(real(T2)) && error("A must have only eigenvalues with negative real part")
   
   U = utriuB(B,Q)
   plyapcs!(AS, U; adj, blocksize)
   return utriuU(U, Q, AS; adj)
end
plyapc(A::Union{Real,Complex}, B::Union{Real,Complex}) =
      real(A) < 0 ? abs(B)/sqrt( -2 * real(A) ) :
      error("A must be a negative number or must have negative real part")

"""
    U = plyapc(A, E, B; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
generalized continuous Lyapunov equation

      AXE' + EXA' + BB' = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with negative real parts.

    U = plyapc(A', E', B'; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the generalized continuous Lyapunov equation

      A'XE + E'XA + B'B = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λE` must have only
eigenvalues with negative real parts.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based generalized Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 

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
function plyapc(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; blocksize::Int = 64)
   # Method

   # A generalization of Bartels-Steward Schur form based method is employed [1],
   # with the modifications proposed by Hammarling [2] and Penzl [3]. 
   # For `BlasFloat` type data, the generalized Sylvester equations 
   # are solved using the recursive blocking based algorithm of [4].

   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] Penzl, T.
   #     Numerical solution of generalized Lyapunov equations.
   #     Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   # [4] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems — 
   #        Part II: Two-sided and generalized Sylvester and Lyapunov matrix equations, 
   #        ACM Trans. Math. Software, 28 (2002), pp. 416–435.

   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)) && (return plyapc(A, B; blocksize))
   
   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))

   adj = isa(A,Adjoint)
   (xor(adj,isa(E,Adjoint)) || xor(adj,isa(B,Adjoint))) &&
      error("Only calls with A, E and B or with A', E' and B' allowed")

   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(E) == T2 || (adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E))
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   # Reduce (A,E) to generalized Schur form and transform C
   # (AS,ES) = (Q'*A*Z, Q'*E*Z)
   if adj
      AS, ES, Q, Z, α, β = schur(A.parent,E.parent)
   else
      AS, ES, Q, Z, α, β = schur(A,E)
   end

   maximum(real(α./β)) >= zero(real(T2)) && error("A-λE must have only eigenvalues with negative real parts")

   U = adj ? utriuB(B,Z) : utriuB(B,Q)
   plyapcs!(AS, ES, U; adj, blocksize)
   return adj ? utriuU(U, Q, AS; adj) : utriuU(U, Z, AS; adj)
end
plyapc(A::Union{Real,Complex}, E::Union{Real,Complex}, B::Union{Real,Complex}) =
      real(A*E') < 0 ? abs(B)/sqrt( -2 * real(A*E') ) :
      error("A*E' must be a negative number or must have negative real part")
"""
    U = plyapd(A, B; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of
the discrete Lyapunov equation

      AXA' - X + BB' = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of rows as `A`. `A` must have only eigenvalues with moduli less than one.

    U = plyapd(A', B'; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the discrete Lyapunov equation

      A'XA - X + B'B = 0,

where `A` is a square real or complex matrix and `B` is a matrix with the same
number of columns as `A`. `A` must have only eigenvalues with moduli less than one.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 

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
function plyapd(A::AbstractMatrix, B::AbstractMatrix; blocksize = 64)
   # Method

   # The Bartels-Steward Schur form based method is employed [1], with the
   # modifications proposed by Hammarling in [2] and [3]. For `BlasFloat` type data, 
   # the discrete Sylvester equations are solved using the recursive blocking based algorithm of [4].

   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] Hammarling, S.J. Numerical solution of the discrete-time, convergent,
   #     non-negative definite Lyapunov equation.
   #     Systems & Control Letters 17 (1991) 137-139.
   # [4] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems — 
   #        Part II: Two-sided and generalized Sylvester and Lyapunov matrix equations, 
   #        ACM Trans. Math. Software, 28 (2002), pp. 416–435.
  
   T2 = promote_type(eltype(A), eltype(B))
   adiag = isdiag(A)
   aDiag = isa(A, Diagonal)
   aHerm = isa(A,Hermitian)
   aSym = isa(A,Symmetric)
   adj = isa(B,Adjoint)
   if aDiag
      eltype(A) <: Complex &&
      (xor(adj,isa(parent(parent(A)),Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   elseif !aHerm && !aSym
      (xor(adj,isa(A,Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   if eltype(A) != T2 
      if adiag
         adj ? A = convert(Diagonal{T2},Diagonal(parent(parent(A))))' : A = convert(Diagonal{T2},A)
      elseif aHerm || aSym
         A = LinearAlgebra.copy_oftype(A,T2)  
      else       
         adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
      end   
   end 
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   adiag && (return plyapds!(Diagonal(A), utriuB(B); adj))

   # Reduce A to Schur form and transform B
   if ishermitian(A)
      # Reduce A to diagonal form and transform C
      AS, Q, EV = schur(Hermitian(A))
   else
      # Reduce A to Schur form and transform C
      if adj
         AS, Q, EV = schur(A.parent)
      else
         AS, Q, EV = schur(A)
      end
   end
   maximum(abs.(EV)) >= one(real(T2)) && error("A must have only eigenvalues with moduli less than one")

   U = utriuB(B,Q)
   plyapds!(AS, U; adj, blocksize)
   return utriuU(U, Q, AS; adj)
end
plyapd(A::Union{Real,Complex}, B::Union{Real,Complex}) =
      abs(A) < real(one(A)) ? real(abs(B)/sqrt( (one(A)-abs(A))*(one(A)+abs(A)) )) :
      error("A must be a subunitary number")
"""
    U = plyapd(A, E, B; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
generalized discrete Lyapunov equation

      AXA' - EXE' + BB' = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with moduli less than one.

    U = plyapd(A', E', B'; blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the generalized discrete Lyapunov equation

      A'XA - E'XE + B'B = 0,

where `A` and `E` are square real or complex matrices and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λE` must have only
eigenvalues with moduli less than one.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based generalized Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 

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
function plyapd(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; blocksize = 64)
   # Method

   # The Bartels-Steward Schur form based method is employed [1], with the
   # modifications proposed by Hammarling in [2] and Penzl in [3].
   # For `BlasFloat` type data, the discrete Sylvester equations 
   # are solved using the recursive blocking based algorithm of [4].

   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] Penzl, T.
   #     Numerical solution of generalized Lyapunov equations.
   #     Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   # [4] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems — 
   #     Part II: Two-sided and generalized Sylvester and Lyapunov matrix equations, 
   #     ACM Trans. Math. Software, 28 (2002), pp. 416–435.

   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)) && (return plyapd(A, B; blocksize))
   
   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))

   adj = isa(A,Adjoint)
   (xor(adj,isa(E,Adjoint)) || xor(adj,isa(B,Adjoint))) &&
      error("Only calls with A, E and B or with A', E' and B' allowed")

   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   
   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(E) == T2 || (adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E))
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   ONE = one(real(T2))

   # Reduce (A,E) to generalized Schur form and transform C
   # (AS,ES) = (Q'*A*Z, Q'*E*Z)
   if adj
      AS, ES, Q, Z, α, β = schur(A.parent,E.parent)
   else
      AS, ES, Q, Z, α, β = schur(A,E)
   end

   maximum(abs.(α./β)) >= ONE && error("A-λE must have only eigenvalues with moduli less than one")

   U = adj ? utriuB(B,Z) : utriuB(B,Q)
   plyapds!(AS, ES, U; adj, blocksize)
   return adj ? utriuU(U, Q, AS; adj) : utriuU(U, Z, AS; adj)
end
plyapd(A::Union{Real,Complex}, E::Union{Real,Complex}, B::Union{Real,Complex}) =
     abs(A) < abs(E) ? real(abs(B)/sqrt( (abs(E)-abs(A))*(abs(E)+abs(A)) )) :
      error("A/E must be a subunitary number")

"""
    U = plyaps(A, B; disc = false, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
continuous Lyapunov equation

      AX + XA' + BB' = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of rows as `A`.
`A` must have only eigenvalues with negative real parts. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A', B', disc = false, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the continuous Lyapunov equation

      A'X + XA + B'B = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of columns as `A`.
`A` must have only eigenvalues with negative real parts. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A, B, disc = true, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
discrete Lyapunov equation

      AXA' - X + BB' = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of rows as `A`.
`A` must have only eigenvalues with moduli less than one. Only the upper
Hessenberg part of `A` is referenced.

    U = plyaps(A', B', disc = true, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the discrete Lyapunov equation

      A'XA - X + B'B = 0,

where `A` is a square real or complex matrix in a real or complex Schur form,
respectively, and `B` is a matrix with the same number of columns as `A`.
`A` must have only eigenvalues with moduli less than one. Only the upper
Hessenberg part of `A` is referenced.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 

"""
function plyaps(A::AbstractMatrix, B::AbstractMatrix; disc = false, blocksize = 64)
   # Method

   # The Bartels-Stewart Schur form based method is employed [1], with the
   # modifications proposed by Hammarling in [2] and [3]. 
   # For `BlasFloat` type data, the continuous Sylvester equations 
   # are solved using the recursive blocking based algorithm of [4], while the 
   # discrete Sylvester equations are solved using the recursive blocking based algorithm of [5].
 
   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] Hammarling, S.J. Numerical solution of the discrete-time, convergent,
   #     non-negative definite Lyapunov equation.
   #     Systems & Control Letters 17 (1991) 137-139.
   # [4] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems—
   #     Part I: One-sided and coupled Sylvester-type matrix equations, ACM Trans. Math. Software, 
   #     28 (2002), pp. 392–415.
   # [5] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems — 
   #     Part II: Two-sided and generalized Sylvester and Lyapunov matrix equations, 
   #     ACM Trans. Math. Software, 28 (2002), pp. 416–435.

   T2 = promote_type(eltype(A), eltype(B))
   adiag = isdiag(A)
   aDiag = isa(A, Diagonal)
   aHerm = isa(A,Hermitian)
   aSym = isa(A,Symmetric)
   adj = isa(B,Adjoint)
   if aDiag
      eltype(A) <: Complex &&
      (xor(adj,isa(parent(parent(A)),Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   elseif !aHerm && !aSym
      (xor(adj,isa(A,Adjoint)) && error("Only calls with A and B or with A' and B' allowed"))
   end

   n = LinearAlgebra.checksquare(A)
   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
   if eltype(A) != T2 
      if adiag
         adj ? A = convert(Diagonal{T2},Diagonal(parent(parent(A))))' : A = convert(Diagonal{T2},A)
      elseif aHerm || aSym
         A = LinearAlgebra.copy_oftype(A,T2)  
      else       
         adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A)
      end   
   end 
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   if adiag 
      if disc
         return plyapds!(Diagonal(A), utriuB(B); adj)
      else
         return plyapcs!(Diagonal(A), utriuB(B); adj)
      end
   end


   U = utriuB(B)
   if adj
      if disc
         plyapds!(A.parent, U; adj, blocksize)
      else
         plyapcs!(A.parent, U; adj, blocksize)
      end
   else
      if disc
         plyapds!(A, U; adj, blocksize)
      else
         plyapcs!(A, U; adj, blocksize)
      end
   end
   return utnormalize!(U,adj)
end
"""
    U = plyaps(A, E, B; disc = false, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
generalized continuous Lyapunov equation

      AXE' + EXA' + BB' = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with negative real parts.

    U = plyaps(A', E', B', disc = false, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the generalized continuous Lyapunov equation

      A'XE + E'XA + B'B = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λE` must have only
eigenvalues with negative real parts.

    U = plyaps(A, E, B, disc = true, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = UU'` of the
generalized discrete Lyapunov equation

      AXA' - EXE' + BB' = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with moduli less than one.

    U = plyaps(A', E', B', disc = true, blocksize = 64)

Compute `U`, the upper triangular factor of the solution `X = U'U` of
the generalized discrete Lyapunov equation

      A'XA - E'XE + B'B = 0,

where `A` and `E` are square real or complex matrices with the pair `(A,E)` in
a generalied real or complex Schur form, respectively,  and `B` is a matrix
with the same number of columns as `A`. The pencil `A - λE` must have only
eigenvalues with moduli less than one.

The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used 
in the recursive blocking based generalizedd Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 
"""
function plyaps(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; disc = false, blocksize = 64)
   # Method

   # Generalizations of Bartels-Stewart Schur form based method is employed [1],
   # with the modifications proposed by Hammarling [2] and Penzl [3].
   # For `BlasFloat` type data, the generalized Sylvester equations 
   # are solved using the recursive blocking based algorithm of [4].

   # Reference:

   # [1] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix
   #     equation AX+XB=C. Comm. ACM, 15:820–826, 1972.
   # [2] Hammarling, S.J. Numerical solution of the stable, non-negative definite
   #     Lyapunov equation. IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [3] Penzl, T.
   #     Numerical solution of generalized Lyapunov equations.
   #     Advances in Comp. Math., vol. 8, pp. 33-48, 1998.
   # [4] I. Jonsson and B. Kågström, Recursive blocked algorithms for solving triangular systems — 
   #     Part II: Two-sided and generalized Sylvester and Lyapunov matrix equations, 
   #     ACM Trans. Math. Software, 28 (2002), pp. 416–435.

   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) &&  size(E,1) == n)) && (return plyaps(A, B))
   
   T2 = promote_type(eltype(A), eltype(E), eltype(B))
   T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))

   adj = isa(A,Adjoint)
   (xor(adj,isa(E,Adjoint)) || xor(adj,isa(B,Adjoint))) &&
      error("Only calls with A, E and B or with A', E' and B' allowed")

   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   
   if adj
      nb, mb = size(B)
      nb == n || throw(DimensionMismatch("B must be a matrix of column dimension $n"))
   else
      mb, nb = size(B)
      mb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))
   end

   eltype(A) == T2 || (adj ? A = convert(Matrix{T2},A.parent)' : A = convert(Matrix{T2},A))
   eltype(E) == T2 || (adj ? E = convert(Matrix{T2},E.parent)' : E = convert(Matrix{T2},E))
   eltype(B) == T2 || (adj ? B = convert(Matrix{T2},B.parent)' : B = convert(Matrix{T2},B))

   U = utriuB(B)
   if adj
      if disc
         plyapds!(A.parent, E.parent, U; adj, blocksize)
      else
         plyapcs!(A.parent, E.parent, U; adj, blocksize)
      end
   else
      if disc
         plyapds!(A, E, U; adj, blocksize)
      else
         plyapcs!(A, E, U; adj, blocksize)
      end
   end
   return utnormalize!(U,adj)
end
"""
    plyapcs!(A,R;adj = false, blocksize = 64)

Solve the positive continuous Lyapunov matrix equation

                op(A)X + Xop(A)' + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
`A` is a square real matrix in a real Schur form  or a square complex matrix in a
complex Schur form and `R` is an upper triangular matrix.
`A` must have only eigenvalues with negative real parts.
`R` contains on output the solution `U`.
The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 
"""
function plyapcs!(A::AbstractMatrix{T1}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Real
   # check for diagonal A
   isdiag(A) && (return plyapcs!(Diagonal(A),R; adj))
   
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   ONE = one(T1)
   ZERO = zero(T1)
   TWO = 2*ONE
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   Wr = Matrix{T1}(undef,n,2)
   Wz = Matrix{T1}(undef,n,2)
   Mα = Matrix{T1}(undef,2,2)
   Mβ = Matrix{T1}(undef,2,2)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = A[j,j]
             λ >= ZERO && error("A is not stable")
             TEMP = sqrt( abs( TWO*λ ) )
             TEMP < SMIN && (TEMP = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
             tα = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/tα
             Mα[1,1] = tα
             Mβ[1,1] = A[j,j]
          else
             plyap2!(view(A,l,l), view(R,l,l), Mβ, Mα, adj = true)
          end
          if ll < p
             dll = 1:dl
             js = j
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,dll)
             z = view(Wz,ir1,dll)
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             # rbar = R[l,j1]'
             transpose!(rbar,view(R,l,j1))
             # z = rbar*α + A[l,j1]'*R[l,l]'
             mul!(z,rbar,α)
             # mul!(z,transpose(view(A,l,j1)),transpose(R[l,l]),ONE,ONE)
             # alternative code exploiting lower triangular form of R'
             k = js+dl-1
             axpy!(R[k,k],view(A,k,j1),view(z,:,dl))
             dl == 1 || (axpy!(R[js,js],view(A,js,j1),view(z,:,1)); axpy!(R[js,js+1],view(A,js+1,j1),view(z,:,1)))

             # Solve S1'*ubar+ubar*β + z = 0
             if T1 <: BlasReal
                sylvcs_blocked!(view(A,j1,j1), β, z; adjA = true, adjB = false, blocksize); 
                transpose!(view(R,l,j1),rmul!(z,-1))
             else
                sylvcs2!(view(A,j1,j1), β, z; adj)
                transpose!(view(R,l,j1),z)
             end
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = rbar - ubar * α'
             mul!(rbar, z, transpose(α), -ONE, ONE)
             #rbar += ubar * α'
             qrupdate!(view(R,j1,j1), rbar)
         end
       end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #        A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = A[j,j]
             λ >= ZERO && error("A is not stable")
             TEMP = sqrt( abs( TWO*λ ) )
             TEMP < SMIN && (TEMP  = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
             tα = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/tα
             Mα[1,1] = tα
             Mβ[1,1] = A[j,j]
          else
             plyap2!(view(A,l,l), view(R,l,l), Mβ, Mα, adj = false)
          end
          if ll > 1
             dll = 1:dl
             js = j
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,dll)
             z = view(Wz,j1,dll)
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             # rbar = R[j1,l]
             copyto!(rbar,view(R,j1,l))
             # z = rbar*α' + A[j1,l]*R[l,l]
             #z = rbar*α'
             mul!(z,rbar,transpose(α))
             #mul!(z,view(A,j1,l),view(R,l,l),ONE,ONE)
             #mul!(z,view(A,j1,l),R[l,l],ONE,ONE)
             # alternative code exploiting upper triangular shape of R
             k = js-dl+1
             axpy!(R[k,k],view(A,j1,k),view(z,:,1))
             dl == 1 || (axpy!(R[js-1,js],view(A,j1,js-1),view(z,:,2)); axpy!(R[js,js],view(A,j1,js),view(z,:,2)))
             # Solve S1*ubar+ubar*β' + z = 0
             if T1 <: BlasReal
                sylvcs_blocked!(view(A,j1,j1), β, z; adjA = false, adjB = true, blocksize); 
                copyto!(view(R,j1,l), rmul!(z,-1))
             else
               sylvcs2!(view(A,j1,j1), β, z; adj)
               copyto!(view(R,j1,l), z)
             end  
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar - ubar*α
             mul!(rbar, z, α, -ONE, ONE)
             rqupdate!(view(R,j1,j1), rbar)
         end
      end
   end
   return R
end
function plyapcs!(A::Diagonal{T1}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Real
   n = size(A,1)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   ZERO = zero(T1)
   any(>=(ZERO), A.diag) && error("A is not stable")

   ONE = one(T1)
   TWO = 2*ONE
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   Wr = Vector{T1}(undef,n)
   Wz = Vector{T1}(undef,n)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = A[j,j]
          TEMP = sqrt( abs( TWO*λ ) )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j < n
             jp1 = j+1
             j1 = jp1:n
             ir1 = 1:n-j
             rbar = view(Wr,ir1)
             z = view(Wz,ir1)
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             # rbar = R[l,j1]'
             k = jp1
             for ii = 1:n-j
                rbar[ii] = R[j,k]
                z[ii] = rbar[ii]*α 
                k += 1
             end  

             # Solve S1'*ubar+ubar*β + z = 0
             solve_in_place!(view(A.diag,j1),β, z)
             #@. z /= (view(A.diag,j1) + β) 
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = rbar - ubar * α'
             k = jp1
             for ii = 1:n-j
                 R[j,k] = -z[ii]
                 rbar[ii] += z[ii] * α
                 k += 1
             end
             #rbar += ubar * α'
             qrupdate!(view(R,j1,j1), rbar)
         end
       end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #        A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = A[j,j]
          TEMP = sqrt( abs( TWO*λ ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j > 1
             jm1 = j-1
             j1 = 1:jm1
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1)
             z = view(Wz,j1)
             # rbar = R[j1,l]
             #copyto!(rbar,view(R,j1,l))
             # z = rbar*α' + A[j1,l]*R[l,l]
             #z = rbar*α'
             #mul!(z,rbar,transpose(α))
             for ii = 1:jm1
               rbar[ii] = R[ii,j]
               z[ii]= rbar[ii]*α + A[ii,j]*R[j,j]
             end
             # Solve S1*ubar+ubar*β' + z = 0
             solve_in_place!(view(A.diag,j1),β, z)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar - ubar*α
             for ii = 1:jm1
                 R[ii,j] = -z[ii]
                 rbar[ii] += z[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
         end
      end
   end
   return R
end
function plyapcs!(A::Diagonal{T1}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Complex
   # if adj = true, A contains A'
   n = size(A,1)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   RZERO = real(zero(T1))
   any(>=(RZERO), real(A.diag)) && error("A is not stable")

   T = real(T1)
   ONE = one(T)
   ZERO = zero(T)
   TWO = 2*ONE
   EPS = eps(T)
   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   Wr = Vector{T1}(undef,n)
   Wz = Vector{T1}(undef,n)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = real(A[j,j])
          TEMP = sqrt( -TWO*λ )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = conj(A[j,j])
          if j < n
             jp1 = j+1
             j1 = jp1:n
             ir1 = 1:n-j
             rbar = view(Wr,ir1)
             z = view(Wz,ir1)
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             # rbar = R[l,j1]'
             k = jp1
             for ii = 1:n-j
                rbar[ii] = R[j,k]'
                z[ii] = rbar[ii]*α 
                k += 1
             end  

             # Solve S1'*ubar+ubar*β + z = 0
             solve_in_place!(view(A.diag,j1),β, z)
             #@. z /= (view(A.diag,j1) + β) 
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = rbar - ubar * α'
             k = jp1
             for ii = 1:n-j
                 R[j,k] = -z[ii]'
                 rbar[ii] = conj(rbar[ii] + z[ii] * α')
                 k += 1
             end
             #rbar += ubar * α'
             qrupdate!(view(R,j1,j1), rbar)
         end
       end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #        A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = real(A[j,j])
          TEMP = sqrt( -TWO*λ  )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j > 1
             jm1 = j-1
             j1 = 1:jm1
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1)
             z = view(Wz,j1)
             # rbar = R[j1,l]
             #copyto!(rbar,view(R,j1,l))
             # z = rbar*α' + A[j1,l]*R[l,l]
             #z = rbar*α'
             #mul!(z,rbar,transpose(α))
             for ii = 1:jm1
               rbar[ii] = R[ii,j]
               z[ii]= rbar[ii]*α' #+ A[ii,j]*R[j,j]
             end
             # Solve S1*ubar+ubar*β' + z = 0
             solve_in_place!(view(A.diag,j1),β', z)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar - ubar*α
             for ii = 1:jm1
                 R[ii,j] = -z[ii]
                 rbar[ii] += z[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
         end
      end
   end
   return R
end
function solve_in_place!(Adiag, β, z; disc = false)
   if disc
      @. z /= (Adiag*β + one(eltype(β))) 
   else
      @. z /= (Adiag + β) 
   end
end
function plyapcs!(A::AbstractMatrix{T1}, R::UpperTriangular{T1}; adj = false, blocksize::Int = 64)  where T1 <: Complex
   # check for diagonal A
   isdiag(A) && (return plyapcs!(Diagonal(A),R; adj))

   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   T = real(T1)
   ONE = one(T)
   ZERO = zero(T)
   TWO = 2*ONE
   EPS = eps(T)
   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   Wr = Vector{T1}(undef,n)
   Wz = similar(Wr,n,1)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = real(A[j,j])
          λ >= ZERO && error("A is not stable")
          TEMP = sqrt( -TWO*λ )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
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
            z = view(Wz,ir1,1:1)
            k = j
            for ii = 1:n-j+1
               rbar[ii] = R[j-1,k]'
               z[ii] = rbar[ii]*α + R[j-1,j-1]*A[j-1,k]'
               k += 1
            end  
            # Solve S1'*ubar+ubar*β + z = 0
            if T1 <: BlasComplex
               sylvcs_blocked!(view(A,j1,j1), β, z; adjA = true, adjB = false, blocksize); 
            else
               sylvcs1!(view(A,j1,j1), β, z; adj)
               rmul!(z,-1)
            end
            k = j
            for ii = 1:n-j+1
               R[j-1,k] = -z[ii]'
               rbar[ii] = conj(rbar[ii] + z[ii] * α')
               k += 1
            end
            # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
            #y = conj(rbar + ubar * α')
            qrupdate!(view(R,j1,j1), rbar)
          end
      end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #       A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = real(A[j,j])
          λ >= ZERO && error("A is not stable")
          TEMP = sqrt( -TWO*λ  )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
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
             z = view(Wz,j1,1:1)
             for ii = 1:j
               rbar[ii] = R[ii,j+1]
               z[ii]= rbar[ii]*α' + A[ii,j+1]*R[j+1,j+1]
             end
             #z = rbar*α' + A[j1,l]*R[l,l]
             # Solve S1*ubar+ubar*β' + z = 0
             if T1 <: BlasComplex
               sylvcs_blocked!(view(A,j1,j1), β, z; adjA = false, adjB = true, blocksize); 
               #  _, scale = LAPACK.trsyl!('N','C', view(A,j1,j1), β, z)
               #  scale == ONE || error("Singular Lyapunov equation")
             else
                sylvcs1!(view(A,j1,j1), β, z; adj)
                rmul!(z,-1)
             end
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar + z*α
             for ii = 1:j
                 R[ii,j+1] = -z[ii]
                 rbar[ii] += z[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
function plyapds!(A::Diagonal{T1}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Real
   n = size(A,1)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   ONE = one(T1)
   Amax = maximum(abs.(A.diag))
   Amax >= ONE && error("A is not convergent")

   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*Amax

   Wr = Vector{T1}(undef,n)
   Wz = Vector{T1}(undef,n)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = abs(A[j,j])
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          #α = copysign( TEMP, R[j,j])
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j < n
             jp1 = j+1
             j1 = jp1:n
             ir1 = 1:n-j
             rbar = view(Wr,ir1)
             z = view(Wz,ir1)
             # Form the right-hand side of (10.16)
             # S = [ s11 0  ]
             #     [ 0   S1 ]
             # rbar = R[j,j1]'
             # z = rbar*α 
             k = jp1
             for ii = 1:n-j
                 rbar[ii] = R[j,k]
                 z[ii] = rbar[ii]*α 
                 k += 1
             end  

             # Solve S1'*ubar*β - ubar + z = 0
             Adiag = view(A.diag,j1)
             solve_in_place!(Adiag, -β, z, disc = true)
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = rbar*β' - S1*z*α
             k = jp1
             for ii = 1:n-j
                 R[j,k] = z[ii]
                 rbar[ii] = rbar[ii] * β - (Adiag[ii]*z[ii]) * α
                 k += 1
             end
             qrupdate!(view(R,j1,j1), rbar)
         end
       end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #        A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = abs(A[j,j])
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j > 1
             jm1 = j-1
             j1 = 1:jm1
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  0  ]
             #     [ 0  s11 ]
             # z = rbar*α' 
             rbar = view(Wr,j1)
             z = view(Wz,j1)
             # rbar = R[j1,j]
             # z = rbar*α' 
             for ii = 1:jm1
               rbar[ii] = R[ii,j]
               z[ii]= rbar[ii]*α 
             end
             # Solve S1*ubar*β'-ubar + z = 0
             Adiag = view(A.diag,j1)
             solve_in_place!(Adiag,-β, z, disc = true)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar*β - S1*z*α
             for ii = 1:jm1
                 R[ii,j] = z[ii]
                 rbar[ii] = rbar[ii]*β - (Adiag[ii]*z[ii])*α
             end
             rqupdate!(view(R,j1,j1), rbar)
         end
      end
   end
   return R
end
function plyapds!(A::Diagonal{T1}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Complex
   # if adj = true, A contains A'
   n = size(A,1)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   T = real(T1)

   ONE = one(T)
   EPS = eps(T)
   Amax = maximum(abs.(A.diag))
   Amax >= ONE && error("A is not convergent")

   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*Amax

   Wr = Vector{T1}(undef,n)
   Wz = Vector{T1}(undef,n)
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L) + X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = abs(A[j,j])
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          #β = A[j,j]
          β = conj(A[j,j])
          if j < n
             jp1 = j+1
             j1 = jp1:n
             ir1 = 1:n-j
             rbar = view(Wr,ir1)
             z = view(Wz,ir1)
             # Form the right-hand side of (10.16)
             # S = [ s11 0  ]
             #     [ 0   S1 ]
             # rbar = R[j,j1]'
             # z = rbar*α 
             k = jp1
             for ii = 1:n-j
                 rbar[ii] = conj(R[j,k])
                 z[ii] = rbar[ii]*α 
                 k += 1
             end  

             # Solve S1'*ubar*β - ubar + z = 0
             Adiag = view(A.diag,j1)
             solve_in_place!(Adiag, -β, z, disc = true)
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = rbar*β' - S1*z*α
             k = jp1
             for ii = 1:n-j
                 R[j,k] = conj(z[ii])
                 rbar[ii] = conj(rbar[ii]) * β - conj(Adiag[ii]*z[ii]) * α
                 k += 1
             end
             qrupdate!(view(R,j1,j1), rbar)
         end
       end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #        A(L,L)*X(L,L) + X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = abs(A[j,j])
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          β = A[j,j]
          if j > 1
             jm1 = j-1
             j1 = 1:jm1
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  0  ]
             #     [ 0  s11 ]
             # z = rbar*α' 
             rbar = view(Wr,j1)
             z = view(Wz,j1)
             # rbar = R[j1,j]
             # z = rbar*α' 
             for ii = 1:jm1
               rbar[ii] = R[ii,j]
               z[ii] = rbar[ii]*conj(α) 
             end
             # Solve S1*ubar*β'-ubar + z = 0
             Adiag = view(A.diag,j1)
             solve_in_place!(Adiag,-conj(β), z, disc = true)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar*β - S1*z*α
             for ii = 1:jm1
                 R[ii,j] = z[ii]
                 rbar[ii] = rbar[ii]*β - (Adiag[ii]*z[ii])*α
             end
             rqupdate!(view(R,j1,j1), rbar)
         end
      end
   end
   return R
end

"""
    plyapcs!(A,E,R;adj = false, blocksize = 64)

Solve the generalized positive continuous Lyapunov matrix equation

                op(A)Xop(E)' + op(E)*Xop(A)' + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
The pair `(A,E)` is in a generalized real/complex Schur form and `R` is an upper
triangular matrix. The pencil `A-λE` must have only eigenvalues with negative
real parts. `R` contains on output the solution `U`.
The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 
"""
function plyapcs!(A::AbstractMatrix{T1}, E::Union{AbstractMatrix{T1},UniformScaling{Bool}},R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Real
   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (plyapcs!(A, R; adj, blocksize); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   ONE = one(T1)
   ZERO = zero(T1)
   TWO = 2*ONE
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))
   
   # determine the structure of the generalized real Schur form
   ba, p = sfstruct(A)

   T1 <: BlasReal && (WS = Matrix{T1}(undef,n,2))
   WB = Matrix{T1}(undef,n,2)
   WD = Matrix{T1}(undef,n,2)
   Wr = Matrix{T1}(undef,n,2)
   Wv = similar(Wr)
   Wz = similar(Wr)
   Mα = Matrix{T1}(undef,2,2)
   Mβ = Matrix{T1}(undef,2,2)
   η = [ ONE ZERO; SMLNUM ONE]
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #       A(L,L)'*X(L,L)*E(L,L) + E(L,L)'*X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = A[j,j]*E[j,j]
             λ >= ZERO && error("A-λE has eigenvalues with non-negative real parts")
             TEMP = sqrt( -TWO*λ )
             TEMP < SMIN && (TEMP = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular generalized Lyapunov equation")
             iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
             R[j,j] = R[j,j]/TEMP
             Mα[1,1] = TEMP/E[j,j]
             Mβ[1,1] = A[j,j]/E[j,j]
          else
             pglyap2!(view(A,l,l), view(E,l,l), view(R,l,l), Mβ, Mα, adj = true)
          end
          if ll < p
             dll = 1:dl
             js = j
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             # Form the right-hand side of (6.2)
             # z = rbar'*α + s'*u11'
             rbar = view(Wr,ir1,dll)
             v = view(Wv,ir1,dll)
             z = view(Wz,ir1,dll)
             #rbar = R[l,j1]'
             transpose!(rbar,view(R,l,j1))
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             mul!(z, rbar, α)
             k = js+dl-1
             # z <- z + A[l,j1]'*R[l,l]' exploiting upper triangular shape of R
             axpy!(R[k,k],view(A,k,j1),view(z,:,dl))
             dl == 1 || (axpy!(R[js,js],view(A,js,j1),view(z,:,1)); axpy!(R[js,js+1],view(A,js+1,j1),view(z,:,1)))
             # v = (R[l,l]*E[l,j1])'  exploiting upper triangular shape of R
             jj = j
             for ii = 1:n-j+1
                 v[ii,dl] = R[k,k]*E[k,jj]
                 jj += 1
             end               
             if dl == 2
               jj = j
               for ii = 1:n-j+1
                   v[ii,1] = R[js,js]*E[js,jj] + R[js,js+1]*E[js+1,jj]
                   jj += 1
               end
             end
             mul!(z, v, β, 1, 1)
             rmul!(z,-1)
             # Solve A[j1,j1]'*ubar+E[j1,j1]'*ubar*β + z = 0
             E1 = view(E,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, view(A,j1,j1), view(η,dll,dll), E1, β, z,
                                 true, false, 1, false, false, blocksize)
             else
                gsylvs!(view(A,j1,j1), view(η,dll,dll), E1, β, z, view(WB,j1,1:2), view(WD,j1,1:2); adjAC = true, adjBD = false)
             end
             #R[l,j1] = z'
             transpose!(view(R,l,j1),z)
             # update the Cholesky factor R2'*R2 <- R2'*R2 + y'*y
             #y = rbar - (E[j1,j1]'*ubar+E[l,j1]'*R[l,l]') * α'
             #y = rbar - (E[j1,j1]'*ubar+v) * α'
             mul!(v, transpose(UpperTriangular(E1)), z, ONE, ONE)
             #rbar -= v * α'
             mul!(rbar, v, transpose(α), -ONE, ONE)
             qrupdate!(view(R,j1,j1), rbar)
         end
      end
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #      A(L,L)*X(L,L)*E(L,L)' + E(L,L)*X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = A[j,j]*E[j,j]
             λ >= ZERO && error("A-λE has eigenvalues with non-negative real parts")
             TEMP = sqrt( -TWO*λ )
             TEMP < SMIN && (TEMP = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular generalized Lyapunov equation")
             iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
             R[j,j] = R[j,j]/TEMP
             Mα[1,1] = TEMP/E[j,j]
             Mβ[1,1] = A[j,j]/E[j,j]
          else
             pglyap2!(view(A,l,l), view(E,l,l), view(R,l,l), Mβ, Mα, adj = false)
          end
          if ll > 1
             dll = 1:dl
             js = j
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,dll)
             v = view(Wv,j1,dll)
             z = view(Wz,j1,dll)
             #rbar = R[j1,l]
             copyto!(rbar,view(R,j1,l))
             #v = E[j1,l]*R[l,l]  - null allocation code exploiting upper triangular shape of R
             #mul!(v,view(E,j1,l),view(R,l,l))
             k = js-dl+1
             for ii = 1:j
                 v[ii,1] = R[k,k]*E[ii,k]
             end               
             if dl == 2
               for ii = 1:j
                   v[ii,2] = R[js-1,js]*E[ii,js-1] + R[js,js]*E[ii,js]
               end
             end
             # z = rbar*α' + A[j1,l]*R[l,l] + v*β'
             mul!(z, rbar, transpose(α))
             # null allocation code exploiting upper triangular shape of R
             axpy!(R[k,k],view(A,j1,k),view(z,:,1))
             dl == 1 || (axpy!(R[js-1,js],view(A,j1,js-1),view(z,:,2)); axpy!(R[js,js],view(A,j1,js),view(z,:,2)))
             mul!(z, v, transpose(β), 1, 1)
             rmul!(z,-1)
             # Solve A[j1,j1]*ubar+E[j1,j1]*ubar*β' + z = 0
             E1 = view(E,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, view(A,j1,j1), view(η,dll,dll), E1, β, z,
                                 false, true, 1, false, false, blocksize)
             else
                gsylvs!(view(A,j1,j1), view(η,dll,dll), E1, β, z, view(WB,j1,1:2), view(WD,j1,1:2); adjAC = false, adjBD = true)
             end
             #R[j1,l] = z
             copyto!(view(R,j1,l), z)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             #y = rbar - (E[j1,j1]*z+v) * α
             #v <-v + E[j1,j1]*z
             #mul!(v,UpperTriangular(E1),z,ONE,ONE)
             mul!(v, E1, z, ONE, ONE)  # null allocation
             #y = rbar - v * α
             mul!(rbar, v, α, -ONE, ONE)
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
function plyapcs!(A::AbstractMatrix{T1}, E::Union{AbstractMatrix{T1},UniformScaling{Bool}},R::UpperTriangular{T1}; adj = false, blocksize::Int = 64)  where T1 <: Complex
   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))
   (typeof(E) == UniformScaling{Bool} || isempty(E) || (isequal(E,I) && size(E,1) == n)) &&
         (plyapcs!(A, R; adj, blocksize); return)

   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))

   T = real(T1)
   ONE = one(T)
   ZERO = zero(T)
   TWO = 2*ONE
   EPS = eps(T)
   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   T1 <: BlasComplex && (WS = Matrix{T1}(undef,n,1))
   WB = Vector{T1}(undef,n)
   WD = Vector{T1}(undef,n)
   Wr = Matrix{T1}(undef,n,1)
   Wv = similar(Wr)
   Wz = similar(Wr)
   η  = complex(fill(ONE,(1,1)))
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner row by row by
      #       A(L,L)'*X(L,L)*E(L,L) + E(L,L)'*X(L,L)*A(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          δ = -TWO*real(A[j,j]'*E[j,j])
          δ <= ZERO && error("A-λE has unstable eigenvalues")
          TEMP = sqrt( δ )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular generalized Lyapunov equation")
          iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
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
             v = view(Wv,ir1,1:1)
             z = view(Wz,ir1,1:1)
             k = j
             for ii = 1:n-j+1
                rbar[ii] = R[j-1,k]'
                v[ii] = R[j-1,j-1]*E[j-1,k]'
                z[ii] = -(rbar[ii]*α + R[j-1,j-1]*A[j-1,k]' + v[ii]*β[1,1])
                k += 1
             end  
             #  rbar = R[l,j1]'
             #  v = (R[l,l]*E[l,j1])'
             #  #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             # Solve A[j1,j1]'*ubar+E[j1,j1]'*ubar*β + z = 0
             E1 = view(E,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, view(A,j1,j1), η, E1, β, z,
                                 true, false, 1, false, false, blocksize)
             else
                gsylvs!(view(A,j1,j1), η, E1, β, z, view(WB,j1), view(WD,j1); adjAC=true, adjBD=false)
             end
             # v <- v + E1'*z
             mul!(v, UpperTriangular(E1)', z, 1, 1)
             #R[l,j1] = z'
             # update the Cholesky factor R2'*R2 <- R2'*R2 + y'*y
             #y = conj(rbar - (E[j1,j1]'*ubar+E[l,j1]'*R[l,l]') * α')
             k = j
             for ii = 1:n-j+1
                R[j-1,k] = z[ii]'
                rbar[ii] = conj(rbar[ii] - v[ii] * α')
                k += 1
             end
             qrupdate!(view(R,j1,j1), rbar)
          end
      end
      return R
   else
      # The (L,L)th block of X is determined starting from
      # bottom-right corner column by column by
      #      A(L,L)*X(L,L)*E(L,L)' + E(L,L)*X(L,L)*A(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          δ = -TWO*real(A[j,j]'*E[j,j])
          δ <= ZERO && error("A-λE has unstable eigenvalues")
          TEMP = sqrt( δ )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP && error("Singular generalized Lyapunov equation")
          iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
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
             v = view(Wv,j1,1:1)
             z = view(Wz,j1,1:1)
             #rbar = R[j1,l]
             #v = E[j1,l]*R[l,l]
             #z = rbar*α + A[l,j1]'*R[l,l]' + E[l,j1]'*R[l,l]'*β
             #z = rbar*α' + A[j1,l]*R[l,l] + v*β'
             for ii = 1:j
                 rbar[ii] = R[ii,j+1]
                 v[ii] = E[ii,j+1]*R[j+1,j+1]
                 z[ii] = -(rbar[ii]*α' + A[ii,j+1]*R[j+1,j+1] + v[ii]*β[1,1]')
             end
             # Solve A[j1,j1]*ubar+E[j1,j1]*ubar*β' + z = 0
             E1 = view(E,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, view(A,j1,j1), η, E1, β, z,
                                 false, true, 1, false, false, blocksize)
             else
                gsylvs!(view(A,j1,j1), η, E1, β, z, view(WB,j1), view(WD,j1); adjAC=false, adjBD=true)
             end
             # v <- v + E1*z
             mul!(v, UpperTriangular(E1), z, 1, 1)
             #R[j1,l] = z
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             # y = rbar - (E[j1,j1]*ubar+v) * α
             #y = rbar - (E1*z+v) * α
             for ii = 1:j
               R[ii,j+1] = z[ii]
               rbar[ii] -= v[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
"""
    plyapds!(A, R; adj = false, blocksize = 64)

Solve the positive discrete Lyapunov matrix equation

                op(A)Xop(A)' - X + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
`A` is a square real matrix in a real Schur form or a square complex matrix in a
complex Schur form and `R` is an upper triangular matrix.
`A` must have only eigenvalues with moduli less than one.
`R` contains on output the upper triangular solution `U`.
The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 
"""
function plyapds!(A::AbstractMatrix{T1}, R::UpperTriangular{T1}; adj = false, blocksize = 64)  where T1 <: Real
   # check for diagonal A
   isdiag(A) && (return plyapds!(Diagonal(A),R; adj))

   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   ONE = one(T1)
   ZERO = zero(T1)
   TWO = 2*ONE
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   Wr = Matrix{T1}(undef,n,2)
   Wv = similar(Wr)
   Wz = similar(Wr)
   Mα = Matrix{T1}(undef,2,2)
   Mβ = Matrix{T1}(undef,2,2)
   WS = Matrix{T1}(undef,n,2)
   WS2 = Matrix{T1}(undef,n,2) 
   isgn = -1

   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #    A(L,L)'*X(L,L)*A(L,L) - X(L,L) = -R(L,L)'*R(L,L),
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             λ = abs(A[j,j])
             λ >= 1 && error("A is not convergent")
             TEMP = sqrt( (ONE - λ)*(ONE + λ) )
             TEMP < SMIN && (TEMP  = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular Lyapunov equation")
             tα = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/tα
             Mα[1,1] = tα
             Mβ[1,1] = A[j,j]
          else
             plyap2!(view(A,l,l), view(R,l,l), Mβ, Mα, adj = true, disc = true)
          end
          if ll < p
             dll = 1:dl
             js = j
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,dll)
             #v = view(Wv,j1,dll)
             v = view(Wv,ir1,dll)
             z = view(Wz,ir1,dll)
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             # Form the right-hand side of (10.16)
             # z = rbar*α + s'*u11*β
             # rbar = R[l,j1]'
             transpose!(rbar,view(R,l,j1))
             # v = (R[l,l]*A[l,j1])'
             #mul!(v,transpose(view(A,l,j1)),transpose(R[l,l]))
             k = js+dl-1
             jj = j
             for ii = 1:n-j+1
                 v[ii,dl] = R[k,k]*A[k,jj]
                 jj += 1
             end               
             if dl == 2
               jj = j
               for ii = 1:n-j+1
                   v[ii,1] = R[js,js]*A[js,jj] + R[js,js+1]*A[js+1,jj]
                   jj += 1
               end
             end
             #z = rbar*α + A[l,j1]'*R[l,l]*β
             #z = rbar*α + v*β
             mul!(z, rbar, α)
             mul!(z, v, β, -ONE, -ONE)
             # Solve S1'*ubar*β+ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._sylvds_blocked!(WS, WS2, S1, β, z, adj, !adj, isgn, blocksize)
             else
                sylvds!(S1, β, z, WS2; adjA = true, adjB = false, isgn)
             end
             #R[l,j1] = ubar'
             transpose!(view(R,l,j1), z)
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # v += S1'*z
             mul!(v, transpose(S1), z, ONE, ONE)
             if dl == 1
                #y = rbar*β - v*α
               #  mul!(y, v, α)
               #  mul!(y,rbar,β,ONE,-ONE)
                rmul!(v, -α[1,1])
                #mul!(v, rbar, β, ONE, ONE)
                #LinearAlgebra.axpy!(β[1,1], rbar, v)
                axpy!(β[1,1], view(Wr,ir1,1), view(Wv,ir1,1))
             else
               #  F = qr([α; β])
               #  vy = [rbar v]*F.Q
               #  y = vy[:,dl+1:end]
                v =  ([rbar v]*qr([α; β]).Q)[:,dl+1:end]
                # alternative formula of Varga
                #y = rbar - (A[l,j1]'*R[l,l]'+S1'*ubar+ubar)*inv(I+β')*α'
             end
             #RR = view(R,j1,j1)
             qrupdate!(view(R,j1,j1), v)
          end
      end
   else
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #      A(L,L)*X(L,L)*A(L,L)' - X(L,L) = -R(L,L)*R(L,L)',
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             λ = abs(A[j,j])
             λ >= 1 && error("A is not convergent")
             TEMP = sqrt( (ONE - λ)*(ONE + λ) )
             TEMP < SMIN && (TEMP  = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular Lyapunov equation")
             tα = copysign( TEMP, R[j,j])
             R[j,j] = R[j,j]/tα
             Mα[1,1] = tα
             Mβ[1,1] = A[j,j]
          else
             plyap2!(view(A,l,l), view(R,l,l), Mβ, Mα, adj = false, disc = true)
          end
          if ll > 1
             dll = 1:dl
             js = j
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,dll)
             z = view(Wz,j1,dll)
             v = view(Wv,j1,dll)
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             #rbar = R[j1,l]
             copyto!(rbar,view(R,j1,l))
             # v = A[j1,l]*R[l,l]
             # v = view(A,j1,l)*R[l,l]
             #mul!(v,view(A,j1,l),view(R,l,l))
             k = js-dl+1
             for ii = 1:j
                 v[ii,1] = R[k,k]*A[ii,k]
             end               
             if dl == 2
               for ii = 1:j
                   v[ii,2] = R[js-1,js]*A[ii,js-1] + R[js,js]*A[ii,js]
               end
             end
             #z = rbar*α' + v*β'
             mul!(z,rbar,transpose(α))
             mul!(z,v,transpose(β),-ONE,-ONE)
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._sylvds_blocked!(WS, WS2, S1, β, z, adj, !adj, isgn, blocksize)
             else
                sylvds!(S1, β, z, WS2; adjA = false, adjB = true, isgn)
             end
             copyto!(view(R,j1,l), z )
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             #v += S1*ubar
             mul!(v, S1, z, ONE, ONE)
             if dl == 1
                #y = rbar*β - v*α
                rmul!(v,-α[1,1])
                mul!(v, rbar, β, ONE, ONE)
               #  mul!(y, v, α)
               #  mul!(y,rbar,β,ONE,-ONE)
             else
               #  F = qr([α'; β'])
               #  vy = [rbar v]*F.Q
               #  y = vy[:,dl+1:end]
                v =  ([rbar v]*qr([α'; β']).Q)[:,dl+1:end]
             end
             #RR = view(R,j1,j1)
             rqupdate!(view(R,j1,j1), v)
          end
       end
   end
   return R
end
function plyapds!(A::AbstractMatrix{T1}, R::UpperTriangular{T1}; adj = false, blocksize = 64)  where T1 <: Complex
   # check for diagonal A
   isdiag(A) && (return plyapds!(Diagonal(A),R; adj))

   n = LinearAlgebra.checksquare(A)
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   T = real(T1)

   ONE = one(T)
   EPS = eps(T)
   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   Wr = Matrix{T1}(undef,n,1)
   Wv = similar(Wr)
   Wz = similar(Wr)
   WS = Matrix{T1}(undef,n,2)
   WS2 = Vector{T1}(undef,n) 
   isgn = 1
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #     A(L,L)'*X(L,L)*A(L,L) - X(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          λ = abs(A[j,j])
          λ >= ONE && error("A is not convergent")
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
             error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j < n
             js = j
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:1)
             v = view(Wv,ir1,1:1)
             z = view(Wz,ir1,1:1)
             # Form the right-hand side of (10.16)
             # z = rbar*α + s'*u11*β
             k = j
             for ii = 1:n-j+1
                rbar[ii] = R[js,k]'
                v[ii] = R[js,js]*A[js,k]'
                z[ii] = rbar[ii]*α + v[ii]*β[1,1]
                k += 1
             end  
             # Solve S1'*ubar*β + ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._sylvds_blocked!(WS, WS2, S1, -β, z, adj, !adj, isgn, blocksize)
             else
                sylvds!(S1, -β, z, WS2; adjA = true, adjB = false)
             end
             # v <- v + S1'*z
             #mul!(v, UpperTriangular(S1)', z, 1, 1) # faster but involves allocations
             mul!(v, S1', z, 1, 1)  # null allocation
             # R[l,j1] = z'
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             # y = conj(rbar*β' - v * α')
             k = j
             t = β[1,1]'
             for ii = 1:n-j+1
                R[j-1,k] = z[ii]'
                rbar[ii] = conj(rbar[ii] * t - v[ii] * α')
                k += 1
             end
             qrupdate!(view(R,j1,j1), rbar)
         end
      end
   else
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #     A(L,L)*X(L,L)*A(L,L)' - X(L,L) = -R(L,L)*R(L,L)',
      for j = n:-1:1
          λ = abs(A[j,j])
          λ >= ONE && error("A is not convergent")
          TEMP = sqrt( (ONE - λ)*(ONE + λ) )
          TEMP < SMIN && (TEMP  = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
             error("Singular Lyapunov equation")
          iszero(DR) ? α = TEMP : α = sign(R[j,j])*TEMP
          R[j,j] = R[j,j]/α
          l = j:j
          β = A[l,l]
          if j > 1
             js = j
             j -= 1
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             rbar = view(Wr,j1,1:1)
             v = view(Wv,j1,1:1)
             z = view(Wz,j1,1:1)
             #  rbar = R[j1,l]
             #  v = A[j1,l]*R[l,l]
             #  z = rbar*α' + v*β'
             for ii = 1:j
                 rbar[ii] = R[ii,js]
                 v[ii] = A[ii,js]*R[js,js]
                 z[ii] = rbar[ii]*α' + v[ii]*β[1,1]'
             end
             # Solve S1*ubar*β'+ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._sylvds_blocked!(WS, WS2, S1, -β, z, adj, !adj, isgn, blocksize)
             else
                sylvds!(S1, -β, z, WS2; adjA = false, adjB = true)
             end
             # v <- v + S1*z
             #mul!(v, UpperTriangular(S1), z, 1, 1) # involves allocations
             mul!(v, S1, z, 1, 1)
             #  R[j1,l] = ubar
             #  update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             #  y = rbar*β - v * α
             for ii = 1:j
                 R[ii,j+1] = z[ii]
                 rbar[ii] = rbar[ii]*β[1,1] - v[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
"""
    plyapds!(A,E,R;adj = false, blocksize = 64)

Solve the generalized positive discrete Lyapunov matrix equation

                op(A)Xop(A)' - op(E)Xop(E)' + op(R)*op(R)' = 0

for `X = op(U)*op(U)'`, where `op(K) = K` if `adj = false` and `op(K) = K'` if `adj = true`.
The pair `(A,E)` of square real or complex matrices is in a generalized Schur form
and `R` is an upper triangular matrix. `A-λE` must have only eigenvalues with
moduli less than one. `R` contains on output the upper triangular solution `U`.
The parameter `blocksize` (Default: `blocksize = 64`) specifies the blocksize to be used in the recursive blocking based Sylvester equation solvers. 
This option can be used only for `BlasFloat` type data. 
"""
function plyapds!(A::AbstractMatrix{T1}, E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, R::UpperTriangular{T1}; adj::Bool = false, blocksize::Int = 64)  where T1 <: Real
   # The method of [1] for the discrete case is implemented.

   # [1] Penzl, T.
   #     Numerical solution of generalized Lyapunov equations.
   #     Advances in Comp. Math., vol. 8, pp. 33-48, 1998.

   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (plyapds!(A, R; adj, blocksize); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   ONE = one(T1)
   ZERO = zero(T1)
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))


   # determine the structure of the real Schur form
   ba, p = sfstruct(A)

   T1 <: BlasReal && (WS = Matrix{T1}(undef,n,2))
   WB = Matrix{T1}(undef,n,2)
   WD = Matrix{T1}(undef,n,2)
   Wr = Matrix{T1}(undef,n,2)
   Wv = similar(Wr)
   Wz = similar(Wr)
   Mα = Matrix{T1}(undef,2,2)
   Mβ = Matrix{T1}(undef,2,2)
   η = [ -ONE ZERO; ZERO -ONE]
   if adj
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #     A(L,L)'*X(L,L)*A(L,L) - E(L,L)'*X(L,L)*E(L,L) = -R(L,L)'*R(L,L),
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          if dl == 1
             abs(A[j,j]) >= abs(E[j,j]) && error("A-λE must have only eigenvalues with moduli less than one")
             TEMP = sqrt( real((E[j,j] - A[j,j])*(E[j,j] + A[j,j])) )
             TEMP < SMIN && (TEMP = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular generalized discrete Lyapunov equation")
             iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
             R[j,j] = R[j,j]/TEMP
             Mα[1,1] = TEMP/E[j,j]
             Mβ[1,1] = A[j,j]/E[j,j]
          else
             pglyap2!(view(A,l,l), view(E,l,l), view(R,l,l), Mβ, Mα, adj = true, disc = true)
          end
          if ll < p
             dll = 1:dl
             js = j
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             j += dl
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,dll)
             v = view(Wv,ir1,dll)
             z = view(Wz,ir1,dll)
             # rbar = R[l,j1]'
             transpose!(rbar,view(R,l,j1))
             # Form the right-hand side of (22) in [1]
             # z = -rbar*α - v*β + (R[l,l]*E[l,j1])' 
             # where v = (R[l,l]*A[l,j1])'
             mul!(z, rbar, α)
             rmul!(z,-1)
             k = js+dl-1
             # z <- z + E[l,j1]'*R[l,l]' exploiting upper triangular shape of R
             axpy!(R[k,k],view(E,k,j1),view(z,:,dl))
             dl == 1 || (axpy!(R[js,js],view(E,js,j1),view(z,:,1)); axpy!(R[js,js+1],view(E,js+1,j1),view(z,:,1)))
             # v = (R[l,l]*A[l,j1])'  exploiting upper triangular shape of R
             jj = j
             for ii = 1:n-j+1
                 v[ii,dl] = R[k,k]*A[k,jj]
                 jj += 1
             end               
             if dl == 2
               jj = j
               for ii = 1:n-j+1
                   v[ii,1] = R[js,js]*A[js,jj] + R[js,js+1]*A[js+1,jj]
                   jj += 1
               end
             end
             mul!(z, v, β, -1, 1)

             # Solve S1'*ubar*β-E[j1,j1]'*ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, S1, β, view(E,j1,j1), view(η,dll,dll), z,
                                 true, false, 1, false, false, blocksize)
             else
                gsylvs!(S1, β, view(E,j1,j1), view(η,dll,dll), z, view(WB,j1,1:2), view(WD,j1,1:2); adjAC = true, adjBD = false)
             end
             # R[l,j1] = z'
             transpose!(view(R,l,j1),z)
             #v += S1'*z
             mul!(v, transpose(S1), z, ONE, ONE)
             # update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             if dl == 1
                #y = rbar*β - v*α
                rmul!(rbar, β[1,1])
                #LinearAlgebra.axpy!(-α[1,1], v, rbar)
                axpy!(-α[1,1], view(Wv,ir1,1), view(Wr,ir1,1))
             else
                rbar =  ([rbar v]*qr([α; β]).Q)[:,dl+1:end]
             end
             qrupdate!(view(R,j1,j1), rbar)
         end
      end
   else
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #     A(L,L)*X(L,L)*A(L,L)' - E(L,L)*X(L,L)*E(L,L)' = -R(L,L)*R(L,L)',
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          if dl == 1
             abs(A[j,j]) >= abs(E[j,j]) && error("A-λE must have only eigenvalues with moduli less than one")
             TEMP = sqrt( real((E[j,j] - A[j,j])*(E[j,j] + A[j,j])) )
             TEMP < SMIN && (TEMP = SMIN)
             DR = abs( R[j,j] )
             TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
                error("Singular generalized discrete Lyapunov equation")
             iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
             R[j,j] = R[j,j]/TEMP
             Mα[1,1] = TEMP/E[j,j]
             Mβ[1,1] = A[j,j]/E[j,j]
          else
             pglyap2!(view(A,l,l), view(E,l,l), view(R,l,l), Mβ, Mα, adj = false, disc = true)
          end
          if ll > 1
             dll = 1:dl
             js = j
             α = view(Mα,dll,dll)
             β = view(Mβ,dll,dll)
             j -= dl
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             # z = rbar*α' + s*u11
             rbar = view(Wr,j1,dll)
             v = view(Wv,j1,dll)
             z = view(Wz,j1,dll)
             #rbar = R[j1,l]
             copyto!(rbar,view(R,j1,l))
             #v = A[j1,l]*R[l,l]
             k = js-dl+1
             for ii = 1:j
                 v[ii,1] = R[k,k]*A[ii,k]
             end               
             if dl == 2
               for ii = 1:j
                   v[ii,2] = R[js-1,js]*A[ii,js-1] + R[js,js]*A[ii,js]
               end
             end
             #z = -rbar*α' - v*β' + E[j1,l]*R[l,l]
             mul!(z, rbar, transpose(α))
             mul!(z, v, transpose(β), -1, -1)
             axpy!(R[k,k],view(E,j1,k),view(z,:,1))
             dl == 1 || (axpy!(R[js-1,js],view(E,j1,js-1),view(z,:,2)); axpy!(R[js,js],view(E,j1,js),view(z,:,2)))

             # Solve S1*ubar*β'-E[j1,j1]*ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasReal
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, S1, β, view(E,j1,j1), view(η,dll,dll), z,
                                 false, true, 1, false, false, blocksize)
             else
                gsylvs!(S1, β, view(E,j1,j1), view(η,dll,dll), z, view(WB,j1,1:2), view(WD,j1,1:2); adjAC = false, adjBD = true)
             end
             #R[j1,l] = z
             copyto!(view(R,j1,l),z)
             # update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             #v += S1*ubar
             mul!(v, S1, z, ONE, ONE)
             if dl == 1
                # y = rbar*β - v*α
                rmul!(rbar, β[1,1])
                axpy!(-α[1,1], view(Wv,j1,1), view(Wr,j1,1))
             else
                rbar =  ([rbar v]*qr([α'; β']).Q)[:,dl+1:end]
             end
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
function plyapds!(A::AbstractMatrix{T1}, E::Union{AbstractMatrix{T1},UniformScaling{Bool}}, R::UpperTriangular{T1}; adj = false, blocksize::Int = 64)  where T1 <: Complex
   n = LinearAlgebra.checksquare(A)
   (typeof(E) == UniformScaling{Bool} || (isequal(E,I) && size(E,1) == n)) && (plyapds!(A, R; adj, blocksize); return)
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   LinearAlgebra.checksquare(R) == n || throw(DimensionMismatch("R must be a $n x $n upper triangular matrix"))

   T = real(T1)
   ONE = one(T)
   EPS = eps(T)
   SMLNUM = sqrt(_safemin(T))/EPS
   BIGNUM = ONE / SMLNUM
   SMIN = EPS*maximum(abs.(A))

   T1 <: BlasComplex && (WS = Matrix{T1}(undef,n,1))
   WB = Vector{T1}(undef,n)
   WD = Vector{T1}(undef,n)
   Wr = Matrix{T1}(undef,n,1)
   Wv = similar(Wr)
   Wz = similar(Wr)
   η  = complex(fill(-ONE,(1,1)))
   if adj
     # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #      A(L,L)'*X(L,L)*A(L,L) - E(L,L)'*X(L,L)*E(L,L) = -R(L,L)'*R(L,L),
      for j = 1:n
          abs(A[j,j]) >= abs(E[j,j]) && error("A-λE must have only eigenvalues with moduli less than one")
          TEMP = sqrt( real((E[j,j]' - A[j,j]')*(E[j,j] + A[j,j])) )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
             error("Singular generalized discrete Lyapunov equation")
          iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          if j < n
             js = j
             j += 1
             j1 = j:n
             ir1 = 1:n-j+1
             rbar = view(Wr,ir1,1:1)
             v = view(Wv,ir1,1:1)
             z = view(Wz,ir1,1:1)
             # Form the right-hand side of (10.16)
             # rbar = R[l,j1]'
             # v = R[l,l]*A[l,j1]'
             # z = rbar*α + A[l,j1]'*R[l,l]*β - E[l,j1]'*R[l,l]
             k = j
             for ii = 1:n-j+1
                 rbar[ii] = R[js,k]'
                 v[ii] = R[js,js]*A[js,k]'
                 z[ii] = -(rbar[ii]*α - R[js,js]*E[js,k]' + v[ii]*β[1,1])
                 k += 1
             end  
             # Solve S1'*ubar*β-E[j1,j1]'*ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, S1, β, view(E,j1,j1), η, z,
                                 true, false, 1, false, false, blocksize)
             else
                gsylvs!(S1, β, view(E,j1,j1), η, z, view(WB,j1), view(WD,j1); adjAC = true, adjBD = false)
             end
             # v <- v + S1'*z
             #mul!(v, UpperTriangular(S1)', z, 1, 1)
             mul!(v, S1', z, 1, 1)  # no allocations
             #  R[l,j1] = z'
             #  update the Cholesky factor R1'*R1 <- R1'*R1 + y'*y
             #  y = conj(rbar*β' - v * α')
             k = j
             t = β[1,1]'
             for ii = 1:n-j+1
                R[js,k] = z[ii]'
                rbar[ii] = conj(rbar[ii] * t - v[ii] * α')
                k += 1
             end
             qrupdate!(view(R,j1,j1), rbar)
          end
      end
   else
      # The (L,L)th block of X is determined starting from
      # upper-left corner column by column by
      #     A(L,L)*X(L,L)*A(L,L)' - E(L,L)*X(L,L)*E(L,L)' = -R(L,L)*R(L,L)',
      for j = n:-1:1
          abs(A[j,j]) >= abs(E[j,j]) && error("A-λE must have only eigenvalues with moduli less than one")
          TEMP = sqrt( real((E[j,j]' - A[j,j]')*(E[j,j] + A[j,j])) )
          TEMP < SMIN && (TEMP = SMIN)
          DR = abs( R[j,j] )
          TEMP < ONE && DR > ONE && DR > BIGNUM*TEMP &&
            error("Singular generalized discrete Lyapunov equation")
          iszero(DR) || (TEMP = sign(R[j,j])*TEMP)
          R[j,j] = R[j,j]/TEMP
          l = j:j
          β = A[l,l]/E[j,j]
          α = TEMP/E[j,j]
          if j > 1
             js = j
             j -= 1
             j1 = 1:j
             # Form the right-hand side corresponding to the dual of (6.2)
             # S = [ S1  s  ]
             #     [ 0  s11 ]
             rbar = view(Wr,j1,1:1)
             v = view(Wv,j1,1:1)
             z = view(Wz,j1,1:1)
             #  rbar = R[j1,l]
             #  v = A[j1,l]*R[l,l]
             #  z = rbar*α + A[j1,l]*R[l,l]*β' - E[j1,l]*R[l,l]
             for ii = 1:j
                 rbar[ii] = R[ii,j+1]
                 v[ii] = A[ii,js]*R[js,js]
                 z[ii] = -rbar[ii]*α' + E[ii,js]*R[js,js] - v[ii]*β[1,1]'
             end
             # Solve S1*ubar*β'-E[j1,j1]*ubar + z = 0
             S1 = view(A,j1,j1)
             if T1 <: BlasComplex
                MatrixEquations._gsylvs_blocked!(WS, WB, WD, S1, β, view(E,j1,j1), η, z,
                                 false, true, 1, false, false, blocksize)
             else
                gsylvs!(S1, β, view(E,j1,j1), η, z, view(WB,j1), view(WD,j1); adjAC = false, adjBD = true)
             end
             # v <- v + S1*z
             #mul!(v, UpperTriangular(S1), z, 1, 1)
             mul!(v, S1, z, 1, 1)  # no allocations
             #  R[j1,l] = z
             #  update the Cholesky factor R1*R1' <- R1*R1' + y*y'
             #  y = rbar*β - v * α
             for ii = 1:j
                 R[ii,j+1] = z[ii]
                 rbar[ii] = rbar[ii] * β[1,1] - v[ii]*α
             end
             rqupdate!(view(R,j1,j1), rbar)
          end
       end
   end
   return R
end
"""
    plyap2!(A, R, β, α; adj = false, disc = false) -> R

Solve for the Cholesky factor  `U`  of  `X`,

     op(U)*op(U)' = X,

where  `U`  is a two-by-two upper triangular matrix, either the
continuous-time two-by-two Lyapunov equation

      op(A)*X + X*op(A)' = -op(R)*op(R)',

when disc = false, or the discrete-time two-by-two Lyapunov equation

      op(A)*X*op(A)' - X = -op(R)*op(R)',

when `disc = true`, where `op(K) = K` if `adj = false` or `op(K) = K'`
if `adj = true`,  `A`  is a two-by-two matrix with complex conjugate eigenvalues,
`R`  is a two-by-two upper triangular matrix.
The routine also computes two matrices, `β` and `α`, so that

      U*A = β*U  and  U*α = R,  if  adj = false, or

      β*U = U*A  and  α*U = R,  if  adj = true,

which are used by the general Lyapunov solver. The computed `U` is returned in `R`.

In the continuous-time case  `A`  must be stable, so that its
eigenvalues must have strictly negative real parts.
In the discrete-time case  `A`  must be convergent, that is, its eigenvalues
must have moduli less than one. These conditions are checked and
an error message is issued if not fulfilled.

If the Lyapunov equation is numerically singular, then small perturbations in `A` can make
one or more of the eigenvalues have a non-negative real part, if `disc = false`, or
can make one or more of the eigenvalues lie outside the unit circle, if `disc = true`.
If this situation is detected, an error message is issued.
"""
function plyap2!(A::AbstractMatrix{T}, R::AbstractMatrix{T}, β::AbstractMatrix{T}, α::AbstractMatrix{T}; adj = false, disc = false) where T<:Real
   errtext = "Singular Lyapunov equation"
   ZERO = zero(T)
   ONE = one(T)
   TWO = 2*ONE
   small = 2*sqrt(_safemin(T))
   BIGNUM = ONE / small
   
   # Fix 1: Pure scalar max instead of abs.(A) broadcast
   SMIN = eps(max(abs(A[1,1]), abs(A[1,2]), abs(A[2,1]), abs(A[2,2])))
   
   noadj = !adj
   S11 = A[1,1]
   S12 = A[1,2]
   S21 = A[2,1]
   S22 = A[2,2]
   
   TEMPR, TEMPI, E1, E2 = _lanv2( S11, S12, S21, S22)
   TEMPI == ZERO && error("A has real eigenvalues")
   ABSB = hypot(E1,E2)
   if disc
      ABSB >= ONE && error("A is not convergent")
   else
      E1 >= ZERO && error("A is not stable")
   end

   TEMP1 = S11 - E1
   noadj ? TEMP2 = -E2 : TEMP2 =  E2
   CSQR, CSQI, SNQ = cgivens2( TEMP1, TEMP2, S21, small )

   TEMP1 = CSQR*S12 - SNQ*S11
   TEMP2 = CSQI*S12
   TEMPR   = CSQR*S22 - SNQ*S21
   TEMPI   = CSQI*S22
   T1      = CSQR*TEMP1 - CSQI*TEMP2 + SNQ*TEMPR
   T2      = CSQR*TEMP2 + CSQI*TEMP1 + SNQ*TEMPI

   if noadj
      TEMP1 =  CSQR*R[2,2] - SNQ*R[1,2]
      TEMP2 = -CSQI*R[2,2]
      CSPR, CSPI, SNP, P1 = cgivens2( TEMP1, TEMP2, -SNQ*R[1,1], small )

      TEMP1 =  CSQR*R[1,2] + SNQ*R[2,2]
      TEMP2 = -CSQI*R[1,2]
      TEMPR   =  CSQR*R[1,1]
      TEMPI   = -CSQI*R[1,1]
      P2R     =  CSPR*TEMP1 - CSPI*TEMP2 + SNP*TEMPR
      P2I     = -CSPR*TEMP2 - CSPI*TEMP1 - SNP*TEMPI
      P3R     =  CSPR*TEMPR   + CSPI*TEMPI   - SNP*TEMP1
      P3I     =  CSPR*TEMPI   - CSPI*TEMPR   - SNP*TEMP2
   else
      TEMP1 = CSQR*R[1,1] + SNQ*R[1,2]
      TEMP2 = CSQI*R[1,1]
      CSPR, CSPI, SNP, P1 = cgivens2( TEMP1, TEMP2, SNQ*R[2,2], small  )

      TEMP1 = CSQR*R[1,2] - SNQ*R[1,1]
      TEMP2 = CSQI*R[1,2]
      TEMPR   = CSQR*R[2,2]
      TEMPI   = CSQI*R[2,2]
      P2R     = CSPR*TEMP1 - CSPI*TEMP2 + SNP*TEMPR
      P2I     = CSPR*TEMP2 + CSPI*TEMP1 + SNP*TEMPI
      P3R     = CSPR*TEMPR   + CSPI*TEMPI   - SNP*TEMP1
      P3I     = CSPI*TEMPR   - CSPR*TEMPI   + SNP*TEMP2
   end

   if P3I == ZERO
      P3  = abs( P3R )
      DP1 = copysign( ONE, P3R )
      DP2 = ZERO
   else
      P3  = hypot(P3R,P3I)
      DP1 = P3R/P3
      DP2 = -P3I/P3
   end

   if disc
      ALPHA = sqrt( abs( ONE - ABSB )*( ONE + ABSB ) )
   else
      ALPHA = sqrt( abs( TWO*E1 ) )
   end

   ALPHA < SMIN && (ALPHA = SMIN)
   ABST = abs( P1 )
   ALPHA < ONE && ABST > ONE && ABST > BIGNUM*ALPHA && error("$errtext")
   V1 = P1/ALPHA

   if disc
      G1 = (ONE - E1 )*( ONE + E1 ) + E2*E2
      G2 = -TWO*E1*E2
      ABSG = hypot(G1,G2)
      ABSG < SMIN && (ABSG = SMIN)
      TEMP1 = ALPHA*P2R + V1*( E1*T1 - E2*T2 )
      TEMP2 = ALPHA*P2I + V1*( E1*T2 + E2*T1 )
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG && error("$errtext")
      TEMP1 = TEMP1/ABSG
      TEMP2 = TEMP2/ABSG

      V2R    = G1*TEMP1 + G2*TEMP2
      V2I    = G1*TEMP2 - G2*TEMP1
      ABST   = max( abs( V2R ), abs( V2I ) )
      ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG && error("$errtext")
      V2R = V2R/ABSG
      V2I = V2I/ABSG

      TEMP1 = P1*T1 - TWO*E2*P2I
      TEMP2 = P1*T2 + TWO*E2*P2R
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG && error("$errtext")
      TEMP1 = TEMP1/ABSG
      TEMP2 = TEMP2/ABSG

      YR  = -( G1*TEMP1 + G2*TEMP2 )
      YI  = -( G1*TEMP2 - G2*TEMP1 )
      ABST    = max( abs( YR ), abs( YI ) )
      ABSG < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSG && error("$errtext")
      YR = YR/ABSG
      YI = YI/ABSG
   else
      ABSB < SMIN && (ABSB = SMIN)
      TEMP1 = ALPHA*P2R + V1*T1
      TEMP2 = ALPHA*P2I + V1*T2
      ABST    = max( abs( TEMP1 ), abs( TEMP2 ) )
      ABSB < ONE  &&  ABST > ONE && ABST > BIGNUM*ABSB && error("$errtext")
      TEMP1 = TEMP1/( TWO*ABSB )
      TEMP2 = TEMP2/( TWO*ABSB )
      V2R     = -(E1*TEMP1 + E2*TEMP2)
      V2I     = -(E1*TEMP2 - E2*TEMP1)
      ABST = max( abs( V2R ), abs( V2I ) )
      ABSB < ONE  &&  ABST > ONE &&  ABST > BIGNUM*ABSB && error("$errtext")
      V2R = V2R/ABSB
      V2I = V2I/ABSB
      YR  = P2R - ALPHA*V2R
      YI  = P2I - ALPHA*V2I
   end

   V3     = hypot3(P3,YR,YI)
   ALPHA < ONE  &&  V3 > ONE && V3 > BIGNUM*ALPHA && error("$errtext")
   V3 = V3/ALPHA

   if noadj
      X11R   =  CSQR*V3
      X11I   =  CSQI*V3
      X21R   =  SNQ*V3
      X21I   =  ZERO
      X12R   =  CSQR*V2R+CSQI*V2I-SNQ*V1
      X12I   = -CSQR*V2I+CSQI*V2R
      X22R   =  CSQR*V1 + SNQ*V2R
      X22I   = -CSQI*V1 - SNQ*V2I
      X22I = -X22I
      CSTR, CSTI, SNT, TMP = cgivens2( X22R, X22I, X21R, small )
      U22 = TMP
      U12 = CSTR*X12R - CSTI*X12I + SNT*X11R
      TEMPR  = CSTR*X11R + CSTI*X11I - SNT*X12R
      TEMPI  = CSTR*X11I - CSTI*X11R - SNT*X12I
      if TEMPI == ZERO
         U11 = abs( TEMPR )
         DT1    = copysign( ONE, TEMPR )
         DT2    = ZERO
      else
         U11 = hypot(TEMPR,TEMPI)
         DT1    = TEMPR/U11
         DT2    = -TEMPI/U11
      end
   else
      X11R   =  CSQR*V1 - SNQ*V2R
      X11I   = -CSQI*V1 + SNQ*V2I
      X21R   = -SNQ*V3
      X21I   =  ZERO
      X12R   =  CSQR*V2R + CSQI*V2I + SNQ*V1
      X12I   = -CSQR*V2I + CSQI*V2R
      X22R   =  CSQR*V3
      X22I   =  CSQI*V3
      CSTR, CSTI, SNT, TMP = cgivens2( X11R, X11I, X21R, small  )
      U11 = TMP
      U12 = CSTR*X12R + CSTI*X12I + SNT*X22R
      TEMPR  = CSTR*X22R - CSTI*X22I - SNT*X12R
      TEMPI  = CSTR*X22I + CSTI*X22R - SNT*X12I
      if TEMPI == ZERO
         U22 = abs( TEMPR )
         DT1    = copysign( ONE, TEMPR )
         DT2    = ZERO
      else
         U22 = hypot(TEMPR,TEMPI)
         DT1    = TEMPR/U22
         DT2    = -TEMPI/U22
      end
   end

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
      X11R =  CSTR*E1 + CSTI*E2
      X11I = -CSTR*E2 + CSTI*E1
      X21R =  SNT*E1
      X21I = -SNT*E2
      X12R =  CSTR*GAMMA1 + CSTI*GAMMA2 - SNT*E1
      X12I = -CSTR*GAMMA2 + CSTI*GAMMA1 - SNT*E2
      X22R =  CSTR*E1 + CSTI*E2 + SNT*GAMMA1
      X22I =  CSTR*E2 - CSTI*E1 - SNT*GAMMA2

      # Mutating pre-allocated β
      β[1,1] = CSTR*X11R + CSTI*X11I - SNT*X12R
      TEMPR  = CSTR*X21R + CSTI*X21I - SNT*X22R
      TEMPI  = CSTR*X21I - CSTI*X21R - SNT*X22I
      β[2,1] = DT1*TEMPR   - DT2*TEMPI
      TEMPR  = CSTR*X12R - CSTI*X12I + SNT*X11R
      TEMPI  = CSTR*X12I + CSTI*X12R + SNT*X11I
      β[1,2] = DT1*TEMPR   + DT2*TEMPI
      β[2,2] = CSTR*X22R - CSTI*X22I + SNT*X21R

      TEMPR  =  DP1*ETA
      TEMPI  = -DP2*ETA
      X11R =  CSPR*TEMPR - CSPI*TEMPI + SNP*DELTA1
      X11I =  CSPR*TEMPI + CSPI*TEMPR - SNP*DELTA2
      X21R =  SNP*ALPHA
      X12R = -SNP*TEMPR + CSPR*DELTA1 - CSPI*DELTA2
      X12I = -SNP*TEMPI - CSPR*DELTA2 - CSPI*DELTA1
      X22R =  CSPR*ALPHA
      X22I = -CSPI*ALPHA

      # Mutating pre-allocated α
      TEMPR  = CSTR*X11R - CSTI*X11I - SNT*X21R
      TEMPI  = CSTR*X22I + CSTI*X22R
      α[1,1] = DT1*TEMPR   + DT2*TEMPI
      TEMPR  = CSTR*X12R - CSTI*X12I - SNT*X22R
      TEMPI  = CSTR*X12I + CSTI*X12R - SNT*X22R
      α[1,2] = DT1*TEMPR   + DT2*TEMPI
      α[2,1] = ZERO
      α[2,2] = CSTR*X22R + CSTI*X22I + SNT*X12R
   else
      X11R =  CSTR*E1 + CSTI*E2
      X11I =  CSTR*E2 - CSTI*E1
      X21R = -SNT*E1
      X21I = -SNT*E2
      X12R =  CSTR*GAMMA1 - CSTI*GAMMA2 + SNT*E1
      X12I = -CSTR*GAMMA2 - CSTI*GAMMA1 - SNT*E2
      X22R =  CSTR*E1 + CSTI*E2 - SNT*GAMMA1
      X22I = -CSTR*E2 + CSTI*E1 + SNT*GAMMA2

      # Mutating pre-allocated β
      β[1,1] = CSTR*X11R - CSTI*X11I + SNT*X12R
      TEMPR  = CSTR*X21R - CSTI*X21I + SNT*X22R
      TEMPI  = CSTR*X21I + CSTI*X21R + SNT*X22I
      β[2,1] = DT1*TEMPR   - DT2*TEMPI
      TEMPR  = CSTR*X12R + CSTI*X12I - SNT*X11R
      TEMPI  = CSTR*X12I - CSTI*X12R - SNT*X11I
      β[1,2] = DT1*TEMPR   + DT2*TEMPI
      β[2,2] = CSTR*X22R + CSTI*X22I - SNT*X21R

      TEMPR  =  DP1*ETA
      TEMPI  = -DP2*ETA
      X11R =  CSPR*ALPHA
      X11I =  CSPI*ALPHA
      X21R =  SNP*ALPHA
      X12R =  CSPR*DELTA1 + CSPI*DELTA2 - SNP*TEMPR
      X12I = -CSPR*DELTA2 + CSPI*DELTA1 - SNP*TEMPI
      X22R =  CSPR*TEMPR + CSPI*TEMPI + SNP*DELTA1
      X22I =  CSPR*TEMPI - CSPI*TEMPR - SNP*DELTA2

      # Mutating pre-allocated α
      α[1,1] = CSTR*X11R - CSTI*X11I + SNT*X12R
      α[2,1] = ZERO
      α[1,2] = CSTR*X12R + CSTI*X12I - SNT*X11R
      TEMPR  = CSTR*X22R + CSTI*X22I - SNT*X21R
      TEMPI  = CSTR*X22I - CSTI*X22R
      α[2,2] = DT1*TEMPR   + DT2*TEMPI
   end

   R[1,1] = U11
   R[1,2] = U12
   R[2,2] = U22
   return R
end
"""
    pglyap2!(A, E, R, β, α; adj = false, disc = false) -> R

Solve for the Cholesky factor  `U`  of  `X`,

     op(U)*op(U)' = X,

where  `U`  is a two-by-two upper triangular matrix, either the
continuous-time two-by-two generalized Lyapunov equation

      op(A)*X*op(E)' + op(E)*X*op(A)' = -op(R)*op(R)',

when disc = false, or the discrete-time two-by-two Lyapunov equation

      op(A)*X*op(A)' - op(E)*X*op(E)' = -op(R)*op(R)',

when `disc = true`, where `op(K) = K` if `adj = false` or `op(K) = K'`
if `adj = true`,  `A` and `E` are two-by-two matrices such that the pencil `A-λE`
has complex conjugate eigenvalues and `R`  is a two-by-two upper triangular matrix.
The routine also computes two matrices, `β` and `α`, so that,
for `adj = true`:

      β*U*E = U*A
      α*U*E = R

for `adj = false`:

      E*U*β = A*U
      E*U*α = R
      α = R'*inv(E')*inv(U')

which are used by the general Lyapunov solver.

The pencil `A-λE` must have a pair of complex conjugate eigenvalues.
In the continuous-time case the eigenvalues must have strictly negative
real parts, while, in the discrete-time case, the eigenvalues must have
moduli less than unity. These conditions are checked and
an error message is issued if not fulfilled.

If the Lyapunov equation is numerically singular, then small perturbations in `A` or `E`
can make one or more of the eigenvalues have a non-negative real part, if `disc = false`, or
can make one or more of the eigenvalues lie outside the unit circle, if `disc = true`.
If this situation is detected, an error message is issued.
"""
function pglyap2!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, R::AbstractMatrix{T1}, β::AbstractMatrix{T1}, α::AbstractMatrix{T1}; adj = false, disc = false) where T1 <: Real
   # This function is based on the SLICOT routine SG03BX, which implements the
   # generalization of the method due to Hammarling ([1], section 6) for Lyapunov
   # equations of order 2. A more detailed description is given in [2].
   # The 2x2 matrix is allowed to be a full matrix.

   # [1] Hammarling S. J.
   #     Numerical solution of the stable, non-negative definite Lyapunov equation.
   #     IMA J. Num. Anal., 2, pp. 303-325, 1982.
   # [2] Penzl, T.
   #     Numerical solution of generalized Lyapunov equations.
   #     Advances in Comp. Math., vol. 8, pp. 33-48, 1998.

   errtext = "Singular Lyapunov equation"

   ONE = one(T1)
   ZERO = zero(T1)
   TWO = 2*ONE
   EPS = eps(T1)
   SMLNUM = sqrt(_safemin(T1))/EPS
   small = SMLNUM

   noadj = !adj
   ISCONT = !disc

   # Extract scalars to avoid array copying or allocations
   AA11, AA12, AA21, AA22 = A[1,1], A[1,2], A[2,1], A[2,2]
   EE11, EE12, EE21, EE22 = E[1,1], E[1,2], E[2,1], E[2,2]
   RR11, RR12, RR21, RR22 = R[1,1], R[1,2], R[2,1], R[2,2]

   if noadj
      AA11, AA22 = AA22, AA11
      EE11, EE22 = EE22, EE11
      RR11, RR22 = RR22, RR11
   end
   if iszero(EE21)
      scale1, scale2, LAMR, W, LAMI = _lag2(AA11, AA12, AA21, AA22, EE11, EE12, EE22, small)
   else
      E11, E12, E21, E22 = EE11, EE12, EE21, EE22
      G, E11 = givens(E11, E21, 1, 2)
      E12_new = G.c * E12 + G.s * E22
      E22_new = -G.s * E12 + G.c * E22
      
      A11_new = G.c * AA11 + G.s * AA21
      A12_new = G.c * AA12 + G.s * AA22
      A21_new = -G.s * AA11 + G.c * AA21
      A22_new = -G.s * AA12 + G.c * AA22

      scale1, scale2, LAMR, W, LAMI = _lag2(A11_new, A12_new, A21_new, A22_new, E11, E12_new, E22_new, small)
   end  
 
   LAMI == ZERO && error("The pair (A,E) has real generalized eigenvalues")
   # Compute left orthogonal transformation matrix Q (modified to cope with nonzero E[2,1])
   CR, CI, SR, SI, L = cgivensc2(scale1*AA11 - EE11*LAMR, -EE11*LAMI, scale1*AA21 - EE21*LAMR, -EE21*LAMI, small)   

   # Explicit scalar 2x2 operations
   # QR = [ CR SR; -SR CR ]
   # QI = [ -CI  -SI; -SI CI ]
   # A := Q * A (AR = QR*AA, AI = QI*AA)
   AR11 =  CR*AA11 + SR*AA21; AR12 =  CR*AA12 + SR*AA22
   AR21 = -SR*AA11 + CR*AA21; AR22 = -SR*AA12 + CR*AA22
   AI11 = -CI*AA11 - SI*AA21; AI12 = -CI*AA12 - SI*AA22
   AI21 = -SI*AA11 + CI*AA21; AI22 = -SI*AA12 + CI*AA22

   # E := Q * E (ER = QR*EE, EI = QI*EE)
   ER11 =  CR*EE11 + SR*EE21; ER12 =  CR*EE12 + SR*EE22
   ER21 = -SR*EE11 + CR*EE21; ER22 = -SR*EE12 + CR*EE22
   EI11 = -CI*EE11 - SI*EE21; EI12 = -CI*EE12 - SI*EE22
   EI21 = -SI*EE11 + CI*EE21; EI22 = -SI*EE12 + CI*EE22
 
   # Compute right orthogonal transformation matrix Z.
   # ZR =  [ CR SR; -SR CR ]
   # ZI =  [ CI -SI; -SI -CI ]
   CR1, CI1, SR1, SI1, L = cgivensc2(ER22, EI22, ER21, EI21, small)
   ZR11, ZR12, ZR21, ZR22 =  CR1,  SR1, -SR1,  CR1
   ZI11, ZI12, ZI21, ZI22 =  CI1, -SI1, -SI1, -CI1

   # E := E * Z
   # E[1,:] := E[1,:] * Z
   # TR = ER[1,:]*ZR - EI[1,:]*ZI
   # TI = ER[1,:]*ZI + EI[1,:]*ZR
   # ER[1,:] = TR, EI[1,:] = TI
   TR11 = ER11*ZR11 + ER12*ZR21 - EI11*ZI11 - EI12*ZI21
   TR12 = ER11*ZR12 + ER12*ZR22 - EI11*ZI12 - EI12*ZI22
   TI11 = ER11*ZI11 + ER12*ZI21 + EI11*ZR11 + EI12*ZR21
   TI12 = ER11*ZI12 + ER12*ZI22 + EI11*ZR12 + EI12*ZR22

   ER11, ER12 = TR11, TR12
   EI11, EI12 = TI11, TI12
   ER21, ER22 = ZERO, L
   EI21, EI22 = ZERO, ZERO  

   # Make main diagonal entries of E real and positive.
   V = hypot(ER11, EI11)
   XR, XI = _ladiv(V, ZERO, ER11, EI11)
   ER11, EI11 = V, ZERO
   YR11, YI11 = ZR11, ZI11
   ZR11 = XR*YR11 - XI*YI11
   ZI11 = XR*YI11 + XI*YR11
   YR21, YI21 = ZR21, ZI21
   ZR21 = XR*YR21 - XI*YI21
   ZI21 = XR*YI21 + XI*YR21

   # # End of QZ-step.
   # BR = RR*ZR
   # BI = RR*ZI

   # A := A * Z (TR = AR*ZR - AI*ZI, TI = AI*ZR + AR*ZI, AR = TR, AI = TI)
   TR11 = AR11*ZR11 + AR12*ZR21 - AI11*ZI11 - AI12*ZI21
   TR12 = AR11*ZR12 + AR12*ZR22 - AI11*ZI12 - AI12*ZI22
   TR21 = AR21*ZR11 + AR22*ZR21 - AI21*ZI11 - AI22*ZI21
   TR22 = AR21*ZR12 + AR22*ZR22 - AI21*ZI12 - AI22*ZI22

   TI11 = AI11*ZR11 + AI12*ZR21 + AR11*ZI11 + AR12*ZI21
   TI12 = AI11*ZR12 + AI12*ZR22 + AR11*ZI12 + AR12*ZI22
   TI21 = AI21*ZR11 + AI22*ZR21 + AR21*ZI11 + AR22*ZI21
   TI22 = AI21*ZR12 + AI22*ZR22 + AR21*ZI12 + AR22*ZI22

   AR11, AR12, AR21, AR22 = TR11, TR12, TR21, TR22
   AI11, AI12, AI21, AI22 = TI11, TI12, TI21, TI22
   # End of QZ-step.

   # B = RR * Z (BR = RR*ZR, BI = RR*ZI)
   BR11 = RR11*ZR11 + RR12*ZR21; BR12 = RR11*ZR12 + RR12*ZR22
   BR21 = RR21*ZR11 + RR22*ZR21; BR22 = RR21*ZR12 + RR22*ZR22
   BI11 = RR11*ZI11 + RR12*ZI21; BI12 = RR11*ZI12 + RR12*ZI22
   BI21 = RR21*ZI11 + RR22*ZI21; BI22 = RR21*ZI12 + RR22*ZI22

   # Overwrite B with the upper triangular matrix of its
   # QR-factorization. The elements on the main diagonal are real
   # and non-negative.

   CR1, CI1, SR1, SI1, L1 = cgivensc2(BR11, BI11, BR21, BI21, small)
   QBR11, QBR12, QBR21, QBR22 =  CR1,  SR1, -SR1, CR1
   QBI11, QBI12, QBI21, QBI22 = -CI1, -SI1, -SI1, CI1

   # TR = QBR*BR[:,2] - QBI*BI[:,2]
   # TI = QBI*BR[:,2] + QBR*BI[:,2]
   # BR[:,2] = TR
   # BI[:,2] = TI
   # BR[1,1] = L
   # BR[2,1] = ZERO
   # BI[1,1] = ZERO
   # BI[2,1] = ZERO

   TR12 = QBR11*BR12 + QBR12*BR22 - QBI11*BI12 - QBI12*BI22
   TR22 = QBR21*BR12 + QBR22*BR22 - QBI21*BI12 - QBI22*BI22
   TI12 = QBI11*BR12 + QBI12*BR22 + QBR11*BI12 + QBR12*BI22
   TI22 = QBI21*BR12 + QBI22*BR22 + QBR21*BI12 + QBR22*BI22

   BR12, BR22 = TR12, TR22
   BI12, BI22 = TI12, TI22
   BR11, BR21, BI11, BI21 = L1, ZERO, ZERO, ZERO

   V = hypot(BR22, BI22)
   if V >= max(EPS*max(BR11, hypot(BR12, BI12)), SMLNUM)
      XR, XI = _ladiv(V, ZERO, BR22, BI22)
      BR22 = V
      QBR21, QBI21 = XR*QBR21 - XI*QBI21, XR*QBI21 + XI*QBR21
      QBR22, QBI22 = XR*QBR22 - XI*QBI22, XR*QBI22 + XI*QBR22
   else
      BR22 = ZERO
   end
   BI22 = ZERO
   
   # Compute the Cholesky factor of the solution of the reduced
   # equation. The solution may be scaled to avoid overflow.

   # Cholesky Step Variables
   UR11 = UR12 = UR22 = ZERO
   UI11 = UI12 = UI22 = ZERO

   M1R11 = M1R12 = M1R21 = M1R22 = ZERO
   M1I11 = M1I12 = M1I21 = M1I22 = ZERO
   M2R11 = M2R12 = M2R21 = M2R22 = ZERO
   M2I11 = M2I12 = M2I21 = M2I22 = ZERO   
   
   if ISCONT
      # Continuous-time equation.

      # Step I:  Compute U[1,1]. Set U[2,1] = 0.
      V = -TWO*(AR11*ER11 + AI11*EI11)
      V <= ZERO && error("The eigenvalues of the pencil A - λE are not in the open right half plane")
      V = sqrt(V)
      T = TWO*abs(BR11)*SMLNUM
      T > V && error("$errtext")
      UR11 = BR11/V

      # Step II:  Compute U[1,2].
      T = max(EPS*max(BR22, hypot(BR12, BI12)), SMLNUM)
      if abs(BR11) >= T
         XR = AR11*ER12 + AI11*EI12 + AR12*ER11 + AI12*EI11
         XI = AI11*ER12 - AR11*EI12 - AI12*ER11 + AR12*EI11
         XR = -BR12*V - XR*UR11
         XI =  BI12*V - XI*UR11
         YR = AR22*ER11 + AI22*EI11 + ER22*AR11 + EI22*AI11
         YI = -AI22*ER11 + AR22*EI11 - EI22*AR11 + ER22*AI11
         T  = TWO*hypot(XR, XI)*SMLNUM
         T > hypot(YR, YI) && error("$errtext")
         UR12, UI12 = _ladiv(XR, XI, YR, YI)
         UI12 = -UI12
      end

      # Step III:  Compute U[2,2].
      XR = (ER12*UR11 + ER22*UR12 - EI22*UI12)*V
      XI = (-EI12*UR11 - ER22*UI12 - EI22*UR12)*V
      T  = TWO*hypot(XR, XI)*SMLNUM
      T > hypot(ER11, EI11) && error("$errtext")
      YR, YI = _ladiv(XR, XI, ER11, -EI11)
      YR =  BR12 - YR
      YI = -BI12 - YI
      V  = -TWO*(AR22*ER22 + AI22*EI22)
      V <= ZERO && error("The eigenvalues of the pencil A - λE have no negative real parts")
      V = sqrt(V)
      W = hypot4(BR22, BI22, YR, YI)
      T = TWO*W*SMLNUM
      T > V && error("$errtext")
      UR22 = W/V

      BETAR, BETAI = _ladiv(AR11, AI11, ER11, EI11)
      M1R11, M1I11 = BETAR, BETAI
      M1R22, M1I22 = BETAR, -BETAI
      ALPHA = sqrt(-TWO*BETAR)
      M2R11 = ALPHA

      V  = ER11*ER22
      XR = (-BR11*ER12 + ER11*BR12)/V
      XI = (-BR11*EI12 + ER11*BI12)/V
      YR =  XR - ALPHA*UR12
      YI = -XI + ALPHA*UI12

      if (abs(YR) > SMLNUM) || (abs(YI) > SMLNUM)
         M2R12 =  YR/UR22
         M2I12 = -YI/UR22
         M2R22 =  BR22/(ER22*UR22)
         M1R12 = -ALPHA*M2R12
         M1I12 = -ALPHA*M2I12
      else
         M2R22 = ALPHA
      end
   else
      T = max(abs(AR11), abs(AI11), abs(ER11), abs(EI11))
      V = (ER11/T)^2 + (EI11/T)^2 - (AR11/T)^2 - (AI11/T)^2
      V <= ZERO && error("The eigenvalues of the pencil A - λE are not inside the unit circle")
      V = T*sqrt(V)
      T = TWO*abs(BR11)*SMLNUM
      T > V && error("$errtext")
      UR11 = BR11/V

      T = max(EPS*max(BR22, hypot(BR12, BI12)), SMLNUM)
      if abs(BR11) >= T
         XR =  AR11*AR12 + AI11*AI12 - ER12*ER11 - EI12*EI11
         XI =  AI11*AR12 - AR11*AI12 + EI12*ER11 - ER12*EI11
         XR = -BR12*V - XR*UR11
         XI =  BI12*V - XI*UR11
         YR =  AR22*AR11 + AI22*AI11 - ER22*ER11 - EI22*EI11
         YI = -AI22*AR11 + AR22*AI11 + EI22*ER11 - ER22*EI11
         T  = TWO*hypot(XR, XI)*SMLNUM
         T > hypot(YR, YI) && error("$errtext")
         t1, t2 = _ladiv(XR, XI, YR, YI)
         UR12, UI12 = t1, -t2
      end

      XR =  ER12*UR11 + ER22*UR12 - EI22*UI12
      XI = -EI12*UR11 - ER22*UI12 - EI22*UR12
      YR =  AR12*UR11 + AR22*UR12 - AI22*UI12
      YI = -AI12*UR11 - AR22*UI12 - AI22*UR12
      V  = ER22^2 + EI22^2 - AR22^2 - AI22^2
      V <= ZERO && error("The eigenvalues of the pencil A - λE are not inside the unit circle")
      V = sqrt(V)
      T = max(abs(BR22), abs(BR12), abs(BI12), abs(XR), abs(XI), abs(YR), abs(YI))
      if T <= SMLNUM
         W = ZERO
      else
         W = (BR22/T)^2 + (BR12/T)^2 + (BI12/T)^2 - (XR/T)^2 - (XI/T)^2 + (YR/T)^2 + (YI/T)^2
         W = W < ZERO ? ZERO : T*sqrt(W)
      end
      T = TWO*W*SMLNUM
      T > V && error("$errtext")
      UR22 = W/V

      B11  = BR11/ER11
      T    = ER11*ER22
      B12R = (ER11*BR12 - BR11*ER12)/T
      B12I = (ER11*BI12 - BR11*EI12)/T
      B22  = BR22/ER22
      BETAR, BETAI = _ladiv(AR11, AI11, ER11, EI11)
      M1R11, M1I11 = BETAR, BETAI
      M1R22, M1I22 = BETAR, -BETAI
      V = hypot(BETAR, BETAI)
      ALPHA = sqrt((ONE - V)*(ONE + V))
      M2R11 = ALPHA
      XR = (AI11*EI12 - AR11*ER12)/T + AR12/ER22
      XI = (AR11*EI12 + AI11*ER12)/T - AI12/ER22
      XR = -TWO*BETAI*B12I - B11*XR
      XI = -TWO*BETAI*B12R - B11*XI
      V  = ONE + (BETAI - BETAR)*(BETAI + BETAR)
      W  = -TWO*BETAI*BETAR
      YR, YI = _ladiv(XR, XI, V, W)

      if (abs(YR) > SMLNUM) || (abs(YI) > SMLNUM)
         M2R12 =  (YR*BETAR - YI*BETAI)/UR22
         M2I12 = -(YI*BETAR + YR*BETAI)/UR22
         M2R22 =  B22/UR22
         M1R12 = -ALPHA*YR/UR22
         M1I12 =  ALPHA*YI/UR22
      else
         M2R22 = ALPHA
      end
   end
 
   # # Transform U back:  U := U * Q.
   # # (Note:  Z is used as workspace.)
   # ZR = UR*QR - UI*QI
   # ZI = UR*QI + UI*QR

   # Transform U back: U := U * Q
   ZR11 =  UR11*CR + UI11*CI - UR12*SR + UI12*SI
   ZR12 =  UR11*SR + UI11*SI + UR12*CR - UI12*CI
   ZR21 = -UR22*SR + UI22*SI
   ZR22 =  UR22*CR - UI22*CI

   ZI11 =  UI11*CR - UR11*CI - UI12*SR - UR12*SI
   ZI12 =  UI11*SR - UR11*SI + UI12*CR + UR12*CI
   ZI21 = -UI22*SR - UR22*SI
   ZI22 =  UI22*CR + UR22*CI

   # Overwrite U with the upper triangular matrix of its
   # QR-factorization. The elements on the main diagonal are real
   # and non-negative.

   CR, CI, SR, SI, L = cgivensc2(ZR11, ZI11, ZR21, ZI21, small)
   QUR11, QUR12, QUR21, QUR22 =  CR,  SR, -SR, CR
   QUI11, QUI12, QUI21, QUI22 = -CI, -SI, -SI, CI

   UR12 = QUR11*ZR12 + QUR12*ZR22 - QUI11*ZI12 - QUI12*ZI22
   UR22 = QUR21*ZR12 + QUR22*ZR22 - QUI21*ZI12 - QUI22*ZI22
   UI12 = QUI11*ZR12 + QUI12*ZR22 + QUR11*ZI12 + QUR12*ZI22
   UI22 = QUI21*ZR12 + QUI22*ZR22 + QUR21*ZI12 + QUR22*ZI22
   U11  = L
   U12 = UR12
   U22  = UR22

   V = hypot(U22, UI22)
   if V > SMLNUM
      XR, XI = _ladiv(V, ZERO, U22, UI22)
      YR = QUR21; YI = QUI21
      QUR21 = XR*YR - XI*YI
      QUI21 = XR*YI + XI*YR
      YR = QUR22; YI = QUI22
      QUR22 = XR*YR - XI*YI
      QUI22 = XR*YI + XI*YR
   end
   U22 = V

   # Transform the matrices M1 and M2 
   # M1 := QU * M1 * QU^H, M2 := QB^H * M2 * QU^H

   # TR = M1R*QUR' + M1I*QUI'
   TR11 = M1R11*QUR11 + M1R12*QUR12 + M1I11*QUI11 + M1I12*QUI12
   TR12 = M1R11*QUR21 + M1R12*QUR22 + M1I11*QUI21 + M1I12*QUI22
   TR21 = M1R21*QUR11 + M1R22*QUR12 + M1I21*QUI11 + M1I22*QUI12
   TR22 = M1R21*QUR21 + M1R22*QUR22 + M1I21*QUI21 + M1I22*QUI22

   # TI = -M1R*QUI' + M1I*QUR'
   TI11 = -M1R11*QUI11 - M1R12*QUI12 + M1I11*QUR11 + M1I12*QUR12
   TI12 = -M1R11*QUI21 - M1R12*QUI22 + M1I11*QUR21 + M1I12*QUR22
   TI21 = -M1R21*QUI11 - M1R22*QUI12 + M1I21*QUR11 + M1I22*QUR12
   TI22 = -M1R21*QUI21 - M1R22*QUI22 + M1I21*QUR21 + M1I22*QUR22

   # β = QUR*TR - QUI*TI
   β[1,1] = QUR11*TR11 + QUR12*TR21 - QUI11*TI11 - QUI12*TI21
   β[1,2] = QUR11*TR12 + QUR12*TR22 - QUI11*TI12 - QUI12*TI22
   β[2,1] = QUR21*TR11 + QUR22*TR21 - QUI21*TI11 - QUI22*TI21
   β[2,2] = QUR21*TR12 + QUR22*TR22 - QUI21*TI12 - QUI22*TI22

   # TR = M2R*QUR' + M2I*QUI'
   TR11 = M2R11*QUR11 + M2R12*QUR12 + M2I11*QUI11 + M2I12*QUI12
   TR12 = M2R11*QUR21 + M2R12*QUR22 + M2I11*QUI21 + M2I12*QUI22
   TR21 = M2R21*QUR11 + M2R22*QUR12 + M2I21*QUI11 + M2I22*QUI12
   TR22 = M2R21*QUR21 + M2R22*QUR22 + M2I21*QUI21 + M2I22*QUI22

   # TI = -M2R*QUI' + M2I*QUR'
   TI11 = -M2R11*QUI11 - M2R12*QUI12 + M2I11*QUR11 + M2I12*QUR12
   TI12 = -M2R11*QUI21 - M2R12*QUI22 + M2I11*QUR21 + M2I12*QUR22
   TI21 = -M2R21*QUI11 - M2R22*QUI12 + M2I21*QUR11 + M2I22*QUR12
   TI22 = -M2R21*QUI21 - M2R22*QUI22 + M2I21*QUR21 + M2I22*QUR22

   # α = QBR'*TR + QBI'*TI
   α[1,1] = QBR11*TR11 + QBR21*TR21 + QBI11*TI11 + QBI21*TI21
   α[1,2] = QBR11*TR12 + QBR21*TR22 + QBI11*TI12 + QBI21*TI22
   α[2,1] = QBR12*TR11 + QBR22*TR21 + QBI12*TI11 + QBI22*TI21
   α[2,2] = QBR12*TR12 + QBR22*TR22 + QBI12*TI12 + QBI22*TI22

   # If the transposed equation (op(K)=K^T, K=A,B,E,U) is to be
   # solved, transpose the matrix U with respect to the
   # anti-diagonal and the matrices M1, M2 with respect to the diagonal
   # and the anti-diagonal.

   if noadj
      U11, U22 = U22, U11
      β[1,1], β[2,2] = β[2,2], β[1,1]
      α[1,1], α[2,2] = α[2,2], α[1,1]
   end

   R[1,1] = U11
   R[1,2] = U12
   R[2,1] = ZERO
   R[2,2] = U22
   return R
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
   ZERO = zero(d)
   ONE = one(d)
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
   iszero(W) ? (return zero(W)) : (return W*sqrt( ( XABS / W )^2+( YABS / W )^2+ ( ZABS / W )^2 ))
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
   ZERO = zero(d)
   ONE = one(d)
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
   iszero(W) ? (return zero(W)) : (return W*sqrt( ( XABS / W )^2+( YABS / W )^2+ ( ZABS / W )^2+ ( TABS / W )^2  ))
end
