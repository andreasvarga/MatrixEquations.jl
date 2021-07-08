"""
    X = sylvc(A,B,C)

Solve the continuous Sylvester matrix equation

                AX + XB = C

using the Bartels-Stewart Schur form based approach. `A` and `B` are
square matrices, and `A` and `-B` must not have common eigenvalues.

The following particular cases are also adressed:

    X = sylvc(α*I,B,C)  or  X = sylvc(α,B,C)

Solve the matrix equation `X(αI+B)  = C`.

    X = sylvc(A,β*I,C)  or  X = sylvc(A,β,C)

Solve the matrix equation `(A+βI)X = C`.

    X = sylvc(α*I,β*I,C)  or  sylvc(α,β,C)

Solve the matrix equation `(α+β)X = C`.

    x = sylvc(α,β,γ)

Solve the equation `(α+β)x = γ`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [-1. -2.; 2. -1.]
2×2 Array{Float64,2}:
 -1.0  -2.0
  2.0  -1.0

julia> X = sylvc(A, B, C)
2×2 Array{Float64,2}:
 -4.46667   1.93333
  3.73333  -1.8

julia> A*X + X*B - C
2×2 Array{Float64,2}:
  2.66454e-15  1.77636e-15
 -3.77476e-15  4.44089e-16
```
"""
function sylvc(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix)
   """
   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """

   m, n = size(C);
   [m; n] == LinearAlgebra.checksquare(A,B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))
 
   T2 = promote_type(eltype(A), eltype(B), eltype(C))
   T2 <: BlasFloat || (T2 = promote_type(Float64,T2))
   eltype(A) == T2 || (A = convert(Matrix{T2},A))
   eltype(B) == T2 || (B = convert(Matrix{T2},B))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))
 
   adjA = isa(A,Adjoint)
   adjB = isa(B,Adjoint)
   if adjA
      RA, QA = schur(A.parent)
   else
      RA, QA = schur(A)
   end
   if adjB
      RB, QB = schur(B.parent)
   else
      RB, QB = schur(B)
   end

   Y = QA' * C * QB

   sylvcs!(RA, RB, Y, adjA = adjA, adjB = adjB)

   mul!(Y, QA, Y*QB')

   return Y
end
# solve X(B+α) = C or (α+β)X = C
sylvc(A::Union{Real,Complex,UniformScaling},B::Union{AbstractMatrix,UniformScaling},C::AbstractMatrix) = C/(A*I+B)
# solve (A+β)X = C
sylvc(A::AbstractMatrix,B::Union{Real,Complex,UniformScaling},C::AbstractMatrix) = (A+B*I)\C
# solve (α+β)X = C
sylvc(A::Union{Real,Complex},B::Union{Real,Complex},C::AbstractMatrix) = A+B == 0 ? throw(SingularException(1)) : C/(A+B)
# solve (α+β)x = γ
sylvc(A::Union{Real,Complex}, B::Union{Real,Complex}, C::Union{Real,Complex}) = A+B == 0 ? throw(SingularException(1)) : C/(A+B)
"""
    X = sylvd(A,B,C)

Solve the discrete Sylvester matrix equation

                AXB + X = C

using an extension of the Bartels-Stewart Schur form based approach.
`A` and `B` are square matrices, and `A` and `-B` must not have
common reciprocal eigenvalues.

The following particular cases are also adressed:

    X = sylvd(α*I,B,C)  or  X = sylvd(α,B,C)

Solve the matrix equation `X(αB+I)  = C`.

    X = sylvd(A,β*I,C)   or  X = sylvd(A,β,C)

Solve the matrix equation `(βA+I)X = C`.

    X = sylvd(α*I,β*I,C)  or  X = sylvd(α,β,C)

Solve the matrix equation `(αβ+1)X = C`.

    x = sylvd(α,β,γ)

Solve the equation `(αβ+1)x = γ`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [-1. -2.; 2. -1.]
2×2 Array{Float64,2}:
 -1.0  -2.0
  2.0  -1.0
   
julia> X = sylvd(A, B, C)
2×2 Array{Float64,2}:
 -2.46667  -2.73333
  2.4       1.86667

julia> A*X*B + X - C
2×2 Array{Float64,2}:
  8.88178e-16   8.88178e-16
 -3.9968e-15   -5.55112e-15
```
"""
function sylvd(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix)
   """
   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """

   m, n = size(C);
   [m; n] == LinearAlgebra.checksquare(A,B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))
   T2 = promote_type(eltype(A), eltype(B), eltype(C))
   T2 <: BlasFloat || (T2 = promote_type(Float64,T2))
   eltype(A) == T2 || (A = convert(Matrix{T2},A))
   eltype(B) == T2 || (B = convert(Matrix{T2},B))
   eltype(C) == T2 || (C = convert(Matrix{T2},C))

   adjA = isa(A,Adjoint)
   adjB = isa(B,Adjoint)
   if adjA
      RA, QA = schur(A.parent)
   else
      RA, QA = schur(A)
   end
   if adjB
      RB, QB = schur(B.parent)
   else
      RB, QB = schur(B)
   end

   Y = QA' * C * QB

   sylvds!(RA, RB, Y, adjA = adjA, adjB = adjB)

   mul!(Y, QA, Y*QB')

   return Y
end
# solve X(αB+I) = C or X(αβ+1) = C
sylvd(A::Union{Real,Complex,UniformScaling},B::Union{AbstractMatrix,UniformScaling},C::AbstractMatrix) = C/(A*B+I)
# solve (Aβ+I)X = C
sylvd(A::AbstractMatrix,B::Union{Real,Complex,UniformScaling},C::AbstractMatrix) = (A*B+I)\C
# solve (αβ+1)X = C
sylvd(A::Union{Real,Complex},B::Union{Real,Complex},C::AbstractMatrix) = A*B+1 == 0 ? throw(SingularException(1)) : C/(A*B+1)
# solve (αβ+1)x = γ
sylvd(A::Union{Real,Complex}, B::Union{Real,Complex}, C::Union{Real,Complex}) = A*B+1 == 0 ? throw(SingularException(1)) : C/(A*B+one(C))
"""
    X = gsylv(A,B,C,D,E)

Solve the generalized Sylvester matrix equation

              AXB + CXD = E

using a generalized Schur form based approach. `A`, `B`, `C` and `D` are
square matrices. The pencils `A-λC` and `D+λB` must be regular and
must not have common eigenvalues.

The following particular cases are also adressed:

    X = gsylv(A,B,E)

Solve the generalized Sylvester matrix equation `AXB  = E`.

    X = gsylv(A,B,γ*I,E)  or  X = gsylv(A,B,γ,E)

Solve the generalized Sylvester matrix equation `AXB +γX = E`.

    X = gsylv(A,B,γ*I,D,E)  or  X = gsylv(A,B,γ,D,E)

Solve the generalized Sylvester matrix equation `AXB +γXD = E`.

    X = gsylv(A,B,C,δ*I,E)  or  X = gsylv(A,B,C,δ,E)

Solve the generalized Sylvester matrix equation `AXB +CXδ = E`.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [-1. -2.; 2. -1.]
2×2 Array{Float64,2}:
 -1.0  -2.0
  2.0  -1.0

julia> D = [1. -2.; -2. -1.]
2×2 Array{Float64,2}:
  1.0  -2.0
 -2.0  -1.0

julia> E = [1. -1.; -2. 2.]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> X = gsylv(A, B, C, D, E)
2×2 Array{Float64,2}:
 -0.52094   -0.0275792
 -0.168539   0.314607
 
julia> A*X*B + C*X*D - E
2×2 Array{Float64,2}:
 4.44089e-16  8.88178e-16
 6.66134e-16  0.0
```
"""
function gsylv(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix,D::AbstractMatrix,E::AbstractMatrix)

    m, n = size(E);
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,C,D) ||
       throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
    T2 = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(E))
    T2 <: BlasFloat || (T2 = promote_type(Float64,T2))
    eltype(A) == T2 || (A = convert(Matrix{T2},A))
    eltype(B) == T2 || (B = convert(Matrix{T2},B))
    eltype(C) == T2 || (C = convert(Matrix{T2},C))
    eltype(D) == T2 || (D = convert(Matrix{T2},D))
    eltype(E) == T2 || (E = convert(Matrix{T2},E))

    adjA = isa(A,Adjoint)
    adjB = isa(B,Adjoint)
    adjC = isa(C,Adjoint)
    adjD = isa(D,Adjoint)
    adjAC = adjA && adjC
    adjBD = adjB && adjD

    if adjAC
       AS, CS, Z1, Q1 = schur(A.parent,C.parent)
    else
       adjA && (A = copy(A))
       adjC && (C = copy(C))
       AS, CS, Q1, Z1 = schur(A,C)
    end
    if adjBD
       BS, DS, Z2, Q2 = schur(B.parent,D.parent)
    else
       adjB && (B = copy(B))
       adjD && (D = copy(D))
       BS, DS, Q2, Z2 = schur(B,D)
    end

    Y = Q1' * E *Z2

    gsylvs!(AS, BS, CS, DS, Y, adjAC = adjAC, adjBD = adjBD)

    mul!(Y, Z1, Y*Q2')

   return Y
end
# solve AXB = C
gsylv(A::Union{AbstractMatrix,UniformScaling,Real,Complex},B::Union{AbstractMatrix,UniformScaling,Real,Complex},E::AbstractMatrix) = (A\E)/B
# solve AXB+γX = E
gsylv(A::AbstractMatrix,B::AbstractMatrix,C::Union{UniformScaling,Real,Complex},E::AbstractMatrix) =
size(A,1) == size(A,2) && size(B,1) == size(B,2) ? gsylv(A,B,Matrix{eltype(C)}(C*I,size(A)),Matrix{eltype(C)}(I,size(B)),E) :
throw(DimensionMismatch("A and B must be square matrices"))
# solve AXB+γXδ = E
gsylv(A::AbstractMatrix,B::AbstractMatrix,C::Union{UniformScaling,Real,Complex},D::Union{UniformScaling,Real,Complex},E::AbstractMatrix) = gsylv(A,B,C*D,E)
# solve AXB+γXD = E
gsylv(A::AbstractMatrix,B::AbstractMatrix,C::Union{UniformScaling,Real,Complex},D::AbstractMatrix,E::AbstractMatrix) =
size(A,1) == size(A,2) ? gsylv(A,B,Matrix{eltype(C)}(C*I,size(A)),D,E) :
throw(DimensionMismatch("A must be a square matrix"))
# solve AXB+CXδ = E
gsylv(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix,D::Union{UniformScaling,Real,Complex},E::AbstractMatrix) =
size(B,1) == size(B,2) ? gsylv(A,B,C,Matrix{eltype(D)}(D*I,size(B)),E) :
throw(DimensionMismatch("B must be a square matrix"))

"""
    (X,Y) = sylvsys(A,B,C,D,E,F)

Solve the Sylvester system of matrix equations

                AX + YB = C
                DX + YE = F,

where `(A,D)`, `(B,E)` are pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.
# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [-1. -2.; 2. -1.]
2×2 Array{Float64,2}:
 -1.0  -2.0
  2.0  -1.0

julia> D = [1. -2.; -2. -1.]
2×2 Array{Float64,2}:
  1.0  -2.0
 -2.0  -1.0

julia> E = [1. -1.; -2. 2.]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> F = [1. -1.; -2. 2.]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> X, Y = sylvsys(A, B, C, D, E, F);

julia> X
2×2 Array{Float64,2}:
  1.388  -1.388
 -0.892   0.892

julia> Y
2×2 Array{Float64,2}:
 -1.788  0.192
  0.236  0.176

julia> A*X + Y*B - C
2×2 Array{Float64,2}:
  6.66134e-16  2.22045e-15
 -3.10862e-15  2.66454e-15

julia> D*X + Y*E - F
2×2 Array{Float64,2}:
  1.33227e-15  -2.22045e-15
 -4.44089e-16   4.44089e-16
```
"""
function sylvsys(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix,D::AbstractMatrix,E::AbstractMatrix,F::AbstractMatrix)

    m, n = size(C);
    (m == size(F,1) && n == size(F,2)) ||
      throw(DimensionMismatch("C and F must have the same dimensions"))
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,D,E) ||
       throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
    T2 = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(E), eltype(F))
    T2 <: BlasFloat || (T2 = promote_type(Float64,T2))
    eltype(A) == T2 || (A = convert(Matrix{T2},A))
    eltype(B) == T2 || (B = convert(Matrix{T2},B))
    eltype(C) == T2 || (C = convert(Matrix{T2},C))
    eltype(D) == T2 || (D = convert(Matrix{T2},D))
    eltype(E) == T2 || (E = convert(Matrix{T2},E))
    eltype(F) == T2 || (F = convert(Matrix{T2},F))

    isa(A,Adjoint) && (A = copy(A))
    isa(B,Adjoint) && (B = copy(B))
    isa(D,Adjoint) && (D = copy(D))
    isa(E,Adjoint) && (E = copy(E))
    
    AS, DS, Q1, Z1 = schur(A,D)
    BS, ES, Q2, Z2 = schur(B,E)

    CS = adjoint(Q1) * (C*Z2)
    FS = adjoint(Q1) * (F*Z2)

    X, Y, scale =  tgsyl!('N',AS,BS,CS,DS,ES,FS)

    (rmul!(Z1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Q2)), inv(-scale)) )
end
"""
    (X,Y) = dsylvsys(A,B,C,D,E,F)

Solve the dual Sylvester system of matrix equations

       AX + DY = C
       XB + YE = F ,

where `(A,D)`, `(B,E)` are pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6.]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [-1. -2.; 2. -1.]
2×2 Array{Float64,2}:
 -1.0  -2.0
  2.0  -1.0

julia> D = [1. -2.; -2. -1.]
2×2 Array{Float64,2}:
  1.0  -2.0
 -2.0  -1.0

julia> E = [1. -1.; -2. 2.]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> F = [1. -1.; -2. 2.]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> X, Y = dsylvsys(A, B, C, D, E, F);

julia> X
2×2 Array{Float64,2}:
  2.472  -1.648
 -1.848   1.232

julia> Y
2×2 Array{Float64,2}:
 -0.496  -0.336
  0.264   0.824

julia> A*X + D*Y - C
2×2 Array{Float64,2}:
  4.44089e-16  0.0
 -3.55271e-15  1.55431e-15
 
julia> X*B + Y*E - F
2×2 Array{Float64,2}:
 -8.88178e-16   0.0
  8.88178e-16  -4.44089e-16
```
"""
function dsylvsys(A::AbstractMatrix,B::AbstractMatrix,C::AbstractMatrix,D::AbstractMatrix,E::AbstractMatrix,F::AbstractMatrix)

    m, n = size(C);
    (m == size(F,1) && n == size(F,2)) ||
      throw(DimensionMismatch("C and F must have the same dimensions"))
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,D,E) ||
       throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))

    T2 = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(E), eltype(F))
    T2 <: BlasFloat || (T2 = promote_type(Float64,T2))
    eltype(A) == T2 || (A = convert(Matrix{T2},A))
    eltype(B) == T2 || (B = convert(Matrix{T2},B))
    eltype(C) == T2 || (C = convert(Matrix{T2},C))
    eltype(D) == T2 || (D = convert(Matrix{T2},D))
    eltype(E) == T2 || (E = convert(Matrix{T2},E))
    eltype(F) == T2 || (F = convert(Matrix{T2},F))
    realcase = T2 <: AbstractFloat
    transsylv = isa(A,Adjoint) && isa(B,Adjoint) && isa(D,Adjoint) && isa(E,Adjoint)
    realcase ? trans = 'T' : trans = 'C'
    if transsylv
       AS, DS, Q1, Z1 = schur(A.parent,D.parent)
       BS, ES, Q2, Z2 = schur(B.parent,E.parent)
       CS = adjoint(Z1) * (C*Z2)
       FS = adjoint(Q1) * (F*Q2)
       X, Y, scale =  tgsyl!(trans,AS,BS,CS,DS,ES,-FS)
       (rmul!(Q1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Z2)), inv(scale)) )
    else
       AS, DS, Q1, Z1 = schur(copy(A'),copy(D'))
       BS, ES, Q2, Z2 = schur(copy(B'),copy(E'))
       CS = adjoint(Z1) * (C*Z2)
       FS = adjoint(Q1) * (F*Q2)

       X, Y, scale =  tgsyl!(trans,AS,BS,CS,DS,ES,-FS)

       (rmul!(Q1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Z2)), inv(scale)) )
    end
end
"""
    sylvcs!(A,B,C; adjA = false, adjB = false)

Solve the continuous Sylvester matrix equation

                op(A)X + Xop(B) =  C,

where `op(A) = A` or `op(A) = A'` if `adjA = false` or `adjA = true`, respectively,
and `op(B) = B` or `op(B) = B'` if `adjB = false` or `adjB = true`, respectively.
`A` and `B` are square matrices in Schur forms, and `A` and `-B` must not have
common eigenvalues. `C` contains on output the solution `X`.
"""
function sylvcs!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}; adjA::Bool = false, adjB::Bool = false) where  T1<:BlasFloat
   """
   This is a wrapper to the LAPACK.trsylv! function, based on the Bartels-Stewart Schur form based approach.
   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   try
      trans = T1 <: Complex ? 'C' : 'T'
      C, scale = LAPACK.trsyl!(adjA ? trans : 'N', adjB ? trans : 'N', A, B, C)
      rmul!(C, inv(scale))
      return C[:,:]
   catch err
      findfirst("LAPACKException(1)",string(err)) === nothing ? rethrow() : 
               throw("ME:SingularException: A has eigenvalue(s) α and B has eigenvalues(s) β such that α+β = 0")
   end
end
function sylvd2!(adjA::Bool,adjB::Bool,C::AbstractMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},Xw::AbstractMatrix{T},Yw::AbstractVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations 
   # encountered in solving discrete Lyapunov equations: 
   # A*X*B + X = C   if adjA = false and adjB = false -> R = kron(B',A) + I 
   # A'*X*B + X = C  if adjA = true  and adjB = false -> R = kron(B',A') + I 
   # A*X*B' + X = C  if adjA = false and adjB = true  -> R = kron(B,A) + I
   # A'*X*B' + X = C if adjA = true  and adjB = true  -> R = kron(B,A') + I
   ONE = one(T)
   if na == 1 && nb == 1
      temp = A[1,1]*B[1,1] + ONE
      rmul!(C,inv(temp))
      any(!isfinite, C) && throw("ME:SingularException: A has eigenvalue(s) α and B has eigenvalues(s) β such that αβ = -1")
      return C
   end
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if adjA && !adjB
      #b = @allocated begin
      if na == 1
         # R12 = 
         # [ a11*b11+1      a11*b21]
         # [     a11*b12  a11*b22+1]
         # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[2,1];
         #                 A[1,1]*B[1,2]  A[1,1]*B[2,2]+ONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
         @inbounds  R[1,2] = A[1,1]*B[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+ONE
         @inbounds  Y[1] = C[1,1]
         @inbounds  Y[2] = C[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11+1      a21*b11]
            # [     a12*b11  a22*b11+1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1];
            #                 A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
         else
            # R = 
            # [ a11*b11+1      a21*b11      a11*b21      a21*b21]
            # [     a12*b11  a22*b11+1      a12*b21      a22*b21]
            # [     a11*b12      a21*b12  a11*b22+1      a21*b22]
            # [     a12*b12      a22*b12      a12*b22  a22*b22+1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]+ONE      A[2,1]*B[2,2];
            # A[1,2]*B[1,2]      A[2,2]*B[1,2]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]
            @inbounds  R[1,4] = A[2,1]*B[2,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  R[2,3] = A[1,2]*B[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]
            @inbounds  R[3,2] = A[2,1]*B[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+ONE
            @inbounds  R[3,4] = A[2,1]*B[2,2]
            @inbounds  R[4,1] = A[1,2]*B[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]
            @inbounds  R[4,3] = A[1,2]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
            @inbounds  Y[3] = C[1,2]
            @inbounds  Y[4] = C[2,2]
         end
      end
      #R = kron(transpose(B),A) + I
   elseif !adjA && adjB
      if na == 1
         # R12 =
         # [ a11*b11+1      a11*b12]
         # [     a11*b21  a11*b22+1]
         # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[1,2];
         #                 A[1,1]*B[2,1]  A[1,1]*B[2,2]+ONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
         @inbounds  R[1,2] = A[1,1]*B[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+ONE
         @inbounds  Y[1] = C[1,1]
         @inbounds  Y[2] = C[1,2]
      else
         if nb == 1
            # R21 =
            #    [ a11*b11+1      a12*b11]
            #    [     a21*b11  a22*b11+1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1];
            #                 A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
         else
            # R = 
            # [ a11*b11+1      a12*b11      a11*b12      a12*b12]
            # [     a21*b11  a22*b11+1      a21*b12      a22*b12]
            # [     a11*b21      a12*b21  a11*b22+1      a12*b22]
            # [     a21*b21      a22*b21      a21*b22  a22*b22+1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1]      A[1,1]*B[1,2]      A[1,2]*B[1,2];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE      A[2,1]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]+ONE      A[1,2]*B[2,2];
            # A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]
            @inbounds  R[1,4] = A[1,2]*B[1,2]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  R[2,3] = A[2,1]*B[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]
            @inbounds  R[3,2] = A[1,2]*B[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+ONE
            @inbounds  R[3,4] = A[1,2]*B[2,2]
            @inbounds  R[4,1] = A[2,1]*B[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]
            @inbounds  R[4,3] = A[2,1]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
            @inbounds  Y[3] = C[1,2]
            @inbounds  Y[4] = C[2,2]
         end
      end
      #R = kron(transpose(B),transpose(A)) + I
   elseif !adjA && !adjB
      if na == 1
         # R12 = 
         # [ a11*b11 + 1,     a11*b21]
         # [     a11*b12, a11*b22 + 1]
         # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[2,1];
         #                 A[1,1]*B[1,2]  A[1,1]*B[2,2]+ONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
         @inbounds  R[1,2] = A[1,1]*B[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+ONE
         @inbounds  Y[1] = C[1,1]
         @inbounds  Y[2] = C[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + 1,     a12*b11]
            # [     a21*b11, a22*b11 + 1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1];
            #                 A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
         else
            # R = 
            # [ a11*b11 + 1,     a12*b11,     a11*b21,     a12*b21]
            # [     a21*b11, a22*b11 + 1,     a21*b21,     a22*b21]
            # [     a11*b12,     a12*b12, a11*b22 + 1,     a12*b22]
            # [     a21*b12,     a22*b12,     a21*b22, a22*b22 + 1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1]      A[1,1]*B[2,1]      A[1,2]*B[2,1];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE      A[2,1]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[1,2]*B[1,2]  A[1,1]*B[2,2]+ONE      A[1,2]*B[2,2];
            # A[2,1]*B[1,2]      A[2,2]*B[1,2]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[1,2]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]
            @inbounds  R[1,4] = A[1,2]*B[2,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  R[2,3] = A[2,1]*B[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]
            @inbounds  R[3,2] = A[1,2]*B[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+ONE
            @inbounds  R[3,4] = A[1,2]*B[2,2]
            @inbounds  R[4,1] = A[2,1]*B[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]
            @inbounds  R[4,3] = A[2,1]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
            @inbounds  Y[3] = C[1,2]
            @inbounds  Y[4] = C[2,2]
         end
      end
      #R = kron(B,A) + I
   else
      if na == 1
         # R12 = 
         # [ a11*b11 + 1,     a11*b12]
         # [     a11*b21, a11*b22 + 1]
         # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[1,2];
         #                 A[1,1]*B[2,1]  A[1,1]*B[2,2]+ONE]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
         @inbounds  R[1,2] = A[1,1]*B[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+ONE
         @inbounds  Y[1] = C[1,1]
         @inbounds  Y[2] = C[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + 1,     a21*b11]
            # [     a12*b11, a22*b11 + 1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1];
            #                 A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
         else
            # R =
            # [ a11*b11 + 1,     a21*b11,     a11*b12,     a21*b12]
            # [     a12*b11, a22*b11 + 1,     a12*b12,     a22*b12]
            # [     a11*b21,     a21*b21, a11*b22 + 1,     a21*b22]
            # [     a12*b21,     a22*b21,     a12*b22, a22*b22 + 1]
            # @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1]      A[1,1]*B[1,2]      A[2,1]*B[1,2];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE      A[1,2]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[2,1]*B[2,1]  A[1,1]*B[2,2]+ONE      A[2,1]*B[2,2];
            # A[1,2]*B[2,1]      A[2,2]*B[2,1]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+ONE]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+ONE
            @inbounds  R[1,2] = A[2,1]*B[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]
            @inbounds  R[1,4] = A[2,1]*B[1,2]
            @inbounds  R[2,1] = A[1,2]*B[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+ONE
            @inbounds  R[2,3] = A[1,2]*B[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]
            @inbounds  R[3,2] = A[2,1]*B[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+ONE
            @inbounds  R[3,4] = A[2,1]*B[2,2]
            @inbounds  R[4,1] = A[1,2]*B[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]
            @inbounds  R[4,3] = A[1,2]*B[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+ONE
            @inbounds  Y[1] = C[1,1]
            @inbounds  Y[2] = C[2,1]
            @inbounds  Y[3] = C[1,2]
            @inbounds  Y[4] = C[2,2]
         end
         #R = kron(B,transpose(A)) + I
      end
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalue(s) α and B has eingenvalu(s) β such that αβ = -1")
   C[:,:] = Y
   return C
end
"""
    sylvds!(A,B,C; adjA = false, adjB = false)

Solve the discrete Sylvester matrix equation

                op(A)Xop(B) + X =  C,

where `op(A) = A` or `op(A) = A'` if `adjA = false` or `adjA = true`, respectively,
and `op(B) = B` or `op(B) = B'` if `adjB = false` or `adjB = true`, respectively.
`A` and `B` are square matrices in Schur forms, and `A` and `-B` must not have
common reciprocal eigenvalues. `C` contains on output the solution `X`.
"""
function sylvds!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}; adjA::Bool = false, adjB::Bool = false) where  T1<:BlasReal
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   (size(C,1) == m && size(C,2) == n ) || throw(DimensionMismatch("C must be an $m x $n matrix"))
   ONE = one(T1)
  
   # determine the structure of the real Schur form of A
   ba, pa = sfstruct(A)
   bb, pb = sfstruct(B)

   W = zeros(T1,m,2)
   G = Matrix{T1}(undef,2,2)
   WA = Matrix{T1}(undef,2,2)

   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if !adjA && !adjB
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-left corner column by column by

      #            A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

      # where
      #                        M
      #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
      #                      J=K+1
      #                        M             L-1
      #                       SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
      #                       J=K            I=1
      # """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = m
          for kk = pa:-1:1
              dk = ba[kk]
              dkk = 1:dk
              i1 = i-dk+1
              k = i1:i
              Ckl = view(C,k,l)
              y = view(G,1:dk,1:dl)
              copyto!(y,Ckl)
              if kk < pa
                 ir = i+1:m
                 W1 = view(WA,dkk,dll)
                 mul!(W1,view(A,k,ir),view(C,ir,l))
                 mul!(y,W1,view(B,l,l),-ONE,ONE)
              end
              if ll > 1
                 ic = i1:m
                 mul!(view(W,k,dll),view(C,k,il1),view(B,il1,l))
                 mul!(y,view(A,k,ic),view(W,ic,dll),-ONE,ONE)
              end
              sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
              copyto!(Ckl,y)
           i -= dk
          end
          j += dl
      end
   elseif !adjA && adjB
         # """
         # The (K,L)th block of X is determined starting from
         # bottom-right corner column by column by

         #             A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

         # where
         #                        M
         #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
         #                      J=K+1
         #                        M              N
         #                       SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
         #                       J=K           I=L+1
         # """
         j = n
         for ll = pb:-1:1
             dl = bb[ll]
             dll = 1:dl
             il1 = j+1:n
             l = j-dl+1:j
             i = m
             for kk = pa:-1:1
                 dk = ba[kk]
                 dkk = 1:dk
                 i1 = i-dk+1
                 k = i1:i
                 Ckl = view(C,k,l)
                 y = view(G,1:dk,1:dl)
                 copyto!(y,Ckl)
                 if kk < pa
                    ir = i+1:m
                    W1 = view(WA,dkk,dll)
                    mul!(W1,view(A,k,ir),view(C,ir,l))
                    mul!(y,W1,adjoint(view(B,l,l)),-ONE,ONE)
                 end
                 if ll < pb
                    ic = i1:m
                    mul!(view(W,k,dll),view(C,k,il1),adjoint(view(B,l,il1)))
                    mul!(y,view(A,k,ic),view(W,ic,dll),-ONE,ONE)
                 end
                 sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
                 copyto!(Ckl,y)
                 i -= dk
             end
             j -= dl
         end
   elseif adjA && !adjB
      # """
      # The (K,L)th block of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
      #                       J=1             I=1
      # """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              Ckl = view(C,k,l)
              y = view(G,1:dk,1:dl)
              copyto!(y,Ckl)
              if kk > 1
                 ir = 1:i-1
                 W1 = view(WA,dkk,dll)
                 mul!(W1,adjoint(view(A,ir,k)),view(C,ir,l))
                 mul!(y,W1,view(B,l,l),-ONE,ONE)
              end
              if ll > 1
                 ic = 1:i1
                 mul!(view(W,k,dll),view(C,k,il1),view(B,il1,l))
                 mul!(y,adjoint(view(A,ic,k)),view(W,ic,dll),-ONE,ONE)
              end
              sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
              copyto!(Ckl,y)
           i += dk
          end
          j += dl
      end
   elseif adjA && adjB
      # """
      # The (K,L)th block of X is determined starting from the
      # lower-left corner column by column by

      #            A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
      #                       J=1            I=L+1
      # """
      j = n
      for ll = pb:-1:1
          dl = bb[ll]
          dll = 1:dl
          il1 = j+1:n
          l = j-dl+1:j
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              Ckl = view(C,k,l)
              y = view(G,1:dk,1:dl)
              copyto!(y,Ckl)
              if kk > 1
                 ir = 1:i-1
                 W1 = view(WA,dkk,dll)
                 mul!(W1,adjoint(view(A,ir,k)),view(C,ir,l))
                 mul!(y,W1,adjoint(view(B,l,l)),-ONE,ONE)
              end
              if ll < pb
                 ic = 1:i1
                 mul!(view(W,k,dll),view(C,k,il1),adjoint(view(B,l,il1)))
                 mul!(y,adjoint(view(A,ic,k)),view(W,ic,dll),-ONE,ONE)
              end
              sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
              copyto!(Ckl,y)
           i += dk
          end
          j -= dl
      end
   end
   return C
end
function sylvds!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}; adjA::Bool = false, adjB::Bool = false) where  T1<:BlasComplex
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   (size(C,1) == m && size(C,2) == n ) || throw(DimensionMismatch("C must be an $m x $n matrix"))
  
   W = zeros(T1,m)
   ONE = one(T1)
   ZERO = zero(T1)
   if !adjA && !adjB
      # """
      # The (K,L)th element of X is determined starting from
      # bottom-left corner column by column by

      #            A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

      # where
      #                        M
      #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
      #                      J=K+1
      #                        M             L-1
      #                       SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
      #                       J=K            I=1
      # """
      for l = 1:n
          il1 = 1:l-1
          for k = m:-1:1
              y = C[k,l]
              if k < m
                 ta = ZERO
                 for ir = k+1:m
                     ta += A[k,ir]*C[ir,l]
                 end
                 y -= ta*B[l,l]
              end
              if l > 1
                 tz = ZERO
                 for ir = il1
                     tz += C[k,ir]*B[ir,l]
                 end
                 W[k] = tz
                 for ic = k:m
                     y -= A[k,ic]*W[ic]
                 end
              end
              C[k,l] = y/(B[l,l]*A[k,k]+ONE)
              isfinite(C[k,l]) || throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
             end
      end
   elseif !adjA && adjB
         # """
         # The (K,L)th element of X is determined starting from
         # bottom-right corner column by column by

         #          A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

         # where
         #                        M
         #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
         #                      J=K+1
         #                        M              N
         #                       SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
         #                       J=K           I=L+1
         # """
         for l = n:-1:1
             il1 = l+1:n
             for k = m:-1:1
                 y = C[k,l]
                 if k < m
                    ta = ZERO
                    for ir = k+1:m
                       ta += A[k,ir]*C[ir,l]
                    end
                    y -= ta*B[l,l]'
                 end
                 if l < n
                    tz = ZERO
                    for ir = il1
                        tz += C[k,ir]*B[l,ir]'
                    end
                    W[k] = tz
                    for ic = k:m
                        y -= A[k,ic]*W[ic]
                    end
                 end
                 C[k,l] = y/(B[l,l]'*A[k,k]+ONE)
                 isfinite(C[k,l]) || throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
             end
         end
   elseif adjA && !adjB
      # """
      # The (K,L)th element of X is determined starting from the
      # upper-left corner column by column by

      #          A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
      #                       J=1             I=1
      # """
      for l = 1:n
          il1 = 1:l-1
          for k = 1:m
              y = C[k,l]
              if k > 1
                 ta = ZERO
                 for ir = 1:k-1
                     ta += A[ir,k]'*C[ir,l]
                 end
                 y -= ta*B[l,l]
              end
              if l > 1
                 ic = 1:m
                 tz = ZERO
                 for ir = il1
                     tz += C[k,ir]*B[ir,l]
                 end
                 W[k] = tz
                 for ic = 1:m
                     y -= A[ic,k]'*W[ic]
                 end
              end
              C[k,l] = y/(B[l,l]*A[k,k]'+ONE)
              isfinite(C[k,l]) || throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
         end
      end
   elseif adjA && adjB
      # """
      # The (K,L)th element of X is determined starting from the
      # upper-right corner column by column by

      #         A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
      #                       J=1            I=L+1
      # """
      for l = n:-1:1
          il1 = l+1:n
          for k = 1:m
              y = C[k,l]
              if k > 1
                 ta = ZERO
                 for ir = 1:k-1
                     ta += A[ir,k]'*C[ir,l]
                 end
                 y -= ta*B[l,l]'
              end
              if l < n
                 ic = 1:m
                 tz = ZERO
                 for ir = il1
                     tz += C[k,ir]*B[l,ir]'
                 end
                 W[k] = tz
                 for ic = 1:m
                     y -= A[ic,k]'*W[ic]
                 end
              end
              C[k,l] = y/(B[l,l]'*A[k,k]'+ONE)
              isfinite(C[k,l]) || throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
          end
      end
   end
   return C
end
"""
    X = gsylvs!(A,B,C,D,E; adjAC=false, adjBD=false, CASchur = false, DBSchur = false)

Solve the generalized Sylvester matrix equation

                op1(A)Xop2(B) + op1(C)Xop2(D) = E,

where `A`, `B`, `C` and `D` are square matrices, and

`op1(A) = A` and `op1(C) = C` if `adjAC = false`;

`op1(A) = A'` and `op1(C) = C'` if `adjAC = true`;

`op2(B) = B` and `op2(D) = D` if `adjBD = false`;

`op2(B) = B'` and `op2(D) = D'` if `adjBD = true`.

The matrix pair `(A,C)` is in a generalized real or complex Schur form.
The matrix pair `(B,D)` is in a generalized real or complex Schur form if `DBSchur = false`
or the matrix pair `(D,B)` is in a generalized real or complex Schur form if `DBSchur = true`.
The pencils `A-λC` and `D+λB` must be regular and must not have common eigenvalues.
"""
function gsylvs!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}, D::AbstractMatrix{T1}, E::AbstractMatrix{T1}; 
                 adjAC::Bool = false, adjBD::Bool = false, CASchur::Bool = false, DBSchur::Bool = false) where T1<:BlasReal
   """
   An extension proposed in [1] of the Bartels-Stewart Schur form based approach [2] is employed.

   References:
   [1] K.-W. E. Chu. The solution of the matrix equation AXB – CXD = E and
       (YA – DZ, YC– BZ) = (E, F). Lin. Alg. Appl., 93:93-105, 1987.
   [2] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
       Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(E);
   [m; n; m; n] == LinearAlgebra.checksquare(A,B,C,D) || 
      throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
   ONE = one(T1)

   # determine the structure of the generalized real Schur form of (A,C)
   CASchur ? ((ba, pa) = sfstruct(C)) : ((ba, pa) = sfstruct(A))
   DBSchur ? ((bb, pb) = sfstruct(D)) : ((bb, pb) = sfstruct(B)) 

   WB = zeros(T1,m,2)
   WD = zeros(T1,m,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if !adjAC && !adjBD
      # """
      # The (K,L)th block of X is determined starting from
      # bottom-left corner column by column by

      #       A(K,K)*X(K,L)*B(L,L) + C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      # where
      #                        M
      #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
      #                      J=K+1
      #                        M             L-1
      #                       SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
      #                       J=K            I=1

      #                        M
      #                     { SUM [C(K,J)*X(J,L)] } * D(L,L) +
      #                      J=K+1
      #                        M             L-1
      #                       SUM { C(K,J) * SUM [X(J,I)*D(I,L)] }.
      #                       J=K            I=1
      # """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = m
          for kk = pa:-1:1
              dk = ba[kk]
              dkk = 1:dk
              i1 = i-dk+1
              k = i1:i
              y = view(E,k,l)
              W1 = view(Xw,dkk,dll)
              if kk < pa
                 ir = i+1:m
                 #W1 = A[k,ir]*E[ir,l]
                 mul!(W1,view(A,k,ir),view(E,ir,l))
                 #y -= W1*B[l,l]
                 mul!(y,W1,view(B,l,l),-ONE,ONE)
                 #W1 = C[k,ir]*E[ir,l]
                 mul!(W1,view(C,k,ir),view(E,ir,l))
                 #y -= W1*D[l,l]
                 mul!(y,W1,view(D,l,l),-ONE,ONE)
              end
              if ll > 1
                 ic = i1:m
                 # WB[k,dll] = E[k,il1]*B[il1,l]
                 mul!(view(WB,k,dll),view(E,k,il1),view(B,il1,l))
                 # WD[k,dll] = E[k,il1]*D[il1,l]
                 mul!(view(WD,k,dll),view(E,k,il1),view(D,il1,l))
                 # y -= (A[k,ic]*WB[ic,dll] + C[k,ic]*WD[ic,dll])
                 mul!(y,view(A,k,ic),view(WB,ic,dll),-ONE,ONE)
                 mul!(y,view(C,k,ic),view(WD,ic,dll),-ONE,ONE)
              end
              gsylv2!(adjAC,adjBD,y,dk,dl,view(A,k,k),view(B,l,l),view(C,k,k),view(D,l,l),Xw,Yw) 
              i -= dk
          end
          j += dl
      end
   elseif !adjAC && adjBD
         # """
         #  The (K,L)th block of X is determined starting from
         #  bottom-right corner column by column by

         #       A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

         #  where
         #                        M
         #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
         #                      J=K+1
         #                        M              N
         #                       SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
         #                       J=K           I=L+1

         #                       M
         #                    { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
         #                     J=K+1
         #                       M              N
         #                      SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
         #                      J=K           I=L+1
         # """
         j = n
         for ll = pb:-1:1
             dl = bb[ll]
             dll = 1:dl
             il1 = j+1:n
             l = j-dl+1:j
             i = m
             for kk = pa:-1:1
                 dk = ba[kk]
                 dkk = 1:dk
                 i1 = i-dk+1
                 k = i1:i
                 y = view(E,k,l)
                 W1 = view(Xw,dkk,dll)
                 if kk < pa
                    ir = i+1:m
                    #W1 = A[k,ir]*E[ir,l]
                    mul!(W1,view(A,k,ir),view(E,ir,l))
                    #y -= W1*B[l,l]'
                    mul!(y,W1,transpose(view(B,l,l)),-ONE,ONE)
                    #W2 = C[k,ir]*E[ir,l]
                    mul!(W1,view(C,k,ir),view(E,ir,l))
                    # y -= W1*D[l,l]'
                    mul!(y,W1,transpose(view(D,l,l)),-ONE,ONE)
                  end
                 if ll < pb
                    ic = i1:m
                    #WB[k,dll] = E[k,il1]*B[l,il1]'
                    mul!(view(WB,k,dll),view(E,k,il1),transpose(view(B,l,il1)))
                    #WD[k,dll] = E[k,il1]*D[l,il1]'
                    mul!(view(WD,k,dll),view(E,k,il1),transpose(view(D,l,il1)))
                    #y -= (A[k,ic]*WB[ic,dll]+C[k,ic]*WD[ic,dll])
                    mul!(y,view(A,k,ic),view(WB,ic,dll),-ONE,ONE)
                    mul!(y,view(C,k,ic),view(WD,ic,dll),-ONE,ONE)
                 end
                 gsylv2!(adjAC,adjBD,y,dk,dl,view(A,k,k),view(B,l,l),view(C,k,k),view(D,l,l),Xw,Yw) 
                 i -= dk
             end
             j -= dl
         end
   elseif adjAC && !adjBD
      # """
      # The (K,L)th block of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
      #                       J=1             I=1

      #                       K-1
      #                     { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
      #                       J=1             I=1
      # """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = view(E,k,l)
              W1 = view(Xw,dkk,dll)
              if kk > 1
                 ir = 1:i-1
                 # W1 = A[ir,k]'*E[ir,l]
                 mul!(W1,transpose(view(A,ir,k)),view(E,ir,l))
                 #y -= W1*B[l,l]
                 mul!(y,W1,view(B,l,l),-ONE,ONE)
                 mul!(W1,transpose(view(C,ir,k)),view(E,ir,l))
                 #y -= W1*D[l,l]
                 mul!(y,W1,view(D,l,l),-ONE,ONE)
              end
              if ll > 1
                 ic = 1:i1
                 #WB[k,dll] = E[k,il1]*B[il1,l]
                 mul!(view(WB,k,dll),view(E,k,il1),view(B,il1,l))
                 #y -= A[ic,k]'*WB[ic,dll]
                 mul!(y,transpose(view(A,ic,k)),view(WB,ic,dll),-ONE,ONE)
                 #WD[k,dll] = E[k,il1]*D[il1,l]
                 mul!(view(WD,k,dll),view(E,k,il1),view(D,il1,l))
                 #y -= C[ic,k]'*WD[ic,dll]
                 mul!(y,transpose(view(C,ic,k)),view(WD,ic,dll),-ONE,ONE)
              end
              gsylv2!(adjAC,adjBD,y,dk,dl,view(A,k,k),view(B,l,l),view(C,k,k),view(D,l,l),Xw,Yw) 
              i += dk
          end
          j += dl
      end
   elseif adjAC && adjBD
      # """
      # The (K,L)th block of X is determined starting from
      # upper-right corner column by column by

      #            A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
      #                       J=1            I=L+1

      #                       K-1
      #                     { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
      #                       J=1            I=L+1
      # """
      j = n
      for ll = pb:-1:1
          dl = bb[ll]
          dll = 1:dl
          il1 = j+1:n
          l = j-dl+1:j
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = view(E,k,l)
              W1 = view(Xw,dkk,dll)
              if kk > 1
                 ir = 1:i-1
                 #W1 = A[ir,k]'*E[ir,l]
                 mul!(W1,transpose(view(A,ir,k)),view(E,ir,l))
                 #y -= W1*B[l,l]'
                 mul!(y,W1,transpose(view(B,l,l)),-ONE,ONE)
                 #   W2 = C[ir,k]'*E[ir,l]
               #   y -= W2*D[l,l]'
                 mul!(W1,transpose(view(C,ir,k)),view(E,ir,l))
                 #y -= W1*D[l,l]'
                 mul!(y,W1,transpose(view(D,l,l)),-ONE,ONE)
               end
              if ll < pb
                 ic = 1:i1
                 #WB[k,dll] = E[k,il1]*B[l,il1]'
                 mul!(view(WB,k,dll),view(E,k,il1),transpose(view(B,l,il1)))
                 #WD[k,dll] = E[k,il1]*D[l,il1]'
                 mul!(view(WD,k,dll),view(E,k,il1),transpose(view(D,l,il1)))
                 #y -= (A[ic,k]'*WB[ic,dll] + C[ic,k]'*WD[ic,dll])
                 mul!(y,transpose(view(A,ic,k)),view(WB,ic,dll),-ONE,ONE)
                 mul!(y,transpose(view(C,ic,k)),view(WD,ic,dll),-ONE,ONE)
              end
              gsylv2!(adjAC,adjBD,y,dk,dl,view(A,k,k),view(B,l,l),view(C,k,k),view(D,l,l),Xw,Yw) 
              i += dk
          end
          j -= dl
      end
   end
   return E
end
@inline function gsylv2!(adjAC::Bool,adjBD::Bool,E::StridedMatrix{T},na::Int,nb::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},C::AbstractMatrix{T},D::AbstractMatrix{T},Xw::StridedMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 generalized Sylvester equations: 
   #      A*X*B + C*X*D = E     if adjAC = false and adjBD = false -> R = kron(B',A)  + kron(D',C) 
   #      A'*X*B + C'*X*D = E   if adjAC = true and adjBD = false  -> R = kron(B',A') + kron(D',C')
   #      A*X*B' + C*X*D' = E   if adjAC = false and adjBD = true  -> R = kron(B,A)   + kron(D,C)
   #      A'*X*B' + C'*X*D' = E if adjAC = true and adjBD = true   -> R = kron(B,A')  + kron(D,C')
   if na == 1 && nb == 1
      temp = A[1,1]*B[1,1] + C[1,1]*D[1,1] 
      rmul!(E,inv(temp))
      any(!isfinite, E) &&  throw("ME:SingularException: `A-λC` and `D+λB` have common eigenvalues")
      return E
   end
   i1 = 1:na*nb
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   if !adjAC && !adjBD
      if na == 1
         # R12 = 
         # [ a11*b11 + c11*d11, a11*b21 + c11*d21]
         # [ a11*b12 + c11*d12, a11*b22 + c11*d22]
         # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,1]*B[2,1]+C[1,1]*D[2,1];
         #                 A[1,1]*B[1,2]+C[1,1]*D[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
         @inbounds  R[1,2] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+C[1,1]*D[2,2]
         @inbounds  Y[1] = E[1,1]
         @inbounds  Y[2] = E[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + c11*d11, a12*b11 + c12*d11]
            # [ a21*b11 + c21*d11, a22*b11 + c22*d11]
            # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]+C[1,2]*D[1,1];
            #                 A[2,1]*B[1,1]+C[2,1]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1] ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
         else
            # R = 
            # [ a11*b11 + c11*d11, a12*b11 + c12*d11, a11*b21 + c11*d21, a12*b21 + c12*d21]
            # [ a21*b11 + c21*d11, a22*b11 + c22*d11, a21*b21 + c21*d21, a22*b21 + c22*d21]
            # [ a11*b12 + c11*d12, a12*b12 + c12*d12, a11*b22 + c11*d22, a12*b22 + c12*d22]
            # [ a21*b12 + c21*d12, a22*b12 + c22*d12, a21*b22 + c21*d22, a22*b22 + c22*d22]
            # (iszero(C[2,1]) && iszero(D[2,1]) && iszero(C[1,2]) && iszero(D[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]      A[1,1]*B[2,1]      A[1,2]*B[2,1];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[2,1]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[1,2]*B[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[1,2]*B[2,2];
            # A[2,1]*B[1,2]      A[2,2]*B[1,2]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]+C[1,2]*D[1,1]      A[1,1]*B[2,1]+C[1,1]*D[2,1]      A[1,2]*B[2,1]+C[1,2]*D[2,1];
            # A[2,1]*B[1,1]+C[2,1]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[2,1]*B[2,1]+C[2,1]*D[2,1]      A[2,2]*B[2,1]+C[2,2]*D[2,1];
            # A[1,1]*B[1,2]+C[1,1]*D[1,2]      A[1,2]*B[1,2]+C[1,2]*D[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[1,2]*B[2,2]+C[1,2]*D[2,2];
            # A[2,1]*B[1,2]+C[2,1]*D[1,2]      A[2,2]*B[1,2]+C[2,2]*D[1,2]      A[2,1]*B[2,2]+C[2,1]*D[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]])
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
            @inbounds  R[1,4] = A[1,2]*B[2,1]+C[1,2]*D[2,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  R[2,3] = A[2,1]*B[2,1]+C[2,1]*D[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]+C[2,2]*D[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
            @inbounds  R[3,2] = A[1,2]*B[1,2]+C[1,2]*D[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+C[1,1]*D[2,2] 
            @inbounds  R[3,4] = A[1,2]*B[2,2]+C[1,2]*D[2,2]
            @inbounds  R[4,1] = A[2,1]*B[1,2]+C[2,1]*D[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]+C[2,2]*D[1,2]
            @inbounds  R[4,3] = A[2,1]*B[2,2]+C[2,1]*D[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+C[2,2]*D[2,2]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
            @inbounds  Y[3] = E[1,2]
            @inbounds  Y[4] = E[2,2]
         end
      end
      #R = kron(transpose(B),A) + kron(transpose(D),C)
   elseif adjAC && !adjBD
      if na == 1
         # R12 = 
         # [ a11*b11 + c11*d11, a11*b21 + c11*d21]
         # [ a11*b12 + c11*d12, a11*b22 + c11*d22]
         # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,1]*B[2,1]+C[1,1]*D[2,1];
         #                 A[1,1]*B[1,2]+C[1,1]*D[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
         @inbounds  R[1,2] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
         @inbounds  R[2,1] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+C[1,1]*D[2,2]
         @inbounds  Y[1] = E[1,1]
         @inbounds  Y[2] = E[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + c11*d11, a21*b11 + c21*d11]
            # [ a12*b11 + c12*d11, a22*b11 + c22*d11]
            # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]+C[2,1]*D[1,1];
            #                 A[1,2]*B[1,1]+C[1,2]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1] ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
         else
            # R = 
            # [ a11*b11 + c11*d11, a21*b11 + c21*d11, a11*b21 + c11*d21, a21*b21 + c21*d21]
            # [ a12*b11 + c12*d11, a22*b11 + c22*d11, a12*b21 + c12*d21, a22*b21 + c22*d21]
            # [ a11*b12 + c11*d12, a21*b12 + c21*d12, a11*b22 + c11*d22, a21*b22 + c21*d22]
            # [ a12*b12 + c12*d12, a22*b12 + c22*d12, a12*b22 + c12*d22, a22*b22 + c22*d22]
            # (iszero(C[2,1]) && iszero(D[2,1]) && iszero(C[1,2]) && iszero(D[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            # A[1,1]*B[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[2,1]*B[2,2];
            # A[1,2]*B[1,2]      A[2,2]*B[1,2]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]+C[2,1]*D[1,1]      A[1,1]*B[2,1]+C[1,1]*D[2,1]      A[2,1]*B[2,1]+C[2,1]*D[2,1];
            # A[1,2]*B[1,1]+C[1,2]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[1,2]*B[2,1]+C[1,2]*D[2,1]      A[2,2]*B[2,1]+C[2,2]*D[2,1];
            # A[1,1]*B[1,2]+C[1,1]*D[1,2]      A[2,1]*B[1,2]+C[2,1]*D[1,2]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[2,1]*B[2,2]+C[2,1]*D[2,2];
            # A[1,2]*B[1,2]+C[1,2]*D[1,2]      A[2,2]*B[1,2]+C[2,2]*D[1,2]      A[1,2]*B[2,2]+C[1,2]*D[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) 
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[1,3] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
            @inbounds  R[1,4] = A[2,1]*B[2,1]+C[2,1]*D[2,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  R[2,3] = A[1,2]*B[2,1]+C[1,2]*D[2,1]
            @inbounds  R[2,4] = A[2,2]*B[2,1]+C[2,2]*D[2,1]
            @inbounds  R[3,1] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
            @inbounds  R[3,2] = A[2,1]*B[1,2]+C[2,1]*D[1,2]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+C[1,1]*D[2,2] 
            @inbounds  R[3,4] = A[2,1]*B[2,2]+C[2,1]*D[2,2]
            @inbounds  R[4,1] = A[1,2]*B[1,2]+C[1,2]*D[1,2]
            @inbounds  R[4,2] = A[2,2]*B[1,2]+C[2,2]*D[1,2]
            @inbounds  R[4,3] = A[1,2]*B[2,2]+C[1,2]*D[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+C[2,2]*D[2,2]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
            @inbounds  Y[3] = E[1,2]
            @inbounds  Y[4] = E[2,2]
         end
      end
      #R = kron(transpose(B),transpose(A)) + kron(transpose(D),transpose(C))
   elseif !adjAC && adjBD
      if na == 1
         # R12 = 
         # [ a11*b11 + c11*d11, a11*b12 + c11*d12]
         # [ a11*b21 + c11*d21, a11*b22 + c11*d22]
         # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,1]*B[1,2]+C[1,1]*D[1,2];
         #                 A[1,1]*B[2,1]+C[1,1]*D[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
         @inbounds  R[1,2] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+C[1,1]*D[2,2]
         @inbounds  Y[1] = E[1,1]
         @inbounds  Y[2] = E[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + c11*d11, a12*b11 + c12*d11]
            # [ a21*b11 + c21*d11, a22*b11 + c22*d11]
            # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]+C[1,2]*D[1,1];
            #                 A[2,1]*B[1,1]+C[2,1]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1] ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,1] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
         else
            # R = 
            # [ a11*b11 + c11*d11, a12*b11 + c12*d11, a11*b12 + c11*d12, a12*b12 + c12*d12]
            # [ a21*b11 + c21*d11, a22*b11 + c22*d11, a21*b12 + c21*d12, a22*b12 + c22*d12]
            # [ a11*b21 + c11*d21, a12*b21 + c12*d21, a11*b22 + c11*d22, a12*b22 + c12*d22]
            # [ a21*b21 + c21*d21, a22*b21 + c22*d21, a21*b22 + c21*d22, a22*b22 + c22*d22]
            # (iszero(C[2,1]) && iszero(D[2,1]) && iszero(C[1,2]) && iszero(D[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]      A[1,1]*B[1,2]      A[1,2]*B[1,2];
            # A[2,1]*B[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[2,1]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[1,2]*B[2,2]+C[1,2]*D[2,2];
            # A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,2]*B[1,1]+C[1,2]*D[1,1]      A[1,1]*B[1,2]+C[1,1]*D[1,2]      A[1,2]*B[1,2]+C[1,2]*D[1,2];
            # A[2,1]*B[1,1]+C[2,1]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[2,1]*B[1,2]+C[2,1]*D[1,2]      A[2,2]*B[1,2]+C[2,2]*D[1,2];
            # A[1,1]*B[2,1]+C[1,1]*D[2,1]      A[1,2]*B[2,1]+C[1,2]*D[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[1,2]*B[2,2]+C[1,2]*D[2,2];
            # A[2,1]*B[2,1]+C[2,1]*D[2,1]      A[2,2]*B[2,1]+C[2,2]*D[2,1]      A[2,1]*B[2,2]+C[2,1]*D[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) 
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
            @inbounds  R[1,4] = A[1,2]*B[1,2]+C[1,2]*D[1,2]
            @inbounds  R[2,1] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  R[2,3] = A[2,1]*B[1,2]+C[2,1]*D[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]+C[2,2]*D[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
            @inbounds  R[3,2] = A[1,2]*B[2,1]+C[1,2]*D[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+C[1,1]*D[2,2] 
            @inbounds  R[3,4] = A[1,2]*B[2,2]+C[1,2]*D[2,2]
            @inbounds  R[4,1] = A[2,1]*B[2,1]+C[2,1]*D[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]+C[2,2]*D[2,1]
            @inbounds  R[4,3] = A[2,1]*B[2,2]+C[2,1]*D[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+C[2,2]*D[2,2]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
            @inbounds  Y[3] = E[1,2]
            @inbounds  Y[4] = E[2,2]
         end
      end
      #R = kron(B,A) + kron(D,C)
   else
      if na == 1
         # R12 = 
         # [ a11*b11 + c11*d11, a11*b12 + c11*d12]
         # [ a11*b21 + c11*d21, a11*b22 + c11*d22]
         # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[1,1]*B[1,2]+C[1,1]*D[1,2];
         #                 A[1,1]*B[2,1]+C[1,1]*D[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]]
         @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
         @inbounds  R[1,2] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
         @inbounds  R[2,1] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
         @inbounds  R[2,2] = A[1,1]*B[2,2]+C[1,1]*D[2,2]
         @inbounds  Y[1] = E[1,1]
         @inbounds  Y[2] = E[1,2]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + c11*d11, a21*b11 + c21*d11]
            # [ a12*b11 + c12*d11, a22*b11 + c22*d11]
            # @inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]+C[2,1]*D[1,1];
            #                 A[1,2]*B[1,1]+C[1,2]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1] ]
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[2,1] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
         else
            # R = 
            # [ a11*b11 + c11*d11, a21*b11 + c21*d11, a11*b12 + c11*d12, a21*b12 + c21*d12]
            # [ a12*b11 + c12*d11, a22*b11 + c22*d11, a12*b12 + c12*d12, a22*b12 + c22*d12]
            # [ a11*b21 + c11*d21, a21*b21 + c21*d21, a11*b22 + c11*d22, a21*b22 + c21*d22]
            # [ a12*b21 + c12*d21, a22*b21 + c22*d21, a12*b22 + c12*d22, a22*b22 + c22*d22]
            # (iszero(C[2,1]) && iszero(D[2,1]) && iszero(C[1,2]) && iszero(D[1,2])) ?
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]      A[1,1]*B[1,2]      A[2,1]*B[1,2];
            # A[1,2]*B[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[1,2]*B[1,2]      A[2,2]*B[1,2];
            # A[1,1]*B[2,1]      A[2,1]*B[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[2,1]*B[2,2];
            # A[1,2]*B[2,1]      A[2,2]*B[2,1]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) :
            # (@inbounds R = [ A[1,1]*B[1,1]+C[1,1]*D[1,1]      A[2,1]*B[1,1]+C[2,1]*D[1,1]      A[1,1]*B[1,2]+C[1,1]*D[1,2]      A[2,1]*B[1,2]+C[2,1]*D[1,2];
            # A[1,2]*B[1,1]+C[1,2]*D[1,1]  A[2,2]*B[1,1]+C[2,2]*D[1,1]      A[1,2]*B[1,2]+C[1,2]*D[1,2]      A[2,2]*B[1,2]+C[2,2]*D[1,2];
            # A[1,1]*B[2,1]+C[1,1]*D[2,1]      A[2,1]*B[2,1]+C[2,1]*D[2,1]  A[1,1]*B[2,2]+C[1,1]*D[2,2]      A[2,1]*B[2,2]+C[2,1]*D[2,2];
            # A[1,2]*B[2,1]+C[1,2]*D[2,1]      A[2,2]*B[2,1]+C[2,2]*D[2,1]      A[1,2]*B[2,2]+C[1,2]*D[2,2]  A[2,2]*B[2,2]+C[2,2]*D[2,2]]) 
            @inbounds  R[1,1] = A[1,1]*B[1,1]+C[1,1]*D[1,1]
            @inbounds  R[1,2] = A[2,1]*B[1,1]+C[2,1]*D[1,1]
            @inbounds  R[1,3] = A[1,1]*B[1,2]+C[1,1]*D[1,2]
            @inbounds  R[1,4] = A[2,1]*B[1,2]+C[2,1]*D[1,2]
            @inbounds  R[2,1] = A[1,2]*B[1,1]+C[1,2]*D[1,1]
            @inbounds  R[2,2] = A[2,2]*B[1,1]+C[2,2]*D[1,1]
            @inbounds  R[2,3] = A[1,2]*B[1,2]+C[1,2]*D[1,2]
            @inbounds  R[2,4] = A[2,2]*B[1,2]+C[2,2]*D[1,2]
            @inbounds  R[3,1] = A[1,1]*B[2,1]+C[1,1]*D[2,1]
            @inbounds  R[3,2] = A[2,1]*B[2,1]+C[2,1]*D[2,1]
            @inbounds  R[3,3] = A[1,1]*B[2,2]+C[1,1]*D[2,2] 
            @inbounds  R[3,4] = A[2,1]*B[2,2]+C[2,1]*D[2,2]
            @inbounds  R[4,1] = A[1,2]*B[2,1]+C[1,2]*D[2,1]
            @inbounds  R[4,2] = A[2,2]*B[2,1]+C[2,2]*D[2,1]
            @inbounds  R[4,3] = A[1,2]*B[2,2]+C[1,2]*D[2,2]
            @inbounds  R[4,4] = A[2,2]*B[2,2]+C[2,2]*D[2,2]
            @inbounds  Y[1] = E[1,1]
            @inbounds  Y[2] = E[2,1]
            @inbounds  Y[3] = E[1,2]
            @inbounds  Y[4] = E[2,2]
         end
      end
      #R = kron(B,transpose(A)) + kron(D,transpose(C))
   end
   luslv!(R,Y) && throw("ME:SingularException: A has eigenvalue(s) α and B has eingenvalu(s) β such that αβ = -1")
   E[:,:] = Y
   return E
end

function gsylvs!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}, D::AbstractMatrix{T1}, E::AbstractMatrix{T1}; 
                 adjAC::Bool = false, adjBD::Bool = false, CASchur::Bool = false, DBSchur::Bool = false) where T1<:BlasComplex
   """
   An extension proposed in [1] of the Bartels-Stewart Schur form based approach [2] is employed.

   References:
   [1] K.-W. E. Chu. The solution of the matrix equation AXB – CXD = E and
       (YA – DZ, YC– BZ) = (E, F). Lin. Alg. Appl., 93:93-105, 1987.
   [2] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
       Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(E);
   [m; n; m; n] == LinearAlgebra.checksquare(A,B,C,D) ||
      throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))

   WB = zeros(T1,m)
   WD = zeros(T1,m)
   ZERO = zero(T1)
   if !adjAC && !adjBD
      # """
      # The (K,L)th element of X is determined starting from
      # bottom-left corner column by column by

      #       A(K,K)*X(K,L)*B(L,L) +C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      # where
      #                        M
      #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
      #                      J=K+1
      #                        M             L-1
      #                       SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
      #                       J=K            I=1

      #                       M
      #                    { SUM [C(K,J)*X(J,L)] } * D(L,L) +
      #                     J=K+1
      #                       M             L-1
      #                      SUM { C(K,J) * SUM [X(J,I)*D(I,L)] } +
      #                      J=K            I=1
      # """
      for l = 1:n
          il1 = 1:l-1
          for k = m:-1:1
              y = E[k,l]
              if k < m
                 ta = ZERO
                 tc = ZERO
                 for ir = k+1:m
                     ta += A[k,ir]*E[ir,l]
                     tc += C[k,ir]*E[ir,l]
                 end
                 y -= (ta*B[l,l]+tc*D[l,l])
              end
              if l > 1
                 ta = ZERO
                 tc = ZERO
                 for ir = il1
                     ta += E[k,ir]*B[ir,l]
                     tc += E[k,ir]*D[ir,l]
                 end
                 WB[k] = ta
                 WD[k] = tc
                 for ic = k:m
                     y -= (A[k,ic]*WB[ic]+C[k,ic]*WD[ic])
                 end
              end
              E[k,l] = y/(B[l,l]*A[k,k]+D[l,l]*C[k,k])
              isfinite(E[k,l]) || throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
          end
      end
   elseif !adjAC && adjBD
         # """
         #  The (K,L)th element of X is determined starting from
         #  bottom-right corner column by column by

         #       A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

         #  where
         #                        M
         #            R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
         #                      J=K+1
         #                        M              N
         #                       SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
         #                       J=K           I=L+1

         #                       M
         #                    { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
         #                     J=K+1
         #                       M              N
         #                      SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
         #                      J=K           I=L+1
         # """
         for l = n:-1:1
             il1 = l+1:n
             for k = m:-1:1
                 y = E[k,l]
                 if k < m
                    ta = ZERO
                    tc = ZERO
                    for ir = k+1:m
                        ta += A[k,ir]*E[ir,l]
                        tc += C[k,ir]*E[ir,l]
                    end
                    y -= (ta*B[l,l]'+tc*D[l,l]')
                 end
                 if l < n
                    ta = ZERO
                    tc = ZERO
                    for ir = il1
                        ta += E[k,ir]*B[l,ir]'
                        tc += E[k,ir]*D[l,ir]'
                    end
                    WB[k] = ta
                    WD[k] = tc
                    for ic = k:m
                        y -= (A[k,ic]*WB[ic]+C[k,ic]*WD[ic])
                    end
                 end
                 E[k,l] = y/(B[l,l]'*A[k,k]+D[l,l]'*C[k,k])
                 isfinite(E[k,l]) || throw("ME:SingularException: A-λC and D'+λB' have common or close eigenvalues")
             end
         end
   elseif adjAC && !adjBD
      # """
      # The (K,L)th element of X is determined starting from the
      # upper-left corner column by column by

      # A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
      #                       J=1             I=1

      #                       K-1
      #                     { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
      #                       J=1
      #                        K              L-1
      #                       SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
      #                       J=1             I=1
      # """
      for l = 1:n
          il1 = 1:l-1
          for k = 1:m
              y = E[k,l]
              if k > 1
                 ta = ZERO
                 tc = ZERO
                 for ir = 1:k-1
                     ta += A[ir,k]'*E[ir,l]
                     tc += C[ir,k]'*E[ir,l]
                 end
                 y -= (ta*B[l,l]+tc*D[l,l])
              end
              if l > 1
                 ta = ZERO
                 tc = ZERO
                 for ir = il1
                     ta += E[k,ir]*B[ir,l]
                     tc += E[k,ir]*D[ir,l]
                 end
                 WB[k] = ta
                 WD[k] = tc
                 for ic = 1:m
                     y -= (A[ic,k]'*WB[ic]+C[ic,k]'*WD[ic])
                 end
              end
              E[k,l] = y/(B[l,l]*A[k,k]'+D[l,l]*C[k,k]')
              isfinite(E[k,l]) || throw("ME:SingularException: A'-λC' and D+λB have common or close eigenvalues")
          end
      end
   elseif adjAC && adjBD
      # """
      # The (K,L)th element of X is determined starting from
      # upper-rght corner column by column by

      #       A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      # where
      #                       K-1
      #            R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
      #                       J=1            I=L+1

      #                       K-1
      #                     { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
      #                       J=1
      #                        K               N
      #                       SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
      #                       J=1            I=L+1
      # """
      for l = n:-1:1
          il1 = l+1:n
          for k = 1:m
              y = E[k,l]
              if k > 1
                 ta = ZERO
                 tc = ZERO
                 for ir = 1:k-1
                     ta += A[ir,k]'*E[ir,l]
                     tc += C[ir,k]'*E[ir,l]
                 end
                 y -= (ta*B[l,l]'+tc*D[l,l]')
              end
              if l < n
                 ic = 1:m
                 ta = ZERO
                 tc = ZERO
                 for ir = il1
                     ta += E[k,ir]*B[l,ir]'
                     tc += E[k,ir]*D[l,ir]'
                 end
                 WB[k] = ta
                 WD[k] = tc
                 for ic = 1:m
                     y -= (A[ic,k]'*WB[ic]+C[ic,k]'*WD[ic])
                 end
              end
              E[k,l] = y/(B[l,l]'*A[k,k]'+D[l,l]'*C[k,k]')
              isfinite(E[k,l]) || throw("ME:SingularException: A'-λC' and D'+λB' have common or close eigenvalues")
          end
      end
   end
   return E
end
"""
    (X,Y) = sylvsyss!(A,B,C,D,E,F)

Solve the Sylvester system of matrix equations

                AX + YB = C
                DX + YE = F,

where `(A,D)`, `(B,E)` are pairs of square matrices of the same size in generalized Schur forms.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues. The computed
solution `(X,Y)` is contained in `(C,F)`.

_Note:_ This is an enhanced interface to the `LAPACK.tgsyl!` function to also cover the case when
`A`, `B`, `D` and `E` are real matrices and `C` and `F` are complex matrices.
"""
function sylvsyss!(A::T1, B::T1, C::T1, D::T1, E::T1, F::T1) where {T<:BlasFloat,T1<:Matrix{T}}
   """
   This is a wrapper to the LAPACK.tgsyl! function with `trans = 'N'`.
   """
   C, F, scale =  tgsyl!('N',A,B,C,D,E,F)
   return rmul!(C,inv(scale)), rmul!(F,inv(-scale))
end
"""
    (X,Y) = dsylvsyss!(A,B,C,D,E,F)

Solve the dual Sylvester system of matrix equations

    A'X + D'Y = C
    XB' + YE' = F,

where `(A,D)`, `(B,E)` are pairs of square matrices of the same size in generalized Schur forms.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues. The computed
solution `(X,Y)` is contained in `(C,F)`.
"""
function dsylvsyss!(A::T1, B::T1, C::T1, D::T1, E::T1, F::T1) where {T<:BlasFloat,T1<:Matrix{T}}
   """
   This is an interface to the LAPACK.tgsyl! function with `trans = 'T' or `trans = 'C'`. 
   """
   # MF = -F
   # E, F, scale =  tgsyl!(T <: Complex ? 'C' : 'T', A, B, C, D, E, MF)
   # F = MF
   # return rmul!(C[:,:],inv(scale)), rmul!(F[:,:],inv(scale))
   F = -F
   C, F, scale =  tgsyl!(T <: Complex ? 'C' : 'T', A, B, C, D, E, F)
   return rmul!(C[:,:],inv(scale)), rmul!(F[:,:],inv(scale))
end
