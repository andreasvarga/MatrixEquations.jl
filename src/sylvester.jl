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

   realcase = eltype(A) <: AbstractFloat
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
   D = adjoint(QA) * (C*QB)
   Y = sylvcs!(RA, RB, D, adjA = adjA, adjB = adjB)
   QA*(Y * adjoint(QB))
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
   D = adjoint(QA) * (C*QB)
   Y = sylvds!(RA, RB, D, adjA = adjA, adjB = adjB)
   QA*(Y * adjoint(QB))
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
       if adjA
          A = copy(A)
       end
       if adjC
          C = copy(C)
       end
       AS, CS, Q1, Z1 = schur(A,C)
    end
    if adjBD
       BS, DS, Z2, Q2 = schur(B.parent,D.parent)
    else
      if adjB
          B = copy(B)
      end
      if adjD
          D = copy(D)
      end
       BS, DS, Q2, Z2 = schur(B,D)
    end
    Y = adjoint(Q1) * (E*Z2)
    gsylvs!(AS, BS, CS, DS, Y, adjAC = adjAC, adjBD = adjBD)
    Z1*(Y * adjoint(Q2))
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

    if isa(A,Adjoint)
      A = copy(A)
    end
    if isa(B,Adjoint)
      B = copy(B)
    end
    if isa(D,Adjoint)
      D = copy(D)
    end
    if isa(E,Adjoint)
      E = copy(E)
    end
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
function sylvcs!(A::Matrix{T1}, B::Matrix{T1}, C::Matrix{T1}; adjA = false, adjB = false) where  T1<:BlasFloat
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
   catch err
      findfirst("LAPACKException(1)",string(err)) === nothing ? rethrow() : 
               throw("ME:SingularException: A has eigenvalue(s) α and B has eigenvalues(s) β such that α+β = 0")
   end
end
@inline function sylvd2!(adjA,adjB,C::StridedMatrix{T},na::Int,nb::Int,A::StridedMatrix{T},B::StridedMatrix{T},Xw::StridedMatrix{T},Yw::StridedVector{T}) where T <:BlasReal
   # speed and reduced allocation oriented implementation of a solver for 1x1 and 2x2 Sylvester equations 
   # encountered in solving discrete Lyapunov equations: 
   # A*X*B + X = C   if adjA = false and adjB = false -> R = kron(B',A) + I 
   # A'*X*B + X = C  if adjA = true  and adjB = false -> R = kron(B',A') + I 
   # A*X*B' + X = C  if adjA = false and adjB = true  -> R = kron(B,A) + I
   # A'*X*B' + X = C if adjA = true  and adjB = true  -> R = kron(B,A') + I
   ONE = one(T)
   if na == 1 && nb == 1
      temp = A[1,1]*B[1,1] + ONE
      iszero(temp) && throw("ME:SingularException: A has eigenvalue(s) α and B has eigenvalues(s) β such that αβ ≈ -1")
      return rmul!(C,inv(temp))
   end
   nv = na*nb
   i1 = 1:nv
   R = view(Xw,i1,i1)
   Y = view(Yw,i1)
   Y[:] = C[i1]
   if adjA && !adjB
      if na == 1
         # R12 = 
         # [ a11*b11-1      a11*b21]
         # [     a11*b12  a11*b22-1]
         @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[2,1];
                         A[1,1]*B[1,2]  A[1,1]*B[2,2]+ONE]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11-1      a21*b11]
            # [     a12*b11  a22*b11-1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1];
                            A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE ]
         else
            # R = 
            # [ a11*b11-1      a21*b11      a11*b21      a21*b21]
            # [     a12*b11  a22*b11-1      a12*b21      a22*b21]
            # [     a11*b12      a21*b12  a11*b22-1      a21*b22]
            # [     a12*b12      a22*b12      a12*b22  a22*b22-1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1]      A[1,1]*B[2,1]      A[2,1]*B[2,1];
            A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE      A[1,2]*B[2,1]      A[2,2]*B[2,1];
            A[1,1]*B[1,2]      A[2,1]*B[1,2]  A[1,1]*B[2,2]+ONE      A[2,1]*B[2,2];
            A[1,2]*B[1,2]      A[2,2]*B[1,2]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+ONE]
         end
      end
   elseif !adjA && adjB
      if na == 1
         # R12 =
         # [ a11*b11-1      a11*b12]
         # [     a11*b21  a11*b22-1]
         @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[1,2];
                         A[1,1]*B[2,1]  A[1,1]*B[2,2]+ONE]
      else
         if nb == 1
            # R21 =
            #    [ a11*b11-1      a12*b11]
            #    [     a21*b11  a22*b11-1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1];
                            A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE]
         else
            # R = 
            # [ a11*b11-1      a12*b11      a11*b12      a12*b12]
            # [     a21*b11  a22*b11-1      a21*b12      a22*b12]
            # [     a11*b21      a12*b21  a11*b22-1      a12*b22]
            # [     a21*b21      a22*b21      a21*b22  a22*b22-1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1]      A[1,1]*B[1,2]      A[1,2]*B[1,2];
            A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE      A[2,1]*B[1,2]      A[2,2]*B[1,2];
            A[1,1]*B[2,1]      A[1,2]*B[2,1]  A[1,1]*B[2,2]+ONE      A[1,2]*B[2,2];
            A[2,1]*B[2,1]      A[2,2]*B[2,1]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+ONE]
         end
      end
   elseif !adjA && !adjB
      if na == 1
         # R12 = 
         # [ a11*b11 + 1,     a11*b21]
         # [     a11*b12, a11*b22 + 1]
         @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[2,1];
                         A[1,1]*B[1,2]  A[1,1]*B[2,2]+ONE]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + 1,     a12*b11]
            # [     a21*b11, a22*b11 + 1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1];
                            A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE]
         else
            # R = 
            # [ a11*b11 + 1,     a12*b11,     a11*b21,     a12*b21]
            # [     a21*b11, a22*b11 + 1,     a21*b21,     a22*b21]
            # [     a11*b12,     a12*b12, a11*b22 + 1,     a12*b22]
            # [     a21*b12,     a22*b12,     a21*b22, a22*b22 + 1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,2]*B[1,1]      A[1,1]*B[2,1]      A[1,2]*B[2,1];
            A[2,1]*B[1,1]  A[2,2]*B[1,1]+ONE      A[2,1]*B[2,1]      A[2,2]*B[2,1];
            A[1,1]*B[1,2]      A[1,2]*B[1,2]  A[1,1]*B[2,2]+ONE      A[1,2]*B[2,2];
            A[2,1]*B[1,2]      A[2,2]*B[1,2]      A[2,1]*B[2,2]  A[2,2]*B[2,2]+ONE]
         end
      end
   else
      if na == 1
         # R12 = 
         # [ a11*b11 + 1,     a11*b12]
         # [     a11*b21, a11*b22 + 1]
         @inbounds R = [ A[1,1]*B[1,1]+ONE      A[1,1]*B[1,2];
                         A[1,1]*B[2,1]  A[1,1]*B[2,2]+ONE]
      else
         if nb == 1
            # R21 = 
            # [ a11*b11 + 1,     a21*b11]
            # [     a12*b11, a22*b11 + 1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1];
                            A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE]
         else
            # R =
            # [ a11*b11 + 1,     a21*b11,     a11*b12,     a21*b12]
            # [     a12*b11, a22*b11 + 1,     a12*b12,     a22*b12]
            # [     a11*b21,     a21*b21, a11*b22 + 1,     a21*b22]
            # [     a12*b21,     a22*b21,     a12*b22, a22*b22 + 1]
            @inbounds R = [ A[1,1]*B[1,1]+ONE      A[2,1]*B[1,1]      A[1,1]*B[1,2]      A[2,1]*B[1,2];
            A[1,2]*B[1,1]  A[2,2]*B[1,1]+ONE      A[1,2]*B[1,2]      A[2,2]*B[1,2];
            A[1,1]*B[2,1]      A[2,1]*B[2,1]  A[1,1]*B[2,2]+ONE      A[2,1]*B[2,2];
            A[1,2]*B[2,1]      A[2,2]*B[2,1]      A[1,2]*B[2,2]  A[2,2]*B[2,2]+ONE]
         end
      end
   end
   try
      ldiv!(lu!(R),Y)
   catch 
      throw("ME:SingularException: A has eigenvalue(s) α and B has eingenvalu(s) β such that αβ ≈ -1")
   end
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
function sylvds!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}; adjA = false, adjB = false) where  T1<:BlasReal
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   (size(C,1) == m && size(C,2) == n ) || throw(DimensionMismatch("C must be an $m x $n matrix"))
   
   # determine the structure of the real Schur form of A
   ba, pa = sfstruct(A)
   bb, pb = sfstruct(B)

   W = zeros(T1,m,2)
   Xw = Matrix{T1}(undef,4,4)
   Yw = Vector{T1}(undef,4)
   if !adjA && !adjB
      """
      The (K,L)th block of X is determined starting from
      bottom-left corner column by column by

                 A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
                            J=K            I=1
      """
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
              y = C[k,l]
              if kk < pa
                 ir = i+1:m
                 W1 = A[k,ir]*C[ir,l]
                 y -= W1*B[l,l]
              end
              if ll > 1
                 ic = i1:m
                 W[k,dll] = C[k,il1]*B[il1,l]
                 y -= A[k,ic]*W[ic,dll]
              end
              C[k,l] = sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
            #   Z = (kron(transpose(B[l,l]),A[k,k])+I)\(y[:])
            #   isfinite(maximum(abs.(Z))) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
              i -= dk
          end
          j += dl
      end
   elseif !adjA && adjB
         """
         The (K,L)th block of X is determined starting from
         bottom-right corner column by column by

                     A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

         where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
                               J=K           I=L+1
         """
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
                 y = C[k,l]
                 if kk < pa
                    ir = i+1:m
                    W1 = A[k,ir]*C[ir,l]
                    y -= W1*B[l,l]'
                 end
                 if ll < pb
                    ic = i1:m
                    W[k,dll] = C[k,il1]*B[l,il1]'
                    y -= A[k,ic]*W[ic,dll]
                 end
                 C[k,l] = sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
                 #   Z = (kron(B[l,l],A[k,k])+I)\(y[:])
               #   isfinite(maximum(abs.(Z))) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
                 i -= dk
             end
             j -= dl
         end
   elseif adjA && !adjB
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
                            J=1             I=1
      """
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
              y = C[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*C[ir,l]
                 y -= W1*B[l,l]
              end
              if ll > 1
                 ic = 1:i1
                 W[k,dll] = C[k,il1]*B[il1,l]
                 y -= A[ic,k]'*W[ic,dll]
              end
              C[k,l] = sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
            #   Z = (kron(transpose(B[l,l]),transpose(A[k,k]))+I)\(y[:])
            #   isfinite(maximum(abs.(Z))) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
              i += dk
          end
          j += dl
      end
   elseif adjA && adjB
      """
      The (K,L)th block of X is determined starting from the
      lower-left corner column by column by

                 A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
                            J=1            I=L+1
      """
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
              y = C[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*C[ir,l]
                 y -= W1*B[l,l]'
              end
              if ll < pb
                 ic = 1:i1
                 W[k,dll] = C[k,il1]*B[l,il1]'
                 y -= A[ic,k]'*W[ic,dll]
              end
              C[k,l] = sylvd2!(adjA,adjB,y,dk,dl,view(A,k,k),view(B,l,l),Xw,Yw)  
            #   Z = (kron(B[l,l],transpose(A[k,k]))+I)\(y[:])
            #   isfinite(maximum(abs.(Z))) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
              i += dk
          end
          j -= dl
      end
   end
   return C
end
function sylvds!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::AbstractMatrix{T1}; adjA = false, adjB = false) where  T1<:BlasComplex
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   (size(C,1) == m && size(C,2) == n ) || throw(DimensionMismatch("C must be an $m x $n matrix"))
  
   W = zeros(T1,m,1)
   ONE = one(T1)
   if !adjA && !adjB
      """
      The (K,L)th element of X is determined starting from
      bottom-left corner column by column by

                 A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
                            J=K            I=1
      """
      for l = 1:n
          il1 = 1:l-1
          ll = l:l
          for k = m:-1:1
              y = C[k,l]
              kk = k:k
              if k < m
                 ir = k+1:m
                 W1 = A[kk,ir]*C[ir,ll]
                 y -= W1[1]*B[l,l]
              end
              if l > 1
                 ic = k:m
                 Z = C[kk,il1]*B[il1,ll]
                 W[k,1] = Z[1]
                 TA = A[kk,ic]*W[ic,1]
                 y -= TA[1]
              end
              temp = B[l,l]*A[k,k]+ONE
              iszero(temp) && throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
              C[k,l] = y/temp
            #   Z = y/(B[l,l]*A[k,k]+I)
            #   isfinite(Z) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
             end
      end
   elseif !adjA && adjB
         """
         The (K,L)th element of X is determined starting from
         bottom-right corner column by column by

                  A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

         where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
                               J=K           I=L+1
         """
         for l = n:-1:1
             ll = l:l
             il1 = l+1:n
             for k = m:-1:1
                 kk = k:k
                 y = C[k,l]
                 if k < m
                    ir = k+1:m
                    W1 = A[kk,ir]*C[ir,ll]
                    y -= W1[1]*B[l,l]'
                 end
                 if l < n
                    ic = k:m
                    Z = C[kk,il1]*B[ll,il1]'
                    W[k,1] = Z[1]
                    TA = A[kk,ic]*W[ic,1]
                    y -= TA[1]
                 end
                 temp = B[l,l]'*A[k,k]+ONE
                 iszero(temp) && throw("ME:SingularException: A and -B' have common or close reciprocal eigenvalues")
                 C[k,l] = y/temp
               #   Z = y/(B[l,l]'*A[k,k]+I)
               #   isfinite(Z) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
             end
         end
   elseif adjA && !adjB
      """
      The (K,L)th element of X is determined starting from the
      upper-left corner column by column by

               A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
                            J=1             I=1
      """
      for l = 1:n
          ll = l:l
          il1 = 1:l-1
          for k = 1:m
              kk = k:k
              y = C[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*C[ir,ll]
                 y -= W1[1]*B[l,l]
              end
              if l > 1
                 ic = 1:m
                 Z = C[kk,il1]*B[il1,ll]
                 W[k,1] = Z[1]
                 TA = A[ic,kk]'*W[ic,1]
                 y -= TA[1]
              end
              temp = B[l,l]*A[k,k]'+ONE
              iszero(temp) && throw("ME:SingularException: A' and -B have common or close reciprocal eigenvalues")
              C[k,l] = y/temp
            #   Z = y/(B[l,l]*A[k,k]'+I)
            #   isfinite(Z) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
          end
      end
   elseif adjA && adjB
      """
      The (K,L)th element of X is determined starting from the
      upper-right corner column by column by

              A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
                            J=1            I=L+1
      """
      for l = n:-1:1
          ll = l:l
          il1 = l+1:n
          for k = 1:m
              kk = k:k
              y = C[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*C[ir,ll]
                 y -= W1[1]*B[l,l]'
              end
              if l < n
                 ic = 1:m
                 Z = C[kk,il1]*B[ll,il1]'
                 W[k,1] = Z[1]
                 TA = A[ic,kk]'*W[ic,1]
                 y -= TA[1]
              end
              temp = B[l,l]'*A[k,k]'+ONE
              iszero(temp) && throw("ME:SingularException: A' and -B' have common or close reciprocal eigenvalues")
              C[k,l] = y/temp
            #   Z = y/(B[l,l]'*A[k,k]'+I)
            #   isfinite(Z) ? C[k,l] = Z : throw("ME:SingularException: A and -B have common or close reciprocal eigenvalues")
          end
      end
   end
   return C
end
"""
    X = gsylvs!(A,B,C,D,E; adjAC=false, adjBD=false, DBSchur = false)

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
function gsylvs!(A::T1, B::T1, C::T1, D::T1, E::T1; adjAC = false, adjBD = false, CASchur = false, DBSchur = false) where 
                 {T<:BlasReal,T1<:Matrix{T}}
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
 

   # determine the structure of the generalized real Schur form of (A,C)
   ba = fill(1,m,1)
   pa = 1
   if m > 1
      CASchur ? d = [diag(C,-1);zeros(1)] : d = [diag(A,-1);zeros(1)]
      i = 1
      pa = 0
      while i <= m
         pa += 1
         if d[i] != 0
            ba[pa] = 2
            i += 1
         end
         i += 1
      end
   end
   # determine the structure of the generalized real Schur form of (B,D)
   bb = fill(1,n,1)
   pb = 1
   if n > 1
      DBSchur ? d = [diag(D,-1);zeros(1)] : d = [diag(B,-1);zeros(1)]
      i = 1
      pb = 0
      while i <= n
         pb += 1
         if d[i] != 0
            bb[pb] = 2
            i += 1
         end
         i += 1
      end
   end

   WB = fill(zero(eltype(E)),m,2)
   WD = fill(zero(eltype(E)),m,2)
   if !adjAC && !adjBD
      """
      The (K,L)th block of X is determined starting from
      bottom-left corner column by column by

            A(K,K)*X(K,L)*B(L,L) + C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
                            J=K            I=1

                             M
                          { SUM [C(K,J)*X(J,L)] } * D(L,L) +
                           J=K+1
                             M             L-1
                            SUM { C(K,J) * SUM [X(J,I)*D(I,L)] }.
                            J=K            I=1
      """
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
              y = E[k,l]
              if kk < pa
                 ir = i+1:m
                 W1 = A[k,ir]*E[ir,l]
                 y -= W1*B[l,l]
                 W1 = C[k,ir]*E[ir,l]
                 y -= W1*D[l,l]
              end
              if ll > 1
                 ic = i1:m
                 WB[k,dll] = E[k,il1]*B[il1,l]
                 WD[k,dll] = E[k,il1]*D[il1,l]
                 y -= (A[k,ic]*WB[ic,dll] + C[k,ic]*WD[ic,dll])
              end
              Z = (kron(transpose(B[l,l]),A[k,k])+kron(transpose(D[l,l]),C[k,k]))\(y[:])
              isfinite(maximum(abs.(Z))) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
              i -= dk
          end
          j += dl
      end
   elseif !adjAC && adjBD
         """
          The (K,L)th block of X is determined starting from
          bottom-right corner column by column by

               A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

          where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
                               J=K           I=L+1

                               M
                            { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
                             J=K+1
                               M              N
                              SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
                              J=K           I=L+1
         """
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
                 y = E[k,l]
                 if kk < pa
                    ir = i+1:m
                    W1 = A[k,ir]*E[ir,l]
                    y -= W1*B[l,l]'
                    W2 = C[k,ir]*E[ir,l]
                    y -= W2*D[l,l]'
                 end
                 if ll < pb
                    ic = i1:m
                    WB[k,dll] = E[k,il1]*B[l,il1]'
                    WD[k,dll] = E[k,il1]*D[l,il1]'
                    y -= (A[k,ic]*WB[ic,dll]+C[k,ic]*WD[ic,dll])
                 end
                 Z = (kron(B[l,l],A[k,k])+kron(D[l,l],C[k,k]))\(y[:])
                 isfinite(maximum(abs.(Z))) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
                 i -= dk
             end
             j -= dl
         end
   elseif adjAC && !adjBD
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
                            J=1             I=1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
                            J=1
                             K              L-1
                            SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
                            J=1             I=1
      """
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
              y = E[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*E[ir,l]
                 y -= W1*B[l,l]
                 W2 = C[ir,k]'*E[ir,l]
                 y -= W2*D[l,l]
              end
              if ll > 1
                 ic = 1:i1
                 WB[k,dll] = E[k,il1]*B[il1,l]
                 y -= A[ic,k]'*WB[ic,dll]
                 WD[k,dll] = E[k,il1]*D[il1,l]
                 y -= C[ic,k]'*WD[ic,dll]
              end
              Z = (kron(transpose(B[l,l]),transpose(A[k,k]))+kron(transpose(D[l,l]),transpose(C[k,k])))\(y[:])
              isfinite(maximum(abs.(Z))) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
              i += dk
          end
          j += dl
      end
   elseif adjAC && adjBD
      """
      The (K,L)th block of X is determined starting from
      upper-right corner column by column by

                 A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
                            J=1            I=L+1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
                            J=1
                             K               N
                            SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
                            J=1            I=L+1
      """
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
              y = E[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*E[ir,l]
                 y -= W1*B[l,l]'
                 W2 = C[ir,k]'*E[ir,l]
                 y -= W2*D[l,l]'
              end
              if ll < pb
                 ic = 1:i1
                 WB[k,dll] = E[k,il1]*B[l,il1]'
                 WD[k,dll] = E[k,il1]*D[l,il1]'
                 y -= (A[ic,k]'*WB[ic,dll] + C[ic,k]'*WD[ic,dll])
              end
              Z = (kron(B[l,l],transpose(A[k,k]))+kron(D[l,l],transpose(C[k,k])))\(y[:])
              isfinite(maximum(abs.(Z))) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
              i += dk
          end
          j -= dl
      end
   end
   return E
end
function gsylvs!(A::T1, B::T1, C::T1, D::T1, E::T1; adjAC = false, adjBD = false, CASchur = false, DBSchur = false) where 
   {T<:BlasComplex,T1<:Matrix{T}}
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

   WB = zeros(T,m,1)
   WD = zeros(T,m,1)
   if !adjAC && !adjBD
      """
      The (K,L)th element of X is determined starting from
      bottom-left corner column by column by

            A(K,K)*X(K,L)*B(L,L) +C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
                            J=K            I=1

                            M
                         { SUM [C(K,J)*X(J,L)] } * D(L,L) +
                          J=K+1
                            M             L-1
                           SUM { C(K,J) * SUM [X(J,I)*D(I,L)] } +
                           J=K            I=1
      """
      for l = 1:n
          il1 = 1:l-1
          ll = l:l
          for k = m:-1:1
              y = E[k,l]
              kk = k:k
              if k < m
                 ir = k+1:m
                 W1 = A[kk,ir]*E[ir,ll]
                 W2 = C[kk,ir]*E[ir,ll]
                 y -= (W1[1]*B[l,l]+W2[1]*D[l,l])
              end
              if l > 1
                 ic = k:m
                 ZB = E[kk,il1]*B[il1,ll]
                 ZD = E[kk,il1]*D[il1,ll]
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[kk,ic]*WB[ic,1]
                 TD = C[kk,ic]*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              temp = B[l,l]*A[k,k]+D[l,l]*C[k,k]
              iszero(temp) && throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
              E[k,l] = y/temp
            #   Z = y/(B[l,l]*A[k,k]+D[l,l]*C[k,k])
            #   isfinite(Z) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
          end
      end
   elseif !adjAC && adjBD
         """
          The (K,L)th element of X is determined starting from
          bottom-right corner column by column by

               A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

          where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
                               J=K           I=L+1

                               M
                            { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
                             J=K+1
                               M              N
                              SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
                              J=K           I=L+1
         """
         for l = n:-1:1
             ll = l:l
             il1 = l+1:n
             for k = m:-1:1
                 kk = k:k
                 y = E[k,l]
                 if k < m
                    ir = k+1:m
                    W1 = A[kk,ir]*E[ir,ll]
                    W2 = C[kk,ir]*E[ir,ll]
                    y -= (W1[1]*B[l,l]'+W2[1]*D[l,l]')
                 end
                 if l < n
                    ic = k:m
                    ZB = E[kk,il1]*B[ll,il1]'
                    ZD = E[kk,il1]*D[ll,il1]'
                    WB[k,1] = ZB[1]
                    WD[k,1] = ZD[1]
                    TB = A[kk,ic]*WB[ic,1]
                    TD = C[kk,ic]*WD[ic,1]
                    y -= (TB[1]+TD[1])
                 end
                 temp = B[l,l]'*A[k,k]+D[l,l]'*C[k,k]
                 iszero(temp) && throw("ME:SingularException: A-λC and D'+λB' have common or close eigenvalues")
                 E[k,l] = y/temp
               #   Z = y/(B[l,l]'*A[k,k]+D[l,l]'*C[k,k])
               #   isfinite(Z) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
             end
         end
   elseif adjAC && !adjBD
      """
      The (K,L)th element of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
                            J=1             I=1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
                            J=1
                             K              L-1
                            SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
                            J=1             I=1
      """
      for l = 1:n
          ll = l:l
          il1 = 1:l-1
          for k = 1:m
              kk = k:k
              y = E[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*E[ir,ll]
                 W2 = C[ir,kk]'*E[ir,ll]
                 y -= (W1[1]*B[l,l] + W2[1]*D[l,l])
              end
              if l > 1
                 ic = 1:m
                 ZB = E[kk,il1]*B[il1,ll]
                 ZD = E[kk,il1]*D[il1,ll]
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[ic,kk]'*WB[ic,1]
                 TD = C[ic,kk]'*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              temp = B[l,l]*A[k,k]'+D[l,l]*C[k,k]'
              iszero(temp) && throw("ME:SingularException: A'-λC' and D+λB have common or close eigenvalues")
              E[k,l] = y/temp
            #   Z = y/(B[l,l]*A[k,k]'+D[l,l]*C[k,k]')
            #   isfinite(Z) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
          end
      end
   elseif adjAC && adjBD
      """
      The (K,L)th element of X is determined starting from
      upper-rght corner column by column by

            A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
                            J=1            I=L+1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
                            J=1
                             K               N
                            SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
                            J=1            I=L+1
      """
      for l = n:-1:1
          ll = l:l
          il1 = l+1:n
          for k = 1:m
              kk = k:k
              y = E[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*E[ir,ll]
                 W2 = C[ir,kk]'*E[ir,ll]
                 y -= (W1[1]*B[l,l]'+W2[1]*D[l,l]')
              end
              if l < n
                 ic = 1:m
                 ZB = E[kk,il1]*B[ll,il1]'
                 ZD = E[kk,il1]*D[ll,il1]'
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[ic,kk]'*WB[ic,1]
                 TD = C[ic,kk]'*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              temp = B[l,l]'*A[k,k]'+D[l,l]'*C[k,k]'
              iszero(temp) && throw("ME:SingularException: A'-λC' and D'+λB' have common or close eigenvalues")
              E[k,l] = y/temp
            #   Z = y/(B[l,l]'*A[k,k]'+D[l,l]'*C[k,k]')
            #   isfinite(Z) ? E[k,l] = Z : throw("ME:SingularException: A-λC and D+λB have common or close eigenvalues")
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
   #function sylvsyss!(A::T, B::T, C::T, D::T, E::T, F::T) where {T<:Union{Array{Complex{Float64},2},Array{Complex{Float32},2}}}
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
   #function dsylvsyss!(A::T, B::T, C::T, D::T, E::T, F::T) where {T<:Union{Array{Complex{Float64},2},Array{Complex{Float32},2}}}
   """
   This is an interface to the LAPACK.tgsyl! function with `trans = 'T' or `trans = 'C'`. 
   """
   MF = -F
   E, F, scale =  tgsyl!(T <: Complex ? 'C' : 'T', A, B, C, D, E, MF)
   F = MF
   return rmul!(C[:,:],inv(scale)), rmul!(F[:,:],inv(scale))
end
