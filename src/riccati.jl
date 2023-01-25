"""
    arec(A, G, Q = 0; as = false, rtol::Real = nϵ) -> (X, EVALS, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-GX`.
`Z = [ U; V ]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = V/U`.

`Reference:`
Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> G = [1. 0. 0.; 0. 5. 0.; 0. 0. 10.]
3×3 Array{Float64,2}:
 1.0  0.0   0.0
 0.0  5.0   0.0
 0.0  0.0  10.0

julia> X, CLSEIG = arec(A,G,2I);

julia> X
3×3 Array{Float64,2}:
  0.459589   0.333603   -0.144406
  0.333603   0.65916    -0.0999216
 -0.144406  -0.0999216   0.340483

julia> A'*X+X*A-X*G*X+2I
3×3 Array{Float64,2}:
  2.22045e-16  4.44089e-16  -1.77636e-15
  4.44089e-16  6.66134e-16   1.11022e-16
 -1.77636e-15  1.11022e-16  -1.33227e-15

julia> CLSEIG
3-element Array{Complex{Float64},1}:
 -4.411547592296008 + 2.4222082620381102im
 -4.411547592296008 - 2.4222082620381102im
 -4.337128244724371 + 0.0im

julia> eigvals(A-G*X)
3-element Array{Complex{Float64},1}:
 -4.4115475922960075 - 2.4222082620381076im
 -4.4115475922960075 + 2.4222082620381076im
  -4.337128244724374 + 0.0im
```
"""
function arec(A::AbstractMatrix, G::Union{AbstractMatrix,UniformScaling,Real,Complex}, Q::Union{AbstractMatrix,UniformScaling,Real,Complex} = zero(eltype(A));
              as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
    n = LinearAlgebra.checksquare(A)
    T = promote_type( eltype(A), eltype(G), eltype(Q) )
    if typeof(G) <: AbstractArray
       (LinearAlgebra.checksquare(G) == n && ishermitian(G)) ||
          throw(DimensionMismatch("G must be a symmetric/hermitian matrix of dimension $n"))
    else
       G = G*I
       iszero(imag(G.λ)) || throw("G must be a symmetric/hermitian matrix")
    end
    if typeof(Q) <: AbstractArray
       (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    else
       Q = Q*I
       iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
    end
    T <: BlasFloat || (T = promote_type(Float64,T))
    TR = real(T)
    epsm = eps(TR)
    eltype(A) == T || (A = convert(Matrix{T},A))
    eltype(G) == T || (typeof(G) <: AbstractMatrix ? G = convert(Matrix{T},G) : G = convert(T,G.λ)*I)
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0) )

    S = schur([A  -G; -Q  -copy(A')])

    as ? select = real(S.values) .> 0 : select = real(S.values) .< 0
    n == length(filter(y-> y == true,select)) || error("The Hamiltonian matrix is not dichotomic")
    ordschur!(S, select)

    n2 = n+n
    ix = 1:n
    F = _LUwithRicTest(S.Z[ix, ix],rtol)
    x = S.Z[n+1:n2, ix]/F
    return  (x+x')/2, S.values[ix], S.Z[:,ix]
end
function _LUwithRicTest(Z11::AbstractArray,rtol::Real)
   try
      F = LinearAlgebra.lu(Z11)
      Z11norm = opnorm(Z11,1)
      Z11norm > 2*rtol ? (rcond = LAPACK.gecon!('1',F.factors,Z11norm)) : (rcond = zero(eltype(Z11)))
      rcond <= rtol ? error("no finite solution exists for the Riccati equation") : (return  F)
    catch
      error("no finite solution exists for the Riccati equation")
   end
end
"""
    arec(A, B, R, Q, S; as = false, rtol::Real = nϵ, orth = false) -> (X, EVALS, F, Z)

Computes `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-BF`.
`F` is the stabilizing or anti-stabilizing gain matrix `F = R^(-1)(B'X+S')`.
`Z = [ U; V; W ]` is a basis for the relevant stable/anti-stable deflating subspace such that `X = V/U` and `F = -W/U`.
An orthogonal basis Z can be determined, with an increased computational cost, by setting `orth = true`.

`Reference:`
Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> B = [1. 2.; 2. 0.; 0. 1.]
3×2 Array{Float64,2}:
 1.0  2.0
 2.0  0.0
 0.0  1.0

julia> R = [1. 0.; 0. 5.]
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  5.0

julia> X, CLSEIG, F = arec(A,B,R,2I);

julia> X
3×3 Array{Float64,2}:
  0.522588   0.303007  -0.327227
  0.303007   0.650895  -0.132608
 -0.327227  -0.132608   0.629825

julia> A'*X+X*A-X*B*inv(R)*B'*X+2I
3×3 Array{Float64,2}:
 -2.66454e-15  -1.55431e-15   8.88178e-16
 -1.55431e-15   2.22045e-15  -2.9976e-15
  9.99201e-16  -2.9976e-15    4.44089e-16

julia> CLSEIG
3-element Array{Complex{Float64},1}:
   -4.37703628399912 + 2.8107164873731247im
   -4.37703628399912 - 2.8107164873731247im
 -1.8663764577096091 + 0.0im

julia> eigvals(A-B*F)
3-element Array{Complex{Float64},1}:
  -4.377036283999118 - 2.8107164873731234im
  -4.377036283999118 + 2.8107164873731234im
 -1.8663764577096063 + 0.0im
```
"""
function arec(A::AbstractMatrix, B::AbstractVecOrMat, R::Union{AbstractMatrix,UniformScaling,Real,Complex},
   Q::Union{AbstractMatrix,UniformScaling,Real,Complex}, S::AbstractVecOrMat = zeros(eltype(B),size(B));
   as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false)
   if orth
      return garec(A, I, B, 0, R, Q, S; as = as, rtol = rtol)
   else
      return arec(A, B, 0, R, Q, S; as = as, rtol = rtol)
   end
end
"""
    arec(A, B, G, R, Q, S; as = false, rtol::Real = nϵ, orth = false) -> (X, EVALS, F, Z)

Computes `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `G`, `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `G`, `R` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I`, `R*I` and `Q*I`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-BF-GX`.
`F` is the stabilizing or anti-stabilizing gain matrix `F = R^(-1)(B'X+S')`.
`Z = [ U; V; W ]` is a basis for the relevant stable/anti-stable deflating subspace such that `X = V/U` and `F = -W/U`.
An orthogonal basis Z can be determined, with an increased computational cost, by setting `orth = true`.

`Reference:`
Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.
"""
function arec(A::AbstractMatrix, B::AbstractVecOrMat, G::Union{AbstractMatrix,UniformScaling,Real,Complex},
              R::Union{AbstractMatrix,UniformScaling,Real,Complex}, Q::Union{AbstractMatrix,UniformScaling,Real,Complex},
              S::AbstractVecOrMat; as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false)
    orth && (return garec(A, I, B, G, R, Q, S; as = as, rtol = rtol))

    n = LinearAlgebra.checksquare(A)
    T = promote_type( eltype(A), eltype(B), eltype(G), eltype(Q), eltype(R), eltype(S) )
    typeof(B) <: AbstractVector ? (nb, m) = (length(B), 1) : (nb, m) = size(B)
    n == nb || throw(DimensionMismatch("B must be a matrix with row dimension $n or a vector of length $n"))
    if typeof(G) <: AbstractArray
      (LinearAlgebra.checksquare(G) == n && ishermitian(G)) ||
          throw(DimensionMismatch("G must be a symmetric/hermitian matrix of dimension $n"))
   else
     G = G*I
     iszero(imag(G.λ)) || throw("G must be a symmetric/hermitian matrix")
   end
   if typeof(R) <: AbstractArray
      (LinearAlgebra.checksquare(R) == m && ishermitian(R)) ||
         throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
   else
      R = R*I
      iszero(imag(R.λ)) || throw("R must be a symmetric/hermitian matrix")
   end
   if typeof(Q) <: AbstractArray
     (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
        throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
   else
     Q = Q*I
     iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
   end
   typeof(S) <: AbstractVector ? (ns, ms) = (length(S), 1) : (ns, ms) = size(S)
   (n == ns && m == ms) ||
      throw(DimensionMismatch("S must be a $n x $m matrix or a vector of length $n"))
   T <: BlasFloat || (T = promote_type(Float64,T))
   TR = real(T)
   epsm = eps(TR)
   eltype(A) == T || (A = convert(Matrix{T},A))
   eltype(B) == T || (typeof(B) <: AbstractVector ? B = convert(Vector{T},B) : B = convert(Matrix{T},B))
   if typeof(G) <: AbstractArray
       (LinearAlgebra.checksquare(G) == n && ishermitian(G)) ||
          throw(DimensionMismatch("G must be a symmetric/hermitian matrix of dimension $n"))
   else
      G = G*I
      iszero(imag(G.λ)) || throw("G must be a symmetric/hermitian matrix")
   end
   if typeof(R) <: AbstractArray
      (LinearAlgebra.checksquare(R) == m && ishermitian(R)) ||
         throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
   else
      R = R*I
      iszero(imag(R.λ)) || throw("R must be a symmetric/hermitian matrix")
   end
   if typeof(Q) <: AbstractArray
     (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
        throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
   else
      Q = Q*I
      iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
   end
   if eltype(S) != T
      if typeof(S) <: AbstractVector
         S = convert(Vector{T},S)
      else
         S = convert(Matrix{T},S)
      end
   end

   n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0) )

   S0flag = iszero(S)
   typeof(R) <: UniformScaling && (R = Matrix{T}(R,m,m))
   SR = schur(R)
   D = real(diag(SR.T))
   Da = abs.(D)
   minDa, = findmin(Da)
   maxDa, = findmax(Da)
   minDa <= epsm*maxDa && error("R must be non-singular")
   if minDa > sqrt(epsm)*maxDa && maxDa > 100*eps(max(opnorm(A,1),opnorm(G,1),opnorm(Q,1)))
      #Dinv = diagm(0 => 1 ./ D)
      Dinv = Diagonal(1 ./ D)
      Bu = B*SR.Z
      #G = G + Bu*Dinv*Bu'
      #G = utqu(Dinv,Bu')
      G += utqu(Dinv,Bu')
      if S0flag
         sol = arec(A,G,Q; as = as, rtol = rtol)
         w2 = SR.Z*Dinv*Bu'
         f = w2*sol[1]
         z = [sol[3]; w2*(sol[3])[n+1:end,:]]
      else
         Su = S*SR.Z
         #Q -= Su*Dinv*Su'
         Q -= utqu(Dinv,Su')
         sol = arec(A-Bu*Dinv*Su',G,Q; as = as, rtol = rtol)
         w1 = SR.Z*Dinv*Su'
         w2 = SR.Z*Dinv*Bu'
         f = w1+w2*sol[1]
         #f = SR.Z*Dinv*(Bu'*sol[1]+Su')
         z = [sol[3]; [w1 w2]*sol[3] ]
      end
      return sol[1], sol[2], f, z
   else
      # use implicit form
      @warn "R nearly singular: using the orthogonal reduction method"
      return garec(A, I, B, G, R, Q, S; as = as, rtol = rtol)
   end
end
"""
    garec(A, E, G, Q = 0; as = false, rtol::Real = nϵ) -> (X, EVALS, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - E'XGXE + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators and `E` is a nonsingular matrix.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-GXE,E)`.
`Z = [ U; V ]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = V/(EU)`.

`Reference:`
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, G::Union{AbstractMatrix,UniformScaling,Real,Complex},
               Q::Union{AbstractMatrix,UniformScaling,Real,Complex} = zero(eltype(A));
               as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
    n = LinearAlgebra.checksquare(A)
    T = promote_type( eltype(A), eltype(G), eltype(Q) )
    eident = (E == I)
    if !eident
       LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       eident = isequal(E,I)
       eident ? (E = I) : (T = promote_type(T,eltype(E)))
    end
    if typeof(G) <: AbstractArray
       (LinearAlgebra.checksquare(G) == n && ishermitian(G)) ||
          throw(DimensionMismatch("G must be a symmetric/hermitian matrix of dimension $n"))
    else
      G = G*I
      iszero(imag(G.λ)) || throw("G must be a symmetric/hermitian matrix")
    end
    if typeof(Q) <: AbstractArray
       (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    else
       Q = Q*I
       iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
    end
    T <: BlasFloat || (T = promote_type(Float64,T))
    TR = real(T)
    epsm = eps(TR)
    eltype(A) == T || (A = convert(Matrix{T},A))
    eident || eltype(E) == T || (E = convert(Matrix{T},E))
    eltype(G) == T || (typeof(G) <: AbstractMatrix ? G = convert(Matrix{T},G) : G = convert(T,G.λ)*I)
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0) )

    if !eident
       Et = LinearAlgebra.LAPACK.getrf!(copy(E))
       LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
    end
    #  Method:  A stable/anti-stable deflating subspace Z1 = [Z11; Z21] of the pencil
    #       L -s P := [  A  -G ]  - s [ E  0  ]
    #                 [ -Q  -A']      [ 0  E' ]
    #  is determined and the solution X is computed as X = Z21*inv(E*Z11).
    L = [ A -G; -Q -A']
    P = [ E zeros(T,n,n); zeros(T,n,n) E']
    LPS = schur(L,P)
    as ? select = real.(LPS.α ./ LPS.β) .> 0 : select = real.(LPS.α ./ LPS.β) .< 0
    n == length(filter(y-> y == true,select)) ||
       error("The Hamiltonian/skew-Hamiltonian pencil is not dichotomic")
    ordschur!(LPS, select)
    i1 = 1:n
    i2 = n+1:2n
    F = _LUwithRicTest(LPS.Z[i1, i1],rtol)
    if eident
       x = LPS.Z[i2,i1]/F
    else
       x = LPS.Z[i2,i1]/(E*LPS.Z[i1,i1])
    end

    return  (x+x')/2, LPS.values[i1], LPS.Z[:,i1]
end
"""
    garec(A, E, B, R, Q, S; as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - (E'XB+S)R^(-1)(B'XE+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing/anti-stabilizing gain matrix `F = R^(-1)(B'XE+S')`.
`Z = [ U; V; W ]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace such that `X = V/(EU)` and `F = -W/U`.

`Reference:`
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> E = [10. 3. 0.; 0. 5. -1.; 0. 0. 10.]
3×3 Array{Float64,2}:
 10.0  3.0   0.0
  0.0  5.0  -1.0
  0.0  0.0  10.0

julia> B = [1. 2.; 2. 0.; 0. 1.]
3×2 Array{Float64,2}:
 1.0  2.0
 2.0  0.0
 0.0  1.0

julia> R = [1. 0.; 0. 5.]
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  5.0

julia> X, CLSEIG, F = garec(A,E,B,R,2I);

julia> X
3×3 Array{Float64,2}:
  0.0502214   0.0284089   -0.0303703
  0.0284089   0.111219    -0.00259162
 -0.0303703  -0.00259162   0.0618395

julia> A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+2I
3×3 Array{Float64,2}:
  1.55431e-15  -1.9984e-15   -3.33067e-15
 -1.77636e-15   1.33227e-15  -3.33067e-15
 -2.88658e-15  -3.21965e-15   1.11022e-15

julia> CLSEIG
3-element Array{Complex{Float64},1}:
  -0.6184265391601464 + 0.2913286844595737im
  -0.6184265391601464 - 0.2913286844595737im
 -0.21613059964451786 + 0.0im

julia> eigvals(A-B*F,E)
3-element Array{Complex{Float64},1}:
 -0.6184265391601462 - 0.29132868445957383im
 -0.6184265391601462 + 0.2913286844595739im
  -0.216130599644518 + 0.0im
```
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat, R::Union{AbstractMatrix,UniformScaling,Real,Complex},
   Q::Union{AbstractMatrix,UniformScaling,Real,Complex}, S::AbstractVecOrMat = zeros(eltype(B),size(B));
   as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
   garec(A, E, B, 0, R, Q, S; as = as, rtol = rtol)
end
"""
    garec(A, E, B, G, R, Q, S; as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - E'XGXE - (E'XB+S)R^(-1)(B'XE+S') + Q = 0,

where `G`, `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
Scalar-valued `G`, `R` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I`, `R*I` and `Q*I`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF-GXE,E)`.
`F` is the stabilizing/anti-stabilizing gain matrix `F = R^(-1)(B'XE+S')`.
`Z = [ U; V; W ]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace such that `X = V/(EU)` and `F = -W/U`.

`Reference:`
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat,
               G::Union{AbstractMatrix,UniformScaling,Real,Complex}, R::Union{AbstractMatrix,UniformScaling,Real,Complex},
               Q::Union{AbstractMatrix,UniformScaling,Real,Complex}, S::AbstractVecOrMat;
               as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
    n = LinearAlgebra.checksquare(A)
    T = promote_type( eltype(A), eltype(B), eltype(G), eltype(Q), eltype(R), eltype(S) )
    typeof(B) <: AbstractVector ? (nb, m) = (length(B), 1) : (nb, m) = size(B)
    n == nb || throw(DimensionMismatch("B must be a matrix with row dimension $n or a vector of length $n"))
    eident = (E == I)
    if !eident
       LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       eident = isequal(E,I)
       eident ? (E = I) : (T = promote_type(T,eltype(E)))
    end
    if typeof(G) <: AbstractArray
       (LinearAlgebra.checksquare(G) == n && ishermitian(G)) ||
           throw(DimensionMismatch("G must be a symmetric/hermitian matrix of dimension $n"))
    else
      G = G*I
      iszero(imag(G.λ)) || throw("G must be a symmetric/hermitian matrix")
    end
    if typeof(R) <: AbstractArray
       (LinearAlgebra.checksquare(R) == m && ishermitian(R)) ||
          throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    else
       R = R*I
       iszero(imag(R.λ)) || throw("R must be a symmetric/hermitian matrix")
    end
    if typeof(Q) <: AbstractArray
      (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
         throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    else
      Q = Q*I
      iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
    end
    typeof(S) <: AbstractVector ? (ns, ms) = (length(S), 1) : (ns, ms) = size(S)
    (n == ns && m == ms) || throw(DimensionMismatch("S must be a $n x $m matrix or a vector of length $n"))
    T <: BlasFloat || (T = promote_type(Float64,T))
    TR = real(T)
    epsm = eps(TR)
    eltype(A) == T || (A = convert(Matrix{T},A))
    eident || eltype(E) == T || (E = convert(Matrix{T},E))
    if !eident
       Et = LinearAlgebra.LAPACK.getrf!(copy(E))
       LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
    end
    eltype(B) == T || (typeof(B) <: AbstractVector ? B = convert(Vector{T},B) : B = convert(Matrix{T},B))
    eltype(G) == T || (typeof(G) <: AbstractMatrix ? G = convert(Matrix{T},G) : G = convert(T,G.λ)*I)
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)
    eltype(R) == T || (typeof(R) <: AbstractMatrix ? R = convert(Matrix{T},R) : R = convert(T,R.λ)*I)
    eltype(S) == T || (typeof(S) <: AbstractVector ? S = convert(Vector{T},S) : S = convert(Matrix{T},S))

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0) )

    cond(R)*epsm < 1 || error("R must be non-singular")

    #  Method:  A stable/ant-stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil
    #               [  A  -G    B ]      [ E  0  0 ]
    #      L -s P = [ -Q  -A'  -S ]  - s [ 0  E' 0 ]
    #               [  S'  B'   R ]      [ 0  0  0 ]
    # is determined and the solution X and feedback F are computed as
    #          X = Z21*inv(E*Z11),   F = -Z31*inv(Z11).

    #deflate m simple infinite eigenvalues
    n2 = n+n;
    W = qr(Matrix([S; B; R]));
    cond(W.R) * epsm  < 1 || error("The extended Hamiltonian/skew-Hamiltonian pencil is not regular")

    #z = W.Q[:,m+1:m+n2]
    z = W.Q*[fill(false,m,n2); I ]

    iric = 1:n2
    i1 = 1:n
    i2 = n+1:n2
    i3 = n2+1:n2+m
    L11 = [ A -G B; -Q -A' -S]*z
    P11 = [ E*z[i1,:]; E'*z[i2,:] ]
    LPS = schur(L11,P11)
    as ? select = real.(LPS.α ./ LPS.β) .> 0 : select = real.(LPS.α ./ LPS.β) .< 0
    n == length(filter(y-> y == true,select)) ||
         error("The extended Hamiltonian/skew-Hamiltonian pencil is not dichotomic")
    ordschur!(LPS, select)

    z[:,i1] = z[:,iric]*LPS.Z[:,i1];

    F = _LUwithRicTest(z[i1,i1],rtol)
    if eident
       x = z[n+1:end,i1]/F
       f = -x[n+1:end,:]
       x = x[i1,:]
    else
       f = -z[i3,i1]/F
       x = z[i2,i1]/(E*z[i1,i1])
    end

    return  (x+x')/2, LPS.values[i1] , f, z[:,i1]
end


"""
    ared(A, B, R, Q, S; as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the discrete-time algebraic Riccati equation

    A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable) eigenvalues of `A-BF`.
`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.
`Z = [ U; V; W ]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace such that `X = V/(EU)` and `F = -W/U`.

`Reference:`
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [ 0. 1.; 0. 0. ]
2×2 Array{Float64,2}:
 0.0  1.0
 0.0  0.0

julia> B = [ 0.; sqrt(2.) ]
2-element Array{Float64,1}:
 0.0
 1.4142135623730951

julia> R = 1.
1.0

julia> Q = [ 1. -1.; -1. 1. ]
2×2 Array{Float64,2}:
  1.0  -1.0
 -1.0   1.0

julia> X, CLSEIG, F = ared(A,B,R,Q);

julia> X
2×2 Array{Float64,2}:
  1.0  -1.0
 -1.0   1.5

julia> A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q
2×2 Array{Float64,2}:
  0.0          -3.33067e-16
 -3.33067e-16   8.88178e-16

julia> CLSEIG
2-element Array{Complex{Float64},1}:
 0.4999999999999998 - 0.0im
               -0.0 - 0.0im

julia> eigvals(A-B*F)
2-element Array{Float64,1}:
 -2.7755575615628914e-16
  0.5
```
"""
function ared(A::AbstractMatrix, B::AbstractVecOrMat, R::Union{AbstractMatrix,UniformScaling,Real,Complex},
              Q::Union{AbstractMatrix,UniformScaling,Real,Complex}, S::AbstractVecOrMat = zeros(eltype(B),size(B));
              as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
    gared(A, I, B, R, Q, S; as = as, rtol = rtol)
end
"""
    gared(A, E, B, R, Q, S; as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized discrete-time
algebraic Riccati equation

    A'XA - E'XE - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices, and `E` ist non-singular.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing/anti-stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.
`Z = [ U; V; W ]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace such that `X = V/(EU)` and `F = -W/U`.

`Reference:`
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.

# Example
```jldoctest
julia> using LinearAlgebra

julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> E = [10. 3. 0.; 0. 5. -1.; 0. 0. 10.]
3×3 Array{Float64,2}:
 10.0  3.0   0.0
  0.0  5.0  -1.0
  0.0  0.0  10.0

julia> B = [1. 2.; 2. 0.; 0. 1.]
3×2 Array{Float64,2}:
 1.0  2.0
 2.0  0.0
 0.0  1.0

julia> R = [1. 0.; 0. 5.]
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  5.0

julia> X, CLSEIG, F = gared(A,E,B,R,2I);

julia> X
3×3 Array{Float64,2}:
  0.065865   -0.0147205  -0.0100407
 -0.0147205   0.0885939   0.0101422
 -0.0100407   0.0101422   0.0234425

julia> A'*X*A-E'*X*E-A'*X*B*inv(R+B'*X*B)*B'*X*A+2I
3×3 Array{Float64,2}:
 -1.33227e-15  -2.48412e-15   1.38778e-16
 -2.498e-15    -4.44089e-16  -6.50521e-16
  1.80411e-16  -5.91541e-16  -1.33227e-15

julia> CLSEIG
3-element Array{Complex{Float64},1}:
  -0.084235615751339 - 0.0im
  -0.190533552034239 - 0.0im
 -0.5238922629921539 - 0.0im

julia> eigvals(A-B*F,E)
3-element Array{Float64,1}:
 -0.5238922629921539
 -0.19053355203423886
 -0.08423561575133902
```
"""
function gared(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat,
               R::Union{AbstractMatrix,UniformScaling,Real,Complex}, Q::Union{AbstractMatrix,UniformScaling,Real,Complex},
               S::AbstractVecOrMat = zeros(eltype(B),size(B)); as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
    n = LinearAlgebra.checksquare(A)
    T = promote_type( eltype(A), eltype(B), eltype(R), eltype(Q), eltype(S) )
    typeof(B) <: AbstractVector ? (nb, m) = (length(B), 1) : (nb, m) = size(B)
    n == nb || throw(DimensionMismatch("B must be a matrix with row dimension $n or a vector of length $n"))
    if typeof(E) <: UniformScaling{Bool}
       eident = true
       E = I
    else
       LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       eident = isequal(E,I)
       eident ? E = I : T = promote_type(T,eltype(E))
     end
    if typeof(R) <: AbstractArray
       (LinearAlgebra.checksquare(R) == m && ishermitian(R)) ||
         throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    else
      R = R*I
      iszero(imag(R.λ)) || throw("R must be a symmetric/hermitian matrix")
    end
    if typeof(Q) <: AbstractArray
       (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    else
       Q = Q*I
       iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
    end
    typeof(S) <: AbstractVector ? (ns, ms) = (length(S), 1) : (ns, ms) = size(S)
    (n == ns && m == ms) || throw(DimensionMismatch("S must be a $n x $m matrix or a vector of length $n"))
    T <: BlasFloat || (T = promote_type(Float64,T))
    TR = real(T)
    epsm = eps(TR)
    eltype(A) == T || (A = convert(Matrix{T},A))
    eident || eltype(E) == T || (E = convert(Matrix{T},E))
    if !eident
       Et = LinearAlgebra.LAPACK.getrf!(copy(E))
       LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
    end
    eltype(B) == T || (typeof(B) <: AbstractVector ? B = convert(Vector{T},B) : B = convert(Matrix{T},B))
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)
    eltype(R) == T || (typeof(R) <: AbstractMatrix ? R = convert(Matrix{T},R) : R = convert(T,R.λ)*I)
    eltype(S) == T || (typeof(S) <: AbstractVector ? S = convert(Vector{T},S) : S = convert(Matrix{T},S))

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0) )

    if !eident
      Et = LinearAlgebra.LAPACK.getrf!(copy(E))
      LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
    end
    #  Method:  A stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil
    #                   [  A   0    B ]      [ E  0  0 ]
    #          L -z P = [ -Q   E'  -S ]  - z [ 0  A' 0 ]
    #                   [ S'   0    R ]      [ 0 -B' 0 ]
    #  is computed and the solution X and feedback F are computed as
    #          X = Z21*inv(E*Z11),   F = Z31*inv(Z11).
    n2 = n+n;
    F = qr([A'; -B'])
    L2 = F.Q'*[-Q  E' -S; copy(S') zeros(T,m,n) R]
    P2 = [zeros(T,n,n) F.R zeros(T,n,m)]

    G = qr(L2[n+1:n+m,:]')
    cond(G.R) * epsm  < 1 || error("The extended symplectic pencil is not regular")
    z = (G.Q*I)[:,[m+1:m+n2; 1:m]]

    i1 = 1:n
    i2 = n+1:n2
    i3 = n2+1:n2+m
    iric = 1:n2

    L1 = [ A zeros(T,n,n) B; L2[i1,:]]*z
    P1 = [ E zeros(T,n,n+m); P2]*z

    as ? PLS = schur(L1[iric,iric],P1[iric,iric]) : PLS = schur(P1[iric,iric],L1[iric,iric])
    select = abs.(PLS.α) .> abs.(PLS.β)

    n == length(filter(y-> y == true,select)) || error("The extended symplectic pencil is not dichotomic")

    ordschur!(PLS, select)
    z[:,i1]= z[:,iric]*PLS.Z[:,i1]

    F = _LUwithRicTest(z[i1,i1],rtol)
    if eident
       x = z[n+1:end,i1]/F
       f = -x[n+1:end,:]
       x = x[i1,:]
    else
       f = -z[i3,i1]/F
       x = z[i2,i1]/(E*z[i1,i1])
    end

    as ? iev = i2 : iev = i1
    clseig = PLS.β[iev] ./ PLS.α[iev]
    if as && T <: Complex
      clseig =  conj(clseig)
    end
    return  (x+x')/2, clseig, f, z[:,i1]
end
