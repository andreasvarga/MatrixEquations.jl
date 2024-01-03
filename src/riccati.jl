"""
    arec(A, G, Q = 0; scaling = 'B', as = false, rtol::Real = nϵ) -> (X, EVALS, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.
The Schur method of [1] is used. 

To enhance the accuracy of computations, a block scaling of matrices `G` and `Q` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(G)`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. Scaling can be disabled with the choice `scaling = 'N'.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-GX`.
`Z = [ U; V ]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = V/U`.

`Reference:`

[1] Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
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
              scaling = 'B', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
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
    
    # use block scaling if appropriate
    H, Sx, Sxi = balham(A, G, Q; scaling)
    S = schur!(H)

    as ? select = real(S.values) .> 0 : select = real(S.values) .< 0
    n == count(select) || error("The Hamiltonian matrix is not dichotomic")
    ordschur!(S, select)

    n2 = n+n
    ix = 1:n
    F = _LUwithRicTest(S.Z[ix, ix],rtol)
    x = S.Z[n+1:n2, ix]/F
    lmul!(Sx,x); rmul!(x,Sxi)
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
    arec(A, B, R, Q, S; scaling = 'B', as = false, rtol::Real = nϵ, orth = false) -> (X, EVALS, F, Z)

Computes `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.
The Schur method of [1] is used. 

To enhance the accuracy of computations, a block scaling of matrices `R`, `Q` and `S` 
is performed using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(B)^2/norm(R)`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. Experimentally, if `orth = true`, two scaling procedures 
can be activated using the options `scaling = 'D' and `scaling = 'T'. Scaling can be disabled with the choice `scaling = 'N'.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-BF`.
`F` is the stabilizing or anti-stabilizing gain matrix `F = R^(-1)(B'X+S')`.
`Z = [ U; V; W ]` is a basis for the relevant stable/anti-stable deflating subspace such that `X = V/U` and `F = -W/U`.
An orthogonal basis Z can be determined, with an increased computational cost, by setting `orth = true`.

`Reference:`

[1] Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
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
   scaling = 'B', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false)
   if orth
      X, EVALS, F, Z = garec(A, I, B, 0, R, Q, S; scaling, as, rtol)
   else
      X, EVALS, F, Z = arec(A, B, 0, R, Q, S; scaling, as, rtol)
   end
   return X, EVALS, F, Z
end
"""
    arec(A, B, G, R, Q, S; as = false, rtol::Real = nϵ, orth = false) -> (X, EVALS, F, Z)

Computes `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `G`, `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `G`, `R` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I`, `R*I` and `Q*I`.

To enhance the accuracy of computations, a block oriented scaling of matrices `G`, `Q`, `R` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > max(norm(G), norm(B)^2/norm(R))`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. If `orth = true`, two experimental scaling procedures 
can be activated using the options `scaling = 'D' and `scaling = 'T'. Scaling can be disabled with the choice `scaling = 'N'.

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
              S::AbstractVecOrMat; scaling = 'B', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false)
   orth && (return garec(A, I, B, G, R, Q, S; scaling, as, rtol))

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
         sol = arec(A, G, Q; scaling, as, rtol)
         w2 = SR.Z*Dinv*Bu'
         f = w2*sol[1]
         z = [sol[3]; w2*(sol[3])[n+1:end,:]]
      else
         Su = S*SR.Z
         #Q -= Su*Dinv*Su'
         Q -= utqu(Dinv,Su')
         sol = arec(A-Bu*Dinv*Su', G, Q; scaling, as, rtol)
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
      return garec(A, I, B, G, R, Q, S; scaling, as, rtol)
   end
end
"""
    garec(A, E, G, Q = 0; scaling = 'B', as = false, rtol::Real = nϵ) -> (X, EVALS, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - E'XGXE + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators and `E` is a nonsingular matrix.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.

The generalized Schur method of [1] is used. 
To enhance the accuracy of computations, a block oriented scaling of matrices `G` and `Q` is performed  
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(G)`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. Scaling can be disabled with the choice `scaling = 'N'.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-GXE,E)`.
`Z = [ U; V ]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = V/(EU)`.

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
    Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
    Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, G::Union{AbstractMatrix,UniformScaling,Real,Complex},
               Q::Union{AbstractMatrix,UniformScaling,Real,Complex} = zero(eltype(A));
               scaling = 'B', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
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
    # use block scaling if appropriate
    L, P, Sx, Sxi = MatrixEquations.balham(A, E, G, Q; scaling)
    LPS = schur(L,P)
    as ? select = real.(LPS.α ./ LPS.β) .> 0 : select = real.(LPS.α ./ LPS.β) .< 0
    n == count(select) || error("The Hamiltonian/skew-Hamiltonian pencil is not dichotomic")
    ordschur!(LPS, select)
    i1 = 1:n
    i2 = n+1:2n
    F = _LUwithRicTest(LPS.Z[i1, i1],rtol)
    x = LPS.Z[i2,i1]/F
    lmul!(Sx,x); rmul!(x,Sxi)
    eident || (x = x/E)

    return  (x+x')/2, LPS.values[i1], LPS.Z[:,i1]
end
"""
    garec(A, E, B, R, Q, S; scaling = 'B', as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - (E'XB+S)R^(-1)(B'XE+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`. 

The generalized Schur method of [1] is used. 
To enhance the accuracy of computations, a block oriented scaling of matrices `R,` `Q` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(B)^2/norm(R)`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. Two experimental scaling procedures 
can be activated using the options `scaling = 'D' and `scaling = 'T'. Scaling can be disabled with the choice `scaling = 'N'.

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
   scaling = 'G', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
   garec(A, E, B, 0, R, Q, S; scaling, as, rtol)
end
"""
    garec(A, E, B, G, R, Q, S; scaling = 'B', as = false, rtol::Real = nϵ) -> (X, EVALS, F, Z)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - E'XGXE - (E'XB+S)R^(-1)(B'XE+S') + Q = 0,

where `G`, `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
Scalar-valued `G`, `R` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I`, `R*I` and `Q*I`.

The generalized Schur method of [1] is used. 
To enhance the accuracy of computations, a block oriented scaling of matrices `G,` `R,` `Q` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > max(norm(G), norm(B)^2/norm(R))`.
Alternative scaling can be performed using the options `scaling = 'S', for a special structure preserving scaling, and 
`scaling = 'G', for a general eigenvalue computation oriented scaling. Two experimental scaling procedures 
can be activated using the options `scaling = 'D' and `scaling = 'T'. Scaling can be disabled with the choice `scaling = 'N'.

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
               scaling = 'B', as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))))
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
    H, J, Sx, Sxi, Sr = balham(A, E, B, G, R, Q, S; scaling)
    #deflate m simple infinite eigenvalues
    n2 = n+n;
    iric = 1:n2
    i1 = 1:n
    i2 = n+1:n2
    i3 = n2+1:n2+m
    #W = qr(Matrix([S; B; R]));
    W = qr!(copy(H[i3,:]'))
    cond(W.R) * epsm  < 1 || error("The extended Hamiltonian/skew-Hamiltonian pencil is not regular")

    #z = W.Q[:,m+1:m+n2]
    z = W.Q*[fill(false,m,n2); I ]

    #L11 = [ A -G B; -Q -A' -S]*z
    L11 = H[iric,:]*z
    #P11 = [ E*z[i1,:]; E'*z[i2,:] ]
    P11 = J[iric,:]*z
    LPS = schur(L11,P11)
    as ? select = real.(LPS.α ./ LPS.β) .> 0 : select = real.(LPS.α ./ LPS.β) .< 0
    n == count(select) ||
         error("The extended Hamiltonian/skew-Hamiltonian pencil is not dichotomic")
    ordschur!(LPS, select)

    z[:,i1] = z[:,iric]*LPS.Z[:,i1];

    F = _LUwithRicTest(z[i1,i1],rtol)
    if eident
       x = z[n+1:end,i1]/F
       f = -x[n+1:end,:]; lmul!(Sr,f); rmul!(f,Sxi)
       x = x[i1,:]; lmul!(Sx,x); rmul!(x,Sxi)
    else
       f = -z[i3,i1]/F; lmul!(Sr,f); rmul!(f,Sxi)
       #x = z[i2,i1]/(E*z[i1,i1])
       x = z[i2,i1]/F; lmul!(Sx,x); rmul!(x,Sxi); x = x/E
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

    n == count(select) || error("The extended symplectic pencil is not dichotomic")

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
function balham(A, G, Q; scaling = 'B')
   n = size(A,1)
   H = [A -G; -Q -A']
   if scaling == 'N'
      return H, I, I
   elseif scaling == 'B'
      qs = sqrt(opnorm(Q,1))
      gs = sqrt(opnorm(G,1))
      scaling = (qs > gs) & (gs > 0)
      if scaling
         scal = qs/gs  
         scalsr = sqrt(scal)  
         return [A  -G*scal; -Q/scal  -copy(A')], scalsr*I, scalsr*I
      else    
         return H, I, I
      end
   else
      D = lsbalance!(H).diag
      #@show norm(inv(Diagonal(D))*[A -G; -Q -A']*Diagonal(D)-H)
      if scaling == 'S'
         s = log2.(D); # unconstrained balancing diag(D1,D2)
         # Impose that diagonal scaling must be of the form diag(D,1./D)  
         sx = round.(Int,(-s[1:n]+s[n+1:n+n])/2); # D = sqrt(D1/D2)
         Sx = Diagonal(2. .^(sx)) 
         At = Sx*A/Sx
         return [At  -Sx*G*Sx; -Sx\(Q/Sx)  -copy(At')], Sx, Sx
      else
         return H, Diagonal(D[n+1:2n]), inv(Diagonal(D[1:n])) 
      end
   end
end
function balham(A, E, G, Q; scaling = 'B') 
   n = size(A,1); n2 = 2n
   T = eltype(A)
   H = [A -G; -Q -A']; J = [E zeros(T,n,n); zeros(T,n,n) E']
   if scaling == 'N'
      return H, J, I, I
   elseif scaling == 'B'
      i1 = 1:n; i2 = n+1:n2
      qs = sqrt(opnorm(view(H,i2,i1),1))
      gs = sqrt(opnorm(view(H,i1,i2),1))
      scaling = (qs > gs) & (gs > 0)
      if scaling
         scal = qs/gs  
         scalsr = sqrt(scal)  
         return [A  -G*scal; -Q/scal  -copy(A')], J, scalsr*I, scalsr*I
      else    
         return H, J, I, I
      end
   elseif scaling == 'S'
      i1 = 1:n; i2 = n+1:2n
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = nj * abs.(H) + nh * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      s = lsbalance!(M).diag
      s = log2.(s); # unconstrained balancing diag(D1,D2,DR)
      # Impose the constraint that diagonal scalings must be of the form 
      # diag(D,1./D,DR). 
      sx = round.(Int,(-s[1:n]+s[n+1:n+n])/2); # D=sqrt(D1/D2)
      Sx = Diagonal(2. .^(sx)) 
      At = Sx*A/Sx; Et = Sx*E/Sx;
      return [At  -Sx*G*Sx; -Sx\(Q/Sx)  -copy(At')], [Et zeros(n,n); zeros(n,n) Et'], Sx, Sx
   else
      D1, D2 = regbalance!(H, J; tol = 0.1)
      return H, J, Diagonal(D2.diag[n+1:2n]), inv(Diagonal(D2.diag[1:n])) 
   end
end
function balham(A, E, B, G, R, Q, S; scaling = 'B') 
   n, m = size(B,1), size(B,2); n2 = 2n
   T = eltype(A)
   H = [A -G B; -Q -A' -S; S' B' R] 
   J = [E zeros(T,n,n+m); zeros(T,n,n) E' zeros(T,n,m); zeros(m,n2+m)]
   if scaling == 'N' 
      return H, J, I, I, I
   elseif scaling == 'B'
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      qs = sqrt(opnorm(view(H,i2,i1),1)) + sqrt(opnorm(view(H,i2,i3),1))
      gs = sqrt(opnorm(view(H,i1,i2),1)) + norm(B,1)/sqrt(norm(view(H,i3,i3),1))
      if (qs > gs) && (gs > 0)
         scal = qs/gs  
         scalsr = sqrt(scal)  
         ldiv!(scal,view(H,i2,i1)); lmul!(scal,view(H,i1,i2)) # Q -> Q/scal; G -> G * scal
         ldiv!(scal,view(H,i2,i3)); ldiv!(scal,view(H,i3,i1)) # S -> S/scal 
         ldiv!(scal,view(H,i3,i3))                            # R -> R/scal
         return H, J, scalsr*I, scalsr*I, (1/scalsr)*I
      else    
         return H, J, I, I, I
      end
   elseif scaling == 'D'
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      j2 = 1:n2
      Sr = I(m)
      S1,  = lsbalance!(view(H,j2,j2),view(J,j2,j2),view(H,j2,i3),view(H,i3,j2); tol = 0.001)
      s1 = S1.diag[i1]; s2 = S1.diag[i2];
      sx = round.(Int,(log2.(s1)-log2.(s2))/2); # D=sqrt(D1/D2)
      Sx = Diagonal(2. .^(sx)) 
      # perform preliminary scaling
      At = Sx*A/Sx; Et = Sx*E/Sx; Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; Qt = Sx\(Q/Sx); Gt = Sx*G*Sx; 
      # update Sr if appropriate
      rs = max(norm(Bt,1),norm(St,1))/norm(Rt,1); #rs = 2. ^(round(Int,log2(rs)));  
      Sr = Sr/rs;  
      Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; 
      H = [At -Gt Bt; -Qt  -At' -St; St' Bt' Rt] 
      #J = [Et zeros(T,n,n+m); zeros(T,n,n) Et' zeros(T,n,m); zeros(m,n2+m)]  
      copyto!(view(J,i1,i1),Et); adjoint!(view(J,i2,i2),Et)
      return H, J, Sx, Sx, Sr
   elseif scaling == 'T'
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      j2 = 1:n2
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      Sr = I(m)
      S1 = lsbalance!(view(M,j2,j2),view(M,j2,i3),view(M,i3,j2))
      s1 = S1.diag[i2]; s2 = S1.diag[i1];
      sx = round.(Int,(log2.(s1)-log2.(s2))/2); # D=sqrt(D1/D2)
      Sx = Diagonal(2. .^(sx)) 
      # perform preliminary scaling
      At = Sx*A/Sx; Et = Sx*E/Sx; Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; Qt = Sx\(Q/Sx); Gt = Sx*G*Sx; 
      # update Sr if appropriate
      rs = max(norm(Bt,1),norm(St,1))/norm(Rt,1); #rs = 2. ^(round(Int,log2(rs)));  
      Sr = Sr/rs;  
      Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; 
      H = [At -Gt Bt; -Qt  -At' -St; St' Bt' Rt] 
      #J = [Et zeros(T,n,n+m); zeros(T,n,n) Et' zeros(T,n,m); zeros(m,n2+m)]  
      copyto!(view(J,i1,i1),Et); adjoint!(view(J,i2,i2),Et)
      return H, J, Sx, Sx, Sr
   elseif scaling == 'S'
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      s = lsbalance!(M).diag
      Sr = Diagonal(s[n2+1:end])
      s = log2.(s); # unconstrained balancing diag(D1,D2,DR)
      sx = round.(Int,(-s[i1]+s[i2])/2); # D=sqrt(D1/D2)
      s = 2. .^[sx ; -sx ; -s[i3]]
      Sr = Diagonal(s[i3])

      # Impose the constraint that diagonal scalings must be of the form diag(D,1./D,DR). 
      Sx = Diagonal(2. .^(sx)) 
      At = Sx*A/Sx; Et = Sx*E/Sx; Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; Qt = Sx\(Q/Sx); Gt = Sx*G*Sx; 
      rs = max(norm(Bt,1),norm(St,1))/sqrt(norm(Rt,1)); Sr = Sr/rs;  
      Bt = Sx*B*Sr; St = Sx\(S*Sr); Rt = Sr*R*Sr; 
      H = [At -Gt Bt; -Qt  -At' -St; St' Bt' Rt] 
      #J = [Et zeros(T,n,n+m); zeros(T,n,n) Et' zeros(T,n,m); zeros(m,n2+m)]  
      copyto!(view(J,i1,i1),Et); adjoint!(view(J,i2,i2),Et)
      return H, J, Sx, Sx, Sr
   else
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      _, D2 = regbalance!(H, J; tol = 0.001, maxiter = 1000, pow2 = false)
      return H, J, Diagonal(D2.diag[i2]), inv(Diagonal(D2.diag[i1])), Diagonal(D2.diag[i3]) 
   end
end
# qS1, lsbalance!, regbalance! from MatrixPencils.jl package 
function qS1(M)    
"""
    qs = qS1(M) 

Compute the 1-norm based scaling quality `qs = qS(abs(M))` of a matrix `M`, 
where `qS(⋅)` is the scaling quality measure defined in Definition 5.5 of [1] for 
nonnegative matrices. Nonzero rows and columns in `M` are allowed.   

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
    (size(M,1) == 0 || size(M,2) == 0) && (return one(real(eltype(M))))
    temp = sum(abs,M,dims=1)
    tmax = maximum(temp)
    rmax = tmax == 0 ? one(real(eltype(M))) : tmax/minimum(temp[temp .!= 0])
    temp = sum(abs,M,dims=2)
    tmax = maximum(temp)
    cmax = tmax == 0 ? one(real(eltype(M))) : tmax/minimum(temp[temp .!= 0])
    return max(rmax,cmax)
end
qS1(M::UniformScaling) = 1
function lsbalance!(A::AbstractMatrix{T1}, B::AbstractVecOrMat{T1}, C::AbstractMatrix{T1}; 
                   withB = true, withC = true, maxred = 16) where {T1}
"""
     lsbalance!(A, B, C; withB = true, withC = true, maxred = 16) -> D

Reduce the 1-norm of a system matrix 

             S =  ( A  B )
                  ( C  0 )

corresponding to a standard system triple `(A,B,C)` by balancing. 
This involves a diagonal similarity transformation `inv(D)*A*D` 
to make the rows and columns of 

                  diag(inv(D),I) * S * diag(D,I)
     
as close in norm as possible.
     
The balancing can be optionally performed on the following 
particular system matrices:   

        S = A, if withB = false and withC = false ,
        S = ( A  B ), if withC = false,     or    
        S = ( A ), if withB = false .
            ( C )      

The keyword argument `maxred = s` specifies the maximum allowed reduction in the 1-norm of
`S` (in an iteration) if zero rows or columns are present. 
`s` must be a positive power of `2`, otherwise  `s` is rounded to the nearest power of 2.   

_Note:_ This function is a translation of the SLICOT routine `TB01ID.f`, which 
represents an extensiom of the LAPACK family `*GEBAL.f` and also includes  
the fix proposed in [1]. 

[1]  R.James, J.Langou and B.Lowery. "On matrix balancing and eigenvector computation."
     ArXiv, 2014, http://arxiv.org/abs/1401.5766
"""
    n = LinearAlgebra.checksquare(A)
    n1, m = size(B,1), size(B,2)
    n1 == n || throw(DimensionMismatch("A and B must have the same number of rows"))
    n == size(C,2) || throw(DimensionMismatch("A and C must have the same number of columns"))
    T = real(T1)
    ZERO = zero(T)
    ONE = one(T)
    SCLFAC = T(2.)
    FACTOR = T(0.95); 
    maxred > ONE || throw(ArgumentError("maxred must be greater than 1, got maxred = $maxred"))
    MAXR = T(maxred)
    MAXR = T(2.) ^round(Int,log2(MAXR))
    # Compute the 1-norm of the required part of matrix S and exit if it is zero.
    SNORM = ZERO
    for j = 1:n
        CO = sum(abs,view(A,:,j))
        withC && (CO += sum(abs,view(C,:,j)))
        SNORM = max( SNORM, CO )
    end
    if withB
       for j = 1:m 
           SNORM = max( SNORM, sum(abs,view(B,:,j)) )
       end
    end
    D = fill(ONE,n)
    SNORM == ZERO && (return D)

    SFMIN1 =  T <: BlasFloat ? safemin(T) / eps(T) : safemin(Float64) / eps(Float64)
    SFMAX1 = ONE / SFMIN1
    SFMIN2 = SFMIN1*SCLFAC
    SFMAX2 = ONE / SFMIN2

    SRED = maxred
    SRED <= ZERO && (SRED = MAXR)

    MAXNRM = max( SNORM/SRED, SFMIN1 )

    # Balance the system matrix.

    # Iterative loop for norm reduction.
    NOCONV = true
    while NOCONV
        NOCONV = false
        for i = 1:n
            Aci = view(A,:,i)
            Ari = view(A,i,:)
            Ci = view(C,:,i)
            Bi = view(B,i,:)

            CO = norm(Aci)
            RO = norm(Ari)
            CA = abs(argmax(abs,Aci))
            RA = abs(argmax(abs,Ari))

            if withC
               CO = hypot(CO, norm(Ci))
               CA = max(CA,abs(argmax(abs,Ci)))
            end
            if withB
                RO = hypot(RO, norm(Bi))
                RA = max( RA, abs(argmax(abs,Bi)))
            end
            #  Special case of zero CO and/or RO.
            CO == ZERO && RO == ZERO && continue
            if CO == ZERO 
               RO <= MAXNRM && continue
               CO = MAXNRM
            end
            if RO == ZERO
               CO <= MAXNRM && continue
               RO = MAXNRM
            end

            #  Guard against zero CO or RO due to underflow.
            G = RO / SCLFAC
            F = ONE
            S = CO + RO
            while ( CO < G && max( F, CO, CA ) < SFMAX2 && min( RO, G, RA ) > SFMIN2 ) 
                F  *= SCLFAC
                CO *= SCLFAC
                CA *= SCLFAC
                G  /= SCLFAC
                RO /= SCLFAC
                RA /= SCLFAC   
            end
            G = CO / SCLFAC

            while ( G >= RO && max( RO, RA ) < SFMAX2 && min( F, CO, G, CA ) > SFMIN2 )
                F  /= SCLFAC
                CO /= SCLFAC
                CA /= SCLFAC
                G  /= SCLFAC
                RO *= SCLFAC
                RA *= SCLFAC
            end

            # Now balance.
            CO+RO >= FACTOR*S && continue
            if F < ONE && D[i] < ONE 
               F*D[i] <= SFMIN1 && continue
            end
            if F > ONE && D[i] > ONE 
               D[i] >= SFMAX1 / F  && continue
            end
            G = ONE / F
            D[i] *= F
            NOCONV = true
       
            lmul!(G,Ari)
            rmul!(Aci,F)
            lmul!(G,Bi)
            rmul!(Ci,F)
        end
    end
    return Diagonal(D)
end   
lsbalance!(A::AbstractMatrix{T1}; maxred = 16) where {T1} = lsbalance!(A,zeros(T1,size(A,1),0),zeros(T1,0,size(A,2)); maxred, withB=false, withC=false)  
function lsbalance!(A::AbstractMatrix{T}, E::Union{AbstractMatrix{T},UniformScaling{Bool}}, B::AbstractVecOrMat{T}, C::AbstractMatrix{T}; 
                    withB = true, withC = true, maxred = 16, maxiter = 100, tol = 1, pow2 = true) where {T}
"""
     lsbalance!(A, E, B, C; withB = true, withC = true, pow2, maxiter = 100, tol = 1) -> (D1,D2)

Reduce the 1-norm of the matrix 

             S =  ( abs(A)+abs(E)  abs(B) )
                  (    abs(C)        0    )

corresponding to a descriptor system triple `(A-λE,B,C)` by row and column balancing. 
This involves diagonal similarity transformations `D1*(A-λE)*D2` applied
iteratively to `abs(A)+abs(E)` to make the rows and columns of 
                             
                  diag(D1,I)  * S * diag(D2,I)
     
as close in norm as possible.
     
The balancing can be performed optionally on the following 
particular system matrices:   

        S = abs(A)+abs(E), if withB = false and withC = false ,
        S = ( abs(A)+abs(E)  abs(B) ), if withC = false,     or    
        S = ( abs(A)+abs(E) ), if withB = false .
            (   abs(C)     )       

The keyword argument `maxiter = k` specifies the maximum number of iterations `k` 
allowed in the balancing algorithm. 

The keyword argument `tol = τ`, with `τ ≤ 1`,  specifies the tolerance used in the stopping criterion.   

If the keyword argument `pow2 = true` is specified, then the components of the resulting 
optimal `D1` and `D2` are replaced by their nearest integer powers of 2, in which case the 
scaling of matrices is done exactly in floating point arithmetic. 
If `pow2 = false`, the optimal values `D1` and `D2` are returned.

_Note:_ This function is an extension of the MATLAB function `rowcolsums.m` of [1]. 

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
   radix = real(T)(2.)
   emat = (typeof(E) <: AbstractMatrix)
   eident = !emat || isequal(E,I) 
   n = LinearAlgebra.checksquare(A)
   emat && (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
   n == size(B,1) || throw(DimensionMismatch("A and B must have compatible dimensions"))
   n == size(C,2) || throw(DimensionMismatch("A and C must have compatible dimensions"))

   n == 0 && (return Diagonal(T[]), Diagonal(T[]))
 
   if eident 
      D = lsbalance!(A, B, C; withB, withC, maxred)
      return inv(D), D
   else
      MA = abs.(A)+abs.(E)   
      MB = abs.(B)   
      MC = abs.(C)   
      r = fill(real(T)(n),n); c = copy(r)
      # Scale the matrix to have total sum(sum(M))=sum(c)=sum(r);
      sumcr = sum(c) 
      sumM = sum(MA) 
      withB && (sumM += sum(MB))
      withC && (sumM += sum(MC))
      sc = sumcr/sumM
      lmul!(1/sc,c); lmul!(1/sc,r)
      t = sqrt(sumcr/sumM); Dl = Diagonal(fill(t,n)); Dr = Diagonal(fill(t,n))
      # Scale left and right to make row and column sums equal to r and c
      conv = false
      for i = 1:maxiter
          conv = true
          cr = sum(MA,dims=1)
          withC && (cr .+= sum(MC,dims=1))
          dr = Diagonal(reshape(cr,n)./c)
          rdiv!(MA,dr); rdiv!(MC,dr) 
          er = minimum(dr.diag)/maximum(dr) 
          rdiv!(Dr,dr)
          cl = sum(MA,dims=2)
          withB && (cl .+= sum(MB,dims=2))
          dl = Diagonal(reshape(cl,n)./r)
          ldiv!(dl,MA); ldiv!(dl,MB)
          el = minimum(dl.diag)/maximum(dl) 
          rdiv!(Dl,dl)
         #  @show i, er, el
          max(1-er,1-el) < tol/2 && break
          conv = false
      end
      conv || (@warn "the iterative algorithm did not converge in $maxiter iterations")
      # Finally scale the two scalings to have equal maxima
      scaled = sqrt(maximum(Dr)/maximum(Dl))
      rmul!(Dl,scaled); rmul!(Dr,1/scaled)
      if pow2
         Dl = Diagonal(radix .^(round.(Int,log2.(Dl.diag))))
         Dr = Diagonal(radix .^(round.(Int,log2.(Dr.diag))))
      end
      lmul!(Dl,A); rmul!(A,Dr)
      lmul!(Dl,E); rmul!(E,Dr)
      lmul!(Dl,B)
      rmul!(C,Dr)
      return Dl, Dr  
   end
end
function regbalance!(A::AbstractMatrix{T}, E::AbstractMatrix{T}; maxiter = 100, tol = 1, pow2 = true) where {T}
"""
     regbalance!(A, E; maxiter = 100, tol = 1, pow2 = true) -> (Dl,Dr)

Balance the regular pair `(A,E)` by reducing the 1-norm of the matrix `M := abs(A)+abs(E)`
by row and column balancing. 
This involves diagonal similarity transformations `Dl*(A-λE)*Dr` applied
iteratively to `M` to make the rows and columns of `Dl*M*Dr` as close in norm as possible.
The [Sinkhorn–Knopp algorithm](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) is used 
to reduce `M` to a doubly stochastic matrix. 

The resulting `Dl` and `Dr` are diagonal scaling matrices.  
If the keyword argument `pow2 = true` is specified, then the components of the resulting 
optimal `Dl` and `Dr` are replaced by their nearest integer powers of 2. 
If `pow2 = false`, the optimal values `Dl` and `Dr` are returned.
The resulting `Dl*A*Dr` and `Dl*E*Dr` overwrite `A` and `E`, respectively
    
The keyword argument `tol = τ`, with `τ ≤ 1`,  specifies the tolerance used in the stopping criterion. 
The iterative process is stopped as soon as the incremental scalings are `tol`-close to the identity. 

The keyword argument `maxiter = k` specifies the maximum number of iterations `k` 
allowed in the balancing algorithm. 

_Note:_ This function is based on the MATLAB function `rowcolsums.m` of [1], modified such that
the scaling operations are directly applied to `A` and `E`.  

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
   n = LinearAlgebra.checksquare(A)
   (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))

   n <= 1 && (return Diagonal(ones(T,n)), Diagonal(ones(T,n)))
   TR = real(T)
   radix = TR(2.)
   t = TR(n)
   pow2 && (t = radix^(round(Int,log2(t)))) 
   c = fill(t,n); 
   # Scale the matrix M = abs(A)+abs(E) to have total sum(sum(M)) = sum(c)
   sumcr = sum(c) 
   sumM = sum(abs,A) + sum(abs,E)
   sc = sumcr/sumM
   pow2 && (sc = radix^(round(Int,log2(sc)))) 
   t = sqrt(sc) 
   ispow2(t) || (sc *= 2; t = sqrt(sc))
   lmul!(sc,A); lmul!(sc,E)
   Dl = Diagonal(fill(t,n)); Dr = Diagonal(fill(t,n))

   # Scale left and right to make row and column sums equal to r and c
   conv = false
   for i = 1:maxiter
       conv = true
       cr = sum(abs,A,dims=1) + sum(abs,E,dims=1) 
       dr = pow2 ? Diagonal(radix .^(round.(Int,log2.(reshape(cr,n)./c)))) : Diagonal(reshape(cr,n)./c)
       rdiv!(A,dr); rdiv!(E,dr) 
       er = minimum(dr.diag)/maximum(dr) 
       rdiv!(Dr,dr)
       cl = sum(abs,A,dims=2) + sum(abs,E,dims=2)
       dl = pow2 ? Diagonal(radix .^(round.(Int,log2.(reshape(cl,n)./c)))) : Diagonal(reshape(cl,n)./c)
       ldiv!(dl,A); ldiv!(dl,E)
       el = minimum(dl.diag)/maximum(dl) 
       rdiv!(Dl,dl)
       max(1-er,1-el) < tol/2 && break
       conv = false
   end
   conv || (@warn "the iterative algorithm did not converge in $maxiter iterations")
   # Finally scale the two scalings to have equal maxima
   scaled = sqrt(maximum(Dr)/maximum(Dl))
   pow2 && (scaled = radix^(round(Int,log2(scaled)))) 
   rmul!(Dl,scaled); rmul!(Dr,1/scaled)
   return Dl, Dr  
end



