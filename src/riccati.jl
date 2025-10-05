"""
    arec(A, G, Q = 0; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, Z, scalinfo)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.
The Schur method of [1] is used. 

To enhance the accuracy of computations, a block scaling of matrices `G` and `Q` is performed, if  
the default setting `scaling = 'B'` is used. This scaling is however performed only if `norm(Q) > norm(G)`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, experimental structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting. 
Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-GX`.

`Z = [U; V]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = Sx*(V/U)*Sxi`, 
where `Sx` and `Sxi` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx` and `scalinfo.Sxi`, respectively.

_Note:_ To solve the continuous-time algebraic Riccati equation

     A'X + XA - XBR^(-1)B'X + Q = 0,

with `R` a hermitian/symmetric matrix and `B` a compatible size matrix, `G = BR^(-1)B'` must be provided. 
This approach is not numerically suited when `R` is ill-conditioned and/or `B` has large norm.  

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
              scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
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
    
    # perform scaling if appropriate
    H, Sx, Sxi = balham(A, G, Q; scaling, pow2, nrm)
    S = schur!(H)

    as ? select = real(S.values) .> 0 : select = real(S.values) .< 0
    n == count(select) || error("The Hamiltonian matrix is not dichotomic")
    ordschur!(S, select)

    n2 = n+n
    ix = 1:n
    F = _LUwithRicTest(S.Z[ix, ix],rtol)
    x = S.Z[n+1:n2, ix]/F
    lmul!(Sx,x); rmul!(x,Sxi)
    scalinfo = (Sx = Sx, Sxi = Sxi)
    return  LinearAlgebra._hermitianpart!(x), S.values[ix], S.Z[:,ix], scalinfo
end
function _LUwithRicTest(Z11::AbstractArray{T},rtol::Real) where {T <: BlasFloat}
   try
      F = LinearAlgebra.lu(Z11)
      Z11norm = opnorm(Z11,1)
      Z11norm > 2*rtol ? (rcond = LAPACK.gecon!('1',F.factors,Z11norm)) : (rcond = zero(eltype(Z11)))
      rcond <= rtol ? error("no finite solution exists for the Riccati equation") : (return  F)
    catch
      error("no finite solution exists for the Riccati equation")
   end
end
function _LUwithRicTest(Z11::AbstractArray,rtol::Real)
   try
      F = LinearAlgebra.lu(Z11)
      # Z11norm = opnorm(Z11,1)
      # Z11norm > 2*rtol ? (rcond = LAPACK.gecon!('1',F.factors,Z11norm)) : (rcond = zero(eltype(Z11)))
      # rcond <= rtol ? error("no finite solution exists for the Riccati equation") : (return  F)
    catch
      error("no finite solution exists for the Riccati equation")
   end
end
"""
    arec(A, B, R, Q, S; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, orth = false, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.
The Schur method of [1] is used. 

To enhance the accuracy of computations, a block scaling of matrices `R`, `Q`  and `S` is performed, if  
the default setting `scaling = 'B'` is used. This scaling is however performed only if `norm(Q) > norm(B)^2/norm(R)`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Experimental structure preserving scalings can be performed using the options `scaling = 'D'` 
or `scaling = 'T'`. Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-BF`.

`F` is the stabilizing or anti-stabilizing gain matrix `F = R^(-1)(B'X+S')`.

`Z = [U; V; W]` is a basis for the relevant stable/anti-stable deflating subspace 
such that `X = Sx*(V/U)*Sxi` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.
An orthogonal basis `Z` can be determined, with an increased computational cost, by setting `orth = true`.

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
   scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false, nrm = 1)
   if orth
      return garec(A, I, B, 0, R, Q, S; scaling, pow2, as, rtol, nrm)
   else
      return arec(A, B, 0, R, Q, S; scaling, pow2, as, rtol, nrm)
   end
end
"""
    arec(A, B, G, R, Q, S; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, orth = false, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

Computes `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the continuous-time
algebraic Riccati equation

     A'X + XA - XGX - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `G`, `R` and `Q` are hermitian/symmetric matrices or uniform scaling operators such that `R` is nonsingular.
Scalar-valued `G`, `R` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I`, `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`. 
For well conditioned `R`, the Schur method of [1] is used. For ill-conditioned `R` or if `orth = true`, 
the generalized Schur method of [2] is used. 

To enhance the accuracy of computations, a block oriented scaling of matrices `G`, `R`, `Q` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > max(norm(G), norm(B)^2/norm(R))`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
If `orth = true`, two experimental scaling procedures 
can be activated using the options `scaling = 'D'` and `scaling = 'T'`. 
Scaling can be disabled with the choice `scaling = 'N'`.

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) eigenvalues of `A-BF-GX`.

`F` is the stabilizing or anti-stabilizing gain matrix `F = R^(-1)(B'X+S')`.

`Z = [U; V; W]` is a basis for the relevant stable/anti-stable deflating subspace 
such that `X = Sx*(V/U)*Sxi` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.
An orthogonal basis `Z` can be determined, with an increased computational cost, by setting `orth = true`.

`Reference:`

[1] Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
    IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.

[2] W.F. Arnold, III and A.J. Laub,
    Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
    Proc. IEEE, 72:1746-1754, 1984.
"""
function arec(A::AbstractMatrix, B::AbstractVecOrMat, G::Union{AbstractMatrix,UniformScaling,Real,Complex},
              R::Union{AbstractMatrix,UniformScaling,Real,Complex}, Q::Union{AbstractMatrix,UniformScaling,Real,Complex},
              S::AbstractVecOrMat; scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), orth = false, nrm = 1)
   orth && (return garec(A, I, B, G, R, Q, S; scaling, pow2, as, rtol, nrm))

   T = promote_type( eltype(A), eltype(B), eltype(G), eltype(Q), eltype(R), eltype(S) )

   n = LinearAlgebra.checksquare(A)
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
      eltype(G) == T || (G = convert(Matrix{T},G))
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
      R = Matrix{T}(R,m,m)
   end
   if typeof(Q) <: AbstractArray
      (LinearAlgebra.checksquare(Q) == n && ishermitian(Q)) ||
        throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
      eltype(Q) == T || (Q = convert(Matrix{T},Q))
   else
      Q = Q*I
      iszero(imag(Q.λ)) || throw("Q must be a symmetric/hermitian matrix")
      Q = Matrix{T}(Q,n,n)
   end
   if eltype(S) != T
      if typeof(S) <: AbstractVector
         S = convert(Vector{T},S)
      else
         S = convert(Matrix{T},S)
      end
   end

   n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0), (Sx = Diagonal(zeros(T,0)), Sxi = Diagonal(zeros(T,0)), Sr = Diagonal(zeros(T,m))) ) 

   S0flag = iszero(S)
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
         sol = arec(A, G, Q; scaling, nrm, pow2, as, rtol)
         w2 = SR.Z*Dinv*Bu'
         f = w2*sol[1]
         z = [sol[3]; w2*(sol[3])[n+1:end,:]]
      else
         Su = S*SR.Z
         #Q -= Su*Dinv*Su'
         Q -= utqu(Dinv,Su')
         sol = arec(A-Bu*Dinv*Su', G, Q; scaling, nrm, pow2, as, rtol)
         w1 = SR.Z*Dinv*Su'
         w2 = SR.Z*Dinv*Bu'
         f = w1+w2*sol[1]
         #f = SR.Z*Dinv*(Bu'*sol[1]+Su')
         z = [sol[3]; [w1 w2]*sol[3] ]
      end
      return sol[1], sol[2], f, z, sol[4]
   else
      # use implicit form
      @warn "R nearly singular: using the orthogonal reduction method"
      return garec(A, I, B, G, R, Q, S; scaling, nrm, pow2, as, rtol)
   end
end
tocomplex(A::LinearAlgebra.UniformScaling) = complex(A.λ)*I
tocomplex(A::AbstractArray) = complex(A)
tocomplex(A::Number) = complex(A)
"""
    garec(A, E, G, Q = 0; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, Z, scalinfo)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized continuous-time
algebraic Riccati equation

    A'XE + E'XA - E'XGXE + Q = 0,

where `G` and `Q` are hermitian/symmetric matrices or uniform scaling operators and `E` is a nonsingular matrix.
Scalar-valued `G` and `Q` are interpreted as appropriately sized uniform scaling operators `G*I` and `Q*I`.
The generalized Schur method of [1] is used. 

To enhance the accuracy of computations, a block scaling of matrices `G` and `Q` is performed, if  
the default setting `scaling = 'B'` is used. This scaling is however performed only if `norm(Q) > norm(G)`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-GXE,E)`.

`Z = [U; V]` is an orthogonal basis for the stable/anti-stable deflating subspace such that `X = (Sx*(V/U)*Sxi)/E`, 
where `Sx` and `Sxi` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx` and `scalinfo.Sxi`, respectively.

_Note:_ To solve the continuous-time algebraic Riccati equation

     A'XE + E'XA - E'XBR^(-1)B'XE + Q = 0,

with `R` a hermitian/symmetric matrix and `B` a compatible size matrix, `G = BR^(-1)B'` must be provided. 
This approach is not numerically suited when `R` is ill-conditioned and/or `B` has large norm.  

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
    Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
    Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, G::Union{AbstractMatrix,UniformScaling,Real,Complex},
               Q::Union{AbstractMatrix,UniformScaling,Real,Complex} = zero(eltype(A));
               scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
    T = promote_type( eltype(A), eltype(G), eltype(Q) )
    T <: BlasFloat  || (T = promote_type(Float64,T))

    # use complex version because the generalized Schur form decomposition available only for complex data 
    if !(T <: BlasFloat || T <: Complex) 
       sol = garec(tocomplex(A),tocomplex(E),tocomplex(G),tocomplex(Q); scaling, nrm, pow2, as, rtol)
       return real(sol[1]), sol[2], Matrix(qr([real(sol[3]) imag(sol[3])]).Q)[:,1:size(A,1)], sol[4]
    end

    n = LinearAlgebra.checksquare(A)
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

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), (Sx = Diagonal(zeros(T,0)), Sxi = Diagonal(zeros(T,0))) ) 

    if !eident
      if T <: BlasFloat
         Et = LinearAlgebra.LAPACK.getrf!(copy(E))
         LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
      else
         cond(E)*epsm > 1 && error("E must be non-singular")
      end
    end
    #  Method:  A stable/anti-stable deflating subspace Z1 = [Z11; Z21] of the pencil
    #       L -s P := [  A  -G ]  - s [ E  0  ]
    #                 [ -Q  -A']      [ 0  E' ]
    #  is determined and the solution X is computed as X = Z21*inv(E*Z11).
    # use block scaling if appropriate
    L, P, Sx, Sxi = balham(A, E, G, Q; scaling, nrm)
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

    scalinfo = (Sx = Sx, Sxi = Sxi)
    return  LinearAlgebra._hermitianpart!(x), LPS.values[i1], LPS.Z[:,i1], scalinfo
end
"""
    garec(A, E, B, R, Q, S; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

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
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Experimental structure preserving scalings can be performed using the options `scaling = 'D'` 
or `scaling = 'T'`. Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF,E)`.

`F` is the stabilizing/anti-stabilizing gain matrix `F = R^(-1)(B'XE+S')`.

`Z = [U; V; W]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace 
such that `X = (Sx*(V/U)*Sxi)/E` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
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
   scaling = 'G', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
   garec(A, E, B, 0, R, Q, S; scaling, pow2, as, rtol, nrm)
end
"""
    garec(A, E, B, G, R, Q, S; scaling = 'B', pw2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

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
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Experimental structure preserving scalings can be performed using the options `scaling = 'D'` 
or `scaling = 'T'`. Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF-GXE,E)`.

`F` is the stabilizing/anti-stabilizing gain matrix `F = R^(-1)(B'XE+S')`.

`Z = [U; V; W]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace 
such that `X = (Sx*(V/U)*Sxi)/E` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
    Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
    Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat,
               G::Union{AbstractMatrix,UniformScaling,Real,Complex}, R::Union{AbstractMatrix,UniformScaling,Real,Complex},
               Q::Union{AbstractMatrix,UniformScaling,Real,Complex}, S::AbstractVecOrMat;
               scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
    T = promote_type( eltype(A), eltype(B), eltype(G), eltype(Q), eltype(R), eltype(S) )
    T <: BlasFloat  || (T = promote_type(Float64,T))
   
    # use complex version because the generalized Schur form decomposition available only for complex data 
    if !(T <: BlasFloat || T <: Complex) 
       sol = garec(tocomplex(A),tocomplex(E),tocomplex(B),tocomplex(G),tocomplex(R),tocomplex(Q),tocomplex(S); scaling, pow2, as, rtol, nrm)
       return real(sol[1]), isreal(sol[2]) ? real(sol[2]) : sol[2], real(sol[3]), Matrix(qr([real(sol[4]) imag(sol[4])]).Q)[:,1:size(A,1)], sol[5]
    end

    n = LinearAlgebra.checksquare(A)
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
    eltype(B) == T || (typeof(B) <: AbstractVector ? B = convert(Vector{T},B) : B = convert(Matrix{T},B))
    eltype(G) == T || (typeof(G) <: AbstractMatrix ? G = convert(Matrix{T},G) : G = convert(T,G.λ)*I)
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)
    eltype(R) == T || (typeof(R) <: AbstractMatrix ? R = convert(Matrix{T},R) : R = convert(T,R.λ)*I)
    eltype(S) == T || (typeof(S) <: AbstractVector ? S = convert(Vector{T},S) : S = convert(Matrix{T},S))

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0), (Sx = Diagonal(zeros(T,0)), Sxi = Diagonal(zeros(T,0)), Sr = Diagonal(zeros(T,m))) )

    if !eident
      if T <: BlasFloat
         Et = LinearAlgebra.LAPACK.getrf!(copy(E))
         LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
      else
         cond(E)*epsm < 1 || error("E must be non-singular")
      end
    end
    cond(R)*epsm < 1 || error("R must be non-singular")

    #  Method:  A stable/ant-stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil
    #               [  A  -G    B ]      [ E  0  0 ]
    #      L -s P = [ -Q  -A'  -S ]  - s [ 0  E' 0 ]
    #               [  S'  B'   R ]      [ 0  0  0 ]
    # is determined and the solution X and feedback F are computed as
    #          X = Z21*inv(E*Z11),   F = -Z31*inv(Z11).
    H, J, Sx, Sxi, Sr = balham(A, E, B, G, R, Q, S; scaling, pow2, nrm)
    # deflate m simple infinite eigenvalues
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
    scalinfo = (Sx = Sx, Sxi = Sxi, Sr = Sr)
    return  LinearAlgebra._hermitianpart!(x), LPS.values[i1] , f, z[:,i1], scalinfo
end


"""
    ared(A, B, R, Q, S; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the discrete-time algebraic Riccati equation

    A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

To enhance the accuracy of computations, a block oriented scaling of matrices `R,` `Q` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(B)^2/norm(R)`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Experimental structure preserving scalings can be performed using the options `scaling = 'D'`, `scaling = 'R'` and `scaling = 'T'`. 
Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable) eigenvalues of `A-BF`.

`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

`Z = [U; V; W]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace 
such that `X = Sx*(V/U)*Sxi` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
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
              scaling = 'B', pow2 = false, as = false, rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
    gared(A, I, B, R, Q, S; scaling, nrm, pow2, as, rtol)
end
"""
    gared(A, E, B, R, Q, S; scaling = 'B', pow2 = false, as = false, rtol::Real = nϵ, nrm = 1) -> (X, EVALS, F, Z, scalinfo)

Compute `X`, the hermitian/symmetric stabilizing solution (if `as = false`) or
anti-stabilizing solution (if `as = true`) of the generalized discrete-time
algebraic Riccati equation

    A'XA - E'XE - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `R` and `Q` are hermitian/symmetric matrices, and `E` ist non-singular.
Scalar-valued `R` and `Q` are interpreted as appropriately sized uniform scaling operators `R*I` and `Q*I`.
`S`, if not specified, is set to `S = zeros(size(B))`.

To enhance the accuracy of computations, a block oriented scaling of matrices `R,` `Q` and `S` is performed 
using the default setting `scaling = 'B'`. This scaling is performed only if `norm(Q) > norm(B)^2/norm(R)`.
A general, eigenvalue computation oriented scaling combined with a block scaling is used if `scaling = 'G'` is selected. 
An alternative, structure preserving scaling can be performed using the option `scaling = 'S'`. 
A symmetric matrix equilibration based scaling is employed if `scaling = 'K'`, for which the underlying vector norm 
can be specified using the keyword argument `nrm = p`, where `p = 1` is the default setting.   
Experimental structure preserving scalings can be performed using the options `scaling = 'D'`, `scaling = 'R'` and `scaling = 'T'`. 
Scaling can be disabled with the choice `scaling = 'N'`.
If `pow2 = true`, the scaling elements are enforced to the nearest power of 2 (default: `pow2 = false`).

By default, the lower bound for the 1-norm reciprocal condition number `rtol` is `n*ϵ`, where `n` is the order of `A`
and `ϵ` is the _machine epsilon_ of the element type of `A`.

`EVALS` is a vector containing the (stable or anti-stable) generalized eigenvalues of the pair `(A-BF,E)`.

`F` is the stabilizing/anti-stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

`Z = [U; V; W]` is an orthogonal basis for the relevant stable/anti-stable deflating subspace 
such that `X = (Sx*(V/U)*Sxi)/E` and  `F = -Sr*(W/U)*Sxi`, 
where `Sx`, `Sxi` and `Sr` are diagonal scaling matrices contained in the named tuple `scalinfo` 
as `scalinfo.Sx`, `scalinfo.Sxi` and `scalinfo.Sr`, respectively.

`Reference:`

[1] W.F. Arnold, III and A.J. Laub,
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
               S::AbstractVecOrMat = zeros(eltype(B),size(B)); scaling = 'B', pow2 = false, as = false, 
               rtol::Real = size(A,1)*eps(real(float(one(eltype(A))))), nrm = 1)
    T = promote_type( eltype(A), eltype(B), eltype(R), eltype(Q), eltype(S) )
    T <: BlasFloat  || (T = promote_type(Float64,T))
    # use complex version because the generalized Schur form decomposition available only for complex data 
    if !(T <: BlasFloat || T <: Complex) 
       sol = gared(tocomplex(A),tocomplex(E),tocomplex(B),tocomplex(R),tocomplex(Q),tocomplex(S); scaling, nrm, pow2, as, rtol)
       return real(sol[1]), sol[2], real(sol[3]), Matrix(qr([real(sol[4]) imag(sol[4])]).Q)[:,1:size(A,1)], sol[5]
    end

    n = LinearAlgebra.checksquare(A)
    typeof(B) <: AbstractVector ? (nb, m) = (length(B), 1) : (nb, m) = size(B)
    n == nb || throw(DimensionMismatch("B must be a matrix with row dimension $n or a vector of length $n"))
    eident = (E == I)
    if !eident
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
    TR = real(T)
    epsm = eps(TR)
    eltype(A) == T || (A = convert(Matrix{T},A))
    eident || eltype(E) == T || (E = convert(Matrix{T},E))
    eltype(B) == T || (typeof(B) <: AbstractVector ? B = convert(Vector{T},B) : B = convert(Matrix{T},B))
    eltype(Q) == T || (typeof(Q) <: AbstractMatrix ? Q = convert(Matrix{T},Q) : Q = convert(T,Q.λ)*I)
    eltype(R) == T || (typeof(R) <: AbstractMatrix ? R = convert(Matrix{T},R) : R = convert(T,R.λ)*I)
    eltype(S) == T || (typeof(S) <: AbstractVector ? S = convert(Vector{T},S) : S = convert(Matrix{T},S))

    n == 0 && (return  zeros(T,0,0), zeros(T,0), zeros(T,m,0), zeros(T,m,0) )

    if !eident
      if T <: BlasFloat
         Et = LinearAlgebra.LAPACK.getrf!(copy(E))
         LinearAlgebra.LAPACK.gecon!('1',Et[1],opnorm(E,1))  < epsm && error("E must be non-singular")
      else
         cond(E)*epsm < 1 || error("E must be non-singular")
      end
    end
    #  Method:  A stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil
    #                   [  A   0    B ]      [ E  0  0 ]
    #          H -z J = [ -Q   E'  -S ]  - z [ 0  A' 0 ]
    #                   [ S'   0    R ]      [ 0 -B' 0 ]
    #  is computed and the solution X and feedback F are computed as
    #          X = Z21*inv(E*Z11),   F = Z31*inv(Z11).
    H, J, Sx, Sxi, Sr = balsympl(A, E, B, R, Q, S; scaling, pow2, nrm)
    n2 = n+n;
    iric = 1:n2
    i1 = 1:n
    i2 = n+1:n2
    i2m = n+1:n2+m
    i3 = n2+1:n2+m
    #F = qr([A'; -B'])
    #F = qr(view(J,i2m,i2))
    F = qr(J[i2m,i2])
    #L2 = F.Q'*[-Q  E' -S; copy(S') zeros(T,m,n) R]
    #L2 = F.Q'*view(H,i2m,:)
    L2 = F.Q'*H[i2m,:]
    P2 = [zeros(T,n,n) F.R zeros(T,n,m)]

    G = qr(L2[n+1:n+m,:]')
    cond(G.R) * epsm  < 1 || error("The extended symplectic pencil is not regular")
    z = (G.Q*I)[:,[m+1:m+n2; 1:m]]

    # L1 = [ A zeros(T,n,n) B; L2[i1,:]]*z
    L1 = [ view(H,i1,:); L2[i1,:]]*z
    # P1 = [ E zeros(T,n,n+m); P2]*z
    P1 = [ view(J,i1,:); P2]*z

    as ? PLS = schur(L1[iric,iric],P1[iric,iric]) : PLS = schur(P1[iric,iric],L1[iric,iric])
    select = abs.(PLS.α) .> abs.(PLS.β)

    n == count(select) || error("The extended symplectic pencil is not dichotomic")

    ordschur!(PLS, select)
    z[:,i1]= z[:,iric]*PLS.Z[:,i1]

    F = _LUwithRicTest(z[i1,i1],rtol)
    if eident
       x = z[n+1:end,i1]/F; 
       f = -x[n+1:end,:]; lmul!(Sr,f); rmul!(f,Sxi)
       x = x[i1,:]; lmul!(Sx,x); rmul!(x,Sxi)
    else
       f = -z[i3,i1]/F; lmul!(Sr,f); rmul!(f,Sxi)
       x = z[i2,i1]/F; lmul!(Sx,x); rmul!(x,Sxi); x = x/E
    end

    as ? iev = i2 : iev = i1
    clseig = PLS.β[iev] ./ PLS.α[iev]
    if as && T <: Complex
      clseig =  conj(clseig)
    end
    scalinfo = (Sx = Sx, Sxi = Sxi, Sr = Sr)
    return  LinearAlgebra._hermitianpart!(x), clseig, f, z[:,i1], scalinfo
end
function balham(A, G, Q; scaling = 'B', pow2 = false, nrm = 1)
   # Scaling function to be used in conjunction with arec(A,G,Q)
   H = [A -G; -Q -A']
   scaling == 'N' && (return H, I, I)
   n = size(A,1)
   i1 = 1:n; i2 = n+1:2n
   At1 = view(H,i1,i1)
   Gt = view(H,i1,i2)
   Qt = view(H,i2,i1)
   At2 = view(H,i2,i2)
   T = eltype(H)
   TR = real(T)
   radix = TR(2.)
   if scaling == 'B'
      # block scaling using square-root norms
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if (qs > gs) && (gs > 0)
         scal = qs/gs  
         scalsr = sqrt(scal)  
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scal,Gt); ldiv!(scal,Qt)  # Q -> Q/scal; G -> G * scal
         return H, scalsr*I, scalsr*I
      else    
         return H, I, I
      end
   elseif scaling == 'S'
      # structure preserving scaling enhanced with block scaling
      # unconstrained balancing with D = diag(D1,D2)
      Hd = Diagonal(H)
      if qS1(H) < 10*qS1(H-Hd) 
         d = lsbalance!(H-Hd).diag
      else
         d = lsbalance!(copy(H)).diag
      end
      s = log2.(d) 
      # impose that diagonal scaling has the form diag(Sx,1./Sx)  
      sx = round.(Int,(-s[i1]+s[i2])/2); # Sx = sqrt(D1/D2)
      Sx = Diagonal(radix.^(sx)) 
      lmul!(Sx,Gt); rmul!(Gt,Sx)  # Gt <- Sx*Gt*Sx
      ldiv!(Sx,Qt); rdiv!(Qt,Sx)  # Qt <- Sx\(Qt/Sx)
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scalsr,Sx)
         lmul!(scal,Gt); ldiv!(scal,Qt)
      end      
      lmul!(Sx,At1); rdiv!(At1,Sx) 
      ldiv!(Sx,At2); rmul!(At2,Sx) 
      return H, Sx, Sx
   elseif scaling == 'K'
      # structure preserving scaling enhanced, if appropriate, with block scaling
      M = abs.(H)
      ind = [i2;i1]  
      dl, dr = symscal!(view(M,ind,:); nrm, maxiter = 1000)
      s = log2.(dr) 
      # impose that diagonal scaling has the form diag(Sx,1./Sx)  
      sx = round.(Int,(-s[i1]+s[i2])/2); # Sx = sqrt(D1/D2)
      Sx = Diagonal(radix.^(sx)) 
      lmul!(Sx,Gt); rmul!(Gt,Sx)  # Gt <- Sx*Gt*Sx
      ldiv!(Sx,Qt); rdiv!(Qt,Sx)  # Qt <- Sx\(Qt/Sx)

      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scalsr,Sx)
         lmul!(scal,Gt); ldiv!(scal,Qt)
      end      
      lmul!(Sx,At1); rdiv!(At1,Sx) 
      ldiv!(Sx,At2); rmul!(At2,Sx) 
      return H, Sx, Sx


      Dl = Diagonal(dl); Dr = Diagonal(dr)
      lmul!(Dl,view(H,ind,:)); rmul!(H,Dr)
      lmul!(Dl,view(J,ind,:)); rmul!(J,Dr)

      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         lmul!(scalsr,view(Dr.diag,i2)); ldiv!(scalsr,view(Dr.diag,i1))
      end

      return H, J, Diagonal(Dr.diag[i2]), inv(Diagonal(Dr.diag[i1])) 

   elseif scaling == 'G'
      # unconstrained balancing with D = diag(D1,D2)
      d = lsbalance!(H).diag
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         lmul!(scal,Gt); ldiv!(scal,Qt)
         lmul!(scalsr,view(d,i2)); ldiv!(scalsr,view(d,i1))
      end
      # for convenience, return D2 and inv(D1)
      return H, Diagonal(d[i2]), inv(Diagonal(d[i1])) 
   else
      @warn "No such scaling option: no scaling is performed"
      return H, I, I
   end
end
function balham(A, E, G, Q; scaling = 'B', pow2 = false, nrm = 1) 
   # Scaling function to be used in conjunction with garec(A,E,G,Q)
   n = size(A,1); 
   T = eltype(A)
   H = [A -G; -Q -A']; J = [E zeros(T,n,n); zeros(T,n,n) E']
   scaling == 'N' && (return H, J, I, I)
   n2 = 2n; i1 = 1:n; i2 = n+1:n2
   Gt = view(H,i1,i2)
   Qt = view(H,i2,i1)
   TR = real(T)
   radix = TR(2.)
   if scaling == 'B'
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if (qs > gs) && (gs > 0)
         scal = qs/gs  
         scalsr = sqrt(scal) 
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal 
         return H, J, scalsr*I, scalsr*I
      else    
         return H, J, I, I
      end
   elseif scaling == 'S'
      # structure preserving scaling enhanced, if appropriate, with block scaling
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = nj * abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      d = lsbalance!(M).diag
      s = log2.(d); # unconstrained balancing diag(D1,D2)
      sx = round.(Int,(-s[i1]+s[i2])/2);
      # impose that diagonal scaling has the form diag(Sx,1./Sx)  
      D = Diagonal(radix.^[sx ; -sx])
      lmul!(D,H); rdiv!(H,D)
      lmul!(D,J); rdiv!(J,D)
      Sx = Diagonal(D.diag[i1]) # Sx = sqrt(D2/D1)

      # check if block scaling is appropriate
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scalsr,Sx) 
         lmul!(scal,Gt); ldiv!(scal,Qt)
      end         
      return H, J, Sx, Sx
   elseif scaling == 'K'
      # structure preserving scaling enhanced, if appropriate, with block scaling
      # nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      # nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      # if nh > 0 && nj > 0
      #    M = nj * abs.(H) + nh/nj * abs.(J);
      # else
      #    M = abs.(H) + abs.(J);
      # end   
      M = abs.(H) + abs.(J)
      ind = [i2;i1]  
      dl, dr = symscal!(view(M,ind,:); nrm, maxiter = 1000)
      pow2 && (dr .= radix .^(round.(Int,log2.(dr))); dl = dr) 
      Dl = Diagonal(dl); Dr = Diagonal(dr)
      lmul!(Dl,view(H,ind,:)); rmul!(H,Dr)
      lmul!(Dl,view(J,ind,:)); rmul!(J,Dr)

      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         lmul!(scalsr,view(Dr.diag,i2)); ldiv!(scalsr,view(Dr.diag,i1))
      end

      return H, J, Diagonal(Dr.diag[i2]), inv(Diagonal(Dr.diag[i1])) 
   elseif scaling == 'G'
      # general scaling enhanced, if appropriate, with a block scaling
      _, D2 = regbalance!(H, J; tol = 0.1)
      qs = sqrt(opnorm(Qt,1))
      gs = sqrt(opnorm(Gt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         lmul!(scalsr,view(D2.diag,i2)); ldiv!(scalsr,view(D2.diag,i1))
      end
      return H, J, Diagonal(D2.diag[i2]), inv(Diagonal(D2.diag[i1])) 
   else
      @warn "No such scaling option: no scaling is performed"
      return H, J, I, I
   end
end
function balham(A, E, B, G, R, Q, S; scaling = 'B', pow2 = false, nrm = 1) 
   # Scaling function to be used in conjunction with garec(A, E, B, G, R, Q, S)
   #                   [  A   -G    B ]      [ E  0  0 ]
   #          H -z J = [ -Q   -A'  -S ]  - z [ 0  E' 0 ]
   #                   [ S'    B'   R ]      [ 0  0  0 ]
   
   n, m = size(B,1), size(B,2); n2 = 2n
   T = eltype(A)
   H = [A -G B; -Q -A' -S; S' B' R] 
   J = [E zeros(T,n,n+m); zeros(T,n,n) E' zeros(T,n,m); zeros(T,m,n2+m)]
   scaling == 'N' && (return H, J, I, I, I)
   i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m; j2 = 1:n2
   At = view(H,i1,i1)
   Bt = view(H,i1,i3)
   Gt = view(H,i1,i2)
   Qt = view(H,i2,i1)
   St = view(H,i2,i3)
   St2 = view(H,i3,i1)
   Rt = view(H,i3,i3)
   Et = view(J,i1,i1)  
   T = eltype(H)
   TR = real(T)
   radix = TR(2.)
   if scaling == 'B'
      # block scaling using square-root norms
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(B,1)/sqrt(norm(Rt,1))
      if (qs > gs) && (gs > 0)
         scal = qs/gs  
         scalsr = sqrt(scal)  
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         return H, J, scalsr*I, scalsr*I, (1/scalsr)*I
      else    
         return H, J, I, I, I
      end
   elseif scaling == 'S'
      # structure preserving scaling enhanced, if appropriate, with block scaling
      nh = norm(At-Diagonal(At),1) 
      nj = norm(Et-Diagonal(Et),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      # unconstrained balancing diag(D1,D2,Dr)
      d = lsbalance!(M).diag
      s = log2.(d) 
      sx = round.(Int,(-s[i1]+s[i2])/2); 
      sr = -s[i3]
      # impose that diagonal scaling has the form diag(Sx,1./Sx, Sr)  
      D = Diagonal(radix.^[sx ; -sx ; sr])
      lmul!(D,H); rdiv!(H,D)
      lmul!(D,J); rdiv!(J,D)
      Sx = Diagonal(radix.^sx) # Sx = sqrt(D2/D1)
      Sr = Diagonal(radix.^sr)

      # check if block scaling is appropriate
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scalsr,Sx); ldiv!(scalsr,Sr)
         lmul!(scal,Gt); ldiv!(scal,Qt); ldiv!(scal,Rt)
         ldiv!(scal,St); ldiv!(scal,St2)
      end      

      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,Sr)
      end
      return H, J, Sx, Sx, Sr
   elseif scaling == 'D'
      # descriptor system oriented general scaling enhanced, if appropriate, with block scaling
      sr = ones(T,m)
      _, D2  = lsbalance!(view(H,j2,j2),view(J,j2,j2),view(H,j2,i3),view(H,i3,j2); tol = 0.001)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(D2.diag,i2)); ldiv!(scalsr,view(D2.diag,i1)); 
         ldiv!(scalsr,sr)
      end
      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,sr)
      end

      return H, J, Diagonal(D2.diag[i2]), inv(Diagonal(D2.diag[i1])), Diagonal(sr) 
   elseif scaling == 'T'
      # standard system oriented structure preserving scaling enhanced, if appropriate, with block scaling
      nh = norm(At-Diagonal(At),1) 
      nj = norm(Et-Diagonal(Et),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      # unconstrained standard system balancing with diag(S1,S2,I)
      S = lsbalance!(view(M,j2,j2),view(M,j2,i3),view(M,i3,j2))
      sx = round.(Int,(log2.(S.diag[i2])-log2.(S.diag[i1]))/2); 
      sr = ones(T,m)     
      D = Diagonal(radix.^[sx ; -sx])
      lmul!(D,view(H,j2,:)); rdiv!(view(H,:,j2),D)
      lmul!(D,view(J,j2,j2)); rdiv!(view(J,j2,j2),D)
      Sx = Diagonal(radix.^sx) # Sx = sqrt(S2/S1)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,Sx); ldiv!(scalsr,sr)
      end

      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,sr)
      end
      Sr = Diagonal(sr)
      return H, J, Sx, Sx, Sr
   elseif scaling == 'K'
      # symmetric scaling enhanced, if appropriate, with a block scaling
      nh = norm(At-Diagonal(At),1) 
      nj = norm(Et-Diagonal(Et),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + (nh/nj) * abs.(J)
      else
         M = abs.(H) + abs.(J)
      end     
      ind = [i2;i1;i3]  
      dl, dr = symscal!(view(M,ind,:); nrm, maxiter = 1000)
      pow2 && (dr .= radix .^(round.(Int,log2.(dr))); dl = dr) 

      Dl = Diagonal(dl); Dr = Diagonal(dr)
      lmul!(Dl,view(H,ind,:)); rmul!(H,Dr)
      lmul!(Dl,view(J,ind,:)); rmul!(J,Dr)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(Dr.diag,i2)); ldiv!(scalsr,view(Dr.diag,i1))
         ldiv!(scalsr,view(Dr.diag,i3))
      end
      return H, J, Diagonal(Dr.diag[i2]), inv(Diagonal(Dr.diag[i1])), Diagonal(Dr.diag[i3])   
   elseif scaling == 'G'
      # general scaling enhanced, if appropriate, with a block scaling
      _, D2 = regbalance!(H, J; tol = 0.001, maxiter = 1000, pow2)
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = sqrt(opnorm(Gt,1)) + norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt); lmul!(scal,Gt)  # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(D2.diag,i2)); ldiv!(scalsr,view(D2.diag,i1)); 
         ldiv!(scalsr,view(D2.diag,i3))
      end
      return H, J, Diagonal(D2.diag[i2]), inv(Diagonal(D2.diag[i1])), Diagonal(D2.diag[i3]) 
   else
      @warn "No such scaling option: no scaling is performed"
      return H, J, I, I, I
   end
end
function balsympl(A, E, B, R, Q, S; scaling = 'B', pow2 = false, nrm = 1) 
   # Scaling function to be used in conjunction with gared(A, E, B, R, Q, S)
   n, m = size(B,1), size(B,2); n2 = 2n
   T = eltype(A)
    #                   [  A   0    B ]      [ E  0  0 ]
    #          H -z J = [ -Q   E'  -S ]  - z [ 0  A' 0 ]
    #                   [ S'   0    R ]      [ 0 -B' 0 ]
   H = [A zeros(T,n,n) B; -Q E' -S; S' zeros(T,m,n) R] 
   J = [E zeros(T,n,n+m); zeros(T,n,n) A' zeros(T,n,m); zeros(T,m,n) -B' zeros(T,m,m)]
   scaling == 'N' && (return H, J, I, I, I)
   i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m; j2 = 1:n2
   Qt = view(H,i2,i1)
   St = view(H,i2,i3)
   St2 = view(H,i3,i1)
   Rt = view(H,i3,i3)
   Bt = view(H,i1,i3)
   T = eltype(H)
   TR = real(T)
   radix = TR(2.)
   if scaling == 'B'
      i1 = 1:n; i2 = n+1:n2; i3 = n2+1:n2+m
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(B,1)/sqrt(norm(Rt,1))
      if (qs > gs) && (gs > 0)
         scal = qs/gs  
         scalsr = sqrt(scal)  
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt)                   # Q -> Q/scal; 
         ldiv!(scal,St); ldiv!(scal,St2)  # S -> S/scal 
         ldiv!(scal,Rt)                   # R -> R/scal
         return H, J, scalsr*I, scalsr*I, (1/scalsr)*I
      else    
         return H, J, I, I, I
      end
   elseif scaling == 'S'
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end  
      d = lsbalance!(M).diag
      s = log2.(d) 
      sx = round.(Int,(-s[i1]+s[i2])/2); 
      sr = -s[i3]
      # impose that diagonal scaling has the form diag(Sx,1./Sx, Sr)  
      D = Diagonal(radix.^[sx ; -sx ; sr])
      lmul!(D,H); rdiv!(H,D)
      lmul!(D,J); rdiv!(J,D)
      Sx = Diagonal(radix.^sx) # Sx = sqrt(D2/D1)
      Sr = Diagonal(radix.^sr)


      # check if block scaling is appropriate
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         lmul!(scalsr,Sx); ldiv!(scalsr,Sr)
         ldiv!(scal,Qt); ldiv!(scal,Rt)
         ldiv!(scal,St); ldiv!(scal,St2)
      end      
      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,view(J,i3,:)); lmul!(rs,Sr)
      end
      return H, J, Sx, Sx, Sr
   elseif scaling == 'K'
      # symmetric scaling enhanced, if appropriate, with a block scaling
      M = abs.(H) + abs.(J)
      ind = [i2;i1;i3]  
      dl, dr, info = symscal!(view(M,ind,:); nrm, maxiter = 1000)
      pow2 && (dr .= radix .^(round.(Int,log2.(dr))); dl = dr) 

      Dl = Diagonal(dl); Dr = Diagonal(dr)
      lmul!(Dl,view(H,ind,:)); rmul!(H,Dr)
      lmul!(Dl,view(J,ind,:)); rmul!(J,Dr)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt);                 # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(Dr.diag,i2)); ldiv!(scalsr,view(Dr.diag,i1))
         ldiv!(scalsr,view(Dr.diag,i3))
      end
      return H, J, Diagonal(Dr.diag[i2]), inv(Diagonal(Dr.diag[i1])), Diagonal(Dr.diag[i3])   
   elseif scaling == 'D'
      # descriptor system oriented general scaling enhanced, if appropriate, with block scaling
      sr = ones(T,m)
      copyto!(view(H,i3,i2),view(J,i3,i2))
      _, D2  = lsbalance!(view(H,j2,j2),view(J,j2,j2),view(H,j2,i3),view(H,i3,j2); tol = 0.001)
      copyto!(view(J,i3,i2),view(H,i3,i2)); fill!(view(H,i3,i2),zero(T))

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt);                 # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(D2.diag,i2)); ldiv!(scalsr,view(D2.diag,i1)); 
         ldiv!(scalsr,sr)
      end
      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,view(J,i3,:)); lmul!(rs,sr)
      end

      return H, J, Diagonal(D2.diag[i2]), inv(Diagonal(D2.diag[i1])), Diagonal(sr) 
   elseif scaling == 'R'
      sr = ones(T,m)
      Ht = [A zeros(T,n,n) B; -Q E' -S; S' -B' R] 
      Jt = [E zeros(T,n,n+m); zeros(T,n,n) A' zeros(T,n,m); zeros(T,m,n2+m)]
      S1,  = lsbalance!(view(Ht,j2,j2),view(Jt,j2,j2),view(Ht,j2,i3),view(Ht,i3,j2); tol = 0.001)

      sx = round.(Int,(log2.(S1.diag[i1])-log2.(S1.diag[i2]))/2); # D=sqrt(D1/D2)
      Sx = Diagonal(radix.^(sx)) 
      D = Diagonal(radix.^[sx ; -sx])
      lmul!(D,view(H,j2,:)); rdiv!(view(H,:,j2),D)
      lmul!(D,view(J,j2,j2)); rdiv!(view(J,:,j2),D)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt);                 # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,Sx); ldiv!(scalsr,sr)
      end

      # adjust scaling of R 
      # rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      # if 10*rs < 1 
      #    pow2 && (rs = radix^(round(Int,log2(rs)))) 
      #    lmul!(rs,view(H,i3,:)); lmul!(rs,view(J,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,sr)
      # end
      Sr = Diagonal(sr)
      return H, J, Sx, Sx, Sr
    elseif scaling == 'T'
      nh = norm(view(H,i1,i1)-Diagonal(view(H,i1,i1)),1) + norm(view(H,i2,i2)-Diagonal(view(H,i2,i2)),1) 
      nj = norm(view(J,i1,i1)-Diagonal(view(J,i1,i1)),1) + norm(view(J,i2,i2)-Diagonal(view(J,i2,i2)),1) 
      if nh > 0 && nj > 0
         M = abs.(H) + nh/nj * abs.(J);
      else
         M = abs.(H) + abs.(J);
      end     
      sr = ones(T,m)
      S1 = lsbalance!(view(M,j2,j2),view(M,j2,i3),view(M,i3,j2))
      sx = round.(Int,(log2.(S1.diag[i2])-log2.(S1.diag[i1]))/2); # D=sqrt(D1/D2)
      Sx = Diagonal(radix.^(sx)) 
      D = Diagonal(radix.^[sx ; -sx])
      lmul!(D,view(H,j2,:)); rdiv!(view(H,:,j2),D)
      lmul!(D,view(J,j2,j2)); rdiv!(view(J,:,j2),D)

      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt);                 # Q -> Q/scal; 
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,Sx); ldiv!(scalsr,sr)
      end
      # adjust scaling of R 
      rs = sqrt(norm(Rt,1))/max(norm(Bt,1),norm(St,1)); 
      if 10*rs < 1 
         pow2 && (rs = radix^(round(Int,log2(rs)))) 
         lmul!(rs,view(H,i3,:)); lmul!(rs,view(J,i3,:)); rmul!(view(H,:,i3),rs); lmul!(rs,sr)
      end

      Sr = Diagonal(sr)
      return H, J, Sx, Sx, Sr
   else
      _, D2 = regbalance!(H, J; tol = 0.001, maxiter = 1000, pow2 = false)
      qs = sqrt(opnorm(Qt,1)) + sqrt(opnorm(St,1))
      gs = norm(Bt,1)/sqrt(norm(Rt,1))
      if qs > 10*gs
         scal = qs/gs  
         scalsr = sqrt(scal)
         pow2 && (scalsr = radix^(round(Int,log2(scalsr))); scal = scalsr^2) 
         ldiv!(scal,Qt);                 # Q -> Q/scal; G -> G * scal
         ldiv!(scal,St); ldiv!(scal,St2) # S -> S/scal 
         ldiv!(scal,Rt)                  # R -> R/scal
         lmul!(scalsr,view(D2.diag,i2)); ldiv!(scalsr,view(D2.diag,i1)); 
         ldiv!(scalsr,view(D2.diag,i3))
      end

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
"""
    symscal!(M; maxiter = 100, tol = 1.e-3, nrm = 1) -> (dleft,dright)

Perform the symmetry preserving algorithm to scale 
a non-negative `m×n` matrix `M` such that
 
     Md := diag(dleft)*M*diag(dright)

has all row and column norms equal to one. The employed norms can be selected 
via the keyword argument `nrm`. Presently only `nrm = 1` (default) and `nrm = Inf` are supported.
The iterative process is stopped as soon as the incremental
scalings are tol-close to the identity. The keyword argument `tol` is the
tolerance for the transformation updates. 

The resulting `Md` overwrites the input matrix `M`, and `dleft` and `dright` are 
the diagonals of the left and right scalings. 
`Md` results symmetric provided `M` is a symmetric matrix, and in this case `dleft = dright`.

The scaling algorithm has been proposed in [1]. 

[1] Authors: P. A. Knight, D. Ruiz and B. Uçar.
    A Symmetry Preserving Algorithm for Matrix Scaling. 
    SIAM Journal on Matrix Analysis and Applications, 35:931-955, 2014. 
"""
function symscal!(M::AbstractMatrix{T}; maxiter = 100, tol = 1.e-3, nrm = 1) where {T}
    m, n = size(M) 
    dl = ones(T,m); dr = ones(T,n)
    for i = 1:maxiter
        r = [sqrt(norm(view(M,j,:),nrm)) for j in 1:m]
        c = [sqrt(norm(view(M,:,j),nrm)) for j in 1:n]
        if norm(1 .- r,Inf) < tol && norm(1 .- c,Inf) < tol 
           info = (flag = 1, iter = i)
           return dl, dr, info
        end
        ldiv!(Diagonal(r),M); rdiv!(M,Diagonal(c))
        dl ./= r; dr ./= c
    end
    info = (flag = 2, iter = maxiter)
    return dl, dr, info
end



