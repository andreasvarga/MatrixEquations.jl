"""
    arec(A, R, Q) -> (X, EVALS)

Compute `X`, the hermitian/symmetric stabilizing solution of the continuous-time
algebraic Riccati equation

     A' X + X A - XRX + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) eigenvalues of `A-RX`.

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

julia> R = [1. 0. 0.; 0. 5. 0.; 0. 0. 10.]
3×3 Array{Float64,2}:
 1.0  0.0   0.0
 0.0  5.0   0.0
 0.0  0.0  10.0

julia> X, CLSEIG = arec(A,R,2I);

julia> X
3×3 Array{Float64,2}:
  0.459589   0.333603   -0.144406
  0.333603   0.65916    -0.0999216
 -0.144406  -0.0999216   0.340483

julia> A'*X+X*A-X*R*X+2I
3×3 Array{Float64,2}:
 -1.33227e-15  4.44089e-16  -2.22045e-15
  4.44089e-16  8.88178e-16   1.11022e-16
 -2.22045e-15  0.0          -1.77636e-15

julia> CLSEIG
3-element Array{Complex{Float64},1}:
 -4.411547592296008 + 2.4222082620381102im
 -4.411547592296008 - 2.4222082620381102im
 -4.337128244724371 + 0.0im

julia> eigvals(A-R*X)
3-element Array{Complex{Float64},1}:
 -4.411547592296008 - 2.4222082620381076im
 -4.411547592296008 + 2.4222082620381076im
 -4.337128244724376 + 0.0im
```
"""
function arec(A::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix = zeros(eltype(A),size(A)))
    n = LinearAlgebra.checksquare(A)
    if LinearAlgebra.checksquare(R) != n || !ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(Q) != n || !ishermitian(Q)
      throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    T2 = promote_type(eltype(A), eltype(R), eltype(Q))
    if T2 == Int64 || T2 == Complex{Int64}
      T2 = promote_type(Float64,T2)
    end
    if eltype(A) !== T2
      A = convert(Matrix{T2},A)
    end
    if eltype(R) !== T2
      R = convert(Matrix{T2},R)
    end
    if eltype(Q) !== T2
      Q = convert(Matrix{T2},Q)
    end

    S = schur([A  -R; -Q  -copy(A')])
    
    select = real(S.values) .< 0
    if n != length(filter(y-> y == true,select))
       error("The Hamiltonian matrix is not dichotomic")
   end
    ordschur!(S, select)

    n2 = n+n
    ix = 1:n
    x = S.Z[n+1:n2, ix]/S.Z[ix, ix]
    return  (x+x')/2, S.values[ix], []
end
function arec(A::AbstractMatrix, R::UniformScaling, Q::UniformScaling) 
   n = LinearAlgebra.checksquare(A)
   T = eltype(A)
   return arec(A,Matrix(one(T)*R,n,n),Matrix(one(T)*Q,n,n))
end
function arec(A::AbstractMatrix, R::UniformScaling, Q::AbstractMatrix) 
   n = LinearAlgebra.checksquare(A)
   T = eltype(A)
   return arec(A,Matrix(one(T)*R,n,n),Q)
end
function arec(A::AbstractMatrix, R::AbstractMatrix, Q::UniformScaling) 
   n = LinearAlgebra.checksquare(A)
   T = eltype(A)
   return arec(A,R,Matrix(one(T)*Q,n,n))
end
"""
    arec(A, B, R, Q, S) -> (X, EVALS, F)

Computes `X`, the hermitian/symmetric stabilizing solution of the continuous-time
 algebraic Riccati equation

     A' X + X A - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular.
`EVALS` is a vector containing the (stable) eigenvalues of `A-BF`.
`F` is the stabilizing gain matrix `F = R^(-1)(B'X+S')`.

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
 -2.66454e-15  -1.55431e-15   1.11022e-15
 -1.55431e-15   1.9984e-15   -3.21965e-15
  1.22125e-15  -2.9976e-15    6.66134e-16

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
function arec(A::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix, S::AbstractMatrix = zeros(eltype(B),size(B))) 
    n = LinearAlgebra.checksquare(A)
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix with row dimension $n"))
    end
    if LinearAlgebra.checksquare(Q) !== n || !ishermitian(Q)
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || !ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if (n,m) !== size(S)
       throw(DimensionMismatch("S must be a $n x $m matrix"))
    end
    S0flag = iszero(S)
    SR = schur(R)
    D = real(diag(SR.T))
    Da = abs.(D)
    minDa, = findmin(Da)
    maxDa, = findmax(Da)
    if minDa <= eps()*maxDa
       error("R must be non-singular")
    elseif minDa > sqrt(eps())*maxDa
       #Dinv = diagm(0 => 1 ./ D)
       Dinv = Diagonal(1 ./ D)
       Bu = B*SR.Z
       #G = Bu*Dinv*Bu'
       G = utqu(Dinv,Bu')
       if S0flag
          sol = arec(A,G,Q)
          f = SR.Z*Dinv*Bu'*sol[1]
       else
          Su = S*SR.Z
          #Q -= Su*Dinv*Su'
          Q -= utqu(Dinv,Su')
          sol = arec(A-Bu*Dinv*Su',G,Q)
          f = SR.Z*Dinv*(Bu'*sol[1]+Su')
       end
       return sol[1], sol[2], f
   else
       #UseImplicitForm
       garec(A, I, B, R, Q, S)
   end
end
arec(A::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::UniformScaling) = arec(A,B,R,Matrix(one(eltype(A))*Q,size(A)))

"""
    garec(A, E, B, R, Q, S) -> (X, EVALS, F)

Compute `X`, the hermitian/symmetric
stabilizing solution of the generalized continuous-time algebraic Riccati equation

    A'XE + E'XA - (E'XB+S)R^(-1)(B'XE+S') + Q = 0 ,

where `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
`EVALS` is a vector containing the (stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing gain matrix `F = R^(-1)(B'XE+S')`.

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
function garec(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix, S::AbstractMatrix = zeros(eltype(B),size(B)))
    n = LinearAlgebra.checksquare(A)
    T = promote_type(eltype(A), eltype(B), eltype(Q), eltype(R) )
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix of row dimension $n"))
    end
    if E == I
       eident = true
    else
       if LinearAlgebra.checksquare(E) != n
          throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       end
       eident = isequal(E,I)
       if eident
          E = I
       else
          Et = LinearAlgebra.LAPACK.getrf!(copy(E))
          if LinearAlgebra.LAPACK.gecon!('1',Et[1],norm(E,1))  < eps(1.)
             error("E must be non-singular")
          end
          T = promote_type(T,eltype(E))
       end
    end
    if LinearAlgebra.checksquare(Q) !== n || !ishermitian(Q)
       throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || !ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if cond(R)*eps(1.) > 1.
       error("R must be non-singular")
    end
    if (n,m) !== size(S)
       throw(DimensionMismatch("S must be a $n x $m matrix"))
    end
    T = promote_type(T,eltype(S))
    if eltype(A) !== T
      A = complex(A)
    end
    if !eident && eltype(E) !== T
      E = complex(E)
    end
    if eltype(B) !== T
      B = complex(B)
    end
    if eltype(Q) !== T
      Q = complex(Q)
    end
    if eltype(R) !== T
      R = complex(R)
    end
    if eltype(S) !== T
      S = complex(S)
    end

    """
    Method:  A stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil

                 [  A   0    B ]      [ E  0  0 ]
        L -s P = [ -Q  -A'  -S ]  - s [ 0  E' 0 ]
                 [ S'   B'   R ]      [ 0  0  0 ]

   is determined and the solution X and feedback F are computed as

            X = Z21*inv(E*Z11),   F = Z31*inv(Z11).
    """

    #deflate m simple infinite eigenvalues
    n2 = n+n;
    G = qr([S; B; R]);
    if cond(G.R) * eps(1.)  > 1.
       error("The extended Hamiltonian pencil is not regular")
    end

    #z = G.Q[:,m+1:m+n2]
    z = G.Q*[fill(false,m,n2); I ]

    iric = 1:n2
    L11 = [ A zeros(T,n,n) B; -Q -A' -S]*z
    P11 = [ E zeros(T,n,n); zeros(T,n,n) E']*z[iric,:]
    LPS = schur(L11,P11)
    select = real.(LPS.α ./ LPS.β) .< 0.
    if n !== length(filter(y-> y == true,select))
       error("The extended simplectic pencil is not dichotomic")
    end
    ordschur!(LPS, select)
    i1 = 1:n
    i2 = n+1:n2
    i3 = n2+1:n2+m

    z[:,i1] = z[:,iric]*LPS.Z[:,i1];

    if eident
       x = z[n+1:end,i1]/z[i1,i1]
       f = -x[n+1:end,:]
       x = x[i1,:]
    else
       f = -z[i3,i1]/z[i1,i1]
       x = z[i2,i1]/(E*z[i1,i1])
    end

    return  (x+x')/2, LPS.values[i1] , f
end
function garec(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::UniformScaling, S::AbstractMatrix = zeros(eltype(B),size(B))) 
   garec(A, E, B, R, Matrix(one(eltype(A))*Q,size(A)), S)
end


"""
    gared(A, E, B, R, Q, S) -> (X, EVALS, F)

Compute `X`, the hermitian/symmetric
stabilizing solution of the generalized discrete-time algebraic Riccati equation

    A'XA - E'XE - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

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
 -0.5238922629921537
 -0.19053355203423877
 -0.08423561575133914
```
"""
function gared(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix, S::AbstractMatrix = zeros(eltype(B),size(B)))
    n = LinearAlgebra.checksquare(A)
    T = promote_type(eltype(A), eltype(B), eltype(Q), eltype(R) )
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix with row dimension $n"))
    end
    if typeof(E) <: UniformScaling{Bool} 
       eident = true
       E = I
    else
       if LinearAlgebra.checksquare(E) != n
          throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       end
       eident = isequal(E,I)
       if eident
          E = I
       else
          T = promote_type(T,eltype(E))
       end
    end
    if LinearAlgebra.checksquare(Q) !== n || !ishermitian(Q)
       throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || !ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if (n,m) !== size(S)
      throw(DimensionMismatch("S must be a $n x $m matrix"))
    end
    T = promote_type(T,eltype(S))
    if eltype(A) != T
      A = complex(A)
    end
    if !eident && eltype(E) != T
      E = complex(E)
    end
    if eltype(B) != T
      B = complex(B)
    end
    if eltype(Q) != T
      Q = complex(Q)
    end
    if eltype(R) != T
      R = complex(R)
    end
    if eltype(S) != T
      S = complex(S)
    end

    """
    Method:  A stable deflating subspace Z1 = [Z11; Z21; Z31] of the pencil

                     [  A   0    B ]      [ E  0  0 ]
            L -z P = [ -Q   E'  -S ]  - z [ 0  A' 0 ]
                     [ S'   0    R ]      [ 0 -B' 0 ]

    is computed and the solution X and feedback F are computed as

            X = Z21*inv(E*Z11),   F = Z31*inv(Z11).
    """
    n2 = n+n;
    F = qr([A'; -B'])
    L2 = F.Q'*[-Q  E' -S; copy(S') zeros(T,m,n) R]
    P2 = [zeros(T,n,n) F.R zeros(T,n,m)]

    G = qr(L2[n+1:n+m,:]')
    if cond(G.R) * eps(1.)  > 1.
       error("The extended symplectic pencil is not regular")
    end
    z = (G.Q*I)[:,[m+1:m+n2; 1:m]]

    L1 = [ A zeros(T,n,n) B; L2[1:n,:]]*z
    P1 = [ E zeros(T,n,n+m); P2]*z

    iric = 1:n2
    PLS = schur(P1[iric,iric],L1[iric,iric])
    select = abs.(PLS.α) .> abs.(PLS.β)

    if n !== length(filter(y-> y == true,select))
       error("The extended simplectic pencil is not dichotomic")
    end
    ordschur!(PLS, select)
    z[:,iric]= z[:,iric]*PLS.Z;

    i1 = 1:n
    i2 = n+1:n2
    i3 = n2+1:n2+m
    if eident
       x = z[n+1:end,i1]/z[i1,i1]
       f = -x[n+1:end,:]
       x = x[i1,:]
    else
       f = -z[i3,i1]/z[i1,i1]
       x = z[i2,i1]/(E*z[i1,i1])
    end

    return  (x+x')/2, PLS.β[i1] ./ PLS.α[i1], f
end
function gared(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::UniformScaling, S::AbstractMatrix = zeros(eltype(B),size(B))) 
   gared(A, E, B, R, Matrix(one(eltype(A))*Q,size(A)), S)
end


"""
    ared(A, B, R, Q, S) -> (X, EVALS, F)

Computes `X`, the hermitian/symmetric
stabilizing solution of the discrete-time algebraic Riccati equation

     A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) generalized eigenvalues of `A-BF`.
`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

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

julia> B = [1. 2.; 2. 0.; 0. 1.]
3×2 Array{Float64,2}:
 1.0  2.0
 2.0  0.0
 0.0  1.0

julia> R = [1. 0.; 0. 5.]
2×2 Array{Float64,2}:
 1.0  0.0
 0.0  5.0

julia> X, CLSEIG, F = ared(A,B,R,2I);

julia> X
3×3 Array{Float64,2}:
  109.316   -63.1658  -214.318
  -63.1658  184.047    426.081
 -214.318   426.081   1051.16

julia> A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+2I
3×3 Array{Float64,2}:
 -2.50111e-11   4.16094e-11   1.06184e-10
  4.75211e-11  -8.11724e-11  -2.06228e-10
  1.11186e-10  -2.06455e-10  -5.15001e-10

julia> CLSEIG
3-element Array{Complex{Float64},1}:
 -0.04209841276282689 - 0.0497727979322874im
 -0.04209841276282689 + 0.0497727979322874im
  -0.5354826075405217 - 0.0im

julia> eigvals(A-B*F)
3-element Array{Complex{Float64},1}:
  -0.5354826075397419 + 0.0im
 -0.04209841276292829 - 0.049772797931966324im
 -0.04209841276292829 + 0.049772797931966324im
```
"""
function ared(A::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::AbstractMatrix, S::AbstractMatrix = zeros(eltype(B),size(B))) 
    gared(A, I, B, R, Q, S)
end
function ared(A::AbstractMatrix, B::AbstractMatrix, R::AbstractMatrix, Q::UniformScaling, S::AbstractMatrix = zeros(eltype(B),size(B))) 
   gared(A, I, B, R, Matrix(one(eltype(A))*Q,size(A)), S)
end
