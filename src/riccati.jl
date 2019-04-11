"""
`arec(A, Q, R) -> (X, EVALS)` computes `X`, the hermitian/symmetric stabilizing
solution of the continuous-time algebraic Riccati equation

     A' X + X A - XRX + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) eigenvalues of `A-RX`.

Reference:
Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.
"""
function arec(A, Q, R)
    n = LinearAlgebra.checksquare(A)
    if LinearAlgebra.checksquare(Q) != n || ~ishermitian(Q)
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) != n || ~ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $n"))
    end

    S = schur([A  -R; -Q  -A'])
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

"""
`arec(A, B, R, Q, S) -> (X, EVALS, F)` computes `X`, the hermitian/symmetric
stabilizing solution of the continuous-time algebraic Riccati equation

     A' X + X A - (XB+S)R^(-1)(B'X+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular.
`EVALS` is a vector containing the (stable) eigenvalues of `A-BF`.
`F` is the stabilizing gain matrix `F = R^(-1)(B'X+S')`.

Reference:
Laub, A.J., A Schur Method for Solving Algebraic Riccati equations.
IEEE Trans. Auto. Contr., AC-24, pp. 913-921, 1979.
"""
function arec(A, B, Q, R, S = 0)
    n = LinearAlgebra.checksquare(A)
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix with row dimension $n"))
    end
    if LinearAlgebra.checksquare(Q) !== n || ~ishermitian(Q)
          throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || ~ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if S == 0
       S = zeros(n,m)
       S0flag = true
    else
      if (n,m) !== size(S)
         throw(DimensionMismatch("S must be a $n x $m matrix"))
      end
      S0flag = false
    end
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
       G = utqu(Dinv,Bu,adj=true)
       if S0flag
          sol = arec(A,Q,G)
          f = SR.Z*Dinv*Bu'*sol[1]
       else
          Su = S*SR.Z
          #Q -= Su*Dinv*Su'
          Q -= utqu(Dinv,Su,adj=true)
          sol = arec(A-Bu*Dinv*Su',Q,G)
          f = SR.Z*Dinv*(Bu'*sol[1]+Su')
       end
       return sol[1], sol[2], f
   else
       #UseImplicitForm
       garec(A, I, B, Q, R, S)
   end
end

"""
`garec(A, E, B, Q, R, S) -> (X, EVALS, F)` computes `X`, the hermitian/symmetric
stabilizing solution of the generalized continuous-time algebraic Riccati equation

   A'XE + E'XA - (A'XB+S)R^(-1)(B'XA+S') + Q = 0 ,

where `Q` and `R` are hermitian/symmetric matrices such that `R` is nonsingular, and
`E` is a nonsingular matrix.
`EVALS` is a vector containing the (stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing gain matrix `F = R^(-1)(B'XE+S')`.

Reference:
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.
"""
function garec(A, E, B, Q, R, S = 0)
    n = LinearAlgebra.checksquare(A)
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix of row dimension $n"))
    end
    if (E == I) || isempty(E) || E == Array{eltype(A),2}(I,n,n)
       eident = true
       E = I
    else
       if LinearAlgebra.checksquare(E) != n
          throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       end
       Et = LinearAlgebra.LAPACK.getrf!(copy(E))
       if LinearAlgebra.LAPACK.gecon!('1',Et[1],norm(E,1))  < eps(1.)
          error("E must be non-singular")
       end
       eident = false
    end
    if LinearAlgebra.checksquare(Q) !== n || ~ishermitian(Q)
       throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || ~ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if cond(R)*eps(1.) > 1.
       error("R must be non-singular")
    end
    if S == 0
       S = zeros(n,m)
    else
      if (n,m) !== size(S)
         throw(DimensionMismatch("S must be a $n x $m matrix"))
      end
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
    z = G.Q*[fill(false,m,n2); Matrix(I,n2,n2) ]

    iric = 1:n2
    L11 = [ A zeros(n,n) B; -Q -A' -S]*z
    P11 = [ E zeros(n,n); zeros(n,n) E']*z[iric,:]
    LPS = schur(L11,P11)
    select = (real.(LPS.α ./ LPS.β) .<= -sqrt(eps(1.)));

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


"""
`gared(A, E, B, Q, R, S) -> (X, EVALS, F)` computes `X`, the hermitian/symmetric
stabilizing solution of the generalized discrete-time algebraic Riccati equation

    A'XA - E'XE - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) generalized eigenvalues of the pair `(A-BF,E)`.
`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

Reference:
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.
"""
function gared(A, E, B, Q, R, S = 0)
    n = LinearAlgebra.checksquare(A)
    nb, m = size(B)
    if n !== nb
       throw(DimensionMismatch("B must be a matrix with row dimension $n"))
    end
    if (E == I) || isempty(E) || E == Array{eltype(A),2}(I,n,n)
       eident = true
       E = I
    else
       if LinearAlgebra.checksquare(E) != n
          throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       end
       eident = false
    end
    if LinearAlgebra.checksquare(Q) !== n || ~ishermitian(Q)
       throw(DimensionMismatch("Q must be a symmetric/hermitian matrix of dimension $n"))
    end
    if LinearAlgebra.checksquare(R) !== m || ~ishermitian(R)
       throw(DimensionMismatch("R must be a symmetric/hermitian matrix of dimension $m"))
    end
    if S == 0
       S = zeros(n,m)
    else
      if (n,m) !== size(S)
         throw(DimensionMismatch("S must be a $n x $m matrix"))
      end
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
    L2 = F.Q'*[-Q  E' -S; S' zeros(m,n) R]
    P2 = [zeros(n,n) F.R zeros(n,m)]

    G = qr(L2[n+1:n+m,:]')
    if cond(G.R) * eps(1.)  > 1.
       error("The extended symplectic pencil is not regular")
    end
    z = G.Q*Matrix(I,n2+m,n2+m)
    z = z[:,[m+1:m+n2; 1:m]]

    L1 = [ A zeros(n,n) B; L2[1:n,:]]*z
    P1 = [ E zeros(n,n+m); P2]*z

    iric = 1:n2
    PLS = schur(P1[iric,iric],L1[iric,iric])
    select = abs.(PLS.α) .> abs.((1-sqrt(eps(1.)))*PLS.β)

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


"""
`ared(A, B, Q, R,S) -> (X, EVALS, F)` computes `X`, the hermitian/symmetric
stabilizing solution of the discrete-time algebraic Riccati equation

     A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0,

where `Q` and `R` are hermitian/symmetric matrices.
`EVALS` is a vector containing the (stable) generalized eigenvalues of `A-BF`.
`F` is the stabilizing gain matrix `F = (R+B'XB)^(-1)(B'XA+S')`.

Reference:
W.F. Arnold, III and A.J. Laub,
Generalized Eigenproblem Algorithms and Software for Algebraic Riccati Equations,
Proc. IEEE, 72:1746-1754, 1984.
"""
function ared(A, B, Q, R, S = 0)
    gared(A, I, B, Q, R, S)
end
