"""
    X = sylvckr(A,B,C)

Solve the continuous Sylvester matrix equation

                AX + XB = C

using the Kronecker product expansion of equations. `A` and `B` are
square matrices, and `A` and `-B` must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvckr(A,B,C)

   m, n = size(C);
   [m; n] == LinearAlgebra.checksquare(A,B) || 
             throw(DimensionMismatch("A, B and Q have incompatible dimensions"))
   reshape((kron(Array{eltype(A),2}(I,n,n),A)+
            kron(transpose(B),Array{eltype(B),2}(I,m,m)))\(C[:]),m,n)
end
"""
    X = sylvdkr(A,B,C)

Solve the discrete Sylvester matrix equation

                AXB + X = C

using the Kronecker product expansion of equations. `A` and `B` are
square matrices, and `A` and `-B` must not have common reciprocal eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvdkr(A,B,C)

    m, n = size(C);
    [m; n] == LinearAlgebra.checksquare(A,B) ||
              throw(DimensionMismatch("A, B and C have incompatible dimensions"))
     reshape((kron(transpose(B),A)+I)\(C[:]),m,n)
end
"""
    X = gsylvkr(A,B,C,D,E)

Solve the generalized Sylvester matrix equation

                AXB + CXD = E

using the Kronecker product expansion of equations. `A`, `B`, `C` and `D` are
square matrices. The pencils `A-λC` and `D+λB` must be regular and
must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function gsylvkr(A,B,C,D,E)

    m, n = size(E);
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,C,D) ||
                    throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
    reshape((kron(transpose(B),A)+kron(transpose(D),C))\(E[:]),m,n)
end
"""
    sylvsyskr(A,B,C,D,E,F) -> (X,Y)

Solve the Sylvester system of matrix equations

                AX + YB = C
                DX + YE = F

using the Kronecker product expansion of equations. `(A,D)`, `(B,E)` are
pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvsyskr(A,B,C,D,E,F)

    m, n = size(C);
    (m == size(F,1) && n == size(F,2)) ||
          throw(DimensionMismatch("C and F must have the same dimensions"))
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,D,E) ||
                    throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
     z = [ kron(Array{eltype(A),2}(I,n,n),A) kron(transpose(B),Array{eltype(B),2}(I,m,m)) ;
          kron(Array{eltype(D),2}(I,n,n),D) kron(transpose(E),Array{eltype(E),2}(I,m,m))] \ [C[:];F[:]]
    (reshape(z[1:m*n],m,n),reshape(z[m*n+1:end],m,n))
end
"""
    dsylvsyskr(A,B,C,D,E,F) -> (X,Y)

Solve the dual Sylvester system of matrix equations

       AX + DY = C
       XB + YE = F 

using the Kronecker product expansion of equations. `(A,D)`, `(B,E)` are
pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function dsylvsyskr(A,B,C,D,E,F)

    m, n = size(C);
    (m == size(F,1) && n == size(F,2)) ||
          throw(DimensionMismatch("C and F must have the same dimensions"))
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,D,E) ||
                    throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
    z = [ kron(Array{eltype(A),2}(I,n,n),A) kron(Array{eltype(D),2}(I,n,n),D);
          kron(transpose(B),Array{eltype(B),2}(I,m,m)) kron(transpose(E),Array{eltype(E),2}(I,m,m))] \ [C[:];F[:]]
    (reshape(z[1:m*n],m,n),reshape(z[m*n+1:end],m,n))
end
