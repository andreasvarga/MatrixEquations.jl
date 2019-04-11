"""
`X = sylvc(A,B,Q)` solves the continuous Sylvester matrix equation

                AX + XB = Q

using the Schur form based approach, where `A` and `B` are
square matrices. `A` and `-B` must not have common eigenvalues.
"""
function sylvc(A,B,Q)
   realABQ = isreal(A) & isreal(B) & isreal(Q)
   if isa(A,Union{Adjoint,Transpose})
      if realABQ
         RA, QA = schur(A.parent)
         TA = 'T'
      else
         RA, QA = schur(complex(A.parent))
         TA = 'C'
      end
   else
      if realABQ
         RA, QA = schur(A)
      else
         RA, QA = schur(complex(A))
      end
      TA = 'N'
   end
   if isa(B,Union{Adjoint,Transpose})
      if realABQ
         RB, QB = schur(B.parent)
         TB = 'T'
      else
         RB, QB = schur(complex(B.parent))
         TB = 'C'
      end
   else
      if realABQ
         RB, QB = schur(B)
      else
         RB, QB = schur(complex(B))
      end
      TB = 'N'
   end
   D = adjoint(QA) * (Q*QB)
   if realABQ
      Y, scale = LAPACK.trsyl!(TA,TB, RA, RB, D)
   else
      Y, scale = LAPACK.trsyl!(TA,TB, RA, RB, complex(D))
   end
   rmul!(QA*(Y * adjoint(QB)), inv(scale))
end
"""
`X = sylvckr(A,B,Q)` solves the continuous Sylvester matrix equation

                AX + XB = Q

using the Kronecker product expansion of equations, where `A` and `B` are
square matrices. `A` and `-B` must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvckr(A,B,Q)

   m, n = size(Q);
   if [m; n] != LinearAlgebra.checksquare(A,B)
      throw(DimensionMismatch("A, B and Q have incompatible dimensions"))
   end
   reshape((kron(Array{eltype(A),2}(I,n,n),A)+
            kron(transpose(B),Array{eltype(B),2}(I,m,m)))\(Q[:]),m,n)
end
"""
`X = sylvdkr(A,B,Q)` solves the discrete Sylvester matrix equation

                AXB - X = Q

using the Kronecker product expansion of equations, where `A` and `B` are
square matrices. `A` and `B` must not have common reciprocal eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvdkr(A,B,Q)

    m, n = size(Q);
    if [m; n] != LinearAlgebra.checksquare(A,B)
       throw(DimensionMismatch("A, B and Q have incompatible dimensions"))
    end
    reshape((kron(transpose(B),A)-I)\(Q[:]),m,n)
end
"""
`X = sylvdkr(A,B,C,D,Q,isgn=1)` solves the generalized discrete Sylvester matrix equation

                AXB + isgn*CXD = Q

using the Kronecker product expansion of equations, where `A`, `B`, `C` and `D` are
square matrices and `isgn = 1` or `isgn = -1`.
The pencils `A-λC` and `isgn*D+λB` must be regular and must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvdkr(A,B,C,D,Q,isgn=1)

    m, n = size(Q);
    if [m; n; m; n] != LinearAlgebra.checksquare(A,B,C,D)
       throw(DimensionMismatch("A, B, C, D and Q have incompatible dimensions"))
    end
    if abs(isgn) != 1
       error("isgn must be either 1 or -1")
    end
    if isgn > 0
       reshape((kron(transpose(B),A)+kron(transpose(D),C))\(Q[:]),m,n)
    else
       reshape((kron(transpose(B),A)-kron(transpose(D),C))\(Q[:]),m,n)
    end
end
