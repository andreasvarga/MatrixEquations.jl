"""
`utqu!(Q,U)` efficiently computes the symmetric/hermitian product `U'QU`,
where `Q` is a symmetric/hermitian matrix and `U` is a square matrix.
The resulting product overwrites `Q`.
"""
function utqu!(Q,U)
   n = LinearAlgebra.checksquare(Q)
   if ~ishermitian(Q)
      error("Q must be a symmetric/hermitian matrix")
   end
   if LinearAlgebra.checksquare(U) != n
      throw(DimensionMismatch("U must be a matrix of dimension $n x $n"))
   end

   idiag = diagind(Q)
   Q[idiag] = Q[idiag]/2
   if isa(U,Adjoint)
      tmp = U.parent*(UpperTriangular(Q)*U)
   else
      tmp = Adjoint(U)*(UpperTriangular(Q)*U)
   end
   #Q = tmp+tmp'
   for j = 1:n
      Q[j,j] = tmp[j,j]+tmp[j,j]'
      for i = j+1:n
         Q[i,j] = tmp[i,j]+tmp[j,i]'
         Q[j,i] = Q[i,j]'
      end
   end
   return Q
end
"""
`X = utqu(Q,U)` efficiently computes the symmetric/hermitian product `U'QU`,
where Q is symmetric/hermitian matrix.
"""
function utqu(Q,U)
   n = LinearAlgebra.checksquare(Q)
   if ~ishermitian(Q)
      error("Q must be a symmetric/hermitian matrix")
   end
   adj = isa(U,Adjoint)
   if adj
      m, n1 = size(U.parent)
      if n1 != n
         throw(DimensionMismatch("U must be a matrix of column dimension $n"))
      end
   else
      n1, m = size(U)
      if n1 != n
         throw(DimensionMismatch("U must be a matrix of row dimension $n"))
      end
   end

   t = triu(Q)
   idiag = diagind(Q)
   t[idiag] = t[idiag]/2
   if adj
      t = U.parent*(UpperTriangular(t)*U)
   else
      t = Adjoint(U)*(UpperTriangular(t)*U)
   end
   #X = t+t'
   X = similar(t)
   for j = 1:m
      X[j,j] = t[j,j]+t[j,j]'
      for i = j+1:m
         X[i,j] = t[i,j]+t[j,i]'
         X[j,i] = X[i,j]'
      end
   end
   return X
end
