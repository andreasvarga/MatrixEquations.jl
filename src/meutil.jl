"""
`utqu!(Q,U;adj = false)` computes the symmetric/hermitian product `op(U)'Qop(U)`,
where `Q` is a symmetric/hermitian matrix, `U` is a square matrix, and
`op(U) = U` if `adj = false` and `op(U) = U'` if `adj = true`.
The resulting product overwrites `Q`.
"""
function utqu!(Q,U;adj = false)
   n = LinearAlgebra.checksquare(Q)
   if ~ishermitian(Q)
      error("Q must be a symmetric/hermitian matrix")
   end
   if LinearAlgebra.checksquare(U) != n
      throw(DimensionMismatch("U must be a matrix of dimension $n x $n"))
   end

   idiag = diagind(Q)
   Q[idiag] = Q[idiag]/2
   if adj
      #tmp = (U*UpperTriangular(triu(Q)))*U'
      tmp = (U*UpperTriangular(Q))*U'
   else
      #tmp = U'*(UpperTriangular(triu(Q))*U)
      tmp = U'*(UpperTriangular(Q)*U)
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
`X = utqu(Q,U;adj = false)` computes `X = op(U)'Qop(U)`, where `Q` is a
symmetric/hermitian matrix, `op(U) = U` if `adj = false` and
`op(U) = U'` if `adj = true`.
"""
function utqu(Q,U;adj = false)
   n = LinearAlgebra.checksquare(Q)
   if ~ishermitian(Q)
      error("Q must be a symmetric/hermitian matrix")
   end
   if adj
      m, n1 = size(U)
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
      t = (U*UpperTriangular(t))*U'
   else
      t = U'*(UpperTriangular(t)*U)
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
