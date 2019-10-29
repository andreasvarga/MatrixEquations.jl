"""
     isschur(A::AbstractMatrix) -> Bool

Test whether `A` is a square matrix in a real or complex Schur form.
In the real case, it is only tested whether A is a quasi upper triangular matrix,
which may have 2x2 diagonal blocks, which however must not correspond to
complex conjugate eigenvalues. In the complex case, it is tested if `A`
is upper triangular.
"""
function isschur(A)
   @assert !Base.has_offset_axes(A)
   m, n = size(A)
   if m != n
      return false
   end
   if m == 1
      return true
   end
   if eltype(A)<:Complex
      return istriu(A)
   else
      if m == 2
         return true
      end
      if istriu(A,-1)
         for i = 1:m-2
             if !iszero(A[i+1,i]) & !iszero(A[i+2,i+1])
                return false
             end
          end
          return true
      else
          return false
      end
   end
end
"""
     isschur(A::AbstractMatrix, B::AbstractMatrix) -> Bool

Test whether `(A,B)` is a pair of square matrices in a real or complex
generalized Schur form.
In the real case, it is only tested whether `B` is upper triangular and `A` is
a quasi upper triangular matrix, which may have 2x2 diagonal blocks, which
however must not correspond to complex conjugate eigenvalues.
In the complex case, it is tested if `A` and `B` are both upper triangular.

"""
function isschur(A,B)
   ma, na = size(A)
   mb, nb = size(B)
   if eltype(A) != eltype(B) || ma != na || mb != nb || na != nb
      return false
   end
   return isschur(A) && istriu(B)
end
"""
    utqu!(Q,U) -> Q

Compute efficiently the symmetric/hermitian product `U'QU -> Q`,
where `Q` is a symmetric/hermitian matrix and `U` is a square matrix.
The resulting product overwrites `Q`.
"""
function utqu!(Q,U)
   n = LinearAlgebra.checksquare(Q)
   if !ishermitian(Q)
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
    X = utqu(Q,U)

Compute efficiently the symmetric/hermitian product `X = U'QU`,
where Q is a symmetric/hermitian matrix.
"""
function utqu(Q,U)
   n = LinearAlgebra.checksquare(Q)
   if !ishermitian(Q)
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

"""
    qrupdate!(R, Y) -> R

Update the upper triangular factor `R` by the
upper triangular factor of the QR factorization  of `[ R; Y' ]`, where `Y` is a
low-rank matrix `Y` (typically with one or two columns). The computation of `R`
only uses `O(n^2)` operations (`n` is the size of `R`). The input matrix `R` is
updated in place and the matrix `Y` is destroyed during the computation.
"""
function qrupdate!(R, Y)
    n, m = size(Y)
    if size(R, 1) != n
      throw(DimensionMismatch("updating matrix must fit size of upper triangular matrix"))
    end
    #Y = conj(Y)
    for k = 1:m
        for i = 1:n

            # Compute Givens rotation
            #c, s, r = LinearAlgebra.givensAlgorithm(R[i,i], conj(Y[i,k]))
            c, s, r = LinearAlgebra.givensAlgorithm(R[i,i], Y[i,k])

            # Store new diagonal element
            R[i,i] = r

            # Update remaining elements in row/column
            for j = i + 1:n
                Rij = R[i,j]
                yjk  = Y[j,k]
                R[i,j]  =   c*Rij + s*yjk
                Y[j,k]    = -s'*Rij + c*yjk
            end
        end
    end
    return R
end
"""

    rqupdate!(R, Y) -> R

Update the upper triangular factor `R` by the
upper triangular factor of the RQ factorization  of `[ Y R]`, where `Y` is a
low-rank matrix `Y` (typically with one or two columns). The computation of `R`
only uses `O(n^2)` operations (`n` is the size of `R`). The input matrix `R` is
updated in place and the matrix `Y` is destroyed during the computation.
"""
function rqupdate!(R, Y)
    n, m = size(Y)
    if size(R, 1) != n
        throw(DimensionMismatch("updating matrix must fit size of upper triangular matrix"))
    end

    for k = 1:m
        for j = n:-1:1

            # Compute Givens rotation
            c, s, r = LinearAlgebra.givensAlgorithm(R[j,j], Y[j,k])

            # Store new diagonal element
            R[j,j] = r

            # Update remaining elements in row/column
            for i = 1: j - 1
                Rij = R[i,j]
                yik  = Y[i,k]
                R[i,j]  =   c*Rij + s*yik
                Y[i,k]    = -s'*Rij + c*yik
            end
        end
    end
    return R
end
"""
    x = triu2vec(Q; rowwise = false, her = false)

Reshape the upper triangular part of the `nxn` array `Q` as a one-dimensional column 
vector `x` with `n(n+1)/2` elements. `Q` is assumed symmetric/hermitian if `her = true`.
The elements of `x` correspond to stacking the elements of successive columns
of the upper triangular part of `Q`, if `rowwise = false`, or stacking the elements
of successive rows of the upper triangular part of `Q`, if `rowwise = true`.
"""
function triu2vec(Q::AbstractArray; rowwise = false, her = false)
   n = LinearAlgebra.checksquare(Q)
   if her && !ishermitian(Q)
      error("Q must be a symmetric/hermitian matrix")
   end
   x = Array{eltype(Q),1}(undef,Int(n*(n+1)/2))
   k = 1
   if rowwise
      for i = 1:n
         for j = i:n
             x[k] = Q[i,j]
             k += 1
         end
      end
   else
      for j = 1:n
         for i = 1:j
             x[k] = Q[i,j]
             k += 1
         end
      end
   end
   return x
end
"""
    Q = vec2triu(x; rowwise = false, her = false)

Build from a one-dimensional column vector `x` with `n(n+1)/2` elements
an `nxn` upper triangular matrix `Q` if `her = false` or an `nxn` symetric/hermitian 
array `Q` if `her = true`.
The elements of `x` correspond to stacking the elements of successive columns
of the upper triangular part of `Q`, if `rowwise = false`, or stacking the elements
of successive rows of the upper triangular part of `Q`, if `rowwise = true`.
"""
function vec2triu(x::AbstractVector; rowwise = false, her = false)
   k = length(x)
   n = (sqrt(1+8*k)-1)/2
   isinteger(n) ? n = Int(n) : error("The lenght of x must be of the form n(n+1)/2")
   her ? Q = Array{eltype(x),2}(undef,n,n) : Q = zeros(eltype(x),n,n) 
   k = 1
   if rowwise
      for i = 1:n
         Q[i,i] = x[k]
         k += 1
         for j = i+1:n
            Q[i,j] = x[k]
            k += 1
         end
      end
   else
      for j = 1:n
         for i = 1:j-1
            Q[i,j] = x[k]
            k += 1
         end
         Q[j,j] = x[k]
         k += 1
      end
   end
   if her
      for j = 1:n
         for i = j+1:n
            Q[i,j] = Q[j,i]'
         end
      end
   end
   return Q
end
