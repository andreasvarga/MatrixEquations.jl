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
   m == n || (return false)
   m == 1 && (return true)
   eltype(A)<:Complex && (return istriu(A))
   m == 2 && (return true)
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
"""
     isschur(A::AbstractMatrix, B::AbstractMatrix) -> Bool

Test whether `(A,B)` is a pair of square matrices in a real or complex
generalized Schur form.
In the real case, it is only tested whether `B` is upper triangular and `A` is
a quasi upper triangular matrix, which may have 2x2 diagonal blocks, which
however must not correspond to complex conjugate eigenvalues.
In the complex case, it is tested if `A` and `B` are both upper triangular.

"""
function isschur(A::AbstractMatrix,B::AbstractMatrix)
   ma, na = size(A)
   mb, nb = size(B)
   if eltype(A) != eltype(B) || ma != na || mb != nb || na != nb
      return false
   end
   return isschur(A) && istriu(B)
end
function sfstruct(A)
   # determine the structure of the generalized real Schur form
   n = size(A,1)
   ba = fill(1,n)
   p = 0
   i = 1
   while i < n
      p += 1
      iszero(A[i+1,i]) ? (i += 1) : (ba[p] = 2; i += 2)
   end
   i == n && (p += 1)
   return resize!(ba,p), p
   #return ba[1:p], p
end
"""
    utqu!(Q,U) -> Q

Compute efficiently the symmetric/hermitian product `U'QU -> Q`,
where `Q` is a symmetric/hermitian matrix and `U` is a square matrix.
The resulting product overwrites `Q`.
"""
function utqu!(Q,U)
   n = LinearAlgebra.checksquare(Q)
   ishermitian(Q) || error("Q must be a symmetric/hermitian matrix")
   LinearAlgebra.checksquare(U) == n ||
      throw(DimensionMismatch("U must be a matrix of dimension $n x $n"))

   T = promote_type(eltype(Q),eltype(U))
   if !(T <: BlasFloat) 
      if T == eltype(Q)
         Q[:,:] = U'*Q*U
         return 
      else
         error("TypeError: same type expected for Q and U ")
      end
   end


   Qd = view(Q,diagind(Q))
   rmul!(Qd,one(T)/2)
   if isa(U,Adjoint)
      mul!(Q, U.parent*UpperTriangular(Q), U) 
   else
      mul!(Q, Adjoint(U), UpperTriangular(Q)*U)
   end
   #Q = tmp+tmp'
   @inbounds  begin
      for j = 1:n
       Q[j,j] += Q[j,j]'
       for i = j+1:n
           Q[i,j] += Q[j,i]'
           Q[j,i] = Q[i,j]'
       end
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
   ishermitian(Q) || error("Q must be a symmetric/hermitian matrix")
   adj = isa(U,Adjoint)
   if adj
      m, n1 = size(U.parent)
      n1 == n || throw(DimensionMismatch("U must be a matrix of column dimension $n"))
   else
      n1, m = size(U)
      n1 == n || throw(DimensionMismatch("U must be a matrix of row dimension $n"))
   end

   T = promote_type(eltype(Q),eltype(U))
   if !(T <: BlasFloat)
      return U'*Q*U
   end

   t = UpperTriangular(Q)-Diagonal(Q[diagind(Q)]./2)
   X = similar(Q, T, m, m)
   if adj
      mul!(X, U.parent*t, U) 
   else
      mul!(X, Adjoint(U), t*U)
   end
   @inbounds  begin
   for j = 1:m
       X[j,j] += X[j,j]'
       for i = j+1:m
           X[i,j] += X[j,i]'
           X[j,i] = X[i,j]'
      end
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
    size(R,1) == n || throw(DimensionMismatch("updating matrix must fit size of upper triangular matrix"))
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
    size(R,1) == n || throw(DimensionMismatch("updating matrix must fit size of upper triangular matrix"))

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
function triu2vec(Q::AbstractArray{T}; rowwise::Bool = false, her::Bool = false) where T
   n = LinearAlgebra.checksquare(Q)
   her && !ishermitian(Q) && error("Q must be a symmetric/hermitian matrix")
   x = Array{T,1}(undef,Int(n*(n+1)/2))
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
function vec2triu(x::AbstractVector{T}; rowwise = false, her = false) where T
   k = length(x)
   n = (sqrt(1+8*k)-1)/2
   isinteger(n) ? n = Int(n) : error("The lenght of x must be of the form n(n+1)/2")
   her ? Q = Array{T,2}(undef,n,n) : Q = zeros(T,n,n) 
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
function utnormalize!(U::UpperTriangular{T},adj::Bool) where T 
   # Normalize an upper traiangular matrix U such that its diagonal elements are non-negative
   # using diagonal orthogonal or unitary transformations.
   ZERO = zero(real(T))
   n = size(U,1)
   if adj
      # Make the diagonal elements of U non-negative.
      if T <: Real
         for i = 1:n
            U[i,i] > ZERO || [@inbounds U[i,j] = -U[i,j] for j = i:n]
         end
      else
         for i = 1:n
             d = abs(U[i,i])
             (!iszero(d) && !(iszero(imag(U[i,i])) && real(U[i,i]) > ZERO)) && 
                     (tmp = conj(U[i,i])/d; [@inbounds U[i,j] *= tmp for j = i:n])
         end
      end
   else
      # Make the diagonal elements of U non-negative.
      if T <: Real
         for j = 1:n
            U[j,j] > ZERO || [@inbounds U[i,j] = -U[i,j] for i = 1:j]
         end
      else
         for j = 1:n
             d = abs(U[j,j])
             (!iszero(d) && (iszero(imag(U[j,j])) && real(U[j,j]) > ZERO)) && 
                    (tmp = conj(U[j,j])/d; [@inbounds U[i,j] *= tmp for i = 1:j])
         end
      end
   end
   return U
end
@inline function luslv!(A::AbstractMatrix{T}, B::AbstractVector{T}) where T
   #
   #  fail = luslv!(A,B)
   # 
   # This function is a speed-oriented implementation of a Gaussion-elimination based
   # solver of small order linear equations of the form A*X = B. The computed solution X 
   # overwrites the vector B, while the resulting A contains in its upper triangular part, 
   # the upper triangular factor U of its LU decomposition. 
   # The diagnostic output parameter fail, of type Bool, is set to false in the case 
   # of normal return or is set to true if the exact singularity of A is detected 
   # or if the resulting B has non-finite components.
   #
   n = length(B) 
   @inbounds begin
         for k = 1:n
            # find index max
            kp = k
            if k < n
                amax = abs(A[k, k])
                for i = k+1:n
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            iszero(A[kp,k]) && return true
            if k != kp
               # Interchange
               for i = 1:n
                   tmp = A[k,i]
                   A[k,i] = A[kp,i]
                   A[kp,i] = tmp
               end
               tmp = B[k]
               B[k] = B[kp]
               B[kp] = tmp
            end
            # Scale first column
            Akkinv = inv(A[k,k])
            i1 = k+1:n
            Ak = view(A,i1,k)
            rmul!(Ak,Akkinv)
            # Update the rest of A and B
            for j = k+1:n
                axpy!(-A[k,j],Ak,view(A,i1,j))
            end
            axpy!(-B[k],Ak,view(B,i1))
         end
         ldiv!(UpperTriangular(A), B)
         return any(!isfinite, B)
   end
end

