"""
     isschur(A::AbstractMatrix) -> Bool

Test whether `A` is a square matrix in a real or complex Schur form.
In the real case, it is only tested whether A is a quasi upper triangular matrix,
which may have 2x2 diagonal blocks, which however must not correspond to
complex conjugate eigenvalues. In the complex case, it is tested if `A`
is upper triangular.
"""
function isschur(A)
   require_one_based_indexing(A)
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
       Q[j,j] += conj(Q[j,j])
       for i = j+1:n
           Q[i,j] += conj(Q[j,i])
           Q[j,i] = conj(Q[i,j])
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
   # if !(T <: BlasFloat)
   #    return U'*Q*U
   # end

   t = UpperTriangular(Q)-Diagonal(Q[diagind(Q)]./2)
   X = similar(Q, T, m, m)
   if adj
      mul!(X, U.parent*t, U)
   else
      mul!(X, Adjoint(U), t*U)
   end
   @inbounds  begin
   for j = 1:m
       X[j,j] += conj(X[j,j])
       for i = j+1:m
           X[i,j] += conj(X[j,i])
           X[j,i] = conj(X[i,j])
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
an `nxn` upper triangular matrix `Q` if `her = false` or an `nxn` symmetric/hermitian
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
            Q[i,j] = conj(Q[j,i])
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
               U[i,i] > ZERO || (rmul!(view(U,i,i:n),-1))
            end
      else
         for i = 1:n
            d = abs(U[i,i])
            (!iszero(d) && !(iszero(imag(U[i,i])) && real(U[i,i]) > ZERO)) && 
               (tmp = conj(U[i,i])/d; rmul!(view(U,i,i:n),tmp))
               #(tmp = conj(U[i,i])/d; [@inbounds U[i,j] *= tmp for j = i:n])
         end
      end
   else
      # Make the diagonal elements of U non-negative.
      if T <: Real
         for j = 1:n
            U[j,j] > ZERO || (rmul!(view(U,1:j,j),-1))
         end
      else
         for j = 1:n
            d = abs(U[j,j])
            (!iszero(d) && (iszero(imag(U[j,j])) && real(U[j,j]) > ZERO)) && 
               (tmp = conj(U[j,j]) / d; rmul!(view(U, 1:j, j), tmp))
               #(tmp = conj(U[j,j])/d; [@inbounds U[i,j] *= tmp for i = 1:j])
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
# compute the eigenvalues of a real 2x2 [a b; c d] in e1r, e1i, e2r, e2i 
# extracted from the translation from LAPACK::dlanv2 by Ralph Smith
# Copyright:
# Univ. of Tennessee
# Univ. of California Berkeley
# Univ. of Colorado Denver
# NAG Ltd.
function _lanv2(a::T,b::T,c::T,d::T) where {T <: Real}
   ZERO = zero(T)
   ONE = one(T)
   sgn(x) = (x < 0) ? -ONE : ONE # fortran sign differs from Julia
   half = ONE / 2
   small = 4eps(T) # how big discriminant must be for easy reality check
   if c==0
   elseif b==0
       # swap rows/cols
       a,b,c,d = d,-c,ZERO,a
   elseif ((a-d) == 0) && (b*c < 0)
       # nothing to do
   else
       asubd = a-d
       p = half*asubd
       bcmax = max(abs(b),abs(c))
       bcmis = min(abs(b),abs(c)) * sgn(b) * sgn(c)
       scale = max(abs(p), bcmax)
       z = (p / scale) * p + (bcmax / scale) * bcmis
       # if z is of order machine accuracy: postpone decision
       if z >= small
           # real eigenvalues
           z = p + sqrt(scale) * sqrt(z) * sgn(p)
           a = d + z
           d -= (bcmax / z) * bcmis
           b -= c
           c = ZERO
       else
           # complex or almost equal real eigenvalues
           σ = b + c
           τ = hypot(σ, asubd)
           cs = sqrt(half * (ONE + abs(σ) / τ))
           sn = -(p / (τ * cs)) * sgn(σ)
           # apply rotations
           aa = a*cs + b*sn
           bb = -a*sn + b*cs
           cc = c*cs + d*sn
           dd = -c*sn + d*cs
           a = aa*cs + cc*sn
           b = bb*cs + dd*sn
           c = -aa*sn + cc*cs
           d = -bb*sn + dd*cs
           midad = half * (a+d)
           a = midad
           d = a
           if (c != 0)
               if (b != 0)
                   if b*c >= 0
                       # real eigenvalues
                       sab = sqrt(abs(b))
                       sac = sqrt(abs(c))
                       p = sab*sac*sgn(c)
                       a = midad + p
                       d = midad - p
                       b -= c
                       c = 0
                   end
               else
                   b,c = -c,ZERO
               end
           end
       end
   end

   w1r,w2r = a, d
   if c==0
       w1i,w2i = ZERO,ZERO
   else
       rti = sqrt(abs(b))*sqrt(abs(c))
       w1i,w2i = rti,-rti
   end
   return w1r,w1i,w2r,w2i
end
# compute the Schur decomposition of a real 2x2 matrix H2 in standard form
# return the eigenvalues in e1r, e1i, e2r, e2i and 
# the elements cs and sn of the corresponding Givens rotation
# Translated from LAPACK::dlanv2 by Ralph Smith within the GenericSchur.jl
function _lanv2!(H2::StridedMatrix{T}) where {T <: Real}
   a,b,c,d = H2[1,1], H2[1,2], H2[2,1], H2[2,2]
   ZERO = zero(T)
   ONE = one(T)
   sgn(x) = (x < 0) ? -ONE : ONE # fortran sign differs from Julia
   half = ONE / 2
   small = 4eps(T) # how big discriminant must be for easy reality check
   if c==0
       cs = ONE
       sn = ZERO
   elseif b==0
       # swap rows/cols
       cs = ZERO
       sn = ONE
       a,b,c,d = d,-c,ZERO,a
   elseif ((a-d) == 0) && (b*c < 0)
       # nothing to do
       cs = ONE
       sn = ZERO
   else
       asubd = a-d
       p = half*asubd
       bcmax = max(abs(b),abs(c))
       bcmis = min(abs(b),abs(c)) * sgn(b) * sgn(c)
       scale = max(abs(p), bcmax)
       z = (p / scale) * p + (bcmax / scale) * bcmis
       # if z is of order machine accuracy: postpone decision
       if z >= small
           # real eigenvalues
           z = p + sqrt(scale) * sqrt(z) * sgn(p)
           a = d + z
           d -= (bcmax / z) * bcmis
           τ = hypot(c,z)
           cs = z / τ
           sn = c / τ
           b -= c
           c = ZERO
       else
           # complex or almost equal real eigenvalues
           σ = b + c
           τ = hypot(σ, asubd)
           cs = sqrt(half * (ONE + abs(σ) / τ))
           sn = -(p / (τ * cs)) * sgn(σ)
           # apply rotations
           aa = a*cs + b*sn
           bb = -a*sn + b*cs
           cc = c*cs + d*sn
           dd = -c*sn + d*cs
           a = aa*cs + cc*sn
           b = bb*cs + dd*sn
           c = -aa*sn + cc*cs
           d = -bb*sn + dd*cs
           midad = half * (a+d)
           a = midad
           d = a
           if (c != 0)
               if (b != 0)
                   if b*c >= 0
                       # real eigenvalues
                       sab = sqrt(abs(b))
                       sac = sqrt(abs(c))
                       p = sab*sac*sgn(c)
                       τ = ONE / sqrt(abs(b+c))
                       a = midad + p
                       d = midad - p
                       b -= c
                       c = 0
                       cs1 = sab*τ
                       sn1 = sac*τ
                       cs, sn = cs*cs1 - sn*sn1, cs*sn1 + sn*cs1
                   end
               else
                   b,c = -c,ZERO
                   cs,sn = -sn,cs
               end
           end
       end
   end

   if c==0
       w1r,w1i,w2r,w2i = a,ZERO,d,ZERO
   else
       rti = sqrt(abs(b))*sqrt(abs(c))
       w1r,w1i,w2r,w2i = a,rti,d,-rti
   end
   H2[1,1], H2[1,2], H2[2,1], H2[2,2] = a,b,c,d

   return w1r,w1i,w2r,w2i,cs,sn
end
"""
A "safe" version of `floatmin`, such that `1/sfmin` does not overflow.
(used in the GenericSchur package of Ralph Smith)
"""
function _safemin(T)
   sfmin = floatmin(T)
   small = one(T) / floatmax(T)
   if small >= sfmin
       sfmin = small * (one(T) + eps(T))
   end
   sfmin
end
function _lag2(a::StridedMatrix{T},b::StridedMatrix{T},safmin::T) where {T <: Real}
#
#    _lag2(A, B, SAFMIN) -> (SCALE1, SCALE2, WR1, WR2, WI)
#
# Compute the eigenvalues of a 2-by-2 generalized real eigenvalue problem for
# the matrix pair `(A,B)`, with scaling as necessary to avoid over-/underflow.
# `SAFMIN` is the smallest positive number s.t. `1/SAFMIN` does not overflow.
# If `WI = 0`, `WR1/SCALE1` and `WR2/SCALE2` are the resulting real eigenvalues, while
# if `WI <> 0`, then `(WR1+/-im*WI)/SCALE1` are the resulting complex eigenvalues.
# Conversion of the LAPACK subroutines DLAG2/SLAG2.

   ZERO = zero(T)
   ONE = one(T)
   two = 2*ONE
   half = ONE/2
   fuzzy1 = ONE+ONE/10000

   rtmin = sqrt( safmin )
   rtmax = ONE / rtmin
   safmax = ONE / safmin
#
#     Scale A
#
   anorm = max( abs( a[ 1, 1 ] )+abs( a[ 2, 1 ] ),
                abs( a[ 1, 2 ] )+abs( a[ 2, 2 ] ), safmin )
   ascale = ONE / anorm
   a11 = ascale*a[ 1, 1 ]
   a21 = ascale*a[ 2, 1 ]
   a12 = ascale*a[ 1, 2 ]
   a22 = ascale*a[ 2, 2 ]
#
#     Perturb B if necessary to insure non-singularity
#
   b11 = b[ 1, 1 ]
   b12 = b[ 1, 2 ]
   b22 = b[ 2, 2 ]
   bmin = rtmin*max( abs( b11 ), abs( b12 ), abs( b22 ), rtmin )
   abs( b11 ) < bmin && (b11 = bmin*sign(b11))
   abs( b22 ) < bmin && (b22 = bmin*sign(b22))
#
#     Scale B
#
   bnorm = max( abs( b11 ), abs( b12 )+abs( b22 ), safmin )
   bsize = max( abs( b11 ), abs( b22 ) )
   bscale = ONE / bsize
   b11 = b11*bscale
   b12 = b12*bscale
   b22 = b22*bscale
#
#     Compute larger eigenvalue by method described by C. van Loan
#
#     ( AS is A shifted by -SHIFT*B )
#
   binv11 = ONE / b11
   binv22 = ONE / b22
   s1 = a11*binv11
   s2 = a22*binv22
   if abs( s1 ) <= abs( s2 ) 
      as12 = a12 - s1*b12
      as22 = a22 - s1*b22
      ss = a21*( binv11*binv22 )
      abi22 = as22*binv22 - ss*b12
      pp = half*abi22
      shift = s1
   else
      as12 = a12 - s2*b12
      as11 = a11 - s2*b11
      ss = a21*( binv11*binv22 )
      abi22 = -ss*b12
      pp = half*( as11*binv11+abi22 )
      shift = s2
   end
   qq = ss*as12
   if abs( pp*rtmin ) >= ONE
      discr = ( rtmin*pp )^2 + qq*safmin
      r = sqrt( abs( discr ) )*rtmax
   else
      if pp^2+abs( qq ) <= safmin 
         discr = ( rtmax*pp )^2 + qq*safmax
         r = sqrt( abs( discr ) )*rtmin
      else
         discr = pp^2 + qq
         r = sqrt( abs( discr ) )
      end
   end
#
#     Note: the test of R in the following IF is to cover the case when
#           DISCR is small and negative and is flushed to zero during
#           the calculation of R.  On machines which have a consistent
#           flush-to-zero threshold and handle numbers above that
#           threshold correctly, it would not be necessary.
#
   if discr >= ZERO  ||  r == ZERO 
      t = r*sign(pp)
      sum = pp + t
      diff = pp - t
      wbig = shift + sum
#
#        Compute smaller eigenvalue
#
      wsmall = shift + diff
      if half*abs( wbig ) > max( abs( wsmall ), safmin ) 
         wdet = ( a11*a22-a12*a21 )*( binv11*binv22 )
         wsmall = wdet / wbig
      end
#
#        Choose (real) eigenvalue closest to 2,2 element of A*B**(-1)
#        for WR1.
#
      if pp > abi22 
         wr1 = min( wbig, wsmall )
         wr2 = max( wbig, wsmall )
      else
         wr1 = max( wbig, wsmall )
         wr2 = min( wbig, wsmall )
      end
      wi = ZERO
   else
#
#        Complex eigenvalues
#
      wr1 = shift + pp
      wr2 = wr1
      wi = r
   end
#
#     Further scaling to avoid underflow and overflow in computing
#     SCALE1 and overflow in computing w*B.
#
#     This scale factor (WSCALE) is bounded from above using C1 and C2,
#     and from below using C3 and C4.
#        C1 implements the condition  s A  must never overflow.
#        C2 implements the condition  w B  must never overflow.
#        C3, with C2,
#           implement the condition that s A - w B must never overflow.
#        C4 implements the condition  s    should not underflow.
#        C5 implements the condition  max(s,|w|) should be at least 2.
#
   c1 = bsize*( safmin*max( ONE, ascale ) )
   c2 = safmin*max( ONE, bnorm )
   c3 = bsize*safmin
   if ascale <= ONE && bsize <= ONE 
      c4 = min( ONE, ( ascale / safmin )*bsize )
   else
      c4 = ONE
   end
   if ascale <= ONE  ||  bsize <= ONE 
      c5 = min( ONE, ascale*bsize )
   else
      c5 = ONE
   end
#
#     Scale first eigenvalue
#
   wabs = abs( wr1 ) + abs( wi )
   wsize = max( safmin, c1, fuzzy1*( wabs*c2+c3 ),
                min( c4, half*max( wabs, c5 ) ) )
   if wsize != ONE 
      wscale = ONE / wsize
      if wsize > ONE 
         scale1 = ( max( ascale, bsize )*wscale ) * min( ascale, bsize )
      else
         scale1 = ( min( ascale, bsize )*wscale ) * max( ascale, bsize )
      end
      wr1 = wr1*wscale
      if wi != ZERO 
         wi = wi*wscale
         wr2 = wr1
         scale2 = scale1
      end
   else
      scale1 = ascale*bsize
      scale2 = scale1
   end
#
#     Scale second eigenvalue (if real)
#
   if wi == ZERO 
      wsize = max( safmin, c1, fuzzy1*( abs( wr2 )*c2+c3 ),
                   min( c4, half*max( abs( wr2 ), c5 ) ) )
      if wsize != ONE 
         wscale = ONE / wsize
         if wsize > ONE 
            scale2 = ( max( ascale, bsize )*wscale ) * min( ascale, bsize )
         else
            scale2 = ( min( ascale, bsize )*wscale ) * max( ascale, bsize )
         end
         wr2 = wr2*wscale
      else
         scale2 = ascale*bsize
      end
   end
#
#     End of DLAG2
#
   return scale1, scale2, wr1, wr2, wi
end
function _ladiv(A, B, C, D)
#
#    ladiv(A, B, C, D) -> (P, Q)
#
# Perform the complex division in real arithmetic
#
#  P + iQ = (A+iB)/(C+iD)
#
# by avoiding unnecessary overflow.
# Interface to the LAPACK subroutines DLADIV/SLADIV.
   t = complex(A,B) / complex(C,D)
   return real(t), imag(t)
end
# SUBROUTINE dladiv( A, B, C, D, P, Q )
# *
# *  -- LAPACK auxiliary routine --
# *  -- LAPACK is a software package provided by Univ. of Tennessee,    --
# *  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
# *
# *     .. Scalar Arguments ..
#       DOUBLE PRECISION   A, B, C, D, P, Q
# *     ..
# *
# *  =====================================================================
# *
# *     .. Parameters ..
#       DOUBLE PRECISION   BS
#       parameter( bs = 2.0d0 )
#       DOUBLE PRECISION   HALF
#       parameter( half = 0.5d0 )
#       DOUBLE PRECISION   TWO
#       parameter( two = 2.0d0 )
# *
# *     .. Local Scalars ..
#       DOUBLE PRECISION   AA, BB, CC, DD, AB, CD, S, OV, UN, BE, EPS
# *     ..
# *     .. External Functions ..
#       DOUBLE PRECISION   DLAMCH
#       EXTERNAL           dlamch
# *     ..
# *     .. External Subroutines ..
#       EXTERNAL           dladiv1
# *     ..
# *     .. Intrinsic Functions ..
#       INTRINSIC          abs, max
# *     ..
# *     .. Executable Statements ..
# *
#       aa = a
#       bb = b
#       cc = c
#       dd = d
#       ab = max( abs(a), abs(b) )
#       cd = max( abs(c), abs(d) )
#       s = 1.0d0
 
#       ov = dlamch( 'Overflow threshold' )
#       un = dlamch( 'Safe minimum' )
#       eps = dlamch( 'Epsilon' )
#       be = bs / (eps*eps)
 
#       IF( ab >= half*ov ) THEN
#          aa = half * aa
#          bb = half * bb
#          s  = two * s
#       END IF
#       IF( cd >= half*ov ) THEN
#          cc = half * cc
#          dd = half * dd
#          s  = half * s
#       END IF
#       IF( ab <= un*bs/eps ) THEN
#          aa = aa * be
#          bb = bb * be
#          s  = s / be
#       END IF
#       IF( cd <= un*bs/eps ) THEN
#          cc = cc * be
#          dd = dd * be
#          s  = s * be
#       END IF
#       IF( abs( d ).LE.abs( c ) ) THEN
#          CALL dladiv1(aa, bb, cc, dd, p, q)
#       ELSE
#          CALL dladiv1(bb, aa, dd, cc, p, q)
#          q = -q
#       END IF
#       p = p * s
#       q = q * s
# *
#       RETURN
# *
# *     End of DLADIV
# *
#       END
 
# *> \ingroup doubleOTHERauxiliary
 
 
#       SUBROUTINE dladiv1( A, B, C, D, P, Q )
# *
# *  -- LAPACK auxiliary routine --
# *  -- LAPACK is a software package provided by Univ. of Tennessee,    --
# *  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
# *
# *     .. Scalar Arguments ..
#       DOUBLE PRECISION   A, B, C, D, P, Q
# *     ..
# *
# *  =====================================================================
# *
# *     .. Parameters ..
#       DOUBLE PRECISION   ONE
#       parameter( one = 1.0d0 )
# *
# *     .. Local Scalars ..
#       DOUBLE PRECISION   R, T
# *     ..
# *     .. External Functions ..
#       DOUBLE PRECISION   DLADIV2
#       EXTERNAL           dladiv2
# *     ..
# *     .. Executable Statements ..
# *
#       r = d / c
#       t = one / (c + d * r)
#       p = dladiv2(a, b, c, d, r, t)
#       a = -a
#       q = dladiv2(b, a, c, d, r, t)
# *
#       RETURN
# *
# *     End of DLADIV1
# *
#       END
 
# *> \ingroup doubleOTHERauxiliary
 
#       DOUBLE PRECISION FUNCTION dladiv2( A, B, C, D, R, T )
# *
# *  -- LAPACK auxiliary routine --
# *  -- LAPACK is a software package provided by Univ. of Tennessee,    --
# *  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
# *
# *     .. Scalar Arguments ..
#       DOUBLE PRECISION   a, b, c, d, r, t
# *     ..
# *
# *  =====================================================================
# *
# *     .. Parameters ..
#       DOUBLE PRECISION   zero
#       parameter( zero = 0.0d0 )
# *
# *     .. Local Scalars ..
#       DOUBLE PRECISION   br
# *     ..
# *     .. Executable Statements ..
# *
#       IF( r.NE.zero ) THEN
#          br = b * r
#          IF( br.NE.zero ) THEN
#             dladiv2 = (a + br) * t
#          ELSE
#             dladiv2 = a * t + (b * t) * r
#          END IF
#       ELSE
#          dladiv2 = (a + d * (b / c)) * t
#       END IF
# *
#       RETURN
# *
# *     End of DLADIV2
# *
#       END