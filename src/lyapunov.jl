# Continuous Lyapunov equations
lyapc(A::T1, C::T2) where {T1<:Number,T2<:Number} = -C/(A+A')
glyapc(A::T1, E::T2, C::T3) where {T1<:Number,T2<:Number,T3<:Number} = -C/(A*E'+A'*E)
"""
`X = lyapc(A, C, adj = false)` computes `X`, the hermitian/symmetric solution
of the continuous Lyapunov equation

      op(A)X + Xop(A)' + C = 0,

where `C` is a hermitian/symmetric matrix, and `op(A) = A` if `adj = false`
and `op(A) = A'` if `adj = true`.

Reference
R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
Comm. ACM, 15:820–826, 1972.
"""
function lyapc(A, C; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end

   # Reduce A to Schur form and transform C
   a, z = schur(A)
   #x = z'*C*z
   x = utqu(C,z)
   lyapcs!(a, x, adj = adj)
   #x = z*x*z'
   utqu!(x,z,adj = true)
end
function lyapc(A::Union{Adjoint, Transpose}, C)
   lyapc(A.parent, C, adj = true)
end
function lyapc(A::Array{Complex{Float64},2}, C::Array{Float64,2}; adj = false)
   lyapc(A,convert(Array{Complex{Float64},2},C),adj = adj)
end
"""
`X = glyapc(A, E, C, adj = false)` computes `X`, the hermitian/symmetric
solution of the continuous generalized Lyapunov equation

     op(A)Xop(E)' + op(E)Xop(A)' + C = 0,

where `C` is a hermitian/symmetric matrix, and `op(M) = M` if `adj = false`
and `op(M) = M'` if `adj = true`, for `M = A` or `M = E`.

Reference
T. Penzl. Numerical solution of generalized Lyapunov equations.
Adv. Comput. Math., 8:33–48, 1998.
"""
function glyapc(A, E, C; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end
   if (E == I) || isempty(E) || E == Array{eltype(A),2}(I,n,n)
      lyapc(A, C, adj = adj)
      return
   end

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   as, es, q, z = schur(A,E)
   if adj
      #x = z'*C*z
      x = utqu(C,z)
      glyapcs!(as,es,x,adj = true)
      #x = q*x*q'
      utqu!(x,q,adj = true)
   else
      #x = q'*C*q
      x = utqu(C,q)
      glyapcs!(as,es,x)
      #x = z*x*z'
      utqu!(x,z,adj = true)
   end
end
function glyapc(A::Union{Adjoint, Transpose}, E::Union{Adjoint, Transpose}, C)
   glyapc(A.parent, E.parent, C, adj = true)
end
function glyapc(A::Union{Adjoint, Transpose}, E, C)
   glyapc(copy(A), E, C, adj = false)
end
function glyapc(A, E::Union{Adjoint, Transpose}, C)
   glyapc(A, copy(E), C, adj = false)
end
function glyapc(A::Array{Complex{Float64},2}, E::Array{Complex{Float64},2}, C::Array{Float64,2}; adj = false)
   glyapc(A,E,convert(Array{Complex{Float64},2},C),adj = adj)
end

# Discrete Lyapunov equations
lyapd(A::T1, C::T2) where {T1<:Number,T2<:Number} = C/(one(C)-A'*A)
glyapd(A::T1, E::T3, C::T2) where {T1<:Number,T2<:Number,T3<:Number} = C/(E'*E-A'*A)
"""
`X = lyapd(A, C, adj = false)` computes `X`, the hermitian/symmetric solution
of the discrete Lyapunov equation

     op(A)Xop(A)' + C = X,

where `C` is a hermitian/symmetric matrix, and `op(A) = A` if `adj = false`
and `op(A) = A'` if `adj = true`.

Reference
G. Kitagawa. An Algorithm for solving the matrix equation X = F X F' + S,
International Journal of Control, 25:745-753, 1977.
"""
function lyapd(A, C; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end

   # Reduce A to Schur form and transform C
   a, z = schur(A)
   #x = z'*C*z
   x = utqu(C,z)
   lyapds!(a,x,adj = adj)
   #x = z*x*z'
   utqu!(x,z,adj = true)
end
function lyapd(A::Union{Adjoint, Transpose}, C)
   lyapd(A.parent, C, adj = true)
end
function lyapd(A::Array{Complex{Float64},2}, C::Array{Float64,2}; adj = false)
   lyapd(A,convert(Array{Complex{Float64},2},C),adj = adj)
end
"""
`X = glyapd(A, E, C, adj = false)` computes `X`, the hermitian/symmetric solution
of the discrete generalized Lyapunov equation

         op(A)Xop(A)' + C = op(E)Xop(E)',

where `C` is a hermitian/symmetric matrix, and `op(M) = M` if `adj = false`
and `op(M) = M'` if `adj = true`, for `M = A` or `M = E`.

Reference:
T. Penzl. Numerical solution of generalized Lyapunov equations.
Adv. Comput. Math., 8:33–48, 1998.
"""
function glyapd(A, E, C; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end
   if (E == I) || isempty(E) || E == Array{eltype(A),2}(I,n,n)
      lyapd(A, C, adj = adj)
      return
   end

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   as, es, q, z = schur(A,E)
   if adj
      #x = z'*C*z
      x = utqu(C,z)
      glyapds!(as,es,x,adj = true)
      #x = q*x*q'
      utqu!(x,q,adj = true)
   else
      #x = q'*C*q
      x = utqu(C,q)
      glyapds!(as,es,x)
      #x = z*x*z'
      utqu!(x,z,adj = true)
   end
end
function glyapd(A::Union{Adjoint, Transpose}, E::Union{Adjoint, Transpose}, C)
   glyapd(A.parent, E.parent, C, adj = true)
end
function glyapd(A::Union{Adjoint, Transpose}, E, C)
   glyapd(copy(A), E, C, adj = false)
end
function glyapd(A, E::Union{Adjoint, Transpose}, C)
   glyapd(A, copy(E), C, adj = false)
end
function glyapd(A::Array{Complex{Float64},2}, E::Array{Complex{Float64},2}, C::Array{Float64,2}; adj = false)
   glyapd(A,E,convert(Array{Complex{Float64},2},C),adj = adj)
end
"""
`lyapcs!(A, C, adj = false)` solves the continuous Lyapunov matrix equation

                AX + XA' + C = 0

with `A` in a real or complex Schur form and `C` a symmetric/hermitian matrix.
The computed symmetric/hermitian solution `X` is contained in `C`.

`lyapcs!(A,C,adj = true)` solves the continuous Lyapunov matrix equation

                A'X + XA + C = 0

with `A` in a real or complex Schur form. `C` contains on output the solution `X`.

`lyapcs!(A,C)` is equivalent to  `lyapcs!(A,C,adj = false)`.
"""
function lyapcs!(A::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   """
   lyapcs!(A,C,adj = false) solves the continuous Lyapunov matrix equation
                   A*X + X*A' + C = 0
   with A in a real Schur form. C contains on output the solution X.

   lyapcs(A,C,adj = true) solves the continuous Lyapunov matrix equation
                   A'*X + X*A + C = 0
   with A in a real Schur form. C contains on output the solution X.

   lyapcs!(A,C) is equivalent to  lyapcs(A,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n symmetric/hermitian matrix"))
   end

   # determine the structure of the real Schur form
   ba = fill(1,n,1)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   W = Array{eltype(A),2}(I,2,2)
   if adj
      """
      The (K,L)th block of X is determined starting from
      upper-left corner column by column by

      A(K,K)'*X(K,L) + X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      where
               K-1                    L-1
      R(K,L) = SUM [A(I,K)'*X(I,L)] + SUM [X(K,J)*A(J,L)].
               I=1                    J=1
      """
      j = 1
      for ll = 1:p
          dl = ba[ll]
          l = j:j+dl-1
          i = j
          for kk = ll:p
              dk = ba[kk]
              k = i:i+dk-1
              y = C[k,l]
              if kk > ll
                 ia = j:i-1
                 y += A[ia,k]'*C[ia,l]
              end
              C[k,l] = (kron(W[1:dl,1:dl],transpose(A[k,k]))+kron(transpose(A[l,l]),W[1:dk,1:dk]))\(-y[:])
              if i == j && dk == 2
                 temp = C[k,l]
                 C[k,l] = (temp'+temp)/2
              else
                 C[l,k] = C[k,l]'
              end
              i += dk
           end
           j += dl
           if j <= n
              for jr = j:n
                 for ir = jr:n
                     for lll = l
                         C[ir,jr] += C[ir,lll]*A[lll,jr] + (A[lll,ir]*C[jr,lll])'
                         C[jr,ir] = C[ir,jr]'
                      end
                  end
               end
           end
      end
   else
      """
      The (K,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(K,K)*X(K,L) + X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      where
                N                     N
      R(K,L) = SUM [A(K,I)*X(I,L)] + SUM [X(K,J)*A(L,J)'].
              I=K+1                 J=L+1
      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = j
          for kk = ll:-1:1
              dk = ba[kk]
              i1 = i-dk+1
              k = i1:i
              y = C[k,l]
              if kk < ll
                 ia = i+1:j
                 y += A[k,ia]*C[ia,l]
              end
              C[k,l] = (kron(W[1:dl,1:dl],A[k,k])+kron(A[l,l],W[1:dk,1:dk]))\(-y[:])
              if i == j && dl == 2
                 temp = C[k,l]
                 C[k,l] = (temp'+temp)/2
              else
                 C[l,k] = C[k,l]'
               end
              i = i-dk
           end
           j = j-dl
           if j >= 0
              for jr = 1:j
                 for ir = 1:jr
                     for lll = l
                         C[ir,jr] += C[ir,lll]*A[jr,lll] + A[ir,lll]*C[lll,jr]
                         C[jr,ir] = C[ir,jr]'
                      end
                  end
               end
           end
       end
   end
end

function lyapcs!(A::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}; adj = false)
   """
   lyapcs!(A,C,adj = false) solves the continuous Lyapunov matrix equation
                   A*X + X*A' + C = 0
   with A in a complex Schur form. C contains on output the solution X.

   lyapcs(A,C,adj = true) solves the continuous Lyapunov matrix equation
                   A'*X + X*A + C = 0
   with A in a complex Schur form. C contains on output the solution X.

   lyapcs!(A,C) is equivalent to lyapcs(A,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end

   if adj
      """
      The (K,L)th element of X is determined starting from
      upper-left corner column by column by

      A(K,K)'*X(K,L) + X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      where
               K-1                    L-1
      R(K,L) = SUM [A(I,K)'*X(I,L)] + SUM [X(K,J)*A(J,L)].
               I=1                    J=1
      """
      for l = 1:n
          for k = l:n
              y = C[k,l]
              for ia = l:k-1
                  y += A[ia,k]'*C[ia,l]
              end
              C[k,l] = -y/(A[k,k]'+A[l,l])
              if k != l
                 C[l,k] = C[k,l]'
              end
           end
           for jr = l+1:n
               for ir = jr:n
                   C[ir,jr] += C[ir,l]*A[l,jr] + (A[l,ir]*C[jr,l])'
                   if jr != ir
                      C[jr,ir] = C[ir,jr]'
                   end
               end
           end
      end
   else
      """
      The (K,L)th element of X is determined starting from
      bottom-right corner column by column by

      A(K,K)*X(K,L) + X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      where
                N                     N
      R(K,L) = SUM [A(K,I)*X(I,L)] + SUM [X(K,J)*A(L,J)'].
              I=K+1                 J=L+1
      """
      for l = n:-1:1
          for k = l:-1:1
              y = C[k,l]
              for ia = k+1:l
                 y += A[k,ia]*C[ia,l]
              end
              C[k,l] = -y/(A[k,k]+A[l,l]')
              if k != l
                 C[l,k] = C[k,l]'
               end
           end
           for jr = 1:l-1
               for ir = 1:jr
                  C[ir,jr] += C[ir,l]*A[jr,l]' + A[ir,l]*C[l,jr]
                  if ir != jr
                     C[jr,ir] = C[ir,jr]'
                  end
               end
           end
       end
   end
end

"""
`glyapcs!(A, E, C,adj = false)` solves the generalized continuous Lyapunov
matrix equation

                A*X*E' + E*X*A' + C = 0

with `(A,E)` in a generalized real or complex Schur form and `C` a
symmetric/hermitian matrix. The computed symmetric/hermitian solution `X`
is contained in `C`.

`glyapcs!(A,E,C,adj = true)` solves the generalized continuous Lyapunov
matrix equation

                A'*X*E + E'*X*A + C = 0

with `(A,E)` in a generalized real or complex Schur form and `C` a
symmetric/hermitian matrix. The computed symmetric/hermitian solution `X`
is contained in `C`.

`glyapcs!(A,E,C)` is equivalent to `glyapcs!(A,E,C,adj = false)`.
"""
function glyapcs!(A::Array{Float64,2}, E::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   """
   glyapcs!(A,C,E,adj = false) solves the generalized continuous Lyapunov
   matrix equation
                   A*X*E' + E*X*A' + C = 0
   with (A,E) in a generalized real Schur form. C contains on output the solution X.

   glyapcs!(A,C,E,adj = true) solves the generalized continuous Lyapunov
   matrix equation
                   A'*X*E + E'*X*A + C = 0
   with (A,E) in a generalized real Schur form. C contains on output the solution X.

   glyapcs!(A,E,C) is equivalent to glyapcs!(A,E,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian/symmetric matrix"))
   end
   if (E == I) || isempty(E) || (isone(E) && size(E,1) == n)
      lyapcs!(A, C, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   # determine the structure of the generalized real Schur form
   ba = fill(1,n,1)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   W = Array{eltype(C),2}(undef,n,2)
   if adj
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*E(L,L) + E(K,K)'*X(K,L)*A(L,L) = -C(K,L) - R(K,L),

      where
                K           L-1
      R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*E(J,L)]} +
               I=1          J=1

                K           L-1
               SUM {E(I,K)'*SUM [X(I,J)*A(J,L)]} +
               I=1          J=1

               K-1
              {SUM [A(I,K)'*X(I,L)]}*E(L,L) +
               I=1

               K-1
              {SUM [E(I,K)'*X(I,L)]}*A(L,L).
               I=1
      """
      i = 1
      for kk = 1:p
          dk = ba[kk]
          dkk = 1:dk
          k = i:i+dk-1
          j = 1
          for ll = 1:kk
             j1 = j+ba[ll]-1
             l = j:j1
             y = C[k,l]
             if kk > 1
                 ir = 1:i-1
                 C[l,k] = C[l,ir]*A[ir,k]
                 W[l,dkk] = C[l,ir]*E[ir,k]
                 ic = 1:j1
                 y += C[ic,k]'*E[ic,l] + W[ic,dkk]'*A[ic,l]
             end
             C[k,l] = ((kron(E[l,l],A[k,k])+kron(A[l,l],E[k,k]))')\(-y[:])
             if i == j
                if dk == 2
                   temp = C[k,l]
                   C[k,l] = Hermitian((temp'+temp)/2)
                else
                   C[k,l] = real(C[k,l])
                end
             end
             j += ba[ll]
             if j <= i
                C[l,k] += C[k,l]'*A[k,k]
                W[l,dkk] += C[k,l]'*E[k,k]
             end
          end
          if kk > 1
             ir = 1:i-1
             C[ir,k] = C[k,ir]'
          end
          i += dk
      end
   else
      """
      The (K,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(K,K)*X(K,L)*E(L,L)' + E(K,K)*X(K,L)*A(L,L)' = -C(K,L) - R(K,L),

      where

                N            N
      R(K,L) = SUM {A(K,I)* SUM [X(I,J)*E(L,J)']} +
               I=K         J=L+1

                N            N
               SUM {E(K,I)* SUM [X(I,J)*A(L,J)']} +
               I=K         J=L+1

                  N
               { SUM [A(K,J)*X(J,L)]}*E(L,L)' +
                J=K+1

                  N
               { SUM [E(K,J)*X(J,L)]}*A(L,L)'
                J=K+1
      """
      j = n
      for ll = p:-1:1
        dl = ba[ll]
        l = j-dl+1:j
        dll = 1:dl
        i = n
        for kk = p:-1:ll
            dk = ba[kk]
            i1 = i-dk+1
            k = i1:i
            y = C[l,k]
            if ll < p
               ir = j+1:n
               C[k,l] = C[k,ir]*A[l,ir]'
               W[k,dll] = C[k,ir]*E[l,ir]'
               ic = i1:n
               y += (E[k,ic]*C[ic,l] + A[k,ic]*W[ic,dll])'
            end
            C[l,k] = (kron(E[k,k],A[l,l])+kron(A[k,k],E[l,l]))\(-y[:])
            if i == j
               if dk == 2
                  temp = C[l,k]
                  C[l,k] = Hermitian((temp'+temp)/2)
               else
                  C[l,k] = real(C[l,k])
               end
            end
            i = i-dk
            if i >= j
               C[k,l] += (A[l,l]*C[l,k])'
               W[k,dll] += (E[l,l]*C[l,k])'
            else
               break
            end
        end
        if ll < p
           ir = j+1:n
           C[ir,l] = C[l,ir]'
        end
        j = j-dl
      end
   end
end
function glyapcs!(A::Array{Complex{Float64},2}, E::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}; adj = false)
   """
   glyapcs!(A,C,E,adj = false) solves the generalized continuous Lyapunov
   matrix equation
                   A*X*E' + E*X*A' + C = 0
   with (A,E) in a generalized complex Schur form. C contains on output the solution X.

   glyapcs!(A,C,E,adj = true) solves the generalized continuous Lyapunov
   matrix equation
                   A'*X*E + E'*X*A + C = 0
   with (A,E) in a generalized complex Schur form. C contains on output the solution X.

   glyapcs!(A,E,C) is equivalent to glyapcs!(A,E,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end
   if (E == I) || isempty(E) ||  (isone(E) && size(E,1) == n)
      lyapcs!(A, C, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   W = Array{Complex{Float64},1}(undef,n)
   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               W[l] = C[l,1]*E[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
                  W[l] += C[l,ir]*E[ir,k]
               end
               for ic = 1:l
                   y += C[ic,k]'*E[ic,l] + W[ic]'*A[ic,l]
                end
            end
            C[k,l] = -y/(A[k,k]'*E[l,l]+E[k,k]'*A[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
               W[l] += C[k,l]'*E[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               W[k] = C[k,l+1]*E[l,l+1]'
               for ir = l+2:n
                  C[k,l] += C[k,ir]*A[l,ir]'
                  W[k] += C[k,ir]*E[l,ir]'
               end
               for ic = k:n
                   y += (E[k,ic]*C[ic,l] + A[k,ic]*W[ic])'
               end
            end
            C[l,k] = -y/(E[k,k]'*A[l,l]+A[k,k]'*E[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
               W[k] += (E[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end

"""
`lyapds!(A, C, adj = false)` solves the discrete Lyapunov matrix equation

                AXA' + C = X

with `A` in a real or complex Schur form and `C` a symmetric/hermitian matrix.
The computed symmetric/hermitian solution `X` is contained in `C`.

`lyapds!(A,C,adj = true)` solves the discrete Lyapunov matrix equation

                AXA' + C = X

with `A` in a real or complex Schur form. `C` contains on output the solution `X`.

`lyapds!(A,C)` is equivalent to  `lyapds!(A,C,adj = false)`.
"""
function lyapds!(A::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   """
   lyapds!(A,C,adj = false) solves the discrete Lyapunov matrix equation
                   A*X*A' + C = X
   with A in a real Schur form. C contains on output the solution X.

   lyapds(A,C,adj = true) solves the discrete Lyapunov matrix equation
                   A'*X*A + C = X
   with A in a real Schur form. C contains on output the solution X.

   lyapds!(A,C) is equivalent to lyapds(A,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n symmetric/hermitian matrix"))
   end

   # determine the structure of the real Schur form
   ba = fill(1,n,1)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   if adj
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*A(L,L) - X(K,L) = -C(K,L) - R(K,L),

      where
                K           L-1
      R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*A(J,L)]} +
               I=1          J=1

                K-1
               {SUM [A(I,K)'*X(I,L)]}*A(L,L).
                I=1
      """
      i = 1
      for kk = 1:p
          dk = ba[kk]
          k = i:i+dk-1
          j = 1
          for ll = 1:kk
              j1 = j+ba[ll]-1
              l = j:j1
              y = C[k,l]
              if kk > 1
                 ir = 1:i-1
                 C[l,k] = C[l,ir]*A[ir,k]
                 ic = 1:j1
                 y += C[ic,k]'*A[ic,l]
              end
              C[k,l] = (I-kron(A[l,l]',A[k,k]'))\y[:]
              if i == j
                 if dk == 2
                    temp = C[k,l]
                    C[k,l] = Hermitian((temp'+temp)/2)
                 else
                    C[k,l] = real(C[k,l])
                 end
              end
              j += ba[ll]
              if j <= i
                 C[l,k] += C[k,l]'*A[k,k]
              end
          end
          if kk > 1
             ir = 1:i-1
             C[ir,k] = C[k,ir]'
          end
          i += dk
      end
   else
      """
      The (K,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(K,K)*X(K,L)*A(L,L)' - X(K,L) = -C(K,L) - R(K,L),

      where

                N            N
      R(K,L) = SUM {A(K,I)* SUM [X(I,J)*A(L,J)']} +
               I=K         J=L+1

                  N
               { SUM [A(K,J)*X(J,L)]}*A(L,L)'
                J=K+1
      """
      j = n
      for ll = p:-1:1
          dl = ba[ll]
          l = j-dl+1:j
          i = n
          for kk = p:-1:ll
              i1 = i-ba[kk]+1
              k = i1:i
              y = C[l,k]
              if ll < p
                 ir = j+1:n
                 C[k,l] = C[k,ir]*A[l,ir]'
                 ic = i1:n
                 y += (A[k,ic]*C[ic,l])'
              end
              C[l,k] = (I-kron(A[k,k],A[l,l]))\y[:]
              if i == j
                 if dl == 2
                    temp = C[l,k]
                    C[l,k] = Hermitian((temp'+temp)/2)
                 else
                    C[l,k] = real(C[l,k])
                 end
              end
              i = i-ba[kk]
              if i >= j
                 C[k,l] += (A[l,l]*C[l,k])'
              else
                 break
              end
          end
          if ll < p
             ir = i+2:n
             C[ir,l] = C[l,ir]'
          end
          j = j-dl
      end
   end
end

function lyapds!(A::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}; adj = false)
   """
   lyapds!(A,C,adj = false) solves the discrete Lyapunov matrix equation
                   A*X*A' + C = X
   with A in a complex Schur form. C contains on output the solution X.

   lyapds(A,C,adj = true) solves the discrete Lyapunov matrix equation
                   A'*X*A + C = X
   with A in a complex Schur form. C contains on output the solution X.

   lyapds!(A,C) is equivalent to lyapds(A,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n  || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end

   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
               end
               for ic = 1:l
                   y += C[ic,k]'*A[ic,l]
                end
            end
            C[k,l] = y/(I-A[k,k]'*A[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               for ir = l+2:n
                  C[k,l] += C[k,ir]*A[l,ir]'
               end
               for ic = k:n
                   y += (A[k,ic]*C[ic,l])'
               end
            end
            C[l,k] = y/(I-A[k,k]'*A[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end

"""
`glyapds!(A, E, C,adj = false)` solves the generalized discrete Lyapunov
matrix equation

                A*X*A' + C = E*X*E'

with `(A,E)` in a generalized real or complex Schur form and `C` a
symmetric/hermitian matrix. The computed symmetric/hermitian solution `X`
is contained in `C`.

`glyapds!(A,E,C,adj = true)` solves the generalized discrete Lyapunov
matrix equation

                A'*X*A + C = E'*X*E

with `(A,E)` in a generalized real or complex Schur form and `C` a
symmetric/hermitian matrix. The computed symmetric/hermitian solution `X`
is contained in `C`.

`glyapds!(A,E,C)` is equivalent to `glyapds!(A,E,C,adj = false)`.
"""
function glyapds!(A::Array{Float64,2}, E::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   """
   glyapds!(A,C,E,adj = false) solves the generalized discrete Lyapunov
   matrix equation
                   A*X*E' + E*X*A' + C = 0
   with (A,E) in a generalized real Schur form. C contains on output the solution X.

   glyapds!(A,C,E,adj = true) solves the generalized discrete Lyapunov
   matrix equation
                   A'*X*E + E'*X*A + C = 0
   with (A,E) in a generalized real Schur form. C contains on output the solution X.

   glyapds!(A,E,C) is equivalent to glyapds!(A,E,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian/symmetric matrix"))
   end
   if (E == I) || isempty(E) || (isone(E) && size(E,1) == n)
      lyapds!(A, C, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   # determine the structure of the real Schur form
   ba = fill(1,n,1)
   p = 1
   if n > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      p = 0
      while i <= n
         p += 1
         if d[i] != 0
            ba[p] = 2
            i += 1
         end
         i += 1
      end
   end

   W = Array{eltype(C),2}(undef,n,2)
   if adj
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*A(L,L) - E(K,K)'*X(K,L)*E(L,L) = -C(K,L) - R(K,L),

      where
                K           L-1
      R(K,L) = SUM {A(I,K)'*SUM [X(I,J)*A(J,L)]} -
               I=1          J=1

                K           L-1
               SUM {E(I,K)'*SUM [X(I,J)*E(J,L)]} +
               I=1          J=1

                K-1
               {SUM [A(I,K)'*X(I,L)]}*A(L,L) -
                I=1

                K-1
               {SUM [E(I,K)'*X(I,L)]}*E(L,L).
                I=1
      """
      i = 1
      for kk = 1:p
          dk = ba[kk]
          dkk = 1:dk
          k = i:i+dk-1
          j = 1
          for ll = 1:kk
             j1 = j+ba[ll]-1
             l = j:j1
             y = C[k,l]
             if kk > 1
                 ir = 1:i-1
                 C[l,k] = C[l,ir]*A[ir,k]
                 W[l,dkk] = C[l,ir]*E[ir,k]
                 ic = 1:j1
                 y += C[ic,k]'*A[ic,l] -W[ic,dkk]'*E[ic,l]
             end
             C[k,l] = (kron(E[l,l]',E[k,k]')-kron(A[l,l]',A[k,k]'))\y[:]
             if i == j
                if dk == 2
                   temp = C[k,l]
                   C[k,l] = Hermitian((temp'+temp)/2)
                else
                   C[k,l] = real(C[k,l])
                end
             end
             j += ba[ll]
             if j <= i
                 C[l,k] += C[k,l]'*A[k,k]
                 W[l,dkk] += C[k,l]'*E[k,k]
             end
          end
          if kk > 1
             ir = 1:i-1
             C[ir,k] = C[k,ir]'
          end
          i += dk
      end
   else
      """
      The (K,L)th block of X is determined starting from
      bottom-right corner column by column by

      A(K,K)*X(K,L)*A(L,L)' - E(K,K)*X(K,L)*E(L,L)' = -C(K,L) - R(K,L),

      where

                N            N
      R(K,L) = SUM {A(K,I)* SUM [X(I,J)*A(L,J)']} -
               I=K         J=L+1

                N            N
               SUM {E(K,I)* SUM [X(I,J)*E(L,J)']} +
               I=K         J=L+1

                  N
               { SUM [A(K,J)*X(J,L)]}*A(L,L)' -
                J=K+1

                  N
               { SUM [E(K,J)*X(J,L)]}*E(L,L)'
                J=K+1
   """
      j = n
      for ll = p:-1:1
        dl = ba[ll]
        l = j-dl+1:j
        dll = 1:dl
        i = n
        for kk = p:-1:ll
            i1 = i-ba[kk]+1
            k = i1:i
            y = C[l,k]
            if ll < p
               ir = j+1:n
               C[k,l] = C[k,ir]*A[l,ir]'
               W[k,dll] = C[k,ir]*E[l,ir]'
               ic = i1:n
               y += (A[k,ic]*C[ic,l] - E[k,ic]*W[ic,dll])'
            end
            C[l,k] = (kron(E[k,k],E[l,l])-kron(A[k,k],A[l,l]))\y[:]
            if i == j
               if dl == 2
                  temp = C[l,k]
                  C[l,k] = Hermitian((temp'+temp)/2)
               else
                  C[l,k] = real(C[l,k])
               end
            end
            i = i-ba[kk]
            if i >= j
               C[k,l] += (A[l,l]*C[l,k])'
               W[k,dll] += (E[l,l]*C[l,k])'
            else
               break
            end
        end
        if ll < p
           ir = i+2:n
           C[ir,l] = C[l,ir]'
        end
        j = j-dl
      end
   end
end

function glyapds!(A::Array{Complex{Float64},2}, E::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}; adj = false)
   """
   glyapds!(A,C,E,adj = false) solves the generalized discrete Lyapunov
   matrix equation
                   A*X*E' + E*X*A' + C = 0
   with (A,E) in a generalized complex Schur form. C contains on output the solution X.

   glyapds!(A,C,E,adj = true) solves the generalized discrete Lyapunov
   matrix equation
                   A'*X*E + E'*X*A + C = 0
   with (A,E) in a generalized complex Schur form. C contains on output the solution X.

   glyapds!(A,E,C) is equivalent to glyapds!(A,E,C,adj = false).
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n  || ~ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end
   if (E == I) || isempty(E) || (isone(E) && size(E,1) == n)
      lyapds!(A, C, adj = adj)
      return
   end
   if LinearAlgebra.checksquare(E) != n
      throw(DimensionMismatch("E must be a $n x $n matrix or I"))
   end

   W = Array{Complex{Float64},1}(undef,n)
   # Compute the hermitian solution
   if adj
      for k = 1:n
         for l = 1:k
            y = C[k,l]
            if k > 1
               C[l,k] = C[l,1]*A[1,k]
               W[l] = C[l,1]*E[1,k]
               for ir = 2:k-1
                  C[l,k] +=  C[l,ir]*A[ir,k]
                  W[l] += C[l,ir]*E[ir,k]
               end
               for ic = 1:l
                   y += C[ic,k]'*A[ic,l] - W[ic]'*E[ic,l]
                end
            end
            C[k,l] = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if l < k
               C[l,k] += C[k,l]'*A[k,k]
               W[l] += C[k,l]'*E[k,k]
            end
         end
         for ir = 1:k-1
             C[ir,k] = C[k,ir]'
         end
      end
   else
      for l = n:-1:1
        for k = n:-1:l
            y = C[l,k]
            if l < n
               C[k,l] = C[k,l+1]*A[l,l+1]'
               W[k] = C[k,l+1]*E[l,l+1]'
               for ir = l+2:n
                  C[k,l] += C[k,ir]*A[l,ir]'
                  W[k] += C[k,ir]*E[l,ir]'
               end
               for ic = k:n
                   y += (A[k,ic]*C[ic,l] - E[k,ic]*W[ic])'
               end
            end
            C[l,k] = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            if k == l
               C[k,l] = real(C[k,l])
            end
            if k > l
               C[k,l] += (A[l,l]*C[l,k])'
               W[k] += (E[l,l]*C[l,k])'
            end
        end
        if l < n
           for ir = l+1:n
               C[ir,l] = C[l,ir]'
           end
        end
      end
   end
end
