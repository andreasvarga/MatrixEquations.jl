# Continuous Lyapunov equations
lyapc(A::T1, C::T2) where {T1<:Number,T2<:Number} = -C/(A+A')
lyapc(A::T1, E::T2, C::T3) where {T1<:Number,T2<:Number,T3<:Number} = -C/(A*E'+A'*E)
"""
    X = lyapc(A, C)

Compute `X`, the symmetric or hermitian solution of the continuous Lyapunov equation

      AX + XA' + C = 0,

where `A` is a square real or complex matrix and `C` is a symmetric or hermitian
matrix. `A` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
"""
function lyapc(A, C)
   """
   The Bartels-Steward Schur form based method is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end

   adj = isa(A,Adjoint)

   # Reduce A to Schur form and transform C
   if adj
      if (typeof(A.parent) == Array{Float64,2})
         AS, Q = schur(A.parent)
      else
         AS, Q = schur(complex(A.parent))
      end
   else
      if (typeof(A) == Array{Float64,2})
         AS, Q = schur(A)
      else
         AS, Q = schur(complex(A))
      end
   end

   #X = Q'*C*Q
   X = utqu(C,Q)
   lyapcs!(AS, X, adj = adj)
   #X <- Q*X*Q'
   utqu!(X,Q')
end
"""
    X = lyapc(A, E, C)

Compute `X`, the symmetric or hermitian solution of the
continuous generalized Lyapunov equation

     AXE' + EXA' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a symmetric or
hermitian matrix. The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
"""
function lyapc(A, E, C)
   """
   The extension of the Bartels-Steward method based on the generalized Schur form
   is employed.

   Reference:
   T. Penzl. Numerical solution of generalized Lyapunov equations.
   Adv. Comput. Math., 8:33–48, 1998.
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end
   if isequal(E,I) || isempty(E)
      return lyapc(A, C)
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a square matrix of dimension $n"))
      end
   end

   adjA = isa(A,Adjoint)
   adjE = isa(E,Adjoint)
   if adjA && !adjE
      A = copy(A)
      adjA = false
   elseif !adjA && adjE
      E = copy(E)
      adjE = false
   end

   adj = adjA & adjE

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   if adj
      if (typeof(A.parent) == Array{Float64,2}) && (typeof(E.parent) == Array{Float64,2})
         as, es, q, z = schur(A.parent,E.parent)
      else
         as, es, q, z = schur(complex(A.parent),complex(E.parent))
      end
   else
      if (typeof(A) == Array{Float64,2}) && (typeof(E) == Array{Float64,2})
         as, es, q, z = schur(A,E)
      else
         as, es, q, z = schur(complex(A),complex(E))
      end
   end
   if adj
      #x = z'*C*z
      x = utqu(C,z)
      lyapcs!(as,es,x,adj = true)
      #x = q*x*q'
      utqu!(x,q')
   else
      #x = q'*C*q
      x = utqu(C,q)
      lyapcs!(as,es,x)
      #x = z*x*z'
      utqu!(x,z')
   end
end
# Discrete Lyapunov equations
lyapd(A::T1, C::T2) where {T1<:Number,T2<:Number} = C/(one(C)-A'*A)
lyapd(A::T1, E::T3, C::T2) where {T1<:Number,T2<:Number,T3<:Number} = C/(E'*E-A'*A)
"""
    X = lyapd(A, C)

Compute `X`, the symmetric or hermitian solution
of the discrete Lyapunov equation

       AXA' - X + C = 0,

where `A` is a square real or complex matrix and `C` is a symmetric or hermitian
matrix. `A` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
"""
function lyapd(A, C)
   """
   The discrete analog of the Bartels-Steward method based on the Schur form
   is employed.

   Reference:
   G. Kitagawa. An Algorithm for solving the matrix equation X = F X F' + S,
   International Journal of Control, 25:745-753, 1977.
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end
   realAC = isreal(A) & isreal(C)

   # Reduce A to Schur form and transform C
   adj = isa(A,Adjoint)
   if adj
      if (typeof(A.parent) == Array{Float64,2})
         AS, Q = schur(A.parent)
      else
         AS, Q = schur(complex(A.parent))
      end
   else
      if (typeof(A) == Array{Float64,2})
         AS, Q = schur(A)
      else
         AS, Q = schur(complex(A))
      end
   end
   #X = Q'*C*Q
   X = utqu(C,Q)
   lyapds!(AS, X, adj = adj)
   #X <- Q*X*Q'
   utqu!(X,Q')
end

"""
    X = lyapd(A, E, C)

Compute `X`, the symmetric or hermitian solution
of the discrete generalized Lyapunov equation

         AXA' - EXE' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a symmetric
or hermitian matrix. The pencil `A-λE` must not have two eigenvalues `α` and `β`
such that `αβ = 1`.
"""
function lyapd(A, E, C)
   """
   The extension of the Bartels-Steward method based on the generalized Schur form
   is employed.

   Reference:
   T. Penzl. Numerical solution of generalized Lyapunov equations.
   Adv. Comput. Math., 8:33–48, 1998.
   """
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a symmetric/hermitian matrix of dimension $n"))
   end
   if isequal(E,I) || isempty(E)
      return lyapd(A, C)
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
   end

   adjA = isa(A,Adjoint)
   adjE = isa(E,Adjoint)
   if adjA && !adjE
      A = copy(A)
      adjA = false
   elseif !adjA && adjE
      E = copy(E)
      adjE = false
   end

   adj = adjA & adjE

   # Reduce (A,E) to generalized Schur form and transform C
   # (as,es) = (q'*A*z, q'*E*z)
   if adj
      if  (typeof(A.parent) == Array{Float64,2}) &&  (typeof(E.parent) == Array{Float64,2})
         as, es, q, z = schur(A.parent,E.parent)
      else
         as, es, q, z = schur(complex(A.parent),complex(E.parent))
      end
   else
      if  (typeof(A) == Array{Float64,2}) &&  (typeof(E) == Array{Float64,2})
         as, es, q, z = schur(A,E)
      else
         as, es, q, z = schur(complex(A),complex(E))
      end
   end
   if adj
      #x = z'*C*z
      x = utqu(C,z)
      lyapds!(as,es,x,adj = true)
      #x = q*x*q'
      utqu!(x,q')
   else
      #x = q'*C*q
      x = utqu(C,q)
      lyapds!(as,es,x)
      #x = z*x*z'
      utqu!(x,z')
   end
end
"""
    lyapcs!(A,C;adj = false)

Solve the continuous Lyapunov matrix equation

                op(A)X + Xop(A)' + C = 0

where `op(A) = A` if `adj = false` and `op(A) = A'` if `adj = true`.
A is a square real matrix in a real Schur form, or a square complex matrix in a
complex Schur form and `C` is a symmetric or hermitian matrix.
`A` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
`C` contains on output the solution `X`.
"""
function lyapcs!(A::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
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
              Z = (kron(W[1:dl,1:dl],transpose(A[k,k]))+kron(transpose(A[l,l]),W[1:dk,1:dk]))\(-y[:])
              isfinite(maximum(abs.(Z))) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that α+β ≈ 0")
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
              Z = (kron(W[1:dl,1:dl],A[k,k])+kron(A[l,l],W[1:dk,1:dk]))\(-y[:])
              isfinite(maximum(abs.(Z))) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that α+β ≈ 0")
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
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
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
              Z = -y/(A[k,k]'+A[l,l])
              isfinite(Z) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that α+β ≈ 0")
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
              Z = -y/(A[k,k]+A[l,l]')
              isfinite(Z) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that α+β ≈ 0")
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
    lyapcs!(A, E, C; adj = false)

Solve the generalized continuous Lyapunov matrix equation

                op(A)Xop(E)' + op(E)Xop(A)' + C = 0

where `op(A) = A` and `op(E) = E` if `adj = false` and `op(A) = A'` and
`op(E) = E'` if `adj = true`. The pair `(A,E)` is in a generalized real or
complex Schur form and `C` is a symmetric or hermitian matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `α+β = 0`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapcs!(A::Array{Float64,2}, E::Union{UniformScaling{Bool}, Array{Float64,2}}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian/symmetric matrix"))
   end
   if isequal(E,I) || isempty(E)
      lyapcs!(A, C, adj = adj)
      return
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
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
             Z = ((kron(E[l,l],A[k,k])+kron(A[l,l],E[k,k]))')\(-y[:])
             isfinite(maximum(abs.(Z))) ? C[k,l] = Z : error("MESingErr: A-λE has eigenvalues α and β such that α+β ≈ 0")
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
            Z = (kron(E[k,k],A[l,l])+kron(A[k,k],E[l,l]))\(-y[:])
            isfinite(maximum(abs.(Z))) ? C[l,k] = Z : error("MESingErr: A-λE has eigenvalues α and β such that α+β ≈ 0")
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
function lyapcs!(A::Array{Complex{Float64},2}, E::Union{UniformScaling{Bool},Array{Complex{Float64},2}}, C::Array{Complex{Float64},2}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end
   if isequal(E,I) || isempty(E)
      lyapcs!(A, C, adj = adj)
      return
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
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
            Z = -y/(A[k,k]'*E[l,l]+E[k,k]'*A[l,l])
            isfinite(Z) ? C[k,l] = Z : error("MESingErr: A-λE has eigenvalues α and β such that α+β ≈ 0")
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
            Z = -y/(E[k,k]'*A[l,l]+A[k,k]'*E[l,l])
            isfinite(Z) ? C[l,k] = Z : error("MESingErr: A-λE has eigenvalues α and β such that α+β ≈ 0")
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
    lyapds!(A, C; adj = false)

Solve the discrete Lyapunov matrix equation

                op(A)Xop(A)' -X + C = 0 ,

where `op(A) = A` if `adj = false` and `op(A) = A'` if `adj = true`.
`A` is in a real or complex Schur form and `C` a symmetric or hermitian matrix.
`A` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapds!(A::Array{Float64,2}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
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
              Z = (I-kron(A[l,l]',A[k,k]'))\y[:]
              isfinite(maximum(abs.(Z))) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that αβ ≈ 1")
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
              Z = (I-kron(A[k,k],A[l,l]))\y[:]
              isfinite(maximum(abs.(Z))) ? C[l,k] = Z : error("MESingErr: A has eigenvalues α and β such that αβ ≈ 1")
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
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n  || !ishermitian(C)
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
            Z = y/(I-A[k,k]'*A[l,l])
            isfinite(Z) ? C[k,l] = Z : error("MESingErr: A has eigenvalues α and β such that αβ ≈ 1")
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
            Z = y/(I-A[k,k]'*A[l,l])
            isfinite(Z) ? C[l,k] = Z : error("MESingErr: A has eigenvalues α and β such that αβ ≈ 1")
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
    lyapds!(A, E, C; adj = false)

Solve the generalized discrete Lyapunov matrix equation

                op(A)Xop(A)' - op(E)Xop(E)' + C = 0,

where `op(A) = A` and `op(E) = E` if `adj = false` and `op(A) = A'` and
`op(E) = E'` if `adj = true`. The pair `(A,E)` in a generalized real or
complex Schur form and `C` a symmetric or hermitian matrix.
The pencil `A-λE` must not have two eigenvalues `α` and `β` such that `αβ = 1`.
The computed symmetric or hermitian solution `X` is contained in `C`.
"""
function lyapds!(A::Array{Float64,2}, E::Union{UniformScaling{Bool},Array{Float64,2}}, C::Union{Array{Complex{Float64},2}, Array{Float64,2}}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n || !ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian/symmetric matrix"))
   end
   if isequal(E,I) || isempty(E)
      lyapds!(A, C, adj = adj)
      return
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
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
             Z = (kron(E[l,l]',E[k,k]')-kron(A[l,l]',A[k,k]'))\y[:]
             isfinite(maximum(abs.(Z))) ? C[k,l] = Z : error("MESingErr: A-λE has eigenvalues α and β such that αβ ≈ 1")
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
            Z = (kron(E[k,k],E[l,l])-kron(A[k,k],A[l,l]))\y[:]
            isfinite(maximum(abs.(Z))) ? C[l,k] = Z : error("MESingErr: A-λE has eigenvalues α and β such that αβ ≈ 1")
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

function lyapds!(A::Array{Complex{Float64},2}, E::Union{UniformScaling{Bool},Array{Complex{Float64},2}}, C::Array{Complex{Float64},2}; adj = false)
   n = LinearAlgebra.checksquare(A)
   if LinearAlgebra.checksquare(C) != n  || !ishermitian(C)
      throw(DimensionMismatch("C must be a $n x $n hermitian matrix"))
   end
   if isequal(E,I) || isempty(E)
      lyapds!(A, C, adj = adj)
      return
   else
      if LinearAlgebra.checksquare(E) != n
         throw(DimensionMismatch("E must be a $n x $n matrix or I"))
      end
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
            Z = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            isfinite(Z) ? C[k,l] = Z : error("MESingErr: A-λE has eigenvalues α and β such that αβ ≈ 1")
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
            Z = y/(E[k,k]'*E[l,l]-A[k,k]'*A[l,l])
            isfinite(Z) ? C[l,k] = Z : error("MESingErr: A-λE has eigenvalues α and β such that αβ ≈ 1")
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
