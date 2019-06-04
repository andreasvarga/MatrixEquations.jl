sylvc(A::T1, B::T2, C::T3) where {T1<:Number,T2<:Number,T3<:Number} = C/(A+B)
"""
`X = sylvc(A,B,C)` solves the continuous Sylvester matrix equation

                AX + XB = C

using the Bartels-Stewart Schur form based approach. `A` and `B` are
square matrices, and `A` and `-B` must not have common eigenvalues.
"""
function sylvc(A,B,C)
   """
   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(C);
   if [m; n] != LinearAlgebra.checksquare(A,B)
      throw(DimensionMismatch("A, B and C have incompatible dimensions"))
   end
   realcase = isreal(A) && isreal(B) && isreal(C)
   if isa(A,Adjoint)
      if realcase
         RA, QA = schur(A.parent)
         TA = 'T'
      else
         RA, QA = schur(complex(A.parent))
         TA = 'C'
      end
   else
      if realcase
         RA, QA = schur(A)
      else
         RA, QA = schur(complex(A))
      end
      TA = 'N'
   end
   if isa(B,Adjoint)
      if realcase
         RB, QB = schur(B.parent)
         TB = 'T'
      else
         RB, QB = schur(complex(B.parent))
         TB = 'C'
      end
   else
      if realcase
         RB, QB = schur(B)
      else
         RB, QB = schur(complex(B))
      end
      TB = 'N'
   end
   D = adjoint(QA) * (C*QB)
   Y, scale = LAPACK.trsyl!(TA,TB, RA, RB, D)
   rmul!(QA*(Y * adjoint(QB)), inv(scale))
end
sylvd(A::T1, B::T2, C::T3) where {T1<:Number,T2<:Number,T3<:Number} = C/(A*B+one(C))
"""
`X = sylvd(A,B,C)` solves the discrete Sylvester matrix equation

                AXB + X = C

using an extension of the Bartels-Stewart Schur form based approach.
`A` and `B` are square matrices, and `A` and `-B` must not have
common reciprocal eigenvalues.
"""
function sylvd(A,B,C)
   """
   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(C);
   if [m; n] != LinearAlgebra.checksquare(A,B)
      throw(DimensionMismatch("A, B and C have incompatible dimensions"))
   end
   realcase = isreal(A) && isreal(B) && isreal(C)
   adjA = isa(A,Adjoint)
   adjB = isa(B,Adjoint)
   if adjA
      if realcase
         RA, QA = schur(A.parent)
      else
         RA, QA = schur(complex(A.parent))
      end
   else
      if realcase
         RA, QA = schur(A)
      else
         RA, QA = schur(complex(A))
      end
   end
   if adjB
      if realcase
         RB, QB = schur(B.parent)
      else
         RB, QB = schur(complex(B.parent))
      end
   else
      if realcase
         RB, QB = schur(B)
      else
         RB, QB = schur(complex(B))
      end
   end
   D = adjoint(QA) * (C*QB)
   Y = sylvds!(RA, RB, D, adjA = adjA, adjB = adjB)
   QA*(Y * adjoint(QB))
end
"""
`X = gsylv(A,B,C,D,E)` solves the generalized Sylvester matrix equation

                AXB + CXD = E

using generalized Schur form based approach. `A`, `B`, `C` and `D` are
square matrices. The pencils `A-λC` and `D+λB` must be regular and
must not have common eigenvalues.
"""
function gsylv(A,B,C,D,E)

    m, n = size(E);
    if [m; n; m; n] != LinearAlgebra.checksquare(A,B,C,D)
       throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
    end
    realcase = isreal(A) && isreal(B) && isreal(C) && isreal(D) && isreal(E)
    adjAC = isa(A,Adjoint) && isa(C,Adjoint)
    adjBD = isa(B,Adjoint) && isa(D,Adjoint)
    if adjAC
       if realcase
          AS, CS, Z1, Q1 = schur(A.parent,C.parent)
       else
          AS, CS, Z1, Q1 = schur(complex(A.parent),complex(C.parent))
       end
    else
       if isa(A,Adjoint)
          A = copy(A)
       end
       if isa(C,Adjoint)
          C = copy(C)
       end
       if realcase
          AS, CS, Q1, Z1 = schur(A,C)
       else
          AS, CS, Q1, Z1 = schur(complex(A),complex(C))
       end
    end
    if adjBD
       if realcase
          BS, DS, Z2, Q2 = schur(B.parent,D.parent)
       else
          BS, DS, Z2, Q2 = schur(complex(B.parent),complex(D.parent))
       end
    else
      if isa(B,Adjoint)
          B = copy(BLAS)
      end
      if isa(D,Adjoint)
          D = copy(D)
      end
       if realcase
          BS, DS, Q2, Z2 = schur(B,D)
       else
          BS, DS, Q2, Z2 =  schur(complex(B),complex(D))
       end
    end
    Y = adjoint(Q1) * (E*Z2)
    gsylvs!(AS, BS, CS, DS, Y, adjAC = adjAC, adjBD = adjBD)
    Z1*(Y * adjoint(Q2))
end
"""
`(X,Y) = sylvsys(A,B,C,D,E,F)` solves the Sylvester system of
matrix equations

                AX + YB = C
                DX + YE = F,

where `(A,D)`, `(B,E)` are pairs of square matrices of same size.
The pencils `A-λD` and `-isgn*(B-λE)` must be regular and must not have common eigenvalues.
"""
function sylvsys(A,B,C,D,E,F)

    m, n = size(C);
    if m != size(F,1) || n != size(F,2)
      throw(DimensionMismatch("C and F must have the same dimensions"))
    end
    if [m; n; m; n] != LinearAlgebra.checksquare(A,B,D,E)
       throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
    end
    if isa(A,Adjoint)
      A = copy(A)
    end
    if isa(B,Adjoint)
      B = copy(B)
    end
    if isa(D,Adjoint)
      D = copy(D)
    end
    if isa(E,Adjoint)
      E = copy(E)
    end
    if isreal(A) && isreal(B) && isreal(C) && isreal(D) && isreal(E) && isreal(F)
       AS, DS, Q1, Z1 = schur(A,D)
       BS, ES, Q2, Z2 = schur(B,E)
    else
       AS, DS, Q1, Z1 = schur(complex(A),complex(D))
       BS, ES, Q2, Z2 = schur(complex(B),complex(E))
    end
    CS = adjoint(Q1) * (C*Z2)
    FS = adjoint(Q1) * (F*Z2)

    X, Y, scale =  tgsyl!('N',AS,BS,CS,DS,ES,FS)

    (rmul!(Z1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Q2)), inv(-scale)) )
end
"""
`(X,Y) = dsylvsys(A,B,C,D,E,F)` solves the dual Sylvester system of
matrix equations

       AX + DY = C
       XB + YE = F ,

where `(A,D)`, `(B,E)` are pairs of square matrices of same size.
The pencils `A-λD` and `-isgn*(B-λE)` must be regular and must not have common eigenvalues.
"""
function dsylvsys(A,B,C,D,E,F)

    m, n = size(C);
    if m != size(F,1) || n != size(F,2)
      throw(DimensionMismatch("C and F must have the same dimensions"))
    end
    if [m; n; m; n] != LinearAlgebra.checksquare(A,B,D,E)
       throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
    end
    realcase = isreal(A) & isreal(B) & isreal(C) & isreal(D) & isreal(E) & isreal(F)
    transsylv = isa(A,Adjoint) && isa(B,Adjoint) && isa(D,Adjoint) && isa(E,Adjoint)
    if transsylv
       if realcase
          AS, DS, Q1, Z1 = schur(A.parent,D.parent)
          BS, ES, Q2, Z2 = schur(B.parent,E.parent)
          trans = 'T'
       else
          AS, DS, Q1, Z1 = schur(complex(A.parent),complex(D.parent))
          BS, ES, Q2, Z2 = schur(complex(B.parent),complex(E.parent))
          trans = 'C'
       end
       CS = adjoint(Z1) * (C*Z2)
       FS = adjoint(Q1) * (F*Q2)

       X, Y, scale =  tgsyl!(trans,AS,BS,CS,DS,ES,-FS)

       (rmul!(Q1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Z2)), inv(scale)) )
    else
       if realcase
          AS, DS, Q1, Z1 = schur(copy(A'),copy(D'))
          BS, ES, Q2, Z2 = schur(copy(B'),copy(E'))
          trans = 'T'
       else
          AS, DS, Q1, Z1 = schur(complex(copy(A')),complex(copy(D')))
          BS, ES, Q2, Z2 = schur(complex(copy(B')),complex(copy(E')))
          trans = 'C'
       end
       CS = adjoint(Z1) * (C*Z2)
       FS = adjoint(Q1) * (F*Q2)

       X, Y, scale =  tgsyl!(trans,AS,BS,CS,DS,ES,-FS)

       (rmul!(Q1*(X * adjoint(Z2)), inv(scale)), rmul!(Q1*(Y * adjoint(Z2)), inv(scale)) )
    end
end
"""
`sylvds!(A,B,C; adjA = false, adjB = false)` solves the discrete Sylvester
matrix equation

                op(A)Xop(B) + X =  C

where `op(A) = A` or `op(A) = A'` if `adjA = false` or `adjA = true`, respectively,
and `op(B) = B` or `op(B) = B'` if `adjB = false` or `adjB = true`, respectively.
`A` and `B` are square matrices in Schur forms, and `A` and `-B` must not have
common reciprocal eigenvalues. `C` contains on output the solution `X`.
"""
function sylvds!(A::Array{Float64,2}, B::Array{Float64,2}, C::Array{Float64,2}; adjA = false, adjB = false)
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   if size(C,1) != m || size(C,2) != n
      throw(DimensionMismatch("C must be an $m x $n matrix"))
   end

   # determine the structure of the real Schur form of A
   ba = fill(1,m,1)
   pa = 1
   if m > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      pa = 0
      while i <= m
         pa += 1
         if d[i] != 0
            ba[pa] = 2
            i += 1
         end
         i += 1
      end
   end
   # determine the structure of the real Schur form of B
   bb = fill(1,n,1)
   pb = 1
   if n > 1
      d = [diag(B,-1);zeros(1)]
      i = 1
      pb = 0
      while i <= n
         pb += 1
         if d[i] != 0
            bb[pb] = 2
            i += 1
         end
         i += 1
      end
   end

   W = fill(zero(eltype(C)),m,2)
   if ~adjA && ~adjB
      """
              The (K,L)th block of X is determined starting from
              bottom-left corner column by column by

                 A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

              where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
                            J=K            I=1
      """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = m
          for kk = pa:-1:1
              dk = ba[kk]
              dkk = 1:dk
              i1 = i-dk+1
              k = i1:i
              y = C[k,l]
              if kk < pa
                 ir = i+1:m
                 W1 = A[k,ir]*C[ir,l]
                 y -= W1*B[l,l]
              end
              if ll > 1
                 ic = i1:m
                 W[k,dll] = C[k,il1]*B[il1,l]
                 y -= A[k,ic]*W[ic,dll]
              end
              C[k,l] = (kron(transpose(B[l,l]),A[k,k])+I)\(y[:])
              i -= dk
          end
          j += dl
      end
   elseif ~adjA && adjB
         """
                 The (K,L)th block of X is determined starting from
                 bottom-right corner column by column by

                     A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

                 where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
                               J=K           I=L+1
         """
         j = n
         for ll = pb:-1:1
             dl = bb[ll]
             dll = 1:dl
             il1 = j+1:n
             l = j-dl+1:j
             i = m
             for kk = pa:-1:1
                 dk = ba[kk]
                 dkk = 1:dk
                 i1 = i-dk+1
                 k = i1:i
                 y = C[k,l]
                 if kk < pa
                    ir = i+1:m
                    W1 = A[k,ir]*C[ir,l]
                    y -= W1*B[l,l]'
                 end
                 if ll < pb
                    ic = i1:m
                    W[k,dll] = C[k,il1]*B[l,il1]'
                    y -= A[k,ic]*W[ic,dll]
                 end
                 C[k,l] = (kron(B[l,l],A[k,k])+I)\(y[:])
                 i -= dk
             end
             j -= dl
         end
   elseif adjA && ~adjB
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
                            J=1             I=1
      """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = C[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*C[ir,l]
                 y -= W1*B[l,l]
              end
              if ll > 1
                 ic = 1:i1
                 W[k,dll] = C[k,il1]*B[il1,l]
                 y -= A[ic,k]'*W[ic,dll]
              end
              C[k,l] = (kron(transpose(B[l,l]),transpose(A[k,k]))+I)\(y[:])
              i += dk
          end
          j += dl
      end
   elseif adjA && adjB
      """
                 A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

              where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
                            J=1            I=L+1
      """
      j = n
      for ll = pb:-1:1
          dl = bb[ll]
          dll = 1:dl
          il1 = j+1:n
          l = j-dl+1:j
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = C[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*C[ir,l]
                 y -= W1*B[l,l]'
              end
              if ll < pb
                 ic = 1:i1
                 W[k,dll] = C[k,il1]*B[l,il1]'
                 y -= A[ic,k]'*W[ic,dll]
              end
              C[k,l] = (kron(B[l,l],transpose(A[k,k]))+I)\(y[:])
              i += dk
          end
          j -= dl
      end
   end
   return C
end
function sylvds!(A::T, B::T, C::T; adjA = false, adjB = false) where {T<:Array{Complex{Float64},2}}
   """
   An extension of the Bartels-Stewart Schur form based approach is employed.

   Reference:
   R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
   Comm. ACM, 15:820–826, 1972.
   """
   m, n = LinearAlgebra.checksquare(A,B)
   if size(C,1) != m || size(C,2) != n
      throw(DimensionMismatch("C must be an $m x $n matrix"))
   end


   W = fill(zero(eltype(C)),m,1)
   if ~adjA && ~adjB
      """
      The (K,L)th element of X is determined starting from
      bottom-left corner column by column by

                 A(K,K)*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] }.
                            J=K            I=1
      """
      for l = 1:n
          il1 = 1:l-1
          ll = l:l
          for k = m:-1:1
              y = C[k,l]
              kk = k:k
              if k < m
                 ir = k+1:m
                 W1 = A[kk,ir]*C[ir,ll]
                 y -= W1[1]*B[l,l]
              end
              if l > 1
                 ic = k:m
                 Z = C[kk,il1]*B[il1,ll]
                 W[k,1] = Z[1]
                 T = A[kk,ic]*W[ic,1]
                 y -= T[1]
              end
              C[k,l] = y/(B[l,l]*A[k,k]+I)
             end
      end
   elseif ~adjA && adjB
         """
         The (K,L)th element of X is determined starting from
         bottom-right corner column by column by

                  A(K,K)*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

         where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] }.
                               J=K           I=L+1
         """
         for l = n:-1:1
             ll = l:l
             il1 = l+1:n
             for k = m:-1:1
                 kk = k:k
                 y = C[k,l]
                 if k < m
                    ir = k+1:m
                    W1 = A[kk,ir]*C[ir,ll]
                    y -= W1[1]*B[l,l]'
                 end
                 if l < n
                    ic = k:m
                    Z = C[kk,il1]*B[ll,il1]'
                    W[k,1] = Z[1]
                    T = A[kk,ic]*W[ic,1]
                    y -= T[1]
                 end
                 C[k,l] = y/(B[l,l]'*A[k,k]+I)
             end
         end
   elseif adjA && ~adjB
      """
      The (K,L)th element of X is determined starting from the
      upper-left corner column by column by

               A(K,K)'*X(K,L)*B(L,L) + X(K,L) = C(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] }.
                            J=1             I=1
      """
      for l = 1:n
          ll = l:l
          il1 = 1:l-1
          for k = 1:m
              kk = k:k
              y = C[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*C[ir,ll]
                 y -= W1[1]*B[l,l]
              end
              if l > 1
                 ic = 1:m
                 Z = C[kk,il1]*B[il1,ll]
                 W[k,1] = Z[1]
                 T = A[ic,kk]'*W[ic,1]
                 y -= T[1]
              end
              C[k,l] = y/(B[l,l]*A[k,k]'+I)
          end
      end
   elseif adjA && adjB
      """
      The (K,L)th element of X is determined starting from the
      upper-right corner column by column by

              A(K,K)'*X(K,L)*B(L,L)' + X(K,L) = C(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }.
                            J=1            I=L+1
      """
      for l = n:-1:1
          ll = l:l
          il1 = l+1:n
          for k = 1:m
              kk = k:k
              y = C[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*C[ir,ll]
                 y -= W1[1]*B[l,l]'
              end
              if l < n
                 ic = 1:m
                 Z = C[kk,il1]*B[ll,il1]'
                 W[k,1] = Z[1]
                 T = A[ic,kk]'*W[ic,1]
                 y -= T[1]
              end
              C[k,l] = y/(B[l,l]'*A[k,k]'+I)
          end
      end
   end
   return C
end
"""
`X = gsylvs!(A,B,C,D,E;adjAC=false,adjBD=false)` solves the generalized
Sylvester matrix equation

                op1(A)Xop2(B) + op1(C)Xop2(D) = E ,

where `A`, `B`, `C` and `D` are square matrices, and
op1(A) = A and op1(C) = C if adjAC = false;
op1(A) = A' and op1(C) = C' if adjAC = true;
op2(B) = B and op2(D) = D if adjBD = false;
op2(B) = B' and op2(D) = D' if adjBD = true.
The matrix pairs (A,C) and (B,D) are in generalized real or complex Schur forms.
The pencils `A-λC` and `D+λB` must be regular and must not have common eigenvalues.
"""
function gsylvs!(A::T, B::T, C::T, D::T, E::T; adjAC = false, adjBD = false) where {T<:Array{Float64,2}}
   """
   An extension proposed in [1] of the Bartels-Stewart Schur form based approach [2] is employed.

   References:
   [1] K.-W. E. Chu. The solution of the matrix equation AXB – CXD = E and
       (YA – DZ, YC– BZ) = (E, F). Lin. Alg. Appl., 93:93-105, 1987.
   [2] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
       Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(E);
   if [m; n; m; n] != LinearAlgebra.checksquare(A,B,C,D)
      throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
   end

   # determine the structure of the generalized real Schur form of (A,C)
   ba = fill(1,m,1)
   pa = 1
   if m > 1
      d = [diag(A,-1);zeros(1)]
      i = 1
      pa = 0
      while i <= m
         pa += 1
         if d[i] != 0
            ba[pa] = 2
            i += 1
         end
         i += 1
      end
   end
   # determine the structure of the generalized real Schur form of (B,D)
   bb = fill(1,n,1)
   pb = 1
   if n > 1
      d = [diag(B,-1);zeros(1)]
      i = 1
      pb = 0
      while i <= n
         pb += 1
         if d[i] != 0
            bb[pb] = 2
            i += 1
         end
         i += 1
      end
   end

   WB = fill(zero(eltype(E)),m,2)
   WD = fill(zero(eltype(E)),m,2)
   if ~adjAC && ~adjBD
      """
      The (K,L)th block of X is determined starting from
      bottom-left corner column by column by

            A(K,K)*X(K,L)*B(L,L) + C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
                            J=K            I=1

                             M
                          { SUM [C(K,J)*X(J,L)] } * D(L,L) +
                           J=K+1
                             M             L-1
                            SUM { C(K,J) * SUM [X(J,I)*D(I,L)] }.
                            J=K            I=1
      """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = m
          for kk = pa:-1:1
              dk = ba[kk]
              dkk = 1:dk
              i1 = i-dk+1
              k = i1:i
              y = E[k,l]
              if kk < pa
                 ir = i+1:m
                 W1 = A[k,ir]*E[ir,l]
                 y -= W1*B[l,l]
                 W1 = C[k,ir]*E[ir,l]
                 y -= W1*D[l,l]
              end
              if ll > 1
                 ic = i1:m
                 WB[k,dll] = E[k,il1]*B[il1,l]
                 WD[k,dll] = E[k,il1]*D[il1,l]
                 y -= (A[k,ic]*WB[ic,dll] + C[k,ic]*WD[ic,dll])
              end
              E[k,l] = (kron(transpose(B[l,l]),A[k,k])+kron(transpose(D[l,l]),C[k,k]))\(y[:])
              i -= dk
          end
          j += dl
      end
   elseif ~adjAC && adjBD
         """
          The (K,L)th block of X is determined starting from
          bottom-right corner column by column by

               A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

          where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
                               J=K           I=L+1

                               M
                            { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
                             J=K+1
                               M              N
                              SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
                              J=K           I=L+1
         """
         j = n
         for ll = pb:-1:1
             dl = bb[ll]
             dll = 1:dl
             il1 = j+1:n
             l = j-dl+1:j
             i = m
             for kk = pa:-1:1
                 dk = ba[kk]
                 dkk = 1:dk
                 i1 = i-dk+1
                 k = i1:i
                 y = E[k,l]
                 if kk < pa
                    ir = i+1:m
                    W1 = A[k,ir]*E[ir,l]
                    y -= W1*B[l,l]'
                    W2 = C[k,ir]*E[ir,l]
                    y -= W2*D[l,l]'
                 end
                 if ll < pb
                    ic = i1:m
                    WB[k,dll] = E[k,il1]*B[l,il1]'
                    WD[k,dll] = E[k,il1]*D[l,il1]'
                    y -= (A[k,ic]*WB[ic,dll]+C[k,ic]*WD[ic,dll])
                 end
                 E[k,l] = (kron(B[l,l],A[k,k])+kron(D[l,l],C[k,k]))\(y[:])
                 i -= dk
             end
             j -= dl
         end
   elseif adjAC && ~adjBD
      """
      The (K,L)th block of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
                            J=1             I=1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
                            J=1
                             K              L-1
                            SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
                            J=1             I=1
      """
      j = 1
      for ll = 1:pb
          dl = bb[ll]
          dll = 1:dl
          il1 = 1:j-1
          j1 = j+dl-1
          l = j:j1
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = E[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*E[ir,l]
                 y -= W1*B[l,l]
                 W2 = C[ir,k]'*E[ir,l]
                 y -= W2*D[l,l]
              end
              if ll > 1
                 ic = 1:i1
                 WB[k,dll] = E[k,il1]*B[il1,l]
                 y -= A[ic,k]'*WB[ic,dll]
                 WD[k,dll] = E[k,il1]*D[il1,l]
                 y -= C[ic,k]'*WD[ic,dll]
              end
              E[k,l] = (kron(transpose(B[l,l]),transpose(A[k,k]))+kron(transpose(D[l,l]),transpose(C[k,k])))\(y[:])
              i += dk
          end
          j += dl
      end
   elseif adjAC && adjBD
      """
      The (K,L)th block of X is determined starting from
      upper-rght corner column by column by

                 A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
                            J=1            I=L+1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
                            J=1
                             K               N
                            SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
                            J=1            I=L+1
      """
      j = n
      for ll = pb:-1:1
          dl = bb[ll]
          dll = 1:dl
          il1 = j+1:n
          l = j-dl+1:j
          i = 1
          for kk = 1:pa
              dk = ba[kk]
              dkk = 1:dk
              i1 = i+dk-1
              k = i:i1
              y = E[k,l]
              if kk > 1
                 ir = 1:i-1
                 W1 = A[ir,k]'*E[ir,l]
                 y -= W1*B[l,l]'
                 W2 = C[ir,k]'*E[ir,l]
                 y -= W2*D[l,l]'
              end
              if ll < pb
                 ic = 1:i1
                 WB[k,dll] = E[k,il1]*B[l,il1]'
                 WD[k,dll] = E[k,il1]*D[l,il1]'
                 y -= (A[ic,k]'*WB[ic,dll] + C[ic,k]'*WD[ic,dll])
              end
              E[k,l] = (kron(B[l,l],transpose(A[k,k]))+kron(D[l,l],transpose(C[k,k])))\(y[:])
              i += dk
          end
          j -= dl
      end
   end
   return E
end
function gsylvs!(A::T, B::T, C::T, D::T, E::T; adjAC = false, adjBD = false) where {T<:Array{Complex{Float64},2}}
   """
   An extension proposed in [1] of the Bartels-Stewart Schur form based approach [2] is employed.

   References:
   [1] K.-W. E. Chu. The solution of the matrix equation AXB – CXD = E and
       (YA – DZ, YC– BZ) = (E, F). Lin. Alg. Appl., 93:93-105, 1987.
   [2] R. H. Bartels and G. W. Stewart. Algorithm 432: Solution of the matrix equation AX+XB=C.
       Comm. ACM, 15:820–826, 1972.
   """
   m, n = size(E);
   if [m; n; m; n] != LinearAlgebra.checksquare(A,B,C,D)
      throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
   end


   WB = fill(zero(eltype(E)),m,1)
   WD = fill(zero(eltype(E)),m,1)
   if ~adjAC && ~adjBD
      """
      The (K,L)th element of X is determined starting from
      bottom-left corner column by column by

            A(K,K)*X(K,L)*B(L,L) +C(K,K)*X(K,L)*D(L,L) = E(K,L) - R(K,L)

      where
                             M
                 R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L) +
                           J=K+1
                             M             L-1
                            SUM { A(K,J) * SUM [X(J,I)*B(I,L)] } +
                            J=K            I=1

                            M
                         { SUM [C(K,J)*X(J,L)] } * D(L,L) +
                          J=K+1
                            M             L-1
                           SUM { C(K,J) * SUM [X(J,I)*D(I,L)] } +
                           J=K            I=1
      """
      for l = 1:n
          il1 = 1:l-1
          ll = l:l
          for k = m:-1:1
              y = E[k,l]
              kk = k:k
              if k < m
                 ir = k+1:m
                 W1 = A[kk,ir]*E[ir,ll]
                 W2 = C[kk,ir]*E[ir,ll]
                 y -= (W1[1]*B[l,l]+W2[1]*D[l,l])
              end
              if l > 1
                 ic = k:m
                 ZB = E[kk,il1]*B[il1,ll]
                 ZD = E[kk,il1]*D[il1,ll]
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[kk,ic]*WB[ic,1]
                 TD = C[kk,ic]*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              E[k,l] = y/(B[l,l]*A[k,k]+D[l,l]*C[k,k])
             end
      end
   elseif ~adjAC && adjBD
         """
          The (K,L)th element of X is determined starting from
          bottom-right corner column by column by

               A(K,K)*X(K,L)*B(L,L)' + C(K,K)*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

          where
                                M
                    R(K,L) = { SUM [A(K,J)*X(J,L)] } * B(L,L)' +
                              J=K+1
                                M              N
                               SUM { A(K,J) * SUM [X(J,I)*B(L,I)'] } +
                               J=K           I=L+1

                               M
                            { SUM [C(K,J)*X(J,L)] } * D(L,L)' +
                             J=K+1
                               M              N
                              SUM { C(K,J) * SUM [X(J,I)*D(L,I)'] }.
                              J=K           I=L+1
         """
         for l = n:-1:1
             ll = l:l
             il1 = l+1:n
             for k = m:-1:1
                 kk = k:k
                 y = E[k,l]
                 if k < m
                    ir = k+1:m
                    W1 = A[kk,ir]*E[ir,ll]
                    W2 = C[kk,ir]*E[ir,ll]
                    y -= (W1[1]*B[l,l]'+W2[1]*D[l,l]')
                 end
                 if l < n
                    ic = k:m
                    ZB = E[kk,il1]*B[ll,il1]'
                    ZD = E[kk,il1]*D[ll,il1]'
                    WB[k,1] = ZB[1]
                    WD[k,1] = ZD[1]
                    TB = A[kk,ic]*WB[ic,1]
                    TD = C[kk,ic]*WD[ic,1]
                    y -= (TB[1]+TD[1])
                 end
                 E[k,l] = y/(B[l,l]'*A[k,k]+D[l,l]'*C[k,k])
             end
         end
   elseif adjAC && ~adjBD
      """
      The (K,L)th element of X is determined starting from the
      upper-left corner column by column by

      A(K,K)'*X(K,L)*B(L,L) + C(K,K)'*X(K,L)*D(L,L) = E(K,L) - R(K,L),

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L) +
                            J=1
                             K              L-1
                            SUM A(J,K)' * { SUM [X(J,I)*B(I,L)] } +
                            J=1             I=1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L) +
                            J=1
                             K              L-1
                            SUM C(J,K)' * { SUM [X(J,I)*D(I,L)] }.
                            J=1             I=1
      """
      for l = 1:n
          ll = l:l
          il1 = 1:l-1
          for k = 1:m
              kk = k:k
              y = E[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*E[ir,ll]
                 W2 = C[ir,kk]'*E[ir,ll]
                 y -= (W1[1]*B[l,l] + W2[1]*D[l,l])
              end
              if l > 1
                 ic = 1:m
                 ZB = E[kk,il1]*B[il1,ll]
                 ZD = E[kk,il1]*D[il1,ll]
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[ic,kk]'*WB[ic,1]
                 TD = C[ic,kk]'*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              E[k,l] = y/(B[l,l]*A[k,k]'+D[l,l]*C[k,k]')
          end
      end
   elseif adjAC && adjBD
      """
      The (K,L)th element of X is determined starting from
      upper-rght corner column by column by

            A(K,K)'*X(K,L)*B(L,L)' + C(K,K)'*X(K,L)*D(L,L)' = E(K,L) - R(K,L)

      where
                            K-1
                 R(K,L) = { SUM [A(J,K)'*X(J,L)] } * B(L,L)' +
                            J=1
                             K               N
                            SUM A(J,K)' * { SUM [X(J,I)*B(L,I)'] }+
                            J=1            I=L+1

                            K-1
                          { SUM [C(J,K)'*X(J,L)] } * D(L,L)' +
                            J=1
                             K               N
                            SUM C(J,K)' * { SUM [X(J,I)*D(L,I)'] }.
                            J=1            I=L+1
      """
      for l = n:-1:1
          ll = l:l
          il1 = l+1:n
          for k = 1:m
              kk = k:k
              y = E[k,l]
              if k > 1
                 ir = 1:k-1
                 W1 = A[ir,kk]'*E[ir,ll]
                 W2 = C[ir,kk]'*E[ir,ll]
                 y -= (W1[1]*B[l,l]'+W2[1]*D[l,l]')
              end
              if l < n
                 ic = 1:m
                 ZB = E[kk,il1]*B[ll,il1]'
                 ZD = E[kk,il1]*D[ll,il1]'
                 WB[k,1] = ZB[1]
                 WD[k,1] = ZD[1]
                 TB = A[ic,kk]'*WB[ic,1]
                 TD = C[ic,kk]'*WD[ic,1]
                 y -= (TB[1]+TD[1])
              end
              E[k,l] = y/(B[l,l]'*A[k,k]'+D[l,l]'*C[k,k]')
          end
      end
   end
   return E
end
