"""
    trmat(n::Int, m::Int) -> M::LinearOperator

Define the transposition operator `M: X -> X'` for all `n x m` matrices.
"""
function trmat(n::Int,m::Int)
  function prod(x)
    X = reshape(x, n, m)
    return transpose(X)[:]
  end
  function tprod(x)
    X = reshape(x, m, n)
    return transpose(X)[:]
  end
  function ctprod(x)
    X = reshape(x, m, n)
    return transpose(X)[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  m == n ? sym = true : sym = false
  return LinearOperator{Int,F1,F2,F3}(n * m, n * m, sym, sym, prod, tprod, ctprod)
end
trmat(n::Int) = trmat(n,n)
trmat(dims::Tuple{Int,Int}) = trmat(dims[1],dims[2])
"""
    trmat(X::AbstractMatrix) -> M::LinearOperator

Define the transposition operator `M: X -> X'` of all matrices of the size of `X`.
"""
trmat(A::AbstractMatrix) = trmat(size(A))
"""
    lyapop(A :: AbstractMatrix; disc = false, her = false) -> L::LinearOperator

Define, for an `n x n` matrix `A`, the continuous Lyapunov operator `L:X -> AX+XA'`
if `disc = false` or the discrete Lyapunov operator `L:X -> AXA'-X` if `disc = true`.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated `M = Matrix(L)` is a
``n^2 \\times n^2`` matrix such that `vec(Y) = M*vec(X)`.
If `her = true` the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated `M = Matrix(L)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(Y)) = M*vec(triu(X))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function lyapop(A :: AbstractMatrix; disc = false, her = false)
  n = LinearAlgebra.checksquare(A)
  T = eltype(A)
  function prod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A') - X)
      else
        Y = A * X
        return her2vec(Y + Y')
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
        Y = A*X*A' - X
      else
        Y = A*X + X*A'
      end
      return Y[:]
    end
  end
  function tprod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A) - X)
      else
        Y = X * A
        return her2vec(Y + transpose(Y))
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
         Y = transpose(A)*X*A - X
       else
         Y = transpose(A)*X + X*A
       end
       return Y[:]
    end
  end
  function ctprod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A) - X)
      else
        Y = X * A
        return her2vec(Y + Y')
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
        return (A'*X*A - X )[:]
      else
        return (A'*X + X*A)[:]
      end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  her ? N = Int(n*(n+1)/2) : N = n*n
  return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    lyapop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false) -> L::LinearOperator

Define, for a pair `(A,E)` of `n x n` matrices, the continuous Lyapunov operator `L:X -> AXE'+EXA'`
if `disc = false` or the discrete Lyapunov operator `L:X -> AXA'-EXE'` if `disc = true`.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated `M = Matrix(L)` is a
``n^2 \\times n^2`` matrix such that `vec(Y) = M*vec(X)`.
If `her = true` the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated `M = Matrix(L)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(Y)) = M*vec(triu(X))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function lyapop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false)
  n = LinearAlgebra.checksquare(A)
  if n != LinearAlgebra.checksquare(E)
    throw(DimensionMismatch("E must be a square matrix of dimension $n"))
  end
  T = promote_type(eltype(A), eltype(E))
  function prod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A') - utqu(X,E'))
      else
        Y = A * X * E'
        return her2vec(Y + Y')
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
        Y = A*X*A' - E*X*E'
      else
        Y = A*X*E' + E*X*A'
      end
      return Y[:]
    end
  end
  function tprod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A) - utqu(X,E))
      else
        Y = E' * X * A
        return her2vec(Y + transpose(Y))
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
         Y = transpose(A)*X*A - transpose(E)*X*E
       else
         Y = transpose(A)*X*E + transpose(E)*X*A
       end
       return Y[:]
    end
  end
  function ctprod(x)
    if her
      X = vec2her(convert(Vector{T}, x))
      if disc
        return her2vec(utqu(X,A) - utqu(X,E))
      else
        Y = E' * X * A
        return her2vec(Y + Y')
      end
    else
      X = reshape(convert(Vector{T}, x), n, n)
      if disc
        return (A'*X*A - E'*X*E )[:]
      else
        return (A'*X*E + E'*X*A)[:]
      end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  her ? N = Int(n*(n+1)/2) : N = n*n
  return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    invlyapop(A :: AbstractMatrix; disc = false, her = false) -> LINV::LinearOperator

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AX+XA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-X` for `disc = true`, where
`A` is an `n x n` matrix.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated `M = Matrix(LINV)` is a
``n^2 \\times n^2`` matrix such that `vec(X) = M*vec(Y)`.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated `M = Matrix(LINV)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(X)) = M*vec(triu(Y))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function invlyapop(A :: AbstractMatrix; disc = false, her = false)
   n = LinearAlgebra.checksquare(A)
   T = eltype(A)
   function prod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
            return her2vec(lyapd(A,-Y))
         else
             return her2vec(lyapc(A,-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
           return sylvd(-A,A',-Y)[:]
         else
           return sylvc(A,A',Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function tprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
           return her2vec(lyapd(A',-Y))
         else
           return her2vec(lyapc(A',-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
           return sylvd(-A',A,-Y)[:]
         else
            return sylvc(A',A,Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function ctprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
           return her2vec(lyapd(A',-Y))
         else
           return her2vec(lyapc(A',-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
           return sylvd(-A',A,-Y)[:]
         else
           return sylvc(A',A,Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
      end
     end
   end
   F1 = typeof(prod)
   F2 = typeof(tprod)
   F3 = typeof(ctprod)
   her ? N = Int(n*(n+1)/2) : N = n*n
   return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    invlyapop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false) -> LINV::LinearOperator

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AXE'+EXA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-EXE'` for `disc = true`, where
`(A,E)` is a pair of `n x n` matrices.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated `M = Matrix(LINV)` is a
``n^2 \\times n^2`` matrix such that `vec(X) = M*vec(Y)`.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated `M = Matrix(LINV)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(X)) = M*vec(triu(Y))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function invlyapop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false)
   n = LinearAlgebra.checksquare(A)
   if n != LinearAlgebra.checksquare(E)
     throw(DimensionMismatch("E must be a square matrix of dimension $n"))
   end
   T = promote_type(eltype(A), eltype(E))
   function prod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
            return her2vec(lyapd(A,E,-Y))
         else
             return her2vec(lyapc(A,E,-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
           return gsylv(-A,A',E,E',-Y)[:]
         else
           return gsylv(A,E',E,A',Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function tprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
            return her2vec(lyapd(A',E',-Y))
         else
            return her2vec(lyapc(A',E',-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
            return gsylv(-A',A,E',E,-Y)[:]
         else
            return gsylv(A',E,E',A,Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function ctprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, x))
         if disc
           return her2vec(lyapd(A',E',-Y))
         else
           return her2vec(lyapc(A',E',-Y))
         end
       else
         Y = reshape(convert(Vector{T}, x), n, n)
         if disc
           return gsylv(-A',A,E',E,-Y)[:]
         else
           return gsylv(A',E,E',A,Y)[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
      end
     end
   end
   F1 = typeof(prod)
   F2 = typeof(tprod)
   F3 = typeof(ctprod)
   her ? N = Int(n*(n+1)/2) : N = n*n
   return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    invlyapsop(A :: AbstractMatrix; disc = false, her = false) -> LINV::LinearOperator

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AX+XA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-X` for `disc = true`, where
`A` is an `n x n` matrix in Schur form.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated `M = Matrix(LINV)` is a
``n^2 \\times n^2`` matrix such that `vec(X) = M*vec(Y)`.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated `M = Matrix(LINV)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(X)) = M*vec(triu(Y))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function invlyapsop(A :: AbstractMatrix; disc = false, her = false)
   n = LinearAlgebra.checksquare(A)
   T = eltype(A)
   if isa(A,Adjoint)
     error("No calls with adjoint matrices are supported")
   end

   # check A is in Schur form
   if !isschur(A)
       error("The matrix A must be in Schur form")
   end
   function prod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,Y) : lyapcs!(A,Y)
         return her2vec(Y)
       else
         Y = reshape(convert(Vector{T}, -x), n, n)
         if disc
           sylvds!(-A,A,Y,adjB = true)
           return Y[:]
         else
           realcase = eltype(A) <: AbstractFloat
           realcase ? (TA,TB) = ('N','T') : (TA,TB) = ('N','C')
           Y, scale = LAPACK.trsyl!(TA, TB, A, A, Y)
           rmul!(Y, inv(-scale))
           return Y[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function tprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,Y,adj = true) : lyapcs!(A,Y,adj = true)
         return her2vec(Y)
       else
         Y = reshape(convert(Vector{T}, -x), n, n)
         if disc
           sylvds!(-A,A,Y,adjA = true)
           return Y[:]
         else
           realcase = eltype(A) <: AbstractFloat
           realcase ? (TA,TB) = ('T','N') : (TA,TB) = ('C','N')
           Y, scale = LAPACK.trsyl!(TA, TB, A, A, Y)
           rmul!(Y, inv(-scale))
           return Y[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function ctprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,Y,adj = true) : lyapcs!(A,Y,adj = true)
         return her2vec(Y)
       else
         Y = reshape(convert(Vector{T}, -x), n, n)
         if disc
           sylvds!(-A,A,Y,adjA = true)
           return Y[:]
         else
           realcase = eltype(A) <: AbstractFloat
           realcase ? (TA,TB) = ('T','N') : (TA,TB) = ('C','N')
           Y, scale = LAPACK.trsyl!(TA, TB, A, A, Y)
           rmul!(Y, inv(-scale))
           return Y[:]
         end
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
      end
     end
   end
   F1 = typeof(prod)
   F2 = typeof(tprod)
   F3 = typeof(ctprod)
   her ? N = Int(n*(n+1)/2) : N = n*n
   return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    invlyapsop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false) -> LINV::LinearOperator

Define `LINV`, the inverse of the continuous Lyapunov operator `L:X -> AXE'+EXA'` for `disc = false`
or the inverse of the discrete Lyapunov operator `L:X -> AXA'-EXE'` for `disc = true`, where
`(A,E)` is a pair of `n x n` matrices in generalized Schur form.
If `her = false` the inverse Lyapunov operator `LINV:Y -> X` maps general square matrices `Y`
into general square matrices `X`, and the associated `M = Matrix(LINV)` is a
``n^2 \\times n^2`` matrix such that `vec(X) = M*vec(Y)`.
If `her = true` the inverse Lyapunov operator `LINV:Y -> X` maps symmetric/Hermitian matrices `Y`
into symmetric/Hermitian matrices `X`, and the associated `M = Matrix(LINV)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(X)) = M*vec(triu(Y))`.
For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.
"""
function invlyapsop(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false)
   n = LinearAlgebra.checksquare(A)
   if n != LinearAlgebra.checksquare(E)
     throw(DimensionMismatch("E must be a square matrix of dimension $n"))
   end
   T = promote_type(eltype(A), eltype(E))
   if isa(A,Adjoint) || isa(E,Adjoint)
     error("No calls with adjoint matrices are supported")
   end

   # check A is in Schur form
   if !isschur(A,E)
       error("The matrix pair (A,E) must be in generalized Schur form")
   end
   function prod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,E,Y) : lyapcs!(A,E,Y)
         return her2vec(Y)
       else
         Y = copy(reshape(convert(Vector{T}, x), n, n))
         disc ? gsylvs!(A,A,-E,E,Y,adjBD = true) :
                gsylvs!(A,E,E,A,Y,adjBD = true,DBSchur = true)
         return Y[:]
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function tprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,E,Y,adj = true) : lyapcs!(A,E,Y,adj = true)
         return her2vec(Y)
       else
         Y = copy(reshape(convert(Vector{T}, x), n, n))
         disc ? gsylvs!(A,A,-E,E,Y,adjAC = true) :
                gsylvs!(A,E,E,A,Y,adjAC = true,DBSchur = true)
         return Y[:]
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
     end
   end
   function ctprod(x)
     try
       if her
         Y = vec2her(convert(Vector{T}, -x))
         disc ? lyapds!(A,E,Y,adj = true) : lyapcs!(A,E,Y,adj = true)
         return her2vec(Y)
       else
         Y = copy(reshape(convert(Vector{T}, x), n, n))
         disc ? gsylvs!(A,A,-E,E,Y,adjAC = true) :
                gsylvs!(A,E,E,A,Y,adjAC = true,DBSchur = true)
         return Y[:]
       end
     catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
      end
     end
   end
   F1 = typeof(prod)
   F2 = typeof(tprod)
   F3 = typeof(ctprod)
   her ? N = Int(n*(n+1)/2) : N = n*n
   return LinearOperator{T,F1,F2,F3}(N, N, false, false, prod, tprod, ctprod)
end
"""
    sylvop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false) -> M::LinearOperator

Define the continuous Sylvester operator `M: X -> AX+XB` if `disc = false`
or the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`.
"""
function sylvop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    disc ? Y = A * X * B + X : Y = A * X + X * B
    return Y[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    disc ? Y = transpose(A)*X*transpose(B) + X : Y = transpose(A)*X + X*transpose(B)
    return Y[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    disc ? Y = A'*X*B' + X : Y = A'*X + X*B'
    return Y[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false) -> MINV::LinearOperator

Define MINV, the inverse of the continuous Sylvester operator  `M: X -> AX+XB` if `disc = false`
or of the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`.
"""
function invsylvop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
      if disc
        return sylvd(A,B,C)[:]
      else
        return sylvc(A,B,C)[:]
     end
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function tprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
      if disc
        return sylvd(A',B',C)[:]
      else
        return sylvc(A',B',C)[:]
     end
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
         throw("ME:SingularException: Singular operator")
       end
    end
  end
  function ctprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
      if disc
        return sylvd(A',B',C)[:]
      else
        return sylvc(A',B',C)[:]
     end
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
         throw("ME:SingularException: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvsop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false) -> MINV::LinearOperator

Define MINV, the inverse of the continuous Sylvester operator  `M: X -> AX+XB` if `disc = false`
or of the discrete Sylvester operator `M: X -> AXB+X` if `disc = true`, where `A` and `B` are in Schur forms.
"""
function invsylvsop(A :: AbstractMatrix, B :: AbstractMatrix; disc = false)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = eltype(A)
  cmplx = T<:Complex
  if T != eltype(B)
    error("A and B must have the same type")
  end
  adjA = isa(A,Adjoint)
  adjB = isa(B,Adjoint)
  if adjA
     if !isschur(A.parent)
         error("A must be in Schur form")
     end
     !disc && cmplx ? (NA, TA) = ('C','N') : (NA, TA) = ('T','N')
  else
     if !isschur(A)
        error("A must be in Schur form")
     end
     !disc && cmplx ? (NA, TA) = ('N','C') : (NA, TA) = ('N','T')
  end
  if adjB
     if !isschur(B.parent)
         error("B must be in Schur form")
     end
     !disc && cmplx ? (NB, TB) = ('C','N') : (NB, TB) = ('T','N')
  else
     if !isschur(B)
        error("B must be in Schur form")
     end
     !disc && cmplx ? (NB, TB) = ('N','C') : (NB, TB) = ('N','T')
  end
  function prod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if disc
          if !adjA & !adjB
             sylvds!(A, B, C, adjA = false, adjB = false)
          elseif !adjA & adjB
             sylvds!(A, B.parent, C, adjA = false, adjB = true)
          elseif adjA & !adjB
             sylvds!(A.parent, B, C, adjA = true, adjB = false)
          else
             sylvds!(A.parent, B.parent, C, adjA = true, adjB = true)
          end
          return C[:]
       else
          if !adjA & !adjB
             Y, scale = LAPACK.trsyl!(NA, NB, A, B, C)
          elseif !adjA & adjB
             Y, scale = LAPACK.trsyl!(NA, NB, A, B.parent, C)
          elseif adjA & !adjB
             Y, scale = LAPACK.trsyl!(NA, NB, A.parent, B, C)
          else
             Y, scale = LAPACK.trsyl!(NA, NB, A.parent, B.parent, C)
          end
          rmul!(Y, inv(scale))
          return Y[:]
       end
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function tprod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if disc
          if !adjA & !adjB
             sylvds!(A, B, C, adjA = true, adjB = true)
          elseif !adjA & adjB
             sylvds!(A, B.parent, C, adjA = true, adjB = false)
          elseif adjA & !adjB
             sylvds!(A.parent, B, C, adjA = false, adjB = true)
          else
             sylvds!(A.parent, B.parent, C, adjA = false, adjB = false)
          end
          return C[:]
       else
          if !adjA & !adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A, B, C)
          elseif !adjA & adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A, B.parent, C)
          elseif adjA & !adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A.parent, B, C)
          else
             Y, scale = LAPACK.trsyl!(TA, TB, A.parent, B.parent, C)
          end
          rmul!(Y, inv(scale))
          return Y[:]
       end
    catch err
        if isnothing(findfirst("LAPACKException",string(err)))
           rethrow()
        else
           throw("ME:SingularException: Singular operator")
        end
     end
  end
  function ctprod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if disc
          if !adjA & !adjB
             sylvds!(A, B, C, adjA = true, adjB = true)
          elseif !adjA & adjB
             sylvds!(A, B.parent, C, adjA = true, adjB = false)
          elseif adjA & !adjB
             sylvds!(A.parent, B, C, adjA = false, adjB = true)
          else
             sylvds!(A.parent, B.parent, C, adjA = false, adjB = false)
          end
          return C[:]
       else
          if !adjA & !adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A, B, C)
          elseif !adjA & adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A, B.parent, C)
          elseif adjA & !adjB
             Y, scale = LAPACK.trsyl!(TA, TB, A.parent, B, C)
          else
             Y, scale = LAPACK.trsyl!(TA, TB, A.parent, B.parent, C)
          end
          rmul!(Y, inv(scale))
          return Y[:]
       end
    catch err
        if isnothing(findfirst("LAPACKException",string(err)))
           rethrow()
        else
           throw("ME:SingularException: Singular operator")
        end
     end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    sylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> M::LinearOperator

Define the generalized Sylvester operator `M: X -> AXB+CXD`.
"""
function sylvop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
  function prod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A * X * B + C * X * D)[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (transpose(A) * X * transpose(B) + transpose(C) * X * transpose(D) )[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A' * X * B' + C' * X * D' )[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`.
"""
function invsylvop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
  function prod(x)
    E = reshape(convert(Vector{T}, x), m, n)
    try
       return gsylv(A,B,C,D,E)[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
end
  function tprod(x)
    E = reshape(convert(Vector{T}, x), m, n)
    try
       return gsylv(A',B',C',D',E)[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function ctprod(x)
    E = reshape(convert(Vector{T}, x), m, n)
    try
       return gsylv(A',B',C',D',E)[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvsop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; DBSchur = false) -> MINV::LinearOperator

Define MINV, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`,
with the pairs `(A,C)` and `(B,D)` in generalized Schur forms. If DBSchur = true,
the pair `(D,B)` is in generalized Schur form.
"""
function invsylvsop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix; DBSchur = false)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  T = eltype(A)
  cmplx = T<:Complex
  if T != eltype(B) || T != eltype(C) || T != eltype(D)
    error("A, B, C and D must have the same type")
  end
  adjAC = isa(A,Adjoint) & isa(C,Adjoint)
  if adjAC
     if !isschur(A.parent,C.parent)
         error("The pair (A,C) must be in generalized Schur form")
     end
  else
     if !isschur(A,C)
        error("The pair (A,C) must be in generalized Schur form")
     end
  end
  adjBD = isa(B,Adjoint) & isa(D,Adjoint)
  if adjBD
     if DBSchur
       if !isschur(D.parent, B.parent)
           error("The pair (D,B) must be in generalized Schur form")
       end
     else
        if !isschur(B.parent, D.parent)
            error("The pair (B,D) must be in generalized Schur form")
        end
     end
  else
     if DBSchur
        if !isschur(D,B)
           error("The pair (D,B) must be in generalized Schur form")
        end
     else
        if !isschur(B,D)
           error("The pair (B,D) must be in generalized Schur form")
        end
     end
  end
  function prod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjAC & !adjBD
          gsylvs!(A, B, C, D, Y, adjAC = false, adjBD = false, DBSchur = DBSchur)
       elseif !adjAC & adjBD
          gsylvs!(A, B.parent, C, D.parent, Y, adjAC = false, adjBD = true, DBSchur = DBSchur)
       elseif adjAC & !adjBD
          gsylvs!(A.parent, B, C.parent, D, Y, adjAC = true, adjBD = false, DBSchur = DBSchur)
       else
          gsylvs!(A.parent, B.parent, C.parent, D.parent, Y, adjAC = true, adjBD = true, DBSchur = DBSchur)
       end
       return Y[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function tprod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjAC & !adjBD
          gsylvs!(A, B, C, D, Y, adjAC = true, adjBD = true, DBSchur = DBSchur)
       elseif !adjAC & adjBD
          gsylvs!(A, B.parent, C, D.parent, Y, adjAC = true, adjBD = false, DBSchur = DBSchur)
       elseif adjAC & !adjBD
          gsylvs!(A.parent, B, C.parent, D, Y, adjAC = false, adjBD = true, DBSchur = DBSchur)
       else
          gsylvs!(A.parent, B.parent, C.parent, D.parent, Y, adjAC = false, adjBD = false, DBSchur = DBSchur)
       end
       return Y[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function ctprod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjAC & !adjBD
          gsylvs!(A, B, C, D, Y, adjAC = true, adjBD = true, DBSchur = DBSchur)
       elseif !adjAC & adjBD
          gsylvs!(A, B.parent, C, D.parent, Y, adjAC = true, adjBD = false, DBSchur = DBSchur)
       elseif adjAC & !adjBD
          gsylvs!(A.parent, B, C.parent, D, Y, adjAC = false, adjBD = true, DBSchur = DBSchur)
       else
          gsylvs!(A.parent, B.parent, C.parent, D.parent, Y, adjAC = false, adjBD = false, DBSchur = DBSchur)
       end
       return Y[:]
    catch err
       if isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end

"""
    sylvsysop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix) -> M::LinearOperator

Define the operator `M: (X,Y) -> [ AX+YB; CX+YD ]`.
"""
function sylvsysop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  mn = m*n
  function prod(x)
    X = reshape(convert(Vector{T}, x[1:mn]), m, n)
    Y = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    return ([A * X + Y * B C * X + Y * D])[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x[1:mn]), m, n)
    Y = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    return [transpose(A) * X + transpose(C) * Y  X * transpose(B) + Y * transpose(D)][:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x[1:mn]), m, n)
    Y = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    return [A' * X + C' * Y  X * B' + Y * D'][:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(2*mn, 2*mn, false, false, prod, tprod, ctprod)
end
"""
    invsylvsysop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the linear operator `M: (X,Y) -> [ AX+YB; CX+YD ]`.
"""
function invsylvsysop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
  mn = m*n
  function prod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       (X,Y) = sylvsys(A,B,E,C,D,F)
       return [X Y][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function tprod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       (X,Y) = dsylvsys(A',B',E,C',D',F)[:]
       return [X Y][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function ctprod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       (X,Y) = dsylvsys(A',B',E,C',D',F)[:]
       return [X Y][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(2*mn, 2*mn, false, false, prod, tprod, ctprod)
end
"""
    invsylvsyssop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the linear operator `M: (X,Y) -> [ AX+YB; CX+YD ]`,
with the pairs `(A,C)` and `(B,D)` in generalized Schur forms.
"""
function invsylvsyssop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  if [m; n] != LinearAlgebra.checksquare(C,D)
     throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
  end
  T = eltype(A)
  cmplx = T<:Complex
  if T != eltype(B) || T != eltype(C) || T != eltype(D)
    error("A, B, C and D must have the same type")
  end
    if isa(A,Adjoint) || isa(B,Adjoint) || isa(C,Adjoint)  || isa(D,Adjoint)
    error("Only calls with (A, B, C, D) without adjoints are allowed")
  end
  if !isschur(A,C)
     error("The pair (A,C) must be in generalized Schur form")
  end
  if !isschur(B,D)
     error("The pair (B,D) must be in generalized Schur form")
  end
  cmplx ? TA = 'C' : TA = 'T'
  mn = m*n
  function prod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       X, Y, scale =  tgsyl!('N',A,B,E,C,D,F)
       return [rmul!(X,inv(scale)) rmul!(Y,inv(-scale))][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function tprod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       X, Y, scale =  tgsyl!(TA,A,B,E,C,D,-F)
       return [rmul!(X,inv(scale)) rmul!(Y,inv(scale))][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  function ctprod(x)
    E = reshape(convert(Vector{T}, x[1:mn]), m, n)
    F = reshape(convert(Vector{T}, x[mn+1:2*mn]), m, n)
    try
       X, Y, scale =  tgsyl!(TA,A,B,E,C,D,-F)
       return [rmul!(X,inv(scale)) rmul!(Y,inv(scale))][:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          throw("ME:SingularException: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(2*mn, 2*mn, false, false, prod, tprod, ctprod)
end
