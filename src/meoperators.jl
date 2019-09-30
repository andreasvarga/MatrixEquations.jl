"""
    sep = lyapsepest(A :: AbstractMatrix; disc = false)

Compute `sep`, an estimation of the separation of the continuous Lyapunov operator
`M: X -> AX+XA'` if `disc = false` or of the discrete Lyapunov operator
`M: X -> AXA'-X` if `disc = true`, by estimating the least singular value
`σ-min` of the corresponding inverse operator `inv(M)` as the reciprocal of an
estimate of the 1-norm of `inv(M)`. It is expected that in most cases `||inv(M)||₁`,
the true reciprocal 1-norm of `inv(M)` does not differ from `σ-min` by more than a
factor of `n`, where `n` is the order of the square matrix `A`.
The separation operation is defined as

     ``sep = \\min_{X\\neq 0} \\frac{\\|M(X)\\|}{\\|X\\|}``

"""
function lyapsepest(A :: AbstractMatrix; disc = false)
  n = LinearAlgebra.checksquare(A)
  T2 = promote_type(typeof(1.), eltype(A))
  if eltype(A) !== T2
     A = convert(Matrix{T2},A)
  end

  adj = isa(A,Adjoint)

  # fast computation if A is in Schur form
  if (adj && isschur(A.parent)) || (!adj && isschur(A))
      disc ? M = invsylvdsop(-A, A') : M = invsylvcsop(A, A')
      return 1. / opnormest(M)
   end

  # Reduce A to Schur form
  if adj
     if eltype(A) == T2
        AS = schur(A.parent).T
     else
        AS = schur(convert(Matrix{T2},A.parent)).T
     end
     disc ? M = invsylvdsop(-AS', AS) : M = invsylvcsop(AS', AS)
 else
     if eltype(A) == T2
        AS = schur(A).T
     else
        AS = schur(convert(Matrix{T2},A)).T
     end
     disc ? M = invsylvdsop(-AS, AS') : M = invsylvcsop(AS, AS')
  end
  return 1. / opnormest(M)
end
"""
    sep = lyapsepest(A :: AbstractMatrix, E :: AbstractMatrix; disc = false)

Compute `sep`, an estimation of the separation of the continuous Lyapunov operator
`M: X -> AXE'+EXA'` if `disc = false` or of the discrete Lyapunov operator
`M: X -> AXA'-EXE'` if `disc = true`, by estimating the least singular value
`σ-min` of the corresponding inverse operator `inv(M)` as the reciprocal of an
estimate of the 1-norm of `inv(M)`. It is expected that in most cases `||inv(M)||₁`,
the true reciprocal 1-norm of `inv(M)` does not differ from `σ-min` by more than a
factor of `n`, where `n` is the order of the square matrix `A`.
The separation operation is defined as

     ``sep = \\min_{X\\neq 0} \\frac{\\|M(X)\\|}{\\|X\\|}``

"""
function lyapsepest(A :: AbstractMatrix, E :: Union{AbstractMatrix,UniformScaling{Bool},Array{Any,1}}; disc = false)
  n = LinearAlgebra.checksquare(A)
  if isequal(E,I) || isempty(E)
     return lyapsepest(A :: AbstractMatrix; disc = disc)
  end
  if LinearAlgebra.checksquare(E) != n
     throw(DimensionMismatch("E must be a square matrix of dimension $n"))
  end
  T2 = promote_type(typeof(1.), eltype(A), eltype(E))
  if eltype(A) !== T2
     A = convert(Matrix{T2},A)
  end
  if eltype(E) !== T2
     E = convert(Matrix{T2},E)
  end

  adjA = isa(A,Adjoint)
  adjE = isa(E,Adjoint)

  # fast computation if (A,E) is in generalized Schur form
  if (adjA && adjE && isschur(A.parent) && isschur(E.parent)) ||
     (!adjA && !adjE && isschur(A) && isschur(E))
     disc ? M = invgsylvsop(-A, A', E, E') : M = invgsylvsop(A, E', E, A',DBSchur = true)
     return 1. / opnormest(M)
  end

  if adjA && !adjE
      A = copy(A)
      adjA = false
  elseif !adjA && adjE
      E = copy(E)
      adjE = false
  end

  adj = adjA & adjE
  # Reduce (A,E) to generalized Schur form
  if adj
     AS, ES = schur(A.parent,E.parent)
     disc ? M = invgsylvsop(-AS', AS, ES', ES) : M = invgsylvsop(AS', ES, ES', AS,DBSchur = true)
  else
     AS, ES = schur(A,E)
     disc ? M = invgsylvsop(-AS, AS', ES, ES') : M = invgsylvsop(AS, ES', ES, AS',DBSchur = true)
  end
  return 1. / opnormest(M)
end
"""
    sep = sylvsepest(A :: AbstractMatrix, B :: AbstractMatrix; disc = false)

Compute `sep`, an estimation of the separation of the continuous Sylvester operator
`M: X -> AX+XB` if `disc = false` or of the discrete Sylvester operator
`M: X -> AXB+X` if `disc = true`, by estimating the least singular value
`σ-min` of the corresponding inverse operator `inv(M)` as the reciprocal of an
estimate of the 1-norm of `inv(M)`. It is expected that in most cases `||inv(M)||₁`,
the true reciprocal 1-norm of `inv(M)` does not differ from `σ-min` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation operation is defined as

     ``sep = \\min_{X\\neq 0} \\frac{\\|M(X)\\|}{\\|X\\|}``

"""
function sylvsepest(A::AbstractMatrix, B::AbstractMatrix; disc = false)
   m = LinearAlgebra.checksquare(A)
   n = LinearAlgebra.checksquare(B)
   T2 = promote_type(typeof(1.), eltype(A), eltype(B))
   if eltype(A) !== T2
      A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      B = convert(Matrix{T2},B)
   end

   adjA = isa(A,Adjoint)
   adjB = isa(B,Adjoint)

   # fast computation if A and B are in Schur forms
   if (!adjA && !adjB && isschur(A) && isschur(B)) ||
      (adjA && adjB && isschur(A.parent) && isschur(B.parent)) ||
      (!adjA && adjB && isschur(A) && isschur(B.parent)) ||
      (adjA && !adjB && isschur(A.parent) && isschur(B))
      disc ? M = invsylvdsop(A, B) : M = invsylvcsop(A, B)
      return 1. / opnormest(M)
   end

   if adjA
      RA = schur(A.parent).T'
   else
      RA = schur(A).T
   end
   if adjB
      RB = schur(B.parent).T'
   else
      RB = schur(B).T
   end
  disc ? M = invsylvdsop(RA, RB) : M = invsylvcsop(RA, RB)
  return 1. / opnormest(M)
end
"""
    sep = sylvsepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)

Compute `sep`, an estimation of the separation of the generalized Sylvester operator
`M: X -> AXB+CXD`, by estimating the least singular value
`σ-min` of the corresponding inverse operator `inv(M)` as the reciprocal of an
estimate of the 1-norm of `inv(M)`. It is expected that in most cases `||inv(M)||₁`,
the true reciprocal 1-norm of `inv(M)` does not differ from `σ-min` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation operation is defined as

     ``sep = \\min_{X\\neq 0} \\frac{\\|AXB+CXD\\|}{\\|X\\|}``

"""
function sylvsepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
   m = LinearAlgebra.checksquare(A)
   n = LinearAlgebra.checksquare(B)
   if [m; n] != LinearAlgebra.checksquare(C,D)
      throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
   end
   T2 = promote_type(typeof(1.), eltype(A), eltype(B), eltype(C), eltype(D))
   if eltype(A) !== T2
      A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      B = convert(Matrix{T2},B)
   end
   if eltype(C) !== T2
      C = convert(Matrix{T2},C)
   end
   if eltype(D) !== T2
      D = convert(Matrix{T2},D)
   end
   adjA = isa(A,Adjoint)
   adjB = isa(B,Adjoint)
   adjC = isa(C,Adjoint)
   adjD = isa(D,Adjoint)

   adjAC = adjA && adjC
   adjBD = adjB && adjD

   # fast computation if (A,C) and (B,D) are in generalized Schur forms
   if (!adjAC && !adjBD && isschur(A,C) && isschur(B,D)) ||
      (adjAC && adjBD && isschur(A.parent,C.parent) && isschur(B.parent,D.parent)) ||
      (!adjAC && adjBD && isschur(A,C) && isschur(B.parent,D.parent)) ||
      (adjAC && !adjBD && isschur(A.parent,C.parent) && isschur(B,D))
      return 1. / opnormest(invgsylvsop(A, B, C, D) )
   end

   # reduce (A,C) and (B,D) to generalized Schur forms
   if adjAC
      AS, CS = schur(A.parent,C.parent)
   else
      if adjA
         A = copy(A)
      end
      if adjC
         C = copy(C)
      end
      AS, CS = schur(A,C)
   end
   if adjBD
      BS, DS = schur(B.parent,D.parent)
   else
      if adjB
          B = copy(B)
      end
      if adjD
          D = copy(D)
      end
      BS, DS = schur(B,D)
   end
   if !adjAC && !adjBD
      return 1. / opnormest(invgsylvsop(AS, BS, CS, DS))
   elseif adjAC && adjBD
      return 1. / opnormest(invgsylvsop(AS', BS', CS', DS'))
   elseif !adjAC && adjBD
      return 1. / opnormest(invgsylvsop(AS, BS', CS, DS'))
   else
      return 1. / opnormest(invgsylvsop(AS', BS, CS', DS))
    end
end
"""
    sep = sylvsyssepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)

Compute `sep`, an estimation of the separation of the generalized Sylvester operator
`M: (X,Y) -> [ AX+YB; CX+YD ] `, by estimating the least singular value
`σ-min` of the corresponding inverse operator `inv(M)` as the reciprocal of an
estimate of the 1-norm of `inv(M)`. It is expected that in most cases `||inv(M)||₁`,
the true reciprocal 1-norm of `inv(M)` does not differ from `σ-min` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation operation is defined as

     ``sep = \\min_{[X Y]\\neq 0} \\frac{\\|M(X,Y)\\|}{\\|[X Y]\\|}``

"""
function sylvsyssepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)
   m = LinearAlgebra.checksquare(A)
   n = LinearAlgebra.checksquare(B)
   if [m; n] != LinearAlgebra.checksquare(C,D)
      throw(DimensionMismatch("A, B, C and D have incompatible dimensions"))
   end
   T2 = promote_type(typeof(1.), eltype(A), eltype(B), eltype(C), eltype(D))
   if eltype(A) !== T2
      A = convert(Matrix{T2},A)
   end
   if eltype(B) !== T2
      B = convert(Matrix{T2},B)
   end
   if eltype(C) !== T2
      C = convert(Matrix{T2},C)
   end
   if eltype(D) !== T2
      D = convert(Matrix{T2},D)
   end

   if isa(A,Adjoint)
     A = copy(A)
   end
   if isa(B,Adjoint)
     B = copy(B)
   end
   if isa(C,Adjoint)
     C = copy(C)
   end
   if isa(D,Adjoint)
     D = copy(D)
   end

   # fast computation if (A,C) and (B,D) are in generalized Schur forms
   if isschur(A,C) && isschur(B,D)
      return 1. / opnormest(invsylvsysop(A, B, C, D) )
   end

   # reduce (A,C) and (B,D) to generalized Schur forms
   AS, CS = schur(A,C)
   BS, DS = schur(B,D)
   return 1. / opnormest(invsylvsysop(AS, BS, CS, DS) )
end

"""
    γ = opnormest(A :: LinearOperator)

Compute `γ`, a lower bound of the 1-norm of the linear operator `A`, using
reverse communication based computations to evaluate `A*x` and `A'*x`.
It is expected that in most cases `γ > ||A||₁/10`, which is usually
acceptable for estimating condition numbers of linear operators.
"""
function opnormest(A :: LinearOperator)
  m, n = size(A)
  if m != n
    throw(DimensionMismatch("The operator A must be square"))
  end
  BIGNUM = eps(2.) / reinterpret(Float64, 0x2000000000000000)
  cmplx = eltype(A)<:Complex
  V = Array{eltype(A),1}(undef,n)
  X = Array{eltype(A),1}(undef,n)
  cmplx ? ISGN = Array{Int,1}(undef,1) : ISGN = Array{Int,1}(undef,n)
  ISAVE = Array{Int,1}(undef,3)
  ANORM = 0.
  KASE = 0
  finish = false
  while !finish
     if cmplx
        ANORM, KASE = LapackUtil.lacn2!(V, X, ANORM, KASE, ISAVE )
     else
        ANORM, KASE = LapackUtil.lacn2!(V, X, ISGN, ANORM, KASE, ISAVE )
     end
     if !isfinite(ANORM) || ANORM >= BIGNUM
        return Inf
      end
     if KASE != 0
        try
           KASE == 1 ? X = A*X : X = A'*X
        catch err
           if !isnothing(findfirst("MESingOpErr",string(err)))
              return Inf
           else
              rethrow()
           end
        end
      else
        finish = true
     end
  end
  return ANORM
end

"""
    RCOND = oprcondest(ANORM1::Real, AINV :: LinearOperator)

Compute RCOND, an estimation of the 1-norm reciprocal condition number
of a linear operator `A`, where ANORM1 is an estimation of the 1-norm of `A` and
`AINV` is the inverse operator `A^(-1)`. The estimate is computed as
    RCOND = 1 / (ANORM1*opnormest(AINV))
"""
function oprcondest(ANORM1::Real, AINV :: LinearOperator)
  ZERO = zero(0.)
  if ANORM1 == ZERO || size(AINV,1) == 0
     return ZERO
  else
     BIGNUM = eps(2.) / reinterpret(Float64, 0x2000000000000000)
     AINVNORM1 = opnormest(AINV)
     if AINVNORM1 >= BIGNUM
       return ZERO
     end
     return one(1.)/ANORM1/AINVNORM1
  end
end
"""
    trmat(n::Int, m::Int) -> M::LinearOperator

Define the linear permutation operator `M: X -> X'` for the transposition of all
`n x m` matrices.
"""
function trmat(n::Int,m::Int)
  function prod(x)
    X = reshape(x, n, m)
    return transpose(X)[:]
  end
  function tprod(x)
    X = reshape(x, n, m)
    return transpose(X)[:]
  end
  function ctprod(x)
    X = reshape(x, n, m)
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

Define the linear permutation operator `M: X -> X'` for the transposition of
all matrices of the size of `X`.
"""
trmat(A::AbstractMatrix) = trmat(size(A))
"""
    lyapcop(A :: AbstractMatrix) -> M::LinearOperator

Define the continuous Lyapunov operator `M: X -> AX+XA'`.
"""
function lyapcop(A :: AbstractMatrix)
  n = LinearAlgebra.checksquare(A)
  T = eltype(A)
  function prod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A * X + X * A')[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (transpose(A) * X + X * A)[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A' * X + X * A)[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(n * n, n * n, false, false, prod, tprod, ctprod)
end
"""
    invlyapcop(A :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the continuous Lyapunov operator `M: X -> AX+XA'`.
"""
invlyapcop(A) = invsylvcop(A, A')
"""
    invlyapcsop(A :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the continuous Lyapunov operator `M: X -> AX+XA'`,
with `A` in Schur form.
"""
invlyapcsop(A) = invsylvcsop(A, A')
"""
    lyapcop(A :: AbstractMatrix, E :: AbstractMatrix) -> M::LinearOperator

Define the continuous generalized Lyapunov operator `M: X -> AXE'+EXA'`.
"""
function lyapcop(A :: AbstractMatrix, E :: AbstractMatrix)
  n = LinearAlgebra.checksquare(A)
  if n != LinearAlgebra.checksquare(E)
    throw(DimensionMismatch("E must be a square matrix of dimension $n"))
  end
  T = promote_type(eltype(A), eltype(E))
  function prod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A * X * E'+ E * X * A')[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (transpose(A) * X * E + transpose(E) * X * A)[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A' * X * E + E' * X * A)[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(n * n, n * n, false, false, prod, tprod, ctprod)
end
"""
    invlyapcop(A :: AbstractMatrix, E :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the generalized continuous Lyapunov `M: X -> AXE'+EXA'`.
"""
invlyapcop(A,E) = invgsylvop(A, E', E, A')
"""
    invlyapcsop(A :: AbstractMatrix, E :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the continuous generalized Lyapunov `M: X -> AXE'+EXA'`,
with the pair `(A,E)` in generalized Schur form.
"""
invlyapcsop(A,E) = invgsylvsop(A, E', E, A', DBSchur = true)
"""
    lyapdop(A :: AbstractMatrix) -> M::LinearOperator

Define the discrete Lyapunov operator `M: X -> AXA'-X`.
"""
function lyapdop(A :: AbstractMatrix)
  n = LinearAlgebra.checksquare(A)
  T = eltype(A)
  function prod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A * X * A' - X)[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (transpose(A) * X * A - X)[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A' * X * A - X)[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(n * n, n * n, false, false, prod, tprod, ctprod)
end
"""
    invlyapdop(A :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the discrete Lyapunov operator `M: X -> AXA'-X`.
"""
invlyapdop(A) = -invsylvdop(-A, A')
"""
    invlyapdsop(A :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the discrete Lyapunov operator `M: X -> AXA'-X`,
with `A` in Schur form.
"""
invlyapdsop(A) = -invsylvdsop(-A, A')
"""
    lyapdop(A :: AbstractMatrix, E :: AbstractMatrix) -> M::LinearOperator

Define the discrete generalized Lyapunov operator `M: X -> AXA'-EXE'`.
"""
function lyapdop(A :: AbstractMatrix, E :: AbstractMatrix)
  n = LinearAlgebra.checksquare(A)
  if n != LinearAlgebra.checksquare(E)
    throw(DimensionMismatch("E must be a square matrix of dimension $n"))
  end
  T = promote_type(eltype(A), eltype(E))
  function prod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A * X * A' - E * X * E')[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (transpose(A) * X * A - transpose(E) * X * E)[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), n, n)
    return (A' * X * A - E' * X * E)[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(n * n, n * n, false, false, prod, tprod, ctprod)
end
"""
    invlyapdop(A :: AbstractMatrix, E :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the discrete generalized Lyapunov operator `M: X -> AXA'-EXE'`.
"""
invlyapdop(A,E) = -invgsylvop(-A, A',E, E')
"""
    invlyapdsop(A :: AbstractMatrix, E :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the discrete generalized Lyapunov operator `M: X -> AXA'-EXE'`,
with the pair `(A,E)` in generalized Schur form.
"""
invlyapdsop(A,E) = -invgsylvsop(-A, A', E, E')

"""
    sylvcop(A :: AbstractMatrix, B :: AbstractMatrix) -> M::LinearOperator

Define the (continuous) Sylvester operator `M: X -> AX+XB`.
"""
function sylvcop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A * X + X * B)[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (transpose(A) * X + X * transpose(B))[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A' * X + X * B')[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvcop(A :: AbstractMatrix, B :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the (continuous) Sylvester operator  `M: X -> AX+XB`.
"""
function invsylvcop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvc(A,B,C)[:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  function tprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvc(A',B',C)[:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
         error("MESingOpErr: Singular operator")
       end
    end
  end
  function ctprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvc(A',B',C)[:]
    catch err
       if isnothing(findfirst("LAPACKException",string(err))) ||
          isnothing(findfirst("SingularException",string(err)))
          rethrow()
       else
         error("MESingOpErr: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvcsop(A :: AbstractMatrix, B :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the (continuous) Sylvester operator  `M: X -> AX+XB`,
with `A` and `B` in Schur forms.
"""
function invsylvcsop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = eltype(A)
  cmplx = T<:Complex
  if T != eltype(B)
    error("A and B must have the same type")
  end
  adjA = isa(A,Adjoint)
  if adjA
     if !isschur(A.parent)
         error("A must be in Schur form")
     end
     cmplx ? (NA, TA) = ('C','N') : (NA, TA) = ('T','N')
  else
     if !isschur(A)
        error("A must be in Schur form")
     end
     cmplx ? (NA, TA) = ('N','C') : (NA, TA) = ('N','T')
  end
  adjB = isa(B,Adjoint)
  if adjB
     if !isschur(B.parent)
         error("B must be in Schur form")
     end
     cmplx ? (NB, TB) = ('C','N') : (NB, TB) = ('T','N')
  else
     if !isschur(B)
        error("B must be in Schur form")
     end
     cmplx ? (NB, TB) = ('N','C') : (NB, TB) = ('N','T')
  end
  function prod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
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
    catch err
       if isnothing(findfirst("LAPACKException",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  function tprod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
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
     catch err
        if isnothing(findfirst("LAPACKException",string(err)))
           rethrow()
        else
           error("MESingOpErr: Singular operator")
        end
     end
  end
  function ctprod(x)
    C = copy(reshape(convert(Vector{T}, x), m, n))
    try
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
     catch err
        if isnothing(findfirst("LAPACKException",string(err)))
           rethrow()
        else
           error("MESingOpErr: Singular operator")
        end
     end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end

"""
    sylvdop(A :: AbstractMatrix, B :: AbstractMatrix) -> M::LinearOperator

Define the (discrete) Sylvester operator `M: X -> AXB+X`.
"""
function sylvdop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A * X * B + X)[:]
  end
  function tprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (transpose(A) * X * transpose(B) + X )[:]
  end
  function ctprod(x)
    X = reshape(convert(Vector{T}, x), m, n)
    return (A' * X * B' + X )[:]
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvdop(A :: AbstractMatrix, B :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the (discrete) Sylvester operator `M: X -> AXB+X`.
"""
function invsylvdop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = promote_type(eltype(A), eltype(B))
  function prod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvd(A,B,C)[:]
    catch err
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  function tprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvd(A',B',C)[:]
    catch err
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  function ctprod(x)
    C = reshape(convert(Vector{T}, x), m, n)
    try
       return sylvd(A',B',C)[:]
    catch err
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invsylvdsop(A :: AbstractMatrix, B :: AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the (discrete) Sylvester operator `M: X -> AXB+X`,
with `A` and `B` in Schur forms.
"""
function invsylvdsop(A :: AbstractMatrix, B :: AbstractMatrix)
  m = LinearAlgebra.checksquare(A)
  n = LinearAlgebra.checksquare(B)
  T = eltype(A)
  cmplx = T<:Complex
  if T != eltype(B)
    error("A and B must have the same type")
  end
  adjA = isa(A,Adjoint)
  if adjA
     if !isschur(A.parent)
         error("A must be in Schur form")
     end
  else
     if !isschur(A)
        error("A must be in Schur form")
     end
  end
  adjB = isa(B,Adjoint)
  if adjB
     if !isschur(B.parent)
         error("B must be in Schur form")
     end
  else
     if !isschur(B)
        error("B must be in Schur form")
     end
  end
  function prod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjA & !adjB
          sylvds!(A, B, Y, adjA = false, adjB = false)
       elseif !adjA & adjB
          sylvds!(A, B.parent, Y, adjA = false, adjB = true)
       elseif adjA & !adjB
          sylvds!(A.parent, B, Y, adjA = true, adjB = false)
       else
          sylvds!(A.parent, B.parent, Y, adjA = true, adjB = true)
       end
       return Y[:]
     catch err
        if isnothing(findfirst("MESingErr",string(err)))
           rethrow()
        else
           error("MESingOpErr: Singular operator")
        end
     end
  end
  function tprod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjA & !adjB
          sylvds!(A, B, Y; adjA = true, adjB = true)
       elseif !adjA & adjB
          sylvds!(A, B.parent, Y; adjA = true, adjB = false)
       elseif adjA & !adjB
          sylvds!(A.parent, B, Y; adjA = false, adjB = true)
       else
          sylvds!(A.parent, B.parent, Y; adjA = false, adjB = false)
       end
       return Y[:]
     catch err
        if isnothing(findfirst("MESingErr",string(err)))
           rethrow()
        else
           error("MESingOpErr: Singular operator")
        end
     end
  end
  function ctprod(x)
    Y = copy(reshape(convert(Vector{T}, x), m, n))
    try
       if !adjA & !adjB
          sylvds!(A, B, Y; adjA = true, adjB = true)
       elseif !adjA & adjB
          sylvds!(A, B.parent, Y; adjA = true, adjB = false)
       elseif adjA & !adjB
          sylvds!(A.parent, B, Y; adjA = false, adjB = true)
       else
          sylvds!(A.parent, B.parent, Y; adjA = false, adjB = false)
       end
       return Y[:]
    catch err
        if isnothing(findfirst("MESingErr",string(err)))
           rethrow()
        else
           error("MESingOpErr: Singular operator")
        end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    gsylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> M::LinearOperator

Define the generalized Sylvester operator `M: X -> AXB+CXD`.
"""
function gsylvop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
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
    invgsylvop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix) -> MINV::LinearOperator

Define MINV, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`.
"""
function invgsylvop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix)
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
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
end
  function tprod(x)
    E = reshape(convert(Vector{T}, x), m, n)
    try
       return gsylv(A',B',C',D',E)[:]
    catch err
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  function ctprod(x)
    E = reshape(convert(Vector{T}, x), m, n)
    try
       return gsylv(A',B',C',D',E)[:]
    catch err
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(m * n, n * m, false, false, prod, tprod, ctprod)
end
"""
    invgsylvsop(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; DBSchur = false) -> MINV::LinearOperator

Define MINV, the inverse of the generalized Sylvester operator `M: X -> AXB+CXD`,
with the pairs `(A,C)` and `(B,D)` in generalized Schur forms. If DBSchur = true,
the pair `(D,B)` is in generalized Schur form.
"""
function invgsylvsop(A :: AbstractMatrix, B :: AbstractMatrix, C :: AbstractMatrix, D :: AbstractMatrix; DBSchur = false)
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
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
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
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
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
       if isnothing(findfirst("MESingErr",string(err)))
          rethrow()
       else
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
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
          error("MESingOpErr: Singular operator")
       end
    end
  end
  F1 = typeof(prod)
  F2 = typeof(tprod)
  F3 = typeof(ctprod)
  return LinearOperator{T,F1,F2,F3}(2*mn, 2*mn, false, false, prod, tprod, ctprod)
end
