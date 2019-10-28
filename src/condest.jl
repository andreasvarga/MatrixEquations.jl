"""
    sep = lyapsepest(A :: AbstractMatrix; disc = false, her = false)

Compute `sep`, an estimation of the separation of the continuous Lyapunov operator
`L: X -> AX+XA'` if `disc = false` or of the discrete Lyapunov operator
`L: X -> AXA'-X` if `disc = true`, by estimating ``\\sigma_{min}(L^{-1})``,
the least singular value of the corresponding inverse operator ``M^{-1}``,
as the reciprocal of an estimate of ``\\|L^{-1}\\|_1``, the 1-norm of ``L^{-1}``.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated `M := Matrix(L)` is a
``n^2 \\times n^2`` matrix such that `vec(Y) = M*vec(X)`.
If `her = true`, the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated `M := Matrix(L)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(Y)) = M*vec(triu(X))`.

It is expected that in most cases ``1/\\|L^{-1}\\|_1``,
the `true` reciprocal of the 1-norm of ``L^{-1}``, does not differ from
``\\sigma_{min}(L^{-1})`` by more than a
factor of `n`, where `n` is the order of the square matrix `A`.
The separation of the operator `L` is defined as

``\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|L(X)\\|}{\\|X\\|}``

An estimate of the reciprocal condition number of `L` can be computed as `sep```\\|L\\|_1``.

For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.

# Examples
```jldoctest
julia> Ac = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> lyapsepest(Ac)
0.30792472968507323

julia> lyapsepest(Ac,her=true)
0.3936325813328574

julia> Ad = [0.76 0.14 -0.38; 0.42 0.12 0.46; 0.06 0.34 0.72]
3×3 Array{Float64,2}:
 0.76  0.14  -0.38
 0.42  0.12   0.46
 0.06  0.34   0.72

julia> lyapsepest(Ad,disc=true)
0.1215493965942189

julia> lyapsepest(Ad,disc=true,her=true)
0.14437131601027722
```
"""
function lyapsepest(A :: AbstractMatrix; disc = false, her = false)
  n = LinearAlgebra.checksquare(A)
  T2 = promote_type(typeof(1.), eltype(A))
  if eltype(A) !== T2
     A = convert(Matrix{T2},A)
  end

  adj = isa(A,Adjoint)

  # fast computation if A is in Schur form
  if adj && isschur(A.parent)
     M = invlyapsop(A.parent,disc = disc,her = her)'
     return 1. / opnorm1est(M)
  end
  if !adj && isschur(A)
      M = invlyapsop(A,disc = disc,her = her)
       return 1. / opnorm1est(M)
  end

  # Reduce A to Schur form
  if adj
     if eltype(A) == T2
        AS = schur(A.parent).T
     else
        AS = schur(convert(Matrix{T2},A.parent)).T
     end
     M = invlyapsop(AS,disc = disc,her = her)'
  else
     if eltype(A) == T2
        AS = schur(A).T
     else
        AS = schur(convert(Matrix{T2},A)).T
     end
     M = invlyapsop(AS,disc = disc,her = her)
   end
  return 1. / opnorm1est(M)
end
function lyapsepest(A :: Schur; disc = false, her = false)
   M = invlyapsop(A.T,disc = disc,her = her)
   return 1. / opnorm1est(M)
end
"""
    sep = lyapsepest(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false)

Compute `sep`, an estimation of the separation of the continuous Lyapunov operator
`L: X -> AXE'+EXA'` if `disc = false` or of the discrete Lyapunov operator
`L: X -> AXA'-EXE'` if `disc = true`, by estimating ``\\sigma_{min}(L^{-1})``,
the least singular value of the corresponding inverse operator ``L^{-1}``,
as the reciprocal of an estimate of ``\\|L^{-1}\\|_1``, the 1-norm of ``L^{-1}``.
If `her = false` the Lyapunov operator `L:X -> Y` maps general square matrices `X`
into general square matrices `Y`, and the associated `M := Matrix(L)` is a
``n^2 \\times n^2`` matrix such that `vec(Y) = M*vec(X)`.
If `her = true`, the Lyapunov operator `L:X -> Y` maps symmetric/Hermitian matrices `X`
into symmetric/Hermitian matrices `Y`, and the associated `M := Matrix(L)` is a
``n(n+1)/2 \\times n(n+1)/2`` matrix such that `vec(triu(Y)) = M*vec(triu(X))`.

It is expected that in most cases ``1/\\|L^{-1}\\|_1``,
the `true` reciprocal of the 1-norm of ``L^{-1}``, does not differ from
``\\sigma_{min}(L^{-1})`` by more than a
factor of `n`, where `n` is the order of the square matrix `A`.
The separation of the operator `L` is defined as

``\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|L(X)\\|}{\\|X\\|}``

An estimate of the reciprocal condition number of `L` can be computed as `sep```\\|L\\|_1``.

For the definitions of the Lyapunov operators see:

M. Konstantinov, V. Mehrmann, P. Petkov. On properties of Sylvester and Lyapunov
operators. Linear Algebra and its Applications 312:35–71, 2000.

# Examples
```jldoctest
julia> Ac = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> Ec = [10. 3. 0.; 0. 5. -1.; 0. 0. 10.]
3×3 Array{Float64,2}:
 10.0  3.0   0.0
  0.0  5.0  -1.0
  0.0  0.0  10.0

julia> lyapsepest(Ac,Ec)
1.6911585896904682

julia> lyapsepest(Ac,Ec,her=true)
2.225560319078633

julia> Ad = [0.76 0.14 -0.38; 0.42 0.12 0.46; 0.06 0.34 0.72]
3×3 Array{Float64,2}:
 0.76  0.14  -0.38
 0.42  0.12   0.46
 0.06  0.34   0.72

julia> Ed = [1. 3. 0.; 0. 5. -1.; 0. 0. 1.]
3×3 Array{Float64,2}:
 1.0  3.0   0.0
 0.0  5.0  -1.0
 0.0  0.0   1.0

julia> lyapsepest(Ad,Ed,disc=true)
0.08858505235206243

julia> lyapsepest(Ad,Ed,disc=true,her=true)
0.10442981903050726

julia> lyapsepest(Ad,-Ad,disc=true)   # null separation
0.0
```
"""
function lyapsepest(A :: AbstractMatrix, E :: AbstractMatrix; disc = false, her = false)
  n = LinearAlgebra.checksquare(A)
  if isequal(E,I) && size(E,1) == n
     return lyapsepest(A, disc = disc, her = her)
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
  if adjA && adjE && isschur(A.parent,E.parent)
     M = invlyapsop(A.parent, E.parent, disc = disc, her = her)'
     return 1. / opnorm1est(M)
  end
  if !adjA && !adjE && isschur(A,E)
     M = invlyapsop(A, E, disc = disc, her = her)
     return 1. / opnorm1est(M)
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
     M = invlyapsop(AS, ES, disc = disc, her = her)'
  else
     AS, ES = schur(A,E)
     M = invlyapsop(AS, ES, disc = disc, her = her)
  end
  return 1. / opnorm1est(M)
end
lyapsepest(A :: AbstractMatrix, E :: UniformScaling{Bool}; disc = false, her = false) =
lyapsepest(A :: AbstractMatrix, disc = disc, her = her)
function lyapsepest(AE :: GeneralizedSchur; disc = false, her = false)
   M = invlyapsop(AE.S, AE.T, disc = disc, her = her)
   return 1. / opnorm1est(M)
end
"""
    sep = sylvsepest(A :: AbstractMatrix, B :: AbstractMatrix; disc = false)

Compute `sep`, an estimation of the separation of the continuous Sylvester operator
`M: X -> AX+XB` if `disc = false` or of the discrete Sylvester operator
`M: X -> AXB+X` if `disc = true`, by estimating ``\\sigma_{min}(M^{-1})``,
the least singular value of the corresponding inverse operator ``M^{-1}``,
as the reciprocal of an estimate of ``\\|M^{-1}\\|_1``, the 1-norm of ``M^{-1}``.
It is expected that in most cases ``1/\\|M^{-1}\\|_1``,
the `true` reciprocal of the 1-norm of ``M^{-1}``, does not differ from
``\\sigma_{min}(M^{-1})`` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation of the operator `M` is defined as

``\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|M(X)\\|}{\\|X\\|}``

An estimate of the reciprocal condition number of `M` can be computed as `sep```\\|M\\|_1``.

# Examples
```jldoctest
julia> Ac = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> sylvsepest(Ac,Ac')   # same as lyapsepest(Ac)
0.30792472968507323

julia> sylvsepest(Ac,-Ac')  # null separation
0.0

julia> Ad = [0.76 0.14 -0.38; 0.42 0.12 0.46; 0.06 0.34 0.72]
3×3 Array{Float64,2}:
 0.76  0.14  -0.38
 0.42  0.12   0.46
 0.06  0.34   0.72

julia> sylvsepest(Ad,-Ad',disc=true)   # same as lyapsepest(Ad,disc=true)
0.1215493965942189

julia> sylvsepest(Ad,-inv(Ad)',disc=true)  # null separation
6.78969633174597e-17
```
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
      M = invsylvsop(A, B, disc = disc)
      return 1. / opnorm1est(M)
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
   M = invsylvsop(RA, RB, disc = disc)
   return 1. / opnorm1est(M)
end
"""
    sep = sylvsepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)

Compute `sep`, an estimation of the separation of the generalized Sylvester operator
`M: X -> AXB+CXD`, by estimating ``\\sigma_{min}(M^{-1})``,
the least singular value of the corresponding inverse operator ``M^{-1}``,
as the reciprocal of an estimate of ``\\|M^{-1}\\|_1``, the 1-norm of ``M^{-1}``.
It is expected that in most cases ``1/\\|M^{-1}\\|_1``,
the `true` reciprocal of the 1-norm of ``M^{-1}``, does not differ from
``\\sigma_{min}(M^{-1})`` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation operation is defined as

``\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|AXB+CXD\\|}{\\|X\\|}``

An estimate of the reciprocal condition number of `M` can be computed as `sep```\\|M\\|_1``.

# Examples
```jldoctest
julia> Ac = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> Ec = [10. 3. 0.; 0. 5. -1.; 0. 0. 10.]
3×3 Array{Float64,2}:
 10.0  3.0   0.0
  0.0  5.0  -1.0
  0.0  0.0  10.0

julia> sylvsepest(Ac,Ec',Ec,Ac')   # same as lyapsepest(Ac,Ec)
1.6911585896904668

julia> sylvsepest(-Ac,Ec',Ec,Ac')  # null separation
4.504549651611036e-16

julia> Ad = [0.76 0.14 -0.38; 0.42 0.12 0.46; 0.06 0.34 0.72]
3×3 Array{Float64,2}:
 0.76  0.14  -0.38
 0.42  0.12   0.46
 0.06  0.34   0.72

julia> sylvsepest(Ad,-Ad',disc=true)   # same as lyapsepest(Ad,disc=true)
0.1215493965942189

julia> sylvsepest(Ad,-inv(Ad)',disc=true)  # null separation
6.78969633174597e-17
```
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
      return 1. / opnorm1est(invsylvsop(A, B, C, D) )
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
      return 1. / opnorm1est(invsylvsop(AS, BS, CS, DS))
   elseif adjAC && adjBD
      return 1. / opnorm1est(invsylvsop(AS', BS', CS', DS'))
   elseif !adjAC && adjBD
      return 1. / opnorm1est(invsylvsop(AS, BS', CS, DS'))
   else
      return 1. / opnorm1est(invsylvsop(AS', BS, CS', DS))
    end
end
"""
    sep = sylvsyssepest(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix)

Compute `sep`, an estimation of the separation of the generalized Sylvester operator
`M: (X,Y) -> [ AX+YB; CX+YD ] `, by estimating ``\\sigma_{min}(M^{-1})``,
the least singular value of the corresponding inverse operator ``M^{-1}``,
as the reciprocal of an estimate of ``\\|M^{-1}\\|_1``, the 1-norm of ``M^{-1}``.
It is expected that in most cases ``1/\\|M^{-1}\\|_1``,
the `true` reciprocal of the 1-norm of ``M^{-1}``, does not differ from
``\\sigma_{min}(M^{-1})`` by more than a
factor of `sqrt(m*n)`, where `m`  and `n` are the orders of the square matrices
`A` and `B`, respectively.
The separation operation is defined as

``\\text{sep} = \\displaystyle\\min_{[X\\; Y]\\neq 0} \\frac{\\|M(X,Y)\\|}{\\|[X \\; Y]\\|}``

An estimate of the reciprocal condition number of `M` can be computed as `sep```\\|M\\|_1``.

# Example
```jldoctest
julia> A = [3. 4.; 5. 6]
2×2 Array{Float64,2}:
 3.0  4.0
 5.0  6.0

julia> B = [1. 1.; 1. 2.]
2×2 Array{Float64,2}:
 1.0  1.0
 1.0  2.0

julia> C = [1. -2.; -2. -1]
2×2 Array{Float64,2}:
  1.0  -2.0
 -2.0  -1.0

julia> D = [1. -1.; -2. 2]
2×2 Array{Float64,2}:
  1.0  -1.0
 -2.0   2.0

julia> sylvsyssepest(A,B,C,D)
0.23371383344564314
```
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
      return 1. / opnorm1est(invsylvsysop(A, B, C, D) )
   end

   # reduce (A,C) and (B,D) to generalized Schur forms
   AS, CS = schur(A,C)
   BS, DS = schur(B,D)
   return 1. / opnorm1est(invsylvsysop(AS, BS, CS, DS) )
end
"""
    opnorm1(op::AbstractLinearOperator)

Compute the induced operator `1`-norm as the maximum of `1`-norm of the
columns of the `m x n` matrix associated to the linear operator `op`:
```math
\\|op\\|_1 = \\max_{1 ≤ j ≤ n} \\|op * e_j\\|_1
```
with ``e_j`` the `j`-th column of the `n`-th order identity matrix.

# Examples
```jldoctest
julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> opnorm1(lyapop(A))
30.0

julia> opnorm1(invlyapop(A))
3.7666666666666706
```
"""
function opnorm1(op :: AbstractLinearOperator)
  (m, n) = size(op)
  T = eltype(op)
  Tnorm = typeof(float(real(zero(T))))
  Tsum = promote_type(Float64, Tnorm)
  nrm::Tsum = 0
  for j = 1 : n
      ej = zeros(T, n)
      ej[j] = 1
      try
         nrm = max(nrm,norm(op*ej,1))
      catch err
         if isnothing(findfirst("SingularException",string(err))) &&
            isnothing(findfirst("LAPACKException",string(err)))
            rethrow()
         else
            return Inf
         end
      end
  end
  return convert(Tnorm, nrm)
end
"""
    γ = opnorm1est(op :: AbstractLinearOperator)

Compute `γ`, a lower bound of the `1`-norm of the square linear operator `op`, using
reverse communication based computations to evaluate `op * x` and `op' * x`.
It is expected that in most cases ``γ > \\|A\\|_1/10``, which is usually
acceptable for estimating the condition numbers of linear operators.

# Examples
```jldoctest
julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> opnorm1est(lyapop(A))
18.0

julia> opnorm1est(invlyapop(A))
3.76666666666667
```
"""
function opnorm1est(op :: AbstractLinearOperator)
  m, n = size(op)
  if m != n
    throw(DimensionMismatch("The operator op must be square"))
  end
  BIGNUM = eps(2.) / reinterpret(Float64, 0x2000000000000000)
  cmplx = eltype(op)<:Complex
  V = Array{eltype(op),1}(undef,n)
  X = Array{eltype(op),1}(undef,n)
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
           KASE == 1 ? X = op*X : X = op'*X
        catch err
           if isnothing(findfirst("SingularException",string(err))) &&
              isnothing(findfirst("LAPACKException",string(err)))
              rethrow()
           else
              return Inf
           end
        end
      else
        finish = true
     end
  end
  return ANORM
end

"""
    sep = opsepest(opinv :: AbstractLinearOperator; exact = false)

Compute `sep`, an estimation of the `1`-norm separation of a linear operator
`op`, where `opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{sep}  = 1 / \\|opinv\\|_1`` , using an estimate of the `1`-norm, if `exact = false`, or
the computed exact value of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.

The separation of the operator `op` is defined as

``\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|op(X)\\|}{\\|X\\|}``

An estimate of the reciprocal condition number of `op` can be computed as ``\\text{sep}/\\|op\\|_1``.

# Example
```jldoctest
julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> opsepest(invlyapop(A))
0.26548672566371656

julia> 1/opnorm1est(invlyapop(A))
0.26548672566371656

julia> opsepest(invlyapop(A),exact = true)
0.26548672566371656

julia> 1/opnorm1(invlyapop(A))
0.26548672566371656
```
"""
function opsepest(opinv :: AbstractLinearOperator; exact = false)
   ZERO = zero(0.)
   BIGNUM = eps(2.) / reinterpret(Float64, 0x2000000000000000)
   exact ? opinvnrm1 = opnorm1(opinv) : opinvnrm1 = opnorm1est(opinv)
   if opinvnrm1 >= BIGNUM
      return ZERO
   end
   return one(1.)/opinvnrm1
end
"""
    rcond = oprcondest(op::AbstractLinearOperator, opinv :: AbstractLinearOperator; exact = false)

Compute `rcond`, an estimation of the `1`-norm reciprocal condition number
of a linear operator `op`, where `opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{rcond} = 1 / (\\|op\\|_1\\|opinv\\|_1)``, using estimates of the `1`-norm, if `exact = false`, or
computed exact values of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.

Note: No check is performed to verify that `opinv = inv(op)`.

# Examples
```jldoctest
julia> A = [-6. -2. 1.; 5. 1. -1; -4. -2. -1.]
3×3 Array{Float64,2}:
 -6.0  -2.0   1.0
  5.0   1.0  -1.0
 -4.0  -2.0  -1.0

julia> oprcondest(lyapop(A),invlyapop(A))
0.014749262536873142
 
julia> 1/opnorm1est(lyapop(A))/opnorm1est(invlyapop(A))
0.014749262536873142
 
julia> oprcondest(lyapop(A),invlyapop(A),exact = true)
0.008849557522123885
 
julia> 1/opnorm1(lyapop(A))/opnorm1(invlyapop(A))
0.008849557522123885 
```
"""
function oprcondest(op:: LinearOperator, opinv :: LinearOperator; exact = false)
   return opsepest(op, exact = exact)*opsepest(opinv, exact = exact)
end
"""
    rcond = oprcondest(opnrm1::Real, opinv :: AbstractLinearOperator; exact = false)

Compute `rcond`, an estimate of the `1`-norm reciprocal condition number
of a linear operator `op`, where `opnrm1` is an estimate of the `1`-norm of `op` and
`opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{rcond} = 1 / (\\text{opnrm1}\\|opinv\\|_1)``, using an estimate of the `1`-norm, if `exact = false`, or
the computed exact value of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.
"""
function oprcondest(opnrm1::Real, opinv :: AbstractLinearOperator; exact = false)
  ZERO = zero(0.)
  if opnrm1 == ZERO || size(opinv,1) == 0
     return ZERO
  else
     return opsepest(opinv)/opnrm1
  end
end
