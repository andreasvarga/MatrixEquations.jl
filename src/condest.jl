"""
    γ = opnorm1(op)

Compute `γ`, the induced `1`-norm of the linear operator `op`, as the maximum of `1`-norm of the
columns of the associated `m x n` matrix ``M`` `= Matrix(op)`:
```math
\\gamma = \\|op\\|_1 := \\max_{1 ≤ j ≤ n} \\|M_j\\|_1
```
with ``M_j`` the `j`-th column of ``M``. This function is not recommended to be used for large order operators.


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
function opnorm1(op::LinearMaps.LinearMap{T}) where T
  (m, n) = size(op)
  Tnorm = real(T)
  Tsum = promote_type(Float64, Tnorm)
  nrm::Tsum = 0
  for j = 1 : n
      ej = zeros(Tsum, n)
      ej[j] = 1
      try
         nrm = max(nrm,norm(op*ej,1))
      catch err
         # if isnothing(findfirst("SingularException",string(err))) &&
         #    isnothing(findfirst("LAPACKException",string(err)))
         findfirst("SingularException",string(err)) === nothing &&
         findfirst("LAPACKException",string(err)) === nothing ? rethrow() : (return Inf)
      end
  end
  return convert(Tnorm, nrm)
end
"""
    γ = opnorm1est(op)

Compute `γ`, a lower bound of the `1`-norm of the square linear operator `op`, using
reverse communication based computations to evaluate `op * x` and `op' * x`.
It is expected that in most cases ``\\gamma > \\|op\\|_1/10``, which is usually
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
function opnorm1est(op::LinearMaps.LinearMap) 
  m, n = size(op)
  m == n || throw(DimensionMismatch("The operator op must be square"))
  T = promote_type(Float64,eltype(op))
  TR = real(T)
  TR == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
  BIGNUM = 2*eps(TR) / SMLNUM
  cmplx = T<:Complex
  V = Array{T,1}(undef,n)
  X = Array{T,1}(undef,n)
  cmplx ? ISGN = Array{Int,1}(undef,1) : ISGN = Array{Int,1}(undef,n)
  ISAVE = Array{Int,1}(undef,3)
  ANORM = zero(TR)
  KASE = 0
  finish = false
  while !finish
     if cmplx
        ANORM, KASE = LapackUtil.lacn2!(V, X, ANORM, KASE, ISAVE )
     else
        ANORM, KASE = LapackUtil.lacn2!(V, X, ISGN, ANORM, KASE, ISAVE )
     end
     (isfinite(ANORM) && ANORM < BIGNUM) || (return Inf)
     if KASE != 0
        try
           KASE == 1 ? X = op*X : X = op'*X
        catch err
         #   if isnothing(findfirst("SingularException",string(err))) &&
         #      isnothing(findfirst("LAPACKException",string(err)))
           findfirst("SingularException",string(err)) === nothing &&
           findfirst("LAPACKException",string(err)) === nothing ? rethrow() : (return Inf)
        end
      else
        finish = true
     end
  end
  return ANORM
end

"""
    sep = opsepest(opinv; exact = false)

Compute `sep`, an estimation of the `1`-norm separation of a linear operator
`op`, where `opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{sep}  = 1 / \\|opinv\\|_1`` , using an estimate of the `1`-norm, if `exact = false`, or
the computed exact value of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.

The separation of the operator `op` is defined as
```math
\\text{sep} = \\displaystyle\\min_{X\\neq 0} \\frac{\\|op*X\\|}{\\|X\\|}.
```      

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
function opsepest(opinv::LinearMaps.LinearMap; exact = false)
   T = promote_type(Float64,eltype(opinv))
   TR = real(T)
   TR == Float64 ? SMLNUM = reinterpret(Float64, 0x2000000000000000) : SMLNUM = reinterpret(Float32, 0x20000000)
   BIGNUM = 2*eps(TR) / SMLNUM
   ZERO = zero(TR)
   exact ? opinvnrm1 = opnorm1(opinv) : opinvnrm1 = opnorm1est(opinv)
   if opinvnrm1 >= BIGNUM
      return ZERO
   end
   return one(TR)/opinvnrm1
end
"""
    rcond = oprcondest(op, opinv; exact = false)

Compute `rcond`, an estimation of the `1`-norm reciprocal condition number
of a linear operator `op`, where `opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{rcond} = 1 / (\\|op\\|_1\\|opinv\\|_1)``, using estimates of the `1`-norm, if `exact = false`, or
computed exact values of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.

_Note:_ No check is performed to verify that `opinv = inv(op)`.

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
function oprcondest(op::LinearMaps.LinearMap, opinv::LinearMaps.LinearMap; exact = false)
   return opsepest(op, exact = exact)*opsepest(opinv, exact = exact)
end
"""
    rcond = oprcondest(op; exact = false)

Compute `rcond`, an estimation of the `1`-norm reciprocal condition number
of a linear operator `op`, where `op` is one of the defined Lyapunov or Sylvester operators. 
The estimate is computed as
``\\text{rcond} = 1 / (\\|op\\|_1\\|inv(op)\\|_1)``, using estimates of the `1`-norm, if `exact = false`, or
computed exact values of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.
"""
function oprcondest(op::MatrixEquationsMaps{T}; kwargs...) where T
    oprcondest(op, inv(op); kwargs...)
end
