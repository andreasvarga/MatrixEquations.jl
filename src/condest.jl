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
function opnorm1(op::AbstractLinearOperator)
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
function opnorm1est(op::AbstractLinearOperator)
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
function opsepest(opinv::AbstractLinearOperator; exact = false)
   ZERO = zero(0.)
   BIGNUM = eps(2.) / reinterpret(Float64, 0x2000000000000000)
   exact ? opinvnrm1 = opnorm1(opinv) : opinvnrm1 = opnorm1est(opinv)
   if opinvnrm1 >= BIGNUM
      return ZERO
   end
   return one(1.)/opinvnrm1
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
function oprcondest(op:: LinearOperator, opinv::LinearOperator; exact = false)
   return opsepest(op, exact = exact)*opsepest(opinv, exact = exact)
end
"""
    rcond = oprcondest(opnrm1, opinv; exact = false)

Compute `rcond`, an estimate of the `1`-norm reciprocal condition number
of a linear operator `op`, where `opnrm1` is an estimate of the `1`-norm of `op` and
`opinv` is the inverse operator `inv(op)`. The estimate is computed as
``\\text{rcond} = 1 / (\\text{opnrm1}\\|opinv\\|_1)``, using an estimate of the `1`-norm, if `exact = false`, or
the computed exact value of the `1`-norm, if `exact = true`.
The `exact = true` option is not recommended for large order operators.
"""
function oprcondest(opnrm1::Real, opinv::AbstractLinearOperator; exact = false)
  ZERO = zero(0.)
  if opnrm1 == ZERO || size(opinv,1) == 0
     return ZERO
  else
     return opsepest(opinv)/opnrm1
  end
end
