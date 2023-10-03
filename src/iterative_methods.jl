"""
    gtsylvi(A, B, C, D, E; mx, nx, abstol, reltol, maxiter) -> (X,info)

Compute a solution `X` of the generalized T-Sylvester matrix equation

      ∑ A_i*X*B_i + ∑ C_j*transpose(X)*D_j = E, 
      
where `A_i` and `C_j` are matrices having the same row dimension equal to the row dimension of `E` and 
`B_i` and `D_j` are matrices having the same column dimension equal to the column dimension of `E`. 
`A_i` and `B_i` are contained in the `k`-vectors of matrices `A` and `B`, respectively, and 
`C_j` and `D_j` are contained in the `l`-vectors of matrices `C` and `D`, respectively. 
Any of the component matrices can be given as an `UniformScaling`. 
The keyword parameters `mx` and `nx` can be used to specify the row and column dimensions of `X`, 
if they cannot be inferred from the data.

A least-squares solution `X` is determined using a conjugate-gradient based iterative method applied 
to a suitably defined T-Sylvester linear operator `L:X -> Y` such that `L(X) = E` or `norm(L(X) - E)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 

_Note:_ For the derivation of the adjoint equation see reference [1], which also served as motivation to implement a general linear matrix equation solver in Julia.  

[1] Uhlig, F., Xu, A.B. Iterative optimal solutions of linear matrix equations for hyperspectral and multispectral image fusing. Calcolo 60, 26 (2023). 
    [https://doi.org/10.1007/s10092-023-00514-8](https://doi.org/10.1007/s10092-023-00514-8)
"""
function gtsylvi(A::Vector{TA}, B::Vector{TB}, C::Vector{TC}, D::Vector{TD}, E::AbstractArray{T}; mx = -1, nx = -1, abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T,TA,TB,TC,TD}
    LT = gsylvop(A,B,C,D; mx, nx)
    xt, info = cgls(LT,vec(E); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return reshape(xt,LT.mx,LT.nx), info
end
"""
    ghsylvi(A, B, C, D, E; mx, nx, abstol, reltol, maxiter) -> (X,info)

Compute a solution `X` of the generalized H-Sylvester matrix equation

      ∑ A_i*X*B_i + ∑ C_j*X'*D_j = E, 
      
where `A_i` and `C_j` are matrices having the same row dimension equal to the row dimension of `E` and 
`B_i` and `D_j` are matrices having the same column dimension equal to the column dimension of `E`. 
`A_i` and `B_i` are contained in the `k`-vectors of matrices `A` and `B`, respectively, and 
`C_j` and `D_j` are contained in the `l`-vectors of matrices `C` and `D`, respectively. 
Any of the component matrices can be given as an `UniformScaling`. 
The keyword parameters `mx` and `nx` can be used to specify the row and column dimensions of `X`, 
if they cannot be inferred from the data.

A least-squares solution `X` is determined using a conjugate-gradient based iterative method applied 
to a suitably defined T-Sylvester linear operator `L:X -> Y` such that `L(X) = E` or `norm(L(X) - E)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: maxiter = 1000). 

_Note:_ For the derivation of the adjoint equation see reference [1], which also served as motivation to implement a general linear matrix equation solver in Julia.  

[1] Uhlig, F., Xu, A.B. Iterative optimal solutions of linear matrix equations for hyperspectral and multispectral image fusing. Calcolo 60, 26 (2023). 
    [https://doi.org/10.1007/s10092-023-00514-8](https://doi.org/10.1007/s10092-023-00514-8)
"""
function ghsylvi(A::Vector{TA}, B::Vector{TB}, C::Vector{TC}, D::Vector{TD}, E::AbstractArray{T}; mx = -1, nx = -1, abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T,TA,TB,TC,TD}
    LT = gsylvop(A,B,C,D; mx, nx, htype = true)
    xt, info = cgls(LT,vec(E); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return reshape(xt,LT.mx,LT.nx), info
end
"""
    tlyapci(A, C; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for a rectangular `A` and a symmetric `C` a solution `X` of the continuous T-Lyapunov matrix equation

                A*X +transpose(X)*transpose(A) = C   if adj = false, 

or

                A*transpose(X) + X*transpose(A) = C   if adj = true.

For a matrix `A`, a least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: maxiter = 1000). 
"""
function tlyapci(A::AbstractMatrix{T}, C::AbstractMatrix{T}; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    issymmetric(C) || throw(ArgumentError("C must be symmetric"))
    LT = tlyapop(A; adj)
    xt, info = cgls(LT,vec(C); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return adj ? reshape(xt,m,n) : reshape(xt,n,m), info
end
"""
    hlyapci(A, C; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for a rectangular `A` and a hermitian `C` a solution `X` of the continuous H-Lyapunov matrix equation

                A*X +X'*A' = C   if adj = false, 

or

                A*X' + X*A' = C   if adj = true.

For a matrix `A`, a least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: maxiter = 1000). 
"""
function hlyapci(A::AbstractMatrix{T}, C::AbstractMatrix{T}; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    ishermitian(C) || throw(ArgumentError("C must be hermitian"))
    LT = hlyapop(A; adj)
    xt, info = cgls(LT,vec(C); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return adj ? reshape(xt,m,n) : reshape(xt,n,m), info
end

"""
    tulyapci(U, Q; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for an upper triangular `U` and a symmetric `Q` an upper triangular solution `X` of the continuous T-Lyapunov matrix equation

                U*transpose(X) + X*transpose(U) = Q   if adj = false, 

or

                transpose(U)*X + transpose(X)*U = Q   if adj = true.

For a `n×n` upper triangular matrix `U`, a least-squares upper-triangular solution `X` is determined using a conjugate-gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y`, which maps upper triangular matrices `X`
into upper triangular matrices `Y`, and the associated matrix `M = Matrix(L)` is ``n(n+1)/2 \\times n(n+1)/2``. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function tulyapci(U::AbstractMatrix{T}, Q::AbstractMatrix{T}; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    n = LinearAlgebra.checksquare(U)
    n == LinearAlgebra.checksquare(Q) || throw(DimensionMismatch("U and Q have incompatible dimensions"))
    istriu(U) || throw(ArgumentError("U must be upper triangular"))
    issymmetric(Q) || throw(ArgumentError("Q must be symmetric"))
    LT = tulyapop(adj ? U : transpose(U))
    xt, info = cgls(LT,triu2vec(Q); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return vec2triu(xt), info
end
"""
    hulyapci(U, Q; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for an upper triangular `U` and a hermitian `Q` an upper triangular solution `X` of the continuous H-Lyapunov matrix equation

                U*X' + X*U' = Q   if adj = false, 

or

                U'*X + X'*U = Q   if adj = true.

For a `n×n` upper triangular matrix `U`, a least-squares upper-triangular solution `X` is determined using a conjugate-gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y`, which maps upper triangular matrices `X`
into upper triangular matrices `Y`, and the associated matrix `M = Matrix(L)` is ``n(n+1)/2 \\times n(n+1)/2``. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`).  
"""
function hulyapci(U::AbstractMatrix{T}, Q::AbstractMatrix{T}; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    n = LinearAlgebra.checksquare(U)
    n == LinearAlgebra.checksquare(Q) || throw(DimensionMismatch("U and Q have incompatible dimensions"))
    istriu(U) || throw(ArgumentError("U must be upper triangular"))
    ishermitian(Q) || throw(ArgumentError("Q must be symmetric"))
    LT = hulyapop(adj ? U : U')
    xt, info = cgls(LT,triu2vec(Q); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return vec2triu(xt), info
end

"""
     cgls(A, b; shift, tol, maxiter, x0) -> (x, info)

Solve `Ax = b` or minimize `norm(Ax-b)` using `CGLS`, the conjugate gradient method for unsymmetric linear equations and least squares problems. 
`A` can be specified either as a rectangular matrix or as a linear operator, as defined in the `LinearMaps` package.  

The keyword argument `shift` specifies a regularization parameter as `shift = s`. If
`s = 0` (default), then `CGLS` is Hestenes and Stiefel's specialized form of the
conjugate-gradient method for least-squares problems. If `s ≠ 0`, the system `(A'*A + s*I)*b = A'*b` is solved. 

A tolerance `tol` can be specified for stopping the iterative process (default: `tol = 1.e-6`).

The maximum number of iterations can be specified using `maxiter` (default: `maxiter = max(size(A),20)`).

An initial guess for the solution can be specified using the keyword argument vector `x0` (default: `x0 = 0`). 

The resulting named tuple `info` contains `(flag, resNE, iter) `, with convergence related information, as follows: 

     `info.flag`  - convergence flag with values:  
                    1, if convergence occured; 
                    2, if the maximum number of iterations has been reached without convergence;
                    3, if the matrix `A'*A + s*I` seems to be singular or indefinite;
                    4, if instability seems likely meaning `(A'*A + s*I)` indefinite and `norm(x)` decreased;  

     `info.resNE` - the relative residual for the normal equations `norm(A'*b - (A'*A + s*I)*x)/norm(A'*b)`;  
 
     `info.iter`  - the iteration number at which `x` was computed.        

This function is a translation of the MATLAB implementation of `CGLS`, the conjugate gradient method for nonsymmetric linear equations and least squares problems
[`https://web.stanford.edu/group/SOL/software/cgls/`](https://web.stanford.edu/group/SOL/software/cgls/). 
The author of the code is Michael Saunders, with contributions from
Per Christian Hansen, Folkert Bleichrodt and Christopher Fougner.    

_Note:_  Two alternative solvers `lsqr` and `lsmr` available in the [`IterativeSolvers`](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) package can be also employed. 
For example, the following call to `lsqr` can be alternatively used:
        
      using IterativeSolvers
      lsqr(A, b; kwargs...) -> x[, history]

where `kwargs` contains solver-specific keyword arguments. A similar call to  `lsmr` can be used.    
"""
function cgls(A, b; shift = 0, abstol = 0, reltol = 1e-6, maxiter = max(size(A,1),size(A,2),20), x0 = zeros(size(A,2))) 
     
   T = eltype(A)
   x = copy(x0)
   r = b - A*x
   s = A'*r-shift*x
      
   # Initialize
   p      = s
   norms0 = norm(s)
   gamma  = norms0^2
   normx  = norm(x)
   xmax   = normx
   k      = 0
   flag   = 0
   
   indefinite = 0
   resNE = 0
   
   #--------------------------------------------------------------------------
   # Main loop
   #--------------------------------------------------------------------------
   while (k < maxiter) && (flag == 0)
       
       k += 1
       
       q = A*p;
          
       delta = norm(q)^2  +  shift*norm(p)^2
       delta < 0 && (indefinite = 1)
       delta == 0 && (delta = eps(real(float(T))))
       alpha = gamma / delta
       
       x     = x + alpha*p
       r     = r - alpha*q
          
       s = A'*r - shift*x
      
       norms  = norm(s)
       gamma1 = gamma
       gamma  = norms^2
       beta   = gamma / gamma1
       p      = s + beta*p
       
       # Convergence
       normx = norm(x)
       xmax  = max(xmax, normx)
       #flag  = Int((norms <= norms0 * tol) || (normx * tol >= 1))
       flag  = Int((norms <= max(norms0 * reltol, abstol)) || (normx * reltol >= 1))
       
       # Output
       resNE = norms / norms0; 
       isnan(resNE) && (resNE = zero(norms))

   end # while
   
   iter = k;
   
   shrink = normx/xmax;
   if k == maxiter;        flag = 2; end
   if indefinite > 0;      flag = 3; end
   if shrink <= sqrt(reltol); flag = 4; end
   return x, (flag = flag, resNE = resNE, iter = iter)
end
