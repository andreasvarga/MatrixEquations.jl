"""
    lyapci(A, C; abstol, reltol, maxiter) -> (X,info)

Compute for a square `A` and a hermitian/symmetric `C` a solution `X` of the continuous Lyapunov matrix equation

                A*X + X*A' + C = 0.

A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function lyapci(A::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    n = LinearAlgebra.checksquare(A)
    LinearAlgebra.checksquare(C) == n ||
        throw(DimensionMismatch("C must be a square matrix of dimension $n"))
    sym = isreal(A) && isreal(C) && issymmetric(C) 
    her = ishermitian(C)
    LT = lyapop(A; her = sym)
    
    if sym 
       xt, info = cgls(LT,-triu2vec(C); abstol, reltol, maxiter)
    else
       xt, info = cgls(LT,-vec(C); abstol, reltol, maxiter)
    end
    info.flag == 1 || @warn "convergence issues: info = $info"
    if sym
       return vec2triu(xt,her = true), info
    else
       Xt = reshape(xt,n,n); 
       return her ? (Xt+Xt')/2 : Xt, info
    end
end
"""
    lyapci(A, E, C; abstol, reltol, maxiter) -> (X,info)

Compute `X`, the solution of the generalized continuous Lyapunov equation

    AXE' + EXA' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a square matrix.
A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function lyapci(A::AbstractMatrix, E::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    n = LinearAlgebra.checksquare(A)
    LinearAlgebra.checksquare(C) == n ||
       throw(DimensionMismatch("C must be a square matrix of dimension $n"))
    LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a square matrix of dimension $n"))
    sym = isreal(A) && isreal(E) && isreal(C) && issymmetric(C) 
    her = ishermitian(C)
    LT = lyapop(A, E; her = sym)
    
    if sym 
       xt, info = cgls(LT,-triu2vec(C); abstol, reltol, maxiter)
    else
       xt, info = cgls(LT,-vec(C); abstol, reltol, maxiter)
    end
    info.flag == 1 || @warn "convergence issues: info = $info"
    if sym
       return vec2triu(xt,her = true), info
    else
       Xt = reshape(xt,n,n); 
       return her ? (Xt+Xt')/2 : Xt, info
    end
end
"""
    lyapdi(A, C; abstol, reltol, maxiter) -> (X,info)

Compute for a square `A` and a hermitian/symmetric `C` a solution `X` of the discrete Lyapunov matrix equation

                AXA' - X + C = 0.

A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function lyapdi(A::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    n = LinearAlgebra.checksquare(A)
    LinearAlgebra.checksquare(C) == n ||
        throw(DimensionMismatch("C must be a square matrix of dimension $n"))
    sym = isreal(A) && isreal(C) && issymmetric(C) 
    her = ishermitian(C)
    LT = lyapop(A; disc = true, her = sym)
    if sym 
       xt, info = cgls(LT,-triu2vec(C); abstol, reltol, maxiter)
    else
       xt, info = cgls(LT,-vec(C); abstol, reltol, maxiter)
    end
    info.flag == 1 || @warn "convergence issues: info = $info"
    if sym
       return vec2triu(xt,her = true), info
    else
       Xt = reshape(xt,n,n); 
       return her ? (Xt+Xt')/2 : Xt, info
    end
end
"""
    lyapdi(A, E, C; abstol, reltol, maxiter) -> (X,info)

Compute `X`, the solution of the generalized discrete Lyapunov equation

    AXA' - EXE' + C = 0,

where `A` and `E` are square real or complex matrices and `C` is a square matrix.
A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function lyapdi(A::AbstractMatrix, E::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    n = LinearAlgebra.checksquare(A)
    LinearAlgebra.checksquare(C) == n ||
       throw(DimensionMismatch("C must be a square matrix of dimension $n"))
    LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a square matrix of dimension $n"))
    sym = isreal(A) && isreal(E) && isreal(C) && issymmetric(C) 
    her = ishermitian(C)
    LT = lyapop(A, E; disc = true, her = sym)
    
    if sym 
       xt, info = cgls(LT,-triu2vec(C); abstol, reltol, maxiter)
    else
       xt, info = cgls(LT,-vec(C); abstol, reltol, maxiter)
    end
    info.flag == 1 || @warn "convergence issues: info = $info"
    if sym
       return vec2triu(xt,her = true), info
    else
       Xt = reshape(xt,n,n); 
       return her ? (Xt+Xt')/2 : Xt, info
    end
end
lyapci(A::AbstractMatrix, E::UniformScaling{Bool}, C::AbstractMatrix; kwargs...) = lyapci(A, C; kwargs...)
lyapdi(A::AbstractMatrix, E::UniformScaling{Bool}, C::AbstractMatrix; kwargs...) = lyapdi(A, C; kwargs...)

"""
    X = sylvci(A,B,C)

Solve the continuous Sylvester matrix equation

                AX + XB = C ,

where `A` and `B` are square matrices. 

A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function sylvci(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    m, n = size(C);
    [m; n] == LinearAlgebra.checksquare(A,B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    LT = sylvop(A, B)   
    xt, info = cgls(LT,vec(C); abstol, reltol, maxiter)
    return reshape(xt,m,n), info
end
"""
    X = sylvdi(A,B,C)

Solve the discrete Sylvester matrix equation

                AXB + X = C ,

where `A` and `B` are square matrices. 

A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 

"""
function sylvdi(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    m, n = size(C);
    [m; n] == LinearAlgebra.checksquare(A,B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    LT = sylvop(A, B; disc = true)   
    xt, info = cgls(LT,vec(C); abstol, reltol, maxiter)
    return reshape(xt,m,n), info
end
"""
    X = gsylvi(A,B,C,D,E)

Solve the generalized Sylvester matrix equation

    AXB + CXD = E ,

where `A`, `B`, `C` and `D` are square matrices. 

A least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function gsylvi(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix; abstol = zero(float(real(eltype(A)))), reltol = sqrt(eps(float(real(eltype(A))))), maxiter = 1000) 
    m, n = size(E);
    [m; n; m; n] == LinearAlgebra.checksquare(A,B,C,D) ||
        throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
    LT = sylvop(A,B,C,D)
    xt, info = cgls(LT,vec(E); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return reshape(xt,m,n), info
end


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
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 

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
    tlyapci(A, C, isig = +1; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute a solution `X` of the continuous T-Lyapunov matrix equation

                A*X +isig*transpose(X)*transpose(A) = C   if adj = false, 

or

                A*transpose(X) + isig*X*transpose(A) = C   if adj = true,

where for `isig = 1`, `C` is a symmetric matrix and for `isig = -1`, `C` is a skew-symmetric matrix.                     

For a matrix `A`, a least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution and 
the keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function tlyapci(A::AbstractMatrix{T}, C::AbstractMatrix{T}, isig::Int = 1; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    abs(isig) == 1 || error(" isig must be either 1 or -1")
    if isig == 1
       issymmetric(C) || error("C must be symmetric for isig = 1")
       # temporary fix to avoid false results for DoubleFloats 
       # C == transpose(C) || error("C must be symmetric for isig = 1")
    else
       iszero(C+transpose(C)) || error("C must be skew-symmetric for isig = -1")
    end
    LT = lyaplikeop(A; adj, isig, htype = false)
    xt, info = cgls(LT, vec(C); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return adj ? reshape(xt,m,n) : reshape(xt,n,m), info
end
"""
    hlyapci(A, C, isig = +1; adj = false, abstol, reltol, maxiter) -> (X,info)


Compute a solution `X` of the continuous H-Lyapunov matrix equation

                A*X + isig*X'*A' = C   if adj = false, 

or

                A*X' + isig*X*A' = C   if adj = true,

where for `isig = 1`, `C` is a hermitian matrix and for `isig = -1`, `C` is a skew-hermitian matrix.                     

For a matrix `A`, a least-squares solution `X` is determined using a conjugate gradient based iterative method applied 
to a suitably defined T-Lyapunov linear operator `L:X -> Y` such that `L(X) = C` or `norm(L(X) - C)` is minimized. 
The keyword arguments `abstol` (default: `abstol = 0`) and `reltol` (default: `reltol = sqrt(eps())`) can be used to provide the desired tolerance for the accuracy of the computed solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 1000`). 
"""
function hlyapci(A::AbstractMatrix{T}, C::AbstractMatrix{T}, isig::Int = 1; adj = false,  abstol = zero(float(real(T))), reltol = sqrt(eps(float(real(T)))), maxiter = 1000) where {T}
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    abs(isig) == 1 || error(" isig must be either 1 or -1")
    if isig == 1
        ishermitian(C) || error("C must be hermitian for isig = 1")
       # temporary fix to avoid false results for DoubleFloats 
       # C == C' || error("C must be hermitian for isig = 1")
    else
       iszero(C+C') || error("C must be skew-hermitian for isig = -1")
    end
    LT = lyaplikeop(A; adj, isig, htype = true)
    xt, info = cgls(LT,vec(C); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return adj ? reshape(xt,m,n) : reshape(xt,n,m), info
end

"""
    tulyapci(U, Q; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for an upper triangular `U` and a symmetric `Q` an upper triangular solution `X` of the continuous T-Lyapunov matrix equation

      transpose(U)*X + transpose(X)*U = Q   if adj = false,

or 

      U*transpose(X) + X*transpose(U) = Q   if adj = true. 


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
    LT = tulyaplikeop(U; adj)
    xt, info = cgls(LT,triu2vec(Q); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return vec2triu(xt), info
end
"""
    hulyapci(U, Q; adj = false, abstol, reltol, maxiter) -> (X,info)

Compute for an upper triangular `U` and a hermitian `Q` an upper triangular solution `X` of the continuous H-Lyapunov matrix equation

                U'*X + X'*U = Q   if adj = false, 

or

                U*X' + X*U' = Q    if adj = true.

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
    ishermitian(Q) || throw(ArgumentError("Q must be hermitian"))
    LT = hulyaplikeop(U; adj)
    xt, info = cgls(LT,triu2vec(Q); abstol, reltol, maxiter)
    info.flag == 1 || @warn "convergence issues: info = $info"
    return vec2triu(xt), info
end

"""
     cgls(A, b; shift, abstol, reltol, maxiter, x0) -> (x, info)

Solve `Ax = b` or minimize `norm(Ax-b)` using `CGLS`, the conjugate gradient method for unsymmetric linear equations and least squares problems. 
`A` can be specified either as a rectangular matrix or as a linear operator, as defined in the `LinearMaps` package.  
It is desirable that `eltype(A) == eltype(b)`, otherwise errors may result or additional allocations may occur in operator-vector products. 

The keyword argument `shift` specifies a regularization parameter as `shift = s`. If
`s = 0` (default), then `CGLS` is Hestenes and Stiefel's specialized form of the
conjugate-gradient method for least-squares problems. If `s ≠ 0`, the system `(A'*A + s*I)*b = A'*b` is solved. 

An absolute tolerance `abstol` and a relative tolerance `reltol` can be specified for stopping the iterative process (default: `abstol = 0`, `reltol = 1.e-6`).

The maximum number of iterations can be specified using `maxiter` (default: `maxiter = max(size(A),20)`).

An initial guess for the solution can be specified using the keyword argument vector `x0` (default: `x0 = missing`). 

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
The author of the code is Michael Saunders, with contributions from Per Christian Hansen, Folkert Bleichrodt and Christopher Fougner.    

_Note:_  Two alternative solvers `lsqr` and `lsmr`, available in the [`IterativeSolvers`](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) package, can also be employed. 
For example, the following call to `lsqr` can be alternatively used:
        
      using IterativeSolvers
      lsqr(A, b; kwargs...) -> x[, history]

where `kwargs` contains solver-specific keyword arguments. A similar call to  `lsmr` can be used.    
"""
function cgls(A, b; shift = 0, abstol = 0, reltol = 1e-6, maxiter = max(size(A,1),size(A,2),20), x0 = missing ) 
   
    m, n = size(A) 
    length(b) == m || error("Inconsistent problem size")
    T = eltype(A)
    T == eltype(b) || @warn "eltype(A) ≠ eltype(b). This could lead to errors or additional allocations in operator-vector products."
    ismissing(x0) || T == eltype(x0) || @warn "eltype(A) ≠ eltype(x0). This could lead to errors or additional allocations in operator-vector products."
   
 
    if iszero(b)
       return zeros(T,n), (flag = 1, resNE = zero(T), iter = 1)
    end
    T1 = typeof(one(eltype(b))/one(T))
    # allocate vectors 
    x = Vector{T1}(undef,n)
    WS = (Vector{T1}(undef,n),Vector{T1}(undef,n),Vector{T1}(undef,m),Vector{T1}(undef,m))
 
    # the following may fail if eltype(x0) is not a subtype of eltype(x)
    ismissing(x0) ? x .= zero(T1) : x .= x0
    return cgls!(x, WS, A, b; shift, abstol, reltol, maxiter) 
end 
function cgls!(x, WS, A, b; shift = 0, abstol = 0, reltol = 1e-6, maxiter = max(size(A,1),size(A,2),20) ) 
    """
       cgls!(x, WS, A, b; shift, abstol, reltol, maxiter) -> (x, info)

    Solve `Ax = b` or minimize `norm(Ax-b)` using `CGLS`, the conjugate gradient method for unsymmetric linear equations and least squares problems. 
    The initial guess `x`, will be updated in-place. For an `m×n` operator `A`, `WS` is a prealocated working space provided as a tuple of vectors `(p,s,r,q)` of 
    dimensions `(n,n,m,m)`, respectively. See [`cgls`](@ref) for the description of the keyword parameters.  
    """
    T1 = eltype(x)
    # recover allocated vectors 
    (p, s, r, q) = WS 
    adjointA = adjoint(A)
    
    r .= b
    #r = b - A*x
    mul!(r,A,x,-1,1)
 
    #s = A'*r-shift*x
    mul!(s,adjointA,r)
    shift == 0 || axpy!(-shift, x, s)
        
    # Initialize
    p      .= s
    norms0 = norm(s)
    gamma  = norms0^2
    normx  = norm(x)
    xmax   = normx
    k      = 0
    flag   = 0
    
    indefinite = 0
    resNE = 0
    ONE = one(T1)
    
    #--------------------------------------------------------------------------
    # Main loop
    #--------------------------------------------------------------------------
    while (k < maxiter) && (flag == 0)
        
        k += 1
        
        #q = A*p;
        mul!(q, A, p)
           
        delta = norm(q)^2  +  shift*norm(p)^2
        delta < 0 && (indefinite = 1)
        delta == 0 && (delta = eps(real(float(T1))))
        alpha = gamma / delta
        
        #x     = x + alpha*p
        axpy!(alpha,p,x)
        #r     = r - alpha*q
        axpy!(-alpha,q,r)
           
        #s = A'*r - shift*x
        mul!(s,adjointA,r)
        shift == 0 || axpy!(-shift, x, s)
           
        norms  = norm(s)
        gamma1 = gamma
        gamma  = norms^2
        beta   = gamma / gamma1
        #p      = s + beta*p
        axpby!(ONE,s,beta,p)
 
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
# function cgls(A, b; shift = 0, abstol = 0, reltol = 1e-6, maxiter = max(size(A,1),size(A,2),20), x0 = missing ) 
#     """
#        cgls(A, b; shift, abstol, reltol, maxiter, x0) -> (x, info)

#     Solve `Ax = b` or minimize `norm(Ax-b)` using `CGLS`, the conjugate gradient method for unsymmetric linear equations and least squares problems. 
#     Comment out to obtain the original translation of the MATLAB implementation of `CGLS`.
#     """
   
#     m, n = size(A) 
#     length(b) == m || error("Inconsistent problem size")
#     T = eltype(A)
#     T == eltype(b) || @warn "eltype(A) ≠ eltype(b). This could lead to errors or additional allocations in operator-vector products."
#     ismissing(x0) || T == eltype(x0) || @warn "eltype(A) ≠ eltype(x0). This could lead to errors or additional allocations in operator-vector products."
   
 
#     if iszero(b)
#        return zeros(T,n), (flag = 1, resNE = zero(T), iter = 1)
#     end
#     T1 = typeof(one(eltype(b))/one(T))
#     # allocate vectors 
#     x = Vector{T1}(undef,n)
#     p = Vector{T1}(undef,n)
#     s = Vector{T1}(undef,n)
#     r = Vector{T1}(undef,m)
#     q = Vector{T1}(undef,m)
 
#     # the following may fail if eltype(x0) is not a subtype of eltype(x)
#     ismissing(x0) ? x .= zero(T1) : x .= x0
    
#     r .= b
#     #r = b - A*x
#     mul!(r,A,x,-1,1)
 
#     #s = A'*r-shift*x
#     mul!(s,A',r)
#     shift == 0 || axpy!(-shift, x, s)
        
#     # Initialize
#     p      .= s
#     norms0 = norm(s)
#     gamma  = norms0^2
#     normx  = norm(x)
#     xmax   = normx
#     k      = 0
#     flag   = 0
    
#     indefinite = 0
#     resNE = 0
#     ONE = one(T1)
    
#     #--------------------------------------------------------------------------
#     # Main loop
#     #--------------------------------------------------------------------------
#     while (k < maxiter) && (flag == 0)
        
#         k += 1
        
#         #q = A*p;
#         mul!(q, A, p)
           
#         delta = norm(q)^2  +  shift*norm(p)^2
#         delta < 0 && (indefinite = 1)
#         delta == 0 && (delta = eps(real(float(T1))))
#         alpha = gamma / delta
        
#         #x     = x + alpha*p
#         axpy!(alpha,p,x)
#         #r     = r - alpha*q
#         axpy!(-alpha,q,r)
           
#         #s = A'*r - shift*x
#         mul!(s,A',r)
#         shift == 0 || axpy!(-shift, x, s)
           
#         norms  = norm(s)
#         gamma1 = gamma
#         gamma  = norms^2
#         beta   = gamma / gamma1
#         #p      = s + beta*p
#         axpby!(ONE,s,beta,p)
 
#         # Convergence
#         normx = norm(x)
#         xmax  = max(xmax, normx)
#         #flag  = Int((norms <= norms0 * tol) || (normx * tol >= 1))
#         flag  = Int((norms <= max(norms0 * reltol, abstol)) || (normx * reltol >= 1))
        
#         # Output
#         resNE = norms / norms0; 
#         isnan(resNE) && (resNE = zero(norms))
 
#     end # while
    
#     iter = k;
    
#     shrink = normx/xmax;
#     if k == maxiter;        flag = 2; end
#     if indefinite > 0;      flag = 3; end
#     if shrink <= sqrt(reltol); flag = 4; end
#     return x, (flag = flag, resNE = resNE, iter = iter)
# end
