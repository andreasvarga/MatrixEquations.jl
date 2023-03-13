"""
    X = sylvckr(A,B,C)

Solve the continuous Sylvester matrix equation

                AX + XB = C

using the Kronecker product expansion of equations. `A` and `B` are
square matrices, and `A` and `-B` must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvckr(A, B, C)
    m, n = size(C)
    [m; n] == LinearAlgebra.checksquare(A, B) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    reshape((kron(Array{eltype(A),2}(I, n, n), A) +
            kron(transpose(B), Array{eltype(B),2}(I, m, m))) \ (C[:]),m,n)
end

"""
    X = sylvdkr(A,B,C)

Solve the discrete Sylvester matrix equation

                AXB + X = C

using the Kronecker product expansion of equations. `A` and `B` are
square matrices, and `A` and `-B` must not have common reciprocal eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvdkr(A, B, C)
    m, n = size(C)
    [m; n] == LinearAlgebra.checksquare(A, B) ||
            throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    reshape((kron(transpose(B), A) + I) \ (C[:]), m, n)
end

"""
    X = gsylvkr(A,B,C,D,E)

Solve the generalized Sylvester matrix equation

                AXB + CXD = E

using the Kronecker product expansion of equations. `A`, `B`, `C` and `D` are
square matrices. The pencils `A-λC` and `D+λB` must be regular and
must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function gsylvkr(A, B, C, D, E)
   m, n = size(E)
   [m; n; m; n] == LinearAlgebra.checksquare(A, B, C, D) ||
                  throw(DimensionMismatch("A, B, C, D and E have incompatible dimensions"))
   reshape((kron(transpose(B), A) + kron(transpose(D), C)) \ (E[:]), m, n)
end

"""
    sylvsyskr(A,B,C,D,E,F) -> (X,Y)

Solve the Sylvester system of matrix equations

                AX + YB = C
                DX + YE = F

using the Kronecker product expansion of equations. `(A,D)`, `(B,E)` are
pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function sylvsyskr(A, B, C, D, E, F)
   m, n = size(C)
   (m == size(F, 1) && n == size(F, 2)) ||
         throw(DimensionMismatch("C and F must have the same dimensions"))
   [m; n; m; n] == LinearAlgebra.checksquare(A, B, D, E) ||
                  throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
   z = [ kron(Array{eltype(A),2}(I, n, n), A) kron(transpose(B), Array{eltype(B),2}(I, m, m)) ;
         kron(Array{eltype(D),2}(I, n, n), D) kron(transpose(E), Array{eltype(E),2}(I, m, m))] \ [C[:];F[:]]
   (reshape(z[1:m * n], m, n), reshape(z[m * n + 1:end], m, n))
end

"""
    dsylvsyskr(A,B,C,D,E,F) -> (X,Y)

Solve the dual Sylvester system of matrix equations

       AX + DY = C
       XB + YE = F

using the Kronecker product expansion of equations. `(A,D)`, `(B,E)` are
pairs of square matrices of the same size.
The pencils `A-λD` and `-B+λE` must be regular and must not have common eigenvalues.
This function is not recommended for large order matrices.
"""
function dsylvsyskr(A, B, C, D, E, F)
   m, n = size(C)
   (m == size(F, 1) && n == size(F, 2)) ||
         throw(DimensionMismatch("C and F must have the same dimensions"))
   [m; n; m; n] == LinearAlgebra.checksquare(A, B, D, E) ||
                  throw(DimensionMismatch("A, B, C, D, E and F have incompatible dimensions"))
   z = [ kron(Array{eltype(A),2}(I, n, n), A) kron(Array{eltype(D),2}(I, n, n), D);
         kron(transpose(B), Array{eltype(B),2}(I, m, m)) kron(transpose(E), Array{eltype(E),2}(I, m, m))] \ [C[:];F[:]]
   (reshape(z[1:m * n], m, n), reshape(z[m * n + 1:end], m, n))
end
"""
    X = tlyapckr(A,C, isig = 1; atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute for `isig = ±1` a solution of the the continuous T-Lyapunov matrix equation

                A*X + isig*transpose(X)*transpose(A) + C = 0

using the Kronecker product expansion of equations. `A` and `C` are
`m×n` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
The matrix `C` must be symmetric if `isig = 1` and skew-symmetric if `isig = -1`.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`,
  where `N = 4*min(m,n)^2` and ϵ is the machine precision of the element type of `A`.
This function is not recommended for large order matrices.
"""
function tlyapckr(A, C, isig = 1; atol::Real = 0.0, rtol::Real = (4*min(size(A)...)^2*eps(real(float(one(eltype(A))))))*iszero(atol))
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    abs(isig) == 1 || error(" isig must be either 1 or -1")
    if isig == 1
       issymmetric(C) || error("C must be symmetric for isig = 1")
    else
       iszero(C+transpose(C)) || error("C must be skew-symmetric for isig = -1")
    end
    it = [(i-1)*n+j for i in 1:m, j in 1:n][:]    
    T =  kron(Array{eltype(A),2}(I, m, m), A) + isig*kron(A, Array{eltype(A),2}(I, m, m))[:,invperm(it)]  
    mn = m*n
    m > n ? F = svd(T,full = true) : F = svd(T)
    tol = max(atol, rtol*F.S[1])
    r = count(x -> x > tol, F.S)
    Xe = F.U'*C[:]
    norm(view(Xe,r+1:m*m),Inf) <= tol || @warn "Incompatible equation: least-squares solution computed"
    Xe[1:r] ./= F.S[1:r]
    reshape(view(F.Vt,1:r,1:mn)'*view(Xe,1:r),n,m)
end
"""
    X = hlyapckr(A,C, isig = 1; atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute for `isig = ±1` a solution of the continuous H-Lyapunov matrix equation

                A*X + isig*adjoint(X)*adjoint(A) + C = 0

using the Kronecker product expansion of equations. `A` and `C` are
`m×n` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
The matrix `C` must be hermitian if `isig = 1` and skew-hermitian if `isig = -1`.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`,
  where `N = 4*min(m,n)^2` and ϵ is the machine precision of the element type of `A`.
This function is not recommended for large order matrices.
"""
function hlyapckr(A, C, isig = 1; atol::Real = 0.0, rtol::Real = (4*min(size(A)...)^2*eps(real(float(one(eltype(A))))))*iszero(atol))
    promote_type(eltype(A),eltype(C)) <: Real && (return tlyapckr(A, C; atol, rtol))
    m = LinearAlgebra.checksquare(C)
    ma, n = size(A)
    ma == m || throw(DimensionMismatch("A and C have incompatible dimensions"))
    abs(isig) == 1 || error(" isig must be either 1 or -1")
    if isig == 1
       ishermitian(C) || error("C must be symmetric for isig = 1")
    else
       iszero(C+adjoint(C)) || error("C must be skew-hermitian for isig = -1")
    end
    Ae = [real(A) imag(A); -imag(A) real(A)]     
    Ce = [real(C) imag(C); -imag(C) real(C)] 
    Xe = tlyapckr(Ae, Ce, isig; atol, rtol)   
    i1 = 1:n; i2 = n+1:2n; j1 = 1:m; j2 = m+1:2m
    (Xe[i1,j1] ≈ Xe[i2,j2] && Xe[i2,j1] ≈ -Xe[i1,j2]) || @warn "Solution possibly inaccurate"
    return complex.((Xe[i1,j1]+Xe[i2,j2])/2,(Xe[i1,j2]-Xe[i2,j1])/2)
end
"""
    X = tsylvckr(A,B,C; atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute a solution of the continuous T-Sylvester matrix equation

                A*X + transpose(X)*B = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×n`, `n×m` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`,
  where `N = 4*min(m,n)^2` and ϵ is the machine precision of the element type of `A`.
This function is not recommended for large order matrices.
"""
function tsylvckr(A, B, C; atol::Real = 0.0, rtol::Real = (4*min(size(A)...)^2*eps(real(float(one(eltype(A))))))*iszero(atol))
    m = LinearAlgebra.checksquare(C)
    ma, na = size(A)
    n, mb = size(B)
    (ma == m && na == n && mb == m) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    it = [(i-1)*n+j for i in 1:m, j in 1:n][:]    
    T =  kron(Array{eltype(A),2}(I, m, m), A) + kron(transpose(B), Array{eltype(B),2}(I, m, m))[:,invperm(it)]  
    F = svd(T)
    tol = max(atol, rtol*F.S[1])
    r = count(x -> x > tol, F.S)
    Xe = F.U'*C[:]
    Xe[1:r] ./= F.S[1:r]
    reshape(F.Vt'*Xe,n,m)
end
"""
    X = hsylvckr(A,B,C; atol::Real=0, rtol::Real=atol>0 ? 0 : N*ϵ)

Compute a solution of the continuous H-Sylvester matrix equation

                A*X + adjoint(X)*B = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×n`, `n×m` and `m×m` matrices, respectively, and `X` is an `n×m` matrix.
`atol` and `rtol` are the absolute and relative tolerances, respectively, used for rank computation. 
The default relative tolerance is `N*ϵ`,
  where `N = 4*min(m,n)^2` and ϵ is the machine precision of the element type of `A`.
This function is not recommended for large order matrices.
"""
function hsylvckr(A, B, C; atol::Real = 0.0, rtol::Real = (4*min(size(A)...)*min(size(B)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
    promote_type(eltype(A),eltype(B),eltype(C)) <: Real && (return tsylvckr(A, B, C; atol, rtol))
    m = LinearAlgebra.checksquare(C)
    ma, na = size(A)
    n, mb = size(B)
    (ma == m && na == n && mb == m) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    Ae = [real(A) imag(A); -imag(A) real(A)]     
    Be = [real(B) imag(B); -imag(B) real(B)] 
    Ce = [real(C) imag(C); -imag(C) real(C)] 
    Xe = tsylvckr(Ae,Be,Ce; atol, rtol)   
    i1 = 1:n; i2 = n+1:2n; j1 = 1:m; j2 = m+1:2m
    (Xe[i1,j1] ≈ Xe[i2,j2] && Xe[i2,j1] ≈ -Xe[i1,j2]) || @warn "Solution possibly inaccurate"
    return complex.((Xe[i1,j1]+Xe[i2,j2])/2,(Xe[i1,j2]-Xe[i2,j1])/2)
end
"""
    X = csylvckr(A,B,C)

Solve the continuous C-Sylvester matrix equation

                A*X + conj(X)*B = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×m`, `n×n` and `m×n` matrices, respectively, and `X` is an `m×n` matrix.
This function is not recommended for large order matrices.
"""
function csylvckr(A, B, C)
    promote_type(eltype(A),eltype(B),eltype(C)) <: Real && (return sylvckr(A, B, C))
    m, n = size(C)
    [m; n] == LinearAlgebra.checksquare(A, B) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    Ae = [real(A) imag(A); imag(A) -real(A)]     
    Be = [real(B) imag(B); imag(B) -real(B)] 
    Ce = [real(C) imag(C); imag(C) -real(C)] 
    m2 = 2m; n2 = 2n
    Xe = reshape((kron(Array{real(eltype(A)),2}(I, n2, n2), Ae) +
                  kron(transpose(Be), Array{real(eltype(B)),2}(I, m2, m2))) \ (Ce[:]),m2,n2)
    i1 = 1:m; i2 = m+1:m2; j1 = 1:n; j2 = n+1:n2
    (Xe[i1,j1] ≈ Xe[i2,j2] && Xe[i2,j1] ≈ -Xe[i1,j2]) || @warn "Solution possibly inaccurate"              
    return complex.((Xe[i1,j1]+Xe[i2,j2])/2,(Xe[i1,j2]-Xe[i2,j1])/2)
end

"""
    X = tsylvdkr(A,B,C)

Solve the discrete T-Sylvester matrix equation

                A*transpose(X)*B + X = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×n` matrices and `X` is an `m×n` matrix.
This function is not recommended for large order matrices.
"""
function tsylvdkr(A, B, C)
    m, n = size(C)
    ma, na = size(A)
    mb, nb = size(B)
    (ma == m && na == n && mb == m && nb == n) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    it = [(i-1)*m+j for i in 1:n, j in 1:m][:]   
    reshape((kron(transpose(B), A)[:,invperm(it)] + I) \ (C[:]), m, n)
end
"""
    X = hsylvdkr(A,B,C)

Solve the discrete H-Sylvester matrix equation

                A*adjoint(X)*B + X = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×n` matrices and `X` is an `m×n` matrix.
This function is not recommended for large order matrices.
"""
function hsylvdkr(A, B, C)
    promote_type(eltype(A),eltype(B),eltype(C)) <: Real && (return tsylvdkr(A, B, C))
    m, n = size(C)
    ma, na = size(A)
    mb, nb = size(B)
    (ma == m && na == n && mb == m && nb == n) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    Ae = [real(A) imag(A); imag(A) -real(A)]     
    Be = [real(B) imag(B); imag(B) -real(B)] 
    Ce = [real(C) imag(C); imag(C) -real(C)] 
    m2 = 2m; n2 = 2n
    it = [(i-1)*m2+j for i in 1:n2, j in 1:m2][:]   
    Ye = reshape((kron(transpose(Be), Ae)[:,invperm(it)] + I) \ (Ce[:]), m2, n2)
    return complex.((Ye[1:m,1:n]-Ye[m+1:m2,n+1:n2])/2,(Ye[1:m,n+1:n2]+Ye[m+1:m2,1:n])/2)
end


"""
    X = csylvdkr(A,B,C)

Solve the discrete C-Sylvester matrix equation

                A*conj(X)*B + X = C

using the Kronecker product expansion of equations. `A`, `B` and `C` are
`m×m`, `n×n` and `m×n` matrices, respectively, and `X` is an `m×n` matrix.
This function is not recommended for large order matrices.
"""
function csylvdkr(A, B, C)
    promote_type(eltype(A),eltype(B),eltype(C)) <: Real && (return sylvdkr(A, B, C))
    m, n = size(C)
    [m; n] == LinearAlgebra.checksquare(A, B) ||
             throw(DimensionMismatch("A, B and C have incompatible dimensions"))
    Ae = [real(A) imag(A); imag(A) -real(A)]     
    Be = [real(B) imag(B); imag(B) -real(B)] 
    Ce = [real(C) imag(C); imag(C) -real(C)] 
    m2 = 2m; n2 = 2n
    Xe = reshape((kron(transpose(Be), Ae) + I) \ (Ce[:]), m2, n2)
    i1 = 1:m; i2 = m+1:m2; j1 = 1:n; j2 = n+1:n2
    (Xe[i1,j1] ≈ -Xe[i2,j2] && Xe[i2,j1] ≈ Xe[i1,j2]) || @warn "Solution possibly inaccurate"              
    return complex.((Xe[i1,j1]-Xe[i2,j2])/2,(Xe[i1,j2]+Xe[i2,j1])/2)
end


