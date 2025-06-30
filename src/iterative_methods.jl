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
    plyapci(A, E, B; cyclic, abstol, reltol, maxiter, nshifts, shifts, loginf) -> (U, info)
    plyapci(A, B; cyclic, abstol, reltol, maxiter, nshifts, shifts, loginf) -> (U, info)

Compute a low-rank factor `U` of the solution `X = UU'` of the
generalized continuous Lyapunov equation

      AXE' + EXA' + BB' = 0,

where `A` and `E` are square real matrices and `B` is a real matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with negative real parts. `E = I` is assumed in the second call. 


The named tuple `info` contains information related to the execution of the LR-ADI algorithm as follows:
`info.niter` contains the number of performed iterations; 
`info.res_fact` contains the norm of the residual factor; 
`info.res` contains, if `loginf = true`, the vector of normalized residual norms (normalized with respect to the norm of the initial approximation);
`info.rc` contains, if `loginf = true`, the vector of norms of relative changes in building the solution;
`info.used_shift` contains the vector of used shifts.

The keyword argument `abstol` (default: `abstol = 1e-12`) is the tolerance used for convergence test
on the normalized residuals, while the keyword argument
`reltol` (default: `reltol = 0`) is the tolerance for the relative changes of the solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 100`).
The keyword argument `nshifts` specifies the desired number of shifts to be used in an iteration cycle (default: `nshifts = 6`). 
The keyword argument `shifts` can be used to provide a pre-calculated set of complex conjugated shifts to be used
to start the iterations (default: `shifts = missing`).
With the keyword argument `loginf = true`, the normalized residual values and the norms of increments of the solution
can be saved as outputs in the resulting info structure (default: `loginf = false`). 

The low-rank ADI (LR-ADI) method with enhancements proposed in [1] is implemented, based on MATLAB 
codes of the free software described in [2]. If `cyclic = true`, the cyclic low-rank method of [3] is used, with the
pre-calculated shifts provided in the keyword argument `shifts`. 

_References_

[1] P. Kürschner. Efficient Low-Rank Solution of Large-Scale Matrix Equations. 
    Dissertation, Otto-von-Guericke-Universität, Magdeburg, Germany, 2016. Shaker Verlag,

[2] P. Benner, M. Köhler, and J. Saak. “Matrix Equations, Sparse Solvers: M-M.E.S.S.-2.0.1—
    Philosophy, Features, and Application for (Parametric) Model Order Reduction.” 
    In Model Reduction of Complex Dynamical Systems, Eds. P. Benner et.al., 171:369–92, Springer, 2021.

[3] T. Penzl, A cyclic low-rank Smith method for large sparse Lyapunov equations, 
    SIAM J. Sci. Comput. 21 (4) (1999) 1401–1418.
"""
function plyapci(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; cyclic = false,
                 abstol = 1e-12, reltol = 0, maxiter = 100, shifts = missing, nshifts = 6, loginf = false)    
    n = LinearAlgebra.checksquare(A)
    withE = !(E == I)
    if E == I 
       EE = I
       withE = false
    else
       LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       EE = E
       withE = true
    end
       
    T2 = promote_type(eltype(A), eltype(EE), eltype(B))
    T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
 
    adjA = isa(A,Adjoint)
    adjE = withE ? isa(E,Adjoint) : false
    adjB= isa(B,Adjoint)

    AA = A; BB = B
 
    nb, mb = size(B)
    nb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))

    eltype(AA) == T2 || (adjA ? AA = convert(Matrix{T2},AA.parent)' : AA = convert(Matrix{T2},AA))
    withE && (eltype(EE) == T2 || (adjE ? EE = convert(Matrix{T2},EE.parent)' : EE = convert(Matrix{T2},EE)))
    eltype(BB) == T2 || (adjB ? BB = convert(Matrix{T2},BB.parent)' : BB = convert(Matrix{T2},BB))
 
    if cyclic 
       ismissing(shifts) && (cyclic = false; @warn "missing cyclic shifts: option cyclic reset to false")
    end
    # Start LR-ADI procedure
 
    used_shifts = zeros(T2,0)
    if iszero(B) 
       info = (niter = 0, res_fact = zero(T2), res = zeros(T2,0), rc = zeros(T2,0), 
               used_shifts = used_shifts)   
       return zeros(T2,n,0), info
    end

    W = Matrix(BB)   
    ncolZ = maxiter * mb
    Z = zeros(T2, n, ncolZ)
    res0 = norm(W' * W)
    res1 = 1
    U = copy(W)
    nrmZ = 0
    
    if ismissing(shifts) 
       p = zeros(0)
       i = 1
       while isempty(p)
             p = projection_shifts(AA, EE, U, AA*U, zeros(T2,0); nshifts)   
             if isempty(p)
                 if i < 5
                    @warn "Could not compute initial projection shifts. Going to retry with random right hand side"
                    U = rand(size(U)...)
                 else
                    error("Could not compute initial projection shifts")
                 end
             end      
             i = i + 1
       end
    else
        p = shifts
        illegal_shifts = false
        # Check if all shifts are in the open left half plane
        # and if complex pairs of shifts are properly ordered.
        kk = 1
        while kk <= length(p)
            if isreal(p[kk]) 
                illegal_shifts = real(p[kk]) >= 0 
            else
                illegal_shifts = p[kk + 1] != conj(p[kk]) || real(p[kk]) >= 0
                kk = kk + 1
            end
            illegal_shifts && error("Improper shift vector detected")
            kk = kk + 1
        end
    end
    lp = length(p)

    if loginf 
       res = zeros(maxiter)  
       rc = zeros(maxiter)
    else
       res = zeros(T2,0) 
       rc = zeros(T2,0)
    end
    psave = complex(zeros(maxiter))
    
    # start iteration
    jZ = 0
    k = 1
    k_shift = 1
    mb2 = 2*mb
    while k < maxiter + 1 
        if k_shift > lp
           # update shifts
           k_shift = 1
           if !cyclic 
              used_shifts = [used_shifts; p]
              first_dropped = length(used_shifts) - nshifts + 1
              last_kept = first_dropped - 1
              if length(used_shifts) > nshifts && 
                 imag(used_shifts[first_dropped]) > 0 && 
                 (abs(used_shifts[first_dropped] - conj(used_shifts[last_kept])) < eps())
                 # don't cut between pair of complex shifts
                 used_shifts = used_shifts[end - nshifts:end]
              elseif length(used_shifts) > nshifts
                 used_shifts = used_shifts[end - nshifts + 1:end]
              end
              len = size(used_shifts, 1) * mb - 1        
              p = projection_shifts(AA, EE, view(Z,:, jZ-len:jZ), W, used_shifts; nshifts)
              lp = length(p)
           end
        end
        isempty(p) && error("empty shift vector computed; the pair (A,E) is possibly unstable")
        # get current shift
        pc = p[k_shift]
        if isreal(pc) 
            pc = real(pc)
        else
            jZ+1 >= ncolZ && break   
        end
        # perform the actual step computations, i. e. shifted solve
        #V = adj ? (A.parent + pc'*(withE ? E.parent : I))'\W : (A + pc*E)\W
        if adjA && adjE
            V = (AA.parent + pc'*EE.parent)'\W 
        else
            V = (AA + pc*EE)\W 
        end

        if isreal(pc) 
            pc2 = 2*pc
            # update the factor
            #Z[:, (m - 1) * k + 1:m * k] = sqrt(-2.0 * pc) * V     
            Z[:, jZ+1:jZ+mb] .= sqrt(-pc2) .* V     
            jZ += mb
            # update low-rank residual
            if withE
                #EV = E*V
                #W = W - 2.0 * pc * EV
                mul!(W,EE,V,-pc2,1)
            else
                #W = W - 2.0 * pc * V
                W .-= pc2 .* V
            end
            psave[k] = pc
        else
            # perform a double step with the known solution for the conjugate shift
            a = 2.0 * sqrt(-real(pc))
            b = real(pc) / imag(pc)
            #V1 = a * (real(V) + b * imag(V))
            Z1 = view(Z,:,jZ+1:jZ+mb)
            Z2 = view(Z,:,jZ+mb+1:jZ+mb2)
            Z1 .= a .* (real.(V) .+ b .* imag.(V))
            g = (a * sqrt(b * b + 1))
            Z2 .=  g .* imag.(V)
            jZ +=mb2
    
            # update low-rank residual for double step
            if withE
                #EV = E*V1
                #W = W + a * EV
                mul!(W,EE,Z1,a,1)
            else
                W .+= a .* Z1
            end
            psave[k] = pc
            psave[k+1] = conj(pc)
    
            k += 1
            k_shift += 1
    
        end
        # evaluate normalized residual
        resm = norm(W' * W) / res0

        # evaluate norm of correction of solution  
        if isreal(pc) 
            k == 1 && (res1 = resm)
            nrmV = 2.0 * abs(pc) * norm(V)^2
        else # complex double step means 2 blocks added
            k < 2 && (res1 = resm)
            nrmV = norm(Z1)^2+norm(Z2)^2
        end
    
        nrmZ = nrmZ + nrmV
        rcm = sqrt(nrmV / nrmZ)
        if loginf 
           if isreal(pc) 
              res[k] = resm; rc[k] = rcm
           else
              res[k] = resm; res[k-1] = resm
              rc[k] = rcm; rc[k-1] = rcm
           end 
        end

        # check convegence conditions
        if (abstol > 0 && resm < abstol) || (reltol > 0 && rcm < reltol) 
           break
        end 
        # checck for possible divergence
        k > 2 && resm > 100*res1 && error("residual is growing: eigenvalues are probably unstable")
        
        k += 1
        k_shift += 1
    end
    
    niter = k - (k > maxiter)
    info = (niter = niter, res_fact = norm(W), 
            res = loginf ? res[1:niter] : zeros(T2,0), 
            rc = loginf ? rc[1:niter] : zeros(T2,0), 
            used_shifts = isreal(psave) ? real(psave[1:niter]) : psave[1:niter])   
    
    if niter >= maxiter
        @warn "LR-ADI reached maximum iteration number. Results may be inaccurate!"
    end
    return Z[:, 1:niter*mb], info
end
function plyapci(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    plyapci(A::AbstractMatrix, I, B::AbstractMatrix; kwargs...)
end 

function projection_shifts(A, E, V, W, p_old; nshifts = 6)
    # function p = projection_shifts(A, E, V, W, nshifts, p_old) 
    #
    # Computes new shifts by implicitly or explicitly
    # projecting the E and A matrices to the span of V. Note that the
    # width of V must be a multiple of that of of the residual W, V is the newest part
    # of the ADI solution factor Z and the old shift
    # vector p_old passed in must have this multiple as its length.
    #
    # The projection is computed implicitly from the contents of V. 
    #
    # This function is based on the MATLAB function mess_projection_shifts.m, which is
    # part of the M - M.E.S.S. project  (http://www.mpi-magdeburg.mpg.de/projects/mess).
    # Authors: Jens Saak, Martin Koehler, Peter Benner and others.
        
    withE = (E != I)
          
    L = length(p_old)
    cols_V = size(V, 2)
    cols_W = size(W, 2)
    
    if L > 0 && !iszero(p_old)  
        cols_V == L*cols_W || error("V and W have inconsistent no. of columns")
    end
    
    # initialize data
    T1 = eltype(A) 
    if L > 0 && !iszero(p_old)
        T = zeros(T1,L, L)
        K = zeros(T1,1, L)
        D = zeros(T1,0,0)
        Ir = Matrix{T1}(I(cols_W))
        iC = findall(!iszero,imag(p_old))
        iCh = iC[1:2:end]
        iR = findall(iszero,imag(p_old))
        isubdiag = [iR; iCh]
        # process previous shifts
        h = 1   
        while h <= L
            is = isubdiag[isubdiag .< h]
            K[1, h] = 1
            if isreal(p_old[h])  # real shift 
                rpc = real(p_old[h])
                T[h, h] = rpc
                if !isempty(is)
                    T[h, is] = 2 * rpc * ones(1, length(is))
                end
                D = cat(D, sqrt(-2 * rpc), dims=Val((1,2)))
                h = h + 1;
            else # complex conjugated pair of shifts
                rpc = real(p_old[h])
                ipc = imag(p_old[h])
                β = rpc / ipc
                T[h:h + 1, h:h + 1] = [3*rpc -ipc
                                       ipc*(1 + 4 * β^2) -rpc]
                if !isempty(is)
                    T[h:h +  1, is] = [4*rpc; 4*rpc*β] * ones(1, length(is))
                end
                D = cat(D, 2*sqrt(-rpc)*[1  0; β sqrt(1 + β^2)], dims=Val((1,2)))
                h = h + 2;
            end
        end
        S = kron(D \ (T * D), Ir)
        K = kron(K * D, Ir)
    else  # explicit AV 
        S = 0
        K = 1
        !iszero(p_old) && (W = A*V)
    end
    
    # compute projection matrices
    F = eigen(V'*V)
    s = real(F.values)
    v = F.vectors
    r = (s .> eps() * s[end] * cols_V)
    st = v[:,r]*Diagonal(1 ./ s[r].^.5)
    U = V * st
  
    ## Project V and compute Ritz values
    if withE
        E_V = E*V
        G = U' * E_V;
        H = U' * W * K * st + G * (S * st);
        G = G * st;
        p = eigvals(H, G)
    else
        H = U' * (W * K) * st + U' * (V * (S * st))
        p = eigvals(H)
    end
    # permute complex shifts if necessary
    kk = 1
    while kk < length(p)
        if isreal(p[kk]) 
           kk += 1
        else
           if imag(p[kk]) < 0
              p[kk] = conj(p[kk])
              p[kk+1] = conj(p[kk])
           end
           kk += 2
        end
    end
  
    # postprocess new shifts
    
    # remove infinite values
    p = p[isfinite.(p)]
    
    # remove zeros
    p = p[abs.(p) .> 10*eps()]
     
    # make all shifts stable
    p[real.(p) .> 0] = -p[real.(p) .> 0]

    if !isempty(p) 
        # remove small imaginary perturbations
        small_imag = findall(abs.(imag.(p)) ./ abs.(p) .< 1e-12)
        p[small_imag] = real(p[small_imag])
        # sort (s.t. compl. pairs are together)
        #sort!(p; by = v -> (real(v), abs(imag(v))))
        isreal(p) ? sort!(p; by = real) : sort!(p; by = abs)
        # select nshifts 
        length(p) > nshifts && (p = subopt_select(p, nshifts))
    end

    return p
end
function subopt_select(RV, nshifts=length(RV))
    #  Determine for the vector RV of Ritz values and desired number of shifts nshifts, 
    #  nshifts (or occasionally nshifts+1) shift parameters P by selecting suboptimal values of 
    #  the min-max ADI shift parameter problem. 
    # 
    #  This is the function heuristic, borrowed from the DifferentialRiccatiEquations.jl package.
    s(t, P) = prod(abs(t - p) / abs(t + p) for p in P)
    length(RV) >= nshifts || throw(ArgumentError("length(RV) must be at least nshifts = $nshifts "))
    p0 = argmin(RV) do p
        maximum(s(t, (p,)) for t in RV)
    end
    P = isreal(p0) ? [p0] : [p0, conj(p0)]
    while length(P) < nshifts
        p = argmax(RV) do t
            s(t, P)
        end
        if isreal(p)
            push!(P, p)
        else
            append!(P, (p, conj(p)))
        end
    end

    return P
end
"""
    plyapdi(A, E, B; cyclic, abstol, reltol, maxiter, nshifts, shifts, loginf) -> (U, info)
    plyapdi(A, B; cyclic, abstol, reltol, maxiter, nshifts, shifts, loginf) -> (U, info)

Compute a low-rank factor `U` of the solution `X = UU'` of the
generalized discrete Lyapunov equation

      AXA' - EXE' + BB' = 0,

where `A` and `E` are square real matrices and `B` is a real matrix
with the same number of rows as `A`. The pencil `A - λE` must have only
eigenvalues with moduli less than one. `E = I` is assumed in the second call. 


The named tuple `info` contains information related to the execution of the LR-ADI algorithm as follows:
`info.niter` contains the number of performed iterations; 
`info.res_fact` contains the norm of the residual factor; 
`info.res` contains, if `loginf = true`, the vector of normalized residual norms (normalized with respect to the norm of the initial approximation);
`info.rc` contains, if `loginf = true`, the vector of norms of relative changes in building the solution;
`info.used_shift` contains the vector of used shifts.

The keyword argument `abstol` (default: `abstol = 1e-12`) is the tolerance used for convergence test
on the normalized residuals, while the keyword argument
`reltol` (default: `reltol = 0`) is the tolerance for the relative changes of the solution. 
The keyword argument `maxiter` can be used to set the maximum number of iterations (default: `maxiter = 100`).
The keyword argument `nshifts` specifies the desired number of shifts to be used in an iteration cycle (default: `nshifts = 6`). 
The keyword argument `shifts` can be used to provide a pre-calculated set of complex conjugated shifts to be used
to start the iterations (default: `shifts = missing`).
With the keyword argument `loginf = true`, the normalized residual values and the norms of increments of the solution
can be saved as outputs in the resulting info structure (default: `loginf = false`). 

The low-rank ADI (LR-ADI) method with enhancements proposed in [1] is adapted to the discrete case, 
inspired by the MATLAB codes of the free software described in [2]. If `cyclic = true`, the cyclic low-rank method of [3] is adapted, with the
pre-calculated shifts provided in the keyword argument `shifts`. 

_References_

[1] P. Kürschner. Efficient Low-Rank Solution of Large-Scale Matrix Equations. 
    Dissertation, Otto-von-Guericke-Universität, Magdeburg, Germany, 2016. Shaker Verlag,

[2] P. Benner, M. Köhler, and J. Saak. “Matrix Equations, Sparse Solvers: M-M.E.S.S.-2.0.1—
    Philosophy, Features, and Application for (Parametric) Model Order Reduction.” 
    In Model Reduction of Complex Dynamical Systems, Eds. P. Benner et.al., 171:369–92, Springer, 2021.

[3] T. Penzl, A cyclic low-rank Smith method for large sparse Lyapunov equations, 
    SIAM J. Sci. Comput. 21 (4) (1999) 1401–1418.
"""
function plyapdi(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; cyclic = false,
                 abstol = 1e-12, reltol = 0, maxiter = 100, shifts = missing, nshifts = 6, loginf = false)    
    n = LinearAlgebra.checksquare(A)
    withE = !(E == I)
    if E == I 
       EE = I
       withE = false
    else
       LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix or I"))
       EE = E
       withE = true
    end
       
    T2 = promote_type(eltype(A), eltype(EE), eltype(B))
    T2 <: BlasFloat  || (T2 = promote_type(Float64,T2))
 
    adjA = isa(A,Adjoint)
    adjE = withE ? isa(E,Adjoint) : false
    adjB= isa(B,Adjoint)

    AA = A; BB = B
 
    nb, mb = size(B)
    nb == n || throw(DimensionMismatch("B must be a matrix of row dimension $n"))

    eltype(AA) == T2 || (adjA ? AA = convert(Matrix{T2},AA.parent)' : AA = convert(Matrix{T2},AA))
    withE && (eltype(EE) == T2 || (adjE ? EE = convert(Matrix{T2},EE.parent)' : EE = convert(Matrix{T2},EE)))
    eltype(BB) == T2 || (adjB ? BB = convert(Matrix{T2},BB.parent)' : BB = convert(Matrix{T2},BB))
 
    if cyclic 
       ismissing(shifts) && (cyclic = false; @warn "missing cyclic shifts: cyclic reset to false")
    end
    # Start LR-ADI procedure
 
    used_shifts = zeros(T2,0)
    if iszero(B) 
       info = (niter = 0, res_fact = zero(T2), res = zeros(T2,0), rc = zeros(T2,0), 
               used_shifts = used_shifts)   
       return zeros(T2,n,0), info
    end

    W = Matrix(BB)   
    ncolZ = maxiter * mb
    Z = zeros(T2, n, ncolZ)
    res0 = norm(W' * W)
    res1 = 1
    U = copy(W)
    nrmZ = 0
    withE && (temp = similar(W))
    
    if ismissing(shifts) 
       p = zeros(0)
       i = 1
       while isempty(p)
             p = projection_shiftsd(AA, EE, U; nshifts)   
             if isempty(p)
                 if i < 5
                    @warn "Could not compute initial projection shifts. Going to retry with random right hand side"
                    U = rand(size(U)...)
                 else
                    error("Could not compute initial projection shifts")
                 end
             end      
             i = i + 1
       end
    else
        p = shifts
        illegal_shifts = false
        # Check if all shifts are in the open left half plane
        # and if complex pairs of shifts are properly ordered.
        kk = 1
        while kk <= length(p)
            if isreal(p[kk]) 
                illegal_shifts = abs(p[kk]) >= 1 
            else
                illegal_shifts = p[kk + 1] != conj(p[kk]) || abs(p[kk]) >= 1
                kk = kk + 1
            end
            illegal_shifts && error("Improper shift vector detected")
            kk = kk + 1
        end
    end
    lp = length(p)

    if loginf 
       res = zeros(maxiter)  
       rc = zeros(maxiter)
    else
       res = zeros(T2,0) 
       rc = zeros(T2,0)
    end
    psave = complex(zeros(maxiter))
    
    # start iteration
    jZ = 0
    k = 1
    k_shift = 1
    mb2 = 2*mb
    while k < maxiter + 1 
        if k_shift > lp
           # update shifts
           k_shift = 1
           if !cyclic 
              used_shifts = [used_shifts; p]
              first_dropped = length(used_shifts) - nshifts + 1
              last_kept = first_dropped - 1
              if length(used_shifts) > nshifts && 
                 imag(used_shifts[first_dropped]) > 0 && 
                 (abs(used_shifts[first_dropped] - conj(used_shifts[last_kept])) < eps())
                 # don't cut between pair of complex shifts
                 used_shifts = used_shifts[end - nshifts:end]
              elseif length(used_shifts) > nshifts
                 used_shifts = used_shifts[end - nshifts + 1:end]
              end
              len = size(used_shifts, 1) * mb - 1        
              p = projection_shiftsd(AA, EE, view(Z,:, jZ-len:jZ); nshifts)
              lp = length(p)
           end
        end
        isempty(p) && error("empty shift vector computed; the pair (A,E) is possibly unstable")
        # get current shift
        pc = p[k_shift]
        if isreal(pc) 
            pc = real(pc)
        else
            jZ+1 >= ncolZ && break   
        end
        # perform the actual step computations, i. e. shifted solve
        #V = adj ? ((withE ? E.parent : I) - conj(pc)*A.parent)'\W : (E -conj(ps)*A)\W
        if adjA && adjE
            V = (EE.parent - pc*AA.parent)'\W 
        else
            V = (EE-conj(pc)*AA)\W 
        end

        α = real(pc); β = imag(pc)  
        if isreal(pc) 
            τ = 1-α^2
            # update the factor
            #Z[:, (m - 1) * k + 1:m * k] = sqrt(τ) * V     
            Z[:, jZ+1:jZ+mb] .= sqrt(τ) .* V     
            jZ += mb
            # update low-rank residual
            if withE
               mul!(W,EE,V,τ/α,-1/α)
            else
               W .= (τ/α) .* V .- (1/α) .* W
            end
            psave[k] = pc
        else
            as2 = α^2+β^2;   τ = 1-as2  
            # perform a double step with the known solution for the conjugate shift
            θ = α/β
            mu = sqrt(τ/(1+as2))  
            Z1 = view(Z,:,jZ+1:jZ+mb)
            Z2 = view(Z,:,jZ+mb+1:jZ+mb2)
            Z1 .= sqrt(1-as2^2) .* real.(V) .+ (θ*mu*(1-as2)) .* imag.(V)
            Z2 .=  (mu*abs(pc^2-1)/β) .* imag.(V)
            jZ +=mb2
    
            # update low-rank residual for double step
            if withE
               temp .= (as2-1/as2).*real.(V) .- (θ*τ^2/as2) .* imag.(V)
               mul!(W,EE,temp,1,1/as2)
            else
                W .= (1/as2) .* W .+ (as2-1/as2).*real.(V) .- (θ*τ^2/as2) .* imag.(V)
            end
            psave[k] = pc
            psave[k+1] = conj(pc)
    
            k += 1
            k_shift += 1
    
        end
        # evaluate normalized residual
        resm = norm(W' * W) / res0

        # evaluate norm of correction of solution  
        if isreal(pc) 
            k == 1 && (res1 = resm)
            nrmV = 2.0 * abs(pc) * norm(V)^2
        else # complex double step means 2 blocks added
            k < 2 && (res1 = resm)
            nrmV = norm(Z1)^2+norm(Z2)^2
        end
    
        nrmZ = nrmZ + nrmV
        rcm = sqrt(nrmV / nrmZ)
        if loginf 
           if isreal(pc) 
              res[k] = resm; rc[k] = rcm
           else
              res[k] = resm; res[k-1] = resm
              rc[k] = rcm; rc[k-1] = rcm
           end 
        end

        # check convegence conditions
        if (abstol > 0 && resm < abstol) || (reltol > 0 && rcm < reltol) 
           break
        end 
        # checck for possible divergence
        k > 2 && resm > 100*res1 && error("residual is growing: eigenvalues are probably unstable")
        
        k += 1
        k_shift += 1
    end
    
    niter = k - (k > maxiter)
    info = (niter = niter, res_fact = norm(W), 
            res = loginf ? res[1:niter] : zeros(T2,0), 
            rc = loginf ? rc[1:niter] : zeros(T2,0), 
            used_shifts = isreal(psave) ? real(psave[1:niter]) : psave[1:niter])   
    
    if niter >= maxiter
        @warn "LR-ADI reached maximum iteration number. Results may be inaccurate!"
    end
    return Z[:, 1:niter*mb], info
end
function plyapdi(A::AbstractMatrix, B::AbstractMatrix; kwargs...)
    plyapdi(A::AbstractMatrix, I, B::AbstractMatrix; kwargs...)
end 

function projection_shiftsd(A, E, V; nshifts = 6)
    # p = projection_shiftsd(A, E, V; nshifts = 6) 
    #
    # Compute up to nshifts new shifts as Ritz values of of the pair (A,E), 
    # by explicitly projecting the E and A matrices to the span of V, where 
    # is the newest part of the ADI solution factor Z.
        
    withE = (E != I)
             
    # compute projection matrices
    F = eigen(V'*V)
    s = real(F.values)
    v = F.vectors
    r = (s .> eps() * s[end] * size(V,2))
    st = v[:,r]*Diagonal(1 ./ s[r].^.5)
    U = V * st
  
    ## Project V and compute Ritz values
    if withE
        p = eigvals(U'*(A*U), U'*(E*U))
    else
        p = eigvals(U'*(A*U))
    end
    # permute complex shifts if necessary
    kk = 1
    while kk < length(p)
        if isreal(p[kk]) 
           kk += 1
        else
           if imag(p[kk]) < 0
              p[kk] = conj(p[kk])
              p[kk+1] = conj(p[kk])
           end
           kk += 2
        end
    end
  
    # postprocess new shifts
    
    # remove infinite values
    p = p[isfinite.(p)]
    
    # remove zeros
    p = p[abs.(p) .> 10*eps()]
     
    # make all shifts stable
    p[abs.(p) .> 1] = 1 ./ p[abs.(p) .> 1]

    if !isempty(p) 
        # remove small imaginary perturbations
        small_imag = findall(abs.(imag.(p)) ./ abs.(p) .< 1e-12)
        p[small_imag] = real(p[small_imag])
        # sort (s.t. compl. pairs are together)
        #sort!(p; by = v -> (real(v), abs(imag(v))))
        isreal(p) ? sort!(p; by = real) : sort!(p; by = abs)
        # select nshifts 
        length(p) > nshifts && (p = subopt_select(p, nshifts))
    end

    return p
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
        α = gamma / delta
        
        #x     = x + α*p
        axpy!(α,p,x)
        #r     = r - α*q
        axpy!(-α,q,r)
           
        #s = A'*r - shift*x
        mul!(s,adjointA,r)
        shift == 0 || axpy!(-shift, x, s)
           
        norms  = norm(s)
        gamma1 = gamma
        gamma  = norms^2
        β   = gamma / gamma1
        #p      = s + β*p
        axpby!(ONE,s,β,p)
 
        # Convergence
        normx = norm(x)
        xmax  = max(xmax, normx)
        #flag  = Int((norms <= norms0 * tol) || (normx * tol >= 1))
        flag  = Int((norms <= max(norms0 * reltol, abstol)) || (normx * reltol >= 1))
        
        # Output
        resNE = norms / norms0; 
        #@show k, resNE
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
