module Test_iterative

using LinearAlgebra
using MatrixEquations
using GenericSchur
using GenericLinearAlgebra
using DoubleFloats
using Test
using SparseArrays
using IterativeSolvers


@testset "Iterative solvers" begin
    n = 10
    T = ComplexF32
    T = Double64
    @testset "Matrix{$T}" for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat, Double64)
        A = rand(T, n, n)
        #A = A' * A + I
        b = rand(T, n)
        reltol = √eps(real(T))/10

        @time x, info = cgls(A, b; reltol=reltol, maxiter=2n);
        @test norm(A*x - b) / norm(b) ≤ 10*reltol
        @test info.flag == 1

        # If you start from the exact solution, you should converge immediately
        x, info = cgls(A, b; x0 = A \ b, abstol=n*n*eps(real(T)), reltol)
        @test info.flag == 1 && info.iter <= 2

        # All-zeros rhs should give all-zeros lhs
        x0 = cgls(A, zeros(T, n))[1]
        @test x0 == zeros(T, n)
    end

    @testset "T/H-Lyapunov with upper triangular singular U"  begin   
        n = 5
        Ty = Float64
        reltol = 1.e-7
        # real case
        U = triu(rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; #U[7,7] = 0
        X0 = triu(rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
        Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
      
        X, info = tulyapci(U, Q, adj = false, reltol=1.e-14, maxiter=1000); 
        @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 

           
        Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
     
        X, info = tulyapci(U,Q,adj = true, reltol=1.e-14,maxiter=1000); 
        @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 
    

        # complex case
        U = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; 
        X0 = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
        Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
        X, info = tulyapci(U, Q, adj = false, reltol=1.e-14, maxiter=1000); 
        @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 
    
        Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
     
        X, info = tulyapci(U,Q,adj = true, reltol=1.e-14,maxiter=1000); 
        @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 
    
    
        # complex hermitian case
        U = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; 
        X0 = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
        Q = Matrix(Hermitian(U'*X0 + X0'*U))
        @time X, info = hulyapci(U, Q; adj = false)  
        @test norm(U'*X + X'*U - Q)/norm(X) < reltol 
    
        Q = Matrix(Hermitian(U*X0' + X0*U'))
        @time X, info = hulyapci(U, Q; adj = true, reltol = 1.e-14)  
        @test norm(U*X' + X*U'- Q)/norm(X) < reltol 
    end

    @testset "T/H-Lyapunov"  begin   
        m = 3; n = 5
        Ty = Float64
        reltol = 1.e-7
        @testset "Matrix{$T}" for T in (Float32, Float64, BigFloat, Double64)
            # T-Lyapunov
            # real case
            A = rand(Ty,m,n);
            X0 = rand(Ty,n,m);
            C = Matrix(Symmetric(A*X0 + transpose(X0)*transpose(A)))
            @time X, info = tlyapci(A, C, adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X + transpose(X)*transpose(A) - C)/norm(X) < reltol 

            Y = A*X0; C = Y - transpose(Y);
            @time X, info = tlyapci(A, C, -1; adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X - transpose(X)*transpose(A) - C)/norm(X) < reltol 


            X0 = rand(Ty,m,n);
            C = Matrix(Symmetric(A*transpose(X0)+X0*transpose(A)))
            @time X, info = tlyapci(A, C, adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*transpose(X)+X*transpose(A) - C)/norm(X) < reltol 

            Y = A*transpose(X0); C = Y - transpose(Y);
            @time X, info = tlyapci(A, C, -1; adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*transpose(X)-X*transpose(A) - C)/norm(X) < reltol 


            # complex case
            A = rand(Ty,m,n)+im*rand(Ty,m,n); 
            X0 =rand(Ty,n,m)+im*rand(Ty,n,m); 
            C = Matrix(Symmetric(A*X0 + transpose(X0)*transpose(A)))
            @time X, info = tlyapci(A, C, adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X + transpose(X)*transpose(A) - C)/norm(X) < reltol 

            Y = A*X0; C = Y - transpose(Y);
            @time X, info = tlyapci(A, C, -1; adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X - transpose(X)*transpose(A) - C)/norm(X) < reltol 


            X0 =rand(Ty,m,n)+im*rand(Ty,m,n); 
            C = Matrix(Symmetric(A*transpose(X0)+X0*transpose(A)))
            @time X, info = tlyapci(A, C, adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*transpose(X)+X*transpose(A) - C)/norm(X) < reltol 

            Y = A*transpose(X0); C = Y - transpose(Y);
            @time X, info = tlyapci(A, C, -1; adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*transpose(X)-X*transpose(A) - C)/norm(X) < reltol 


            # H-Lyapunov
            # real case
            A = rand(Ty,m,n);
            X0 = rand(Ty,n,m);
            C = Matrix(Hermitian(A*X0 + X0'*A'))
            @time X, info = hlyapci(A, C, adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X +X'*A' - C)/norm(X) < reltol 

            Y = A*X0; C = Y - Y';
            @time X, info = hlyapci(A, C, -1; adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X - X'*A' - C)/norm(X) < reltol 

            X0 = rand(Ty,m,n);
            C = Matrix(Hermitian(A*X0'+X0*A'))
            @time X, info = hlyapci(A, C, adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X'+X*A' - C)/norm(X) < reltol 

            Y = A*X0';  C = Y - Y';
            @time X, info = hlyapci(A, C, -1; adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X'-X*A' - C)/norm(X) < reltol 


            # complex case
            A = rand(Ty,m,n)+im*rand(Ty,m,n); 
            X0 =rand(Ty,n,m)+im*rand(Ty,n,m); 
            C = Matrix(Hermitian(A*X0 + X0'*A'))
            @time X, info = hlyapci(A, C, adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X +X'*A' - C)/norm(X) < reltol 

            Y = A*X0; C = Y - Y';
            @time X, info = hlyapci(A, C, -1; adj = false, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X - X'*A' - C)/norm(X) < reltol 


            X0 = rand(Ty,m,n)+im*rand(Ty,m,n); 
            C = Matrix(Hermitian(A*X0'+X0*A'))
            @time X, info = hlyapci(A, C, adj = true, reltol=1.e-14, maxiter=1000); 
            @test  norm(A*X'+X*A' - C)/norm(X) < reltol 

            Y = A*X0';  C = Y - Y';
            @time X, info = hlyapci(A, C, -1; adj = true, reltol=1.e-14, maxiter=1000); 
            @test norm(A*X'-X*A' - C)/norm(X) < reltol 

        end
    end

    @testset "Generalized T/H-Sylvester"  begin   
        a = [rand(2,3)]; b = [I];  c = [I]; d = [a[1]'];
        L = gsylvop(a,b,c,d;nx=2)
        L1 = lyaplikeop(a[1])
        @test Matrix(L) == Matrix(L1)
        @test Matrix(L') == Matrix(L1')

        m = 3; n = 5; mx = 2; nx = 4; la = 2; lc = 1
        Ty = Float64
        reltol = 1.e-6
        A = [rand(Ty,m,mx) for i in 1:la]
        B = [rand(Ty,nx,n) for i in 1:la]
        C = [rand(Ty,m,nx) for i in 1:la]
        D = [rand(Ty,mx,n) for i in 1:la]
        L = gsylvop(A,B,C,D)
        X0 = rand(Ty,mx,nx);
        # solve with cgls for an exact solution
        e = L*vec(X0);
        @time x, info = cgls(L, e; reltol = 1.e-7);
        @test info.flag == 1 && norm(L*x-e)/norm(x) < reltol
        # solve with cgls for a least-squares solution
        b = rand(m*n); 
        @time x, info = cgls(L, b; reltol = 1.e-7);
        @test info.flag == 1
        # solve with cgls for an underdetermine solution
        b = rand(mx*nx); 
        @time x, info = cgls(L', b; reltol = 1.e-7);
        @test info.flag == 1 && norm(L'*x-b)/norm(x) < reltol

        # solve with lsqr from IterativeSolvers.jl
        @time x1 = lsqr(L, e);
        @test norm(L*x1-e)/norm(x1) < reltol
        # solve with lsmr from IterativeSolvers.jl
        @time x2 = lsmr(L, e);
        @test norm(L*x2-e)/norm(x2) < reltol

        m = 3; n = 5; mx = 2; nx = 4; la = 2; lc = 1
        Ty = Float64
        reltol = 1.e-7
        @testset "Matrix{$Ty}" for Ty in (Float64, BigFloat, Double64)
            reltol = √eps(real(Ty))
            # real data
            A = [rand(Ty,m,mx) for i in 1:la]
            B = [rand(Ty,nx,n) for i in 1:la]
            C = [rand(Ty,m,nx) for i in 1:la]
            D = [rand(Ty,mx,n) for i in 1:la]
            L = gsylvop(A,B,C,D)
            X0 = rand(Ty,mx,nx);
            # solve with cgls for an exact solution
            E = reshape(L*vec(X0),m,n);
            X, info = gtsylvi(A,B,C,D,E; reltol = reltol/10)
            @test norm(L*vec(X)-vec(E))/norm(X) < 10*reltol        
   
            # complex data
            A = [rand(Ty,m,mx)+im*rand(Ty,m,mx) for i in 1:la]
            B = [rand(Ty,nx,n)+im*rand(Ty,nx,n) for i in 1:la]
            C = [rand(Ty,m,nx)+im*rand(Ty,m,nx) for i in 1:la]
            D = [rand(Ty,mx,n)+im*rand(Ty,mx,n) for i in 1:la]
            X0 = rand(Ty,mx,nx)+im*rand(Ty,mx,nx);
            # solve for Ttype with cgls for an exact solution
            L = gsylvop(A,B,C,D)
            E = reshape(L*vec(X0),m,n);
            X, info = gtsylvi(A,B,C,D,E; reltol = reltol/10)
            @test norm(L*vec(X)-vec(E))/norm(X) < 10*reltol       
            # solve for Htype with cgls for an exact solution
            LH = gsylvop(A,B,C,D; htype = true) 
            E = reshape(LH*vec(X0),m,n);
            X, info = ghsylvi(A,B,C,D,E; reltol = reltol/10)
            @test norm(LH*vec(X)-vec(E))/norm(X) < 10*reltol       
        end
    end

    A = [I,Matrix(rand(4,4))]
    B = [I, Matrix(rand(4,4))]
    C = [I,Matrix(rand(4,4))] 
    D = [I,Matrix(rand(4,4))] 
    E = rand(4,4); 
    L = gsylvop(A,B,C,D)
    @time X, info = gtsylvi(A,B,C,D,E; reltol = 1.e-8);
    @test norm(L*vec(X)-vec(E))/norm(X)  < 1.e-4     


    m = 3; n = 5; mx = 2; nx = 4; la = 2; lc = 1
    Ty = Float64
    reltol = √eps(real(Ty))
    # sparse real data
    A = [sprand(Ty,m,mx,0.5) for i in 1:la]
    B = [sprand(Ty,nx,n,0.5) for i in 1:la]
    C = [sprand(Ty,m,nx,0.5) for i in 1:la]
    D = [sprand(Ty,mx,n,0.5) for i in 1:la]
    L = gsylvop(A,B,C,D)
    X0 = sprand(Ty,mx,nx,0.4);
    # solve with cgls for an exact solution
    E = reshape(L*vec(X0),m,n);
    X, info = gtsylvi(A,B,C,D,E; reltol = reltol/10)
    @test norm(L*vec(X)-vec(E))/norm(X) < 10*reltol        

    # solve AX+XD = E for sparse matrices
    mx = 4; nx = 400;
    As = [sprand(Ty,mx,mx,0.9),I]; Bs = [I,sprand(Ty,nx,nx,0.05)];
    Cs = []; Ds = [];
    Ls = gsylvop(As,Bs,Cs,Ds)
    X0 = sprand(Ty,mx,nx,0.4);
    Es = reshape(Ls*vec(X0),mx,nx);
    @time Xs, infos = gtsylvi(As,Bs,Cs,Ds,Es; reltol = 1.e-8, maxiter = 2000);
    @test norm(Ls*vec(Xs)-vec(Es))/norm(Xs) < 1.e-4 

    A = [Matrix(As[1]),I]
    B = [I, Matrix(Bs[2])]
    C = Cs; D = Ds; E = Matrix(Es); 
    L = gsylvop(A,B,C,D)
    @time X, info = gtsylvi(A,B,C,D,E; reltol = 1.e-8, maxiter = 5000);
    @test norm(L*vec(X)-vec(E))/norm(X)  < 1.e-4   
      
    
    n = 5; m = 4;
    Ty = Float64
    Ty = ComplexF64
    Ty = BigFloat
    @testset "Matrix{$Ty}" for Ty in (Float64, ComplexF64, BigFloat, Complex{BigFloat}, Double64, Complex{Double64})
        reltol = √eps(real(Ty))
        A = rand(Ty,n,n); E = rand(Ty,n,n);  
        Q = rand(Ty,n,n); C = Hermitian(Q);           
        # Lyapunov equation, Hermitian case
        X, info = lyapci(A, C)
        @test norm(A*X+X*A'+C)/norm(X)  < 1.e-4   && ishermitian(X)
        X, info = lyapdi(A, C) 
        @test norm(A*X*A' -X+C)/norm(X)  < 1.e-4   && ishermitian(X)
       
        # Lyapunov equation, non-Hermitian case
        X, info = lyapci(A, Q)
        @test norm(A*X+X*A'+Q)/norm(X)  < 1.e-4   
        X, info = lyapdi(A, Q)
        @test norm(A*X*A' -X+Q)/norm(X)  < 1.e-4   

        # generalized Lyapunov equation, Hermitian case
        X, info = lyapci(A, E, C)
        @test norm(A*X*E'+E*X*A'+C)/norm(X)  < 1.e-4   && ishermitian(X)
        X, info = lyapdi(A, E, C) 
        @test norm(A*X*A' -E*X*E'+C)/norm(X)  < 1.e-4   && ishermitian(X)
       
         # generalized Lyapunov equation, non-Hermitian case
        X, info = lyapci(A, E, Q)
        @test norm(A*X*E'+E*X*A'+Q)/norm(X)  < 1.e-4   
        X, info = lyapdi(A, E, Q)
        @test norm(A*X*A' - E*X*E'+Q)/norm(X)  < 1.e-4   

        #  Sylvester equation
        B = rand(Ty,m,m); W = rand(Ty,n,m)
        X, info = sylvci(A, B, W)
        @test norm(A*X+X*B-W)/norm(X)  < 1.e-4   
        X, info = sylvdi(A, B, W)
        @test norm(A*X*B+X-W)/norm(X)  < 1.e-4   
    
        #  generalized Sylvester equation
        C = rand(Ty,n,n); D = rand(Ty,m,m)
        X, info = gsylvi(A, B, C, D, W)
        @test norm(A*X*B+C*X*D-W)/norm(X)  < 1.e-4   
        X, info = gsylvi(A, B', C', D, W)
        @test norm(A*X*B'+C'*X*D-W)/norm(X)  < 1.e-4   
    end



   
 
end


end


