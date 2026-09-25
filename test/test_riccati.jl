module Test_riccati

using Random
using LinearAlgebra
using MatrixEquations
using Test
using GenericSchur
using GenericLinearAlgebra

function randomDoublyStochasticMatrix(n; num_perm = n^2)

    M = zeros(n,n)
    α = rand(num_perm)
    α = α / sum(α)

    for i=1:num_perm
        perm = randperm(n)
        for j=1:n
            M[perm[j],j] += α[i]
        end
    end

    return M
end



println("Test_riccati")

@testset "Testing algebraic Riccati equation solvers" begin

@testset "Nonsymmetri/nonhermitian Riccati equation" begin

# only double precision tests are performed

# unilateral quadratic matrix equation (UQME): Kressner et al. 2019
n = 50; 
A1 = diagm(0 => 30*ones(n),-1=>-10*ones(n-1),1=>-10*ones(n-1) ); A1[1,1] = 20; A1[n,n] = 20. ; 
A2 = zeros(n,n); G = -Matrix{Float64}(I,n,n); 
Q = diagm(0 => 15*ones(n),-1=>-5*ones(n-1),1=>-5*ones(n-1) );
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*x+A1*x+Q) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10 

# left and right stabilization example with Q = 0
A1 = [-2 0 0;0 1 0;0 0 4]; A2 = [5 0; 0 1]; G = [5 1 0; 1 2 1]; Q = zeros(3,2);
K = [0 0; 6 0; -30 5];
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && x ≈ K && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10 && 
      maximum(real(eigvals(A1-x*G))) < 0 && maximum(real(eigvals(A2-G*x))) < 0 

# positive solution: Example 6.1 of Guo (SIMAX 2001)
n = 100;
R = rand(n,n);
W = diagm(R*ones(n))-R; 
α = 0.1; #α = -0.001; α = 0.; 
M = W + α*I;
i1 =1:div(n,2); i2 = div(n,2)+1:n;
A2 = -M[i1,i1]; A1 = -M[i2,i2]; G = -M[i1,i2]; Q = -M[i2,i1];
D = M[i1,i1]; A = M[i2,i2]; C = M[i1,i2]; B = -M[i2,i1];
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*C*x-x*D-A*x+B) < 1e-10 
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10


# positive solution: Example 6.2 of Guo (SIMAX 2001)
m = 50; 
R = randomDoublyStochasticMatrix(2m);
a = 1; 
M = a*(I - R); 
i1 =1:m; i2 = m+1:n;
A2 = -M[i1,i1]; A1 = -M[i2,i2]; G = -M[i1,i2]; Q = -M[i2,i1];
D = M[i1,i1]; A = M[i2,i2]; C = M[i1,i2]; B = -M[i2,i1];
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*C*x-x*D-A*x+B) < 1e-10 
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# doubly stochastic solution: Example 6.2 of Guo (SIMAX 2001)
m = 100
B = C = Matrix{Float64}(I,m,m);
A = D = diagm(1 => -ones(m-1), 0 => 2*ones(m),-m+1 => [-1]);
A1 = -A; A2 = -D; G = -C; Q = B;
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*C*x-x*D-A*x+B) < 1e-10 
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 && 
      norm(x*ones(m)-ones(m),Inf) < 1e-7 && norm(x'*ones(m)-ones(m),Inf) < 1e-7 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# positive solution:  
A1 = A2 = G = -Matrix{Float64}(I,2,2)
Q = [0 1;1 0]
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# positive solution: Example 2.1 SIMAX Guo & Laub (SIMAX 2000)

c1 = c2 = 1/2;
w1 = 3/4;
w2 = 1/4;
c = 1/2;
α = 0.1
α = 0.2
e = ones(2,1); q = [c1/2/w1;c2/2/w2]
δ1 = 1/c/w1/(1+α); δ2 = 1/c/w2/(1+α)
d1 = 1/c/w1/(1-α); d2 = 1/c/w2/(1-α)

A1 = -(diagm([δ1; δ2]) − e*q'); # -A
A2 = -(diagm([d1; d2]) − q*e'); # -D
G = -q*q'; # -C
Q = e*e';  # B
# -XD -AX + XCX + B = 0
x, ev = nare(A1,A2,G,Q; as=false);

@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# positive solution: Example 5.1 SIMAX Guo & Laub (SIMAX 2000)
α = 4.27; α = 6.;
A1 = -[α -2;-1 α]; 
A2 = -[5 -1; -1 4];
G = -[3 4;2 1];
Q = [1 1;2 1]
x, ev = nare(A1,A2,G,Q; as=false);

@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 &&  minimum(x) > 0.0 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10



# positive stochastic solution: Example Guo & Higham   SIMAX 2007
m = n = 100;
A = diagm(0 => 3*ones(m), 1 => -ones(m-1), -m+1 => [-1]); A[m,m]=1.9;   
B = diagm(0 => ones(m), 1 => ones(m-1)); B[m,m] = 0.9
C = diagm(0 => ones(m), -1 => ones(m-1));   
D = diagm(0 => 3*ones(m), 1 => -ones(m-1), -m+1 => [-1]); D[1,1] = 2.;
A1 = -A; A2 = -D; G = -C; Q = B;
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*C*x-x*D-A*x+B) < 1e-10 
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 &&  minimum(x) > 0.0 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10


nm = 10; n = 5; m = nm-n; i1 = 1:n; i2 = n+1:nm
M = [0.92038    0.539384   0.830556  0.856029   0.25361    0.115753  0.0487206  0.034561   0.724532   0.875804
 0.533231   0.0199564  0.97188   0.395389   0.140877   0.770254  0.289926   0.270919   0.219031   0.847851
 0.406041   0.203171   0.195407  0.491316   0.581786   0.869959  0.723029   0.755385   0.230713   0.805813
 0.481145   0.771876   0.294938  0.470606   0.0478576  0.90678   0.754986   0.0837657  0.0170108  0.0886567
 0.0884447  0.172554   0.619271  0.435887   0.935235   0.607263  0.748878   0.560723   0.761095   0.256845
 0.892364   0.767856   0.512534  0.474933   0.462245   0.385894  0.647882   0.651101   0.690611   0.664415
 0.400024   0.357446   0.225022  0.255931   0.762162   0.625031  0.208448   0.833267   0.736438   0.588668
 0.942787   0.0715331  0.50873   0.0556853  0.649507   0.319407  0.070113   0.642313   0.14744    0.0204208
 0.885485   0.594088   0.896687  0.694776   0.235911   0.823385  0.136315   0.346832   0.893713   0.995861
 0.457231   0.881096   0.710535  0.538029   0.455018   0.399169  0.238008   0.884699   0.6813     0.0830576];
@views A2 = M[i1,i1]; G = M[i1,i2]; Q = M[i2,i1]; A1 = M[i2,i2];
#x, ev = nare(M11,M12,M21,M22; as=false);
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10


nm = 10; n = 5; m = nm-n; i1 = 1:n; i2 = n+1:nm
M = [ 0.206894  0.647987   0.978259  0.573942    0.760473  0.721102   0.991653  0.665694    0.49927    0.0283918
 0.485164  0.671879   0.912292  0.962415    0.182261  0.466124   0.558987  0.807104    0.419912   0.629134
 0.834098  0.590364   0.953538  0.766051    0.121736  0.70641    0.263044  0.524945    0.0338247  0.533112
 0.553465  0.0635469  0.83314   0.828064    0.115488  0.0109646  0.948111  0.0772236   0.313495   0.447261
 0.430239  0.0123586  0.712853  0.00642428  0.809065  0.401336   0.559325  0.968344    0.625919   0.338503
 0.521394  0.026751   0.15523   0.0862366   0.203291  0.759901   0.10982   0.00902374  0.305064   0.022878
 0.771858  0.616199   0.268513  0.400727    0.792068  0.526531   0.98098   0.500107    0.234162   0.289052
 0.875481  0.36195    0.808351  0.772763    0.616048  0.212764   0.857495  0.893323    0.021414   0.360048
 0.946474  0.158071   0.285827  0.286346    0.602634  0.545074   0.381267  0.77789     0.483196   0.0706794
 0.186945  0.361477   0.293914  0.386531    0.944161  0.344333   0.587848  0.595932    0.715252   0.839841]
 # A1X + XA2 - XGX + Q = 0.
@views A2 = M[i1,i1]; G = M[i1,i2]; Q = M[i2,i1]; A1 = M[i2,i2];
x, ev = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

x, ev = nare(A1,A2,G,Q; as=true);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

x, ev = nare(A1,A2,G,Q; as=false, disc = true);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

@test_throws "suitable" x, ev = nare(A1,A2,G,Q; as=true, disc = true)
x, ev = nare(A1,A2,G,Q; as=false, disc = true, stab = false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10


A2 = -M[i1,i1]; G = -M[i1,i2]; Q = -M[i2,i1]; A1 = -M[i2,i2];      
x, ev = nare(A1,A2,G,Q; as=true);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

A2 = 2M[i1,i1]; G = 2M[i1,i2]; Q = 2M[i2,i1]; A1 = 2M[i2,i2]; 
x, ev = nare(A1,A2,G,Q; as=true,disc = true);  
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10   


A2 = .2M[i1,i1]; G = .2M[i1,i2]; Q = .2M[i2,i1]; A1 = .2M[i2,i2]; 
x, ev = nare(A1,A2,G,Q; as=false,disc = true);  
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10   

end      

# only double precision tests are performed

#for (Ty,n,p,m) in ((Float64, 30, 10, 10), (Float32, 5, 3, 3))
for (Ty,n,p,m) in ((Float64, 20, 10, 10),  (Float64, 5, 3, 1))

Random.seed!(21235)

#(Ty,n,p,m) = (Float64, 20, 10, 10)
#(Ty,n,p,m) = (Float64, 2, 1, 1)

ar = randn(Ty,n,n)
er = rand(Ty,n,n)
if m == 1
    br = rand(Ty,n)
    bc = rand(Ty,n)+im*rand(Ty,n)
    sc = rand(Complex{Ty},n)/100
    sr = rand(Ty,n)/100
else
    br = rand(Ty,n,m)
    bc = rand(Ty,n,m)+im*rand(Ty,n,m)
    sc = rand(Complex{Ty},n,m)/100
    sr = rand(Ty,n,m)/100
end
cr = rand(Ty,p,n)
ac = randn(Ty,n,n) + im*randn(Ty,n,n)
ec = er+im*rand(Ty,n,n)
cc = rand(Ty,p,n)+im*rand(Ty,p,n)
bc = rand(Ty,n,m)+im*rand(Ty,n,m)
rr1 = rand(Ty,m,m)
rc1 = rand(Ty,m,m)+im*rand(Ty,m,m)
#Ty == Float64 ? rtol = n*n*sqrt(eps(1.)) : rtol = n*sqrt(eps(1.f0))
rtol = n*n*sqrt(eps(1.)) 

qc = cc'*cc
Qr = cr'*cr
gc = bc*bc'
gr = br*br'
rr = rr1*rr1'
rc = rc1*rc1'




@testset "Continuous Riccati equation ($Ty, n = $n, m = $m)" begin

@time x, clseig = arec(ar,gr,Qr)
@test norm(ar'*x+x*ar-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = nare(ar',ar,gr,Qr)
@test norm(ar'*x+x*ar-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr,Qr,as = true)
@test norm(ar'*x+x*ar-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = nare(ar',ar,gr,Qr,as = true)
@test norm(ar'*x+x*ar-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr)
@test norm(ar'*x+x*ar-x*gr*x)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,2I,I)
@test norm(ar'*x+x*ar-x*2*x+I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-2*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-2*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,2,I)
@test norm(ar'*x+x*ar-x*2*x+I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-2*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-2*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr,2I)
@test norm(ar'*x+x*ar-x*gr*x+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr,2)
@test norm(ar'*x+x*ar-x*gr*x+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,2I,Qr)
@test norm(ar'*x+x*ar-x*2*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-2*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-2*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar',gr,Qr)
@test norm(ar*x+x*ar'-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar'-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar'-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac,gc,qc)
@test norm(ac'*x+x*ac-x*gc*x+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-gc*x))))/norm(clseig)  < rtol

@time x, clseig = nare(ac',ac,gc,qc)
@test norm(ac'*x+x*ac-x*gc*x+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac',gc,qc)
@test norm(ac*x+x*ac'-x*gc*x+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac'-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac'-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gc,qc)
@test norm(ar'*x+x*ar-x*gc*x+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac,gr,Qr)
@test norm(ac'*x+x*ac-x*gr*x+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-gr*x))))/norm(clseig)  < rtol

try
    arec(zeros(2,2),[1 1;1 1],[1 0;0 0])  # Hamiltonian not dichotomic
    @test false
catch
    @test true
end

try
    arec(rand(2,2),0I,0I)  # no finite solution 
    @test false
catch
    @test true
end


end



@testset "Generalized continuous Riccati equation ($Ty, n = $n, m = $m)" begin

@time x, clseig = garec(ar,er,gr,Qr)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol

@time x, clseig = garec(ar,er,gr,Qr,as = true)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol

@time x, clseig = garec(ar,er,gr)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol


@time x, clseig = garec(ar,er,gr,2I)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x*er,er))))/norm(clseig)  < rtol


@time x, clseig = garec(ar,er,I,2I)
@test norm(ar'*x*er+er'*x*ar-er'*x*x*er+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-x*er,er))))/norm(clseig)  < rtol


@time x, clseig = garec(ar,I,1,2)
@test norm(ar'*x+x*ar-x*x+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-x))))/norm(clseig)  < rtol

end


@testset "Continuous control Riccati equation ($Ty, n = $n, m = $m)" begin


@time x, clseig, f, z = arec(ar,br,rr,Qr,sr)
@test norm(ar'*x+x*ar-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f, z = arec(ar,br,rr,Qr,sr;orth=true)
@test norm(ar'*x+x*ar-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol
norm(z'*z - I) < rtol

@time x, clseig, f = arec(ar,br,rr,Qr,sr,as=true)
@test norm(ar'*x+x*ar-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,rr,Qr)
@test norm(ar'*x+x*ar-x*br*inv(rr)*br'*x+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,I,Qr)
@test norm(ar'*x+x*ar-x*br*br'*x+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,1,Qr)
@test norm(ar'*x+x*ar-x*br*br'*x+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,rr,0I)
@test norm(ar'*x+x*ar-x*br*inv(rr)*br'*x)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,rr,0)
@test norm(ar'*x+x*ar-x*br*inv(rr)*br'*x)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rc,qc)
@test norm(ac'*x+x*ac-x*bc*inv(rc)*bc'*x+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rc,qc,sc)
@test norm(ac'*x+x*ac-(x*bc+sc)*inv(rc)*(bc'*x+sc')+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rr,Qr,sr)
@test norm(ac'*x+x*ac-(x*bc+sr)*inv(rr)*(bc'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,gr,rr,Qr,sr)
@test norm(ar'*x+x*ar-x*gr*x-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-gr*x))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,gr,rr,Qr,sr,as=true)
@test norm(ar'*x+x*ar-x*gr*x-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-gr*x))))/norm(clseig)  < rtol

g = 2
@time x, clseig, f, z = arec(ar,br,g*I,rr,Qr,sr)
@test norm(ar'*x+x*ar-x*g*x-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol

g = 2
@time x, clseig, f, z = arec(ar,br,g*I,rr,Qr,sr,as=true)
@test norm(ar'*x+x*ar-x*g*x-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol

g = 2; q = 3. *I; r = 1;
@time x, clseig, f, z = arec(ar,br,g*I,r,q,sr,as=true)
@test norm(ar'*x+x*ar-x*g*x-(x*br+sr)*inv(r)*(br'*x+sr')+q)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x))))/norm(clseig)  < rtol

ar1 = rand(Ty,2,2); br1 = rand(Ty,2,2); gr1 = 0*I; r1 = [1.e5 0; 0 1.e-5]; q1 = 0*I; sr1 = zeros(Ty,2,2);
@time x, clseig, f, z = arec(ar1,br1,gr1,r1,q1,sr1)
@test norm(ar1'*x+x*ar1-x*gr1*x-(x*br1+sr1)*inv(r1)*(br1'*x+sr1')+q1)/max(1,norm(x)) < rtol

end

@testset "Generalized continuous control Riccati equation ($Ty, n = $n, m = $m)" begin

@time x, clseig, f = garec(ar,er,br,rr,Qr,sr)
@test norm(ar'*x*er+er'*x*ar-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,rr,Qr,sr,as = true)
@test norm(ar'*x*er+er'*x*ar-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,gr,rr,Qr,sr)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er - (er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-gr*x*er,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,gr,rr,Qr,sr,as = true)
@test norm(ar'*x*er+er'*x*ar-er'*x*gr*x*er - (er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-gr*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-gr*x*er,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,rr,2I,sr)
@test norm(ar'*x*er+er'*x*ar-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+2I)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ac,ec,bc,rc,qc,sc)
@test norm(ac'*x*ec+ec'*x*ac-(ec'x*bc+sc)*inv(rc)*(bc'*x*ec+sc')+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ac,ec,bc,rr,Qr,sr)
@test norm(ac'*x*ec+ec'*x*ac-(ec'x*bc+sr)*inv(rr)*(bc'*x*ec+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,1,rr,Qr,sr)
@test norm(ar'*x*er+er'*x*ar-er'*x*x*er-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-x*er,er))))/norm(clseig)  < rtol

g = 2
@time x, clseig, f, z = garec(ar,er,br,g*I,rr,Qr,sr)
@test norm(ar'*x*er+er'*x*ar-er'*x*g*x*er-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol

g = 2
@time x, clseig, f = garec(ar,er,br,g,rr,Qr,sr,as=true)
@test norm(ar'*x*er+er'*x*ar-er'*x*g*x*er-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol

g = 2; q = 3. *I; r = 1;
@time x, clseig, f = garec(ar,er,br,g,r,q,sr,as=true)
@test norm(ar'*x*er+er'*x*ar-er'*x*g*x*er-(er'x*br+sr)*inv(r)*(br'*x*er+sr')+q)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f-g*x*er,er))))/norm(clseig)  < rtol


end


@testset "Discrete control Riccati equation ($Ty, n = $n, m = $m)" begin
@time x, clseig, f = ared(ar,br,rr,Qr)
@test norm(ar'*x*ar-x-ar'*x*br*inv(rr+br'*x*br*I)*br'*x*ar+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,0I)
@test norm(ar'*x*ar-x-ar'*x*br*inv(rr+br'*x*br*I)*br'*x*ar)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,0I; as = true)
@test norm(ar'*x*ar-x-ar'*x*br*inv(rr+br'*x*br*I)*br'*x*ar)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,Qr,sr)
@test norm(ar'*x*ar-x-(ar'*x*br+sr)*inv(rr+br'*x*br*I)*(br'*x*ar+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,Qr,sr,as = true)  #fails
@test norm(ar'*x*ar-x-(ar'*x*br+sr)*inv(rr+br'*x*br*I)*(br'*x*ar+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ac,bc,rc,qc)
@test norm(ac'*x*ac-x-ac'*x*bc*inv(rc+bc'*x*bc*I)*bc'*x*ac+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(ac)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(ac)  < rtol

@time x, clseig, f = ared(ac,bc,rc,qc,sc)
@test norm(ac'*x*ac-x-(ac'*x*bc+sc)*inv(rc+bc'*x*bc*I)*(bc'*x*ac+sc')+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(ac)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(ac)  < rtol

q = 3.f0 *I; r = 1;
@time x, clseig, f = ared(ar,br,r,q,sr,as=true)
@test norm(ar'*x*ar-x-(ar'*x*br+sr)*inv(r*I+br'*x*br)*(br'*x*ar+sr')+q)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/max(opnorm(ar,1),1)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/max(opnorm(ar,1),1)  < rtol

end


@testset "Generalized discrete control Riccati equation ($Ty, n = $n, m = $m)" begin
@time x, clseig, f = gared(ar,er,br,rr,Qr,sr)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br+sr)*inv(rr+br'*x*br*I)*(br'*x*ar+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol

@time x, clseig, f = gared(ar,er,br,rr,Qr,sr,as = true)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br+sr)*inv(rr+br'*x*br*I)*(br'*x*ar+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol

@time x, clseig, f = gared(ar,er,br,rr,0I)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br)*inv(rr+br'*x*br*I)*(br'*x*ar))/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol

@time x, clseig, f, z = gared(ar,er,br,rr,0I;as = true)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br)*inv(rr+br'*x*br*I)*(br'*x*ar))/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol

@time x, clseig, f = gared(ac,ec,bc,rc,qc,sc)
@test norm(ac'*x*ac-ec'*x*ec-(ac'*x*bc+sc)*inv(rc+bc'*x*bc*I)*(bc'*x*ac+sc')+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol

@time x, clseig, f = gared(ac,ec,bc,rc,qc,sc; as = true) #fails
@test norm(ac'*x*ac-ec'*x*ec-(ac'*x*bc+sc)*inv(rc+bc'*x*bc*I)*(bc'*x*ac+sc')+qc)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol

@time x, clseig, f = gared(ac,ec,bc,rr,Qr,sr)
@test norm(ac'*x*ac-ec'*x*ec-(ac'*x*bc+sr)*inv(rr+bc'*x*bc*I)*(bc'*x*ac+sr')+Qr)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/max(opnorm(ac,1),opnorm(ec,1))  < rtol

q = 3.f0 *I; r = 1;
@time x, clseig, f = gared(ar,er,br,r,q,sr,as=true)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br+sr)*inv(r*I+br'*x*br)*(br'*x*ar+sr')+q)/max(1,norm(x)) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/max(opnorm(ar,1),opnorm(er,1))  < rtol

end

end

@testset "Scaling continuous-time Riccati equation" begin
    
# some continuous-time problems which need scaling
A = [ 0.0        0.0        -0.00135213   0.0
      0.0        0.0         0.0         -0.000755718
      0.0531806  0.0         0.0          0.0378156
      0.0        0.0531806  -0.0709927    0.0];
B = [ 1.0  0.0
 0.0  1.0
 0.0  0.0
 0.0  0.0]; 
Q = [ 0.0  0.0  0.0        0.0
 0.0  0.0  0.0        0.0
 0.0  0.0  1.17474e9  0.0
 0.0  0.0  0.0        4.14026e9]; 
R = [100.0    0.0
   0.0  100.0];
G = B*inv(R)*B'; G = (G+G)/2   
E = [0.3145695364503345 0.28299421375349765 0.7751430938038222 0.3817600131380937; 
0.6175205621578082 0.9791019859058574 0.6388662440424374 0.36327849747268603; 
0.13292183504367217 0.9319918486921431 0.15347895705946646 0.1378470943397626; 
0.1411783703336893 0.6496471027507487 0.7764461576953698 0.2687776918944005];
reltol = sqrt(eps(1000.))

# without scaling
@time X, clseig = arec(A,G,Q; scaling = 'N')
rezn = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol)

# without scaling
@time X, clseig = nare(A',A,G,Q; scaling = 'N')
rezn = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol)

# with block scaling
@time X, clseig = arec(A,G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test  rezb < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with block scaling
@time X, clseig = nare(A',A,G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test  rezb < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# extended precision with block scaling
@time X, clseig = arec(BigFloat.(A),G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test  rezb < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# extended precision with block scaling
@time X, clseig = nare(BigFloat.(A)',BigFloat.(A),G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test  rezb < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with special scaling
@time X, clseig = arec(A,G,Q; scaling = 'S')
rezs = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezs < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with special scaling
@time X, clseig = nare(A',A,G,Q; scaling = 'S')
rezs = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezs < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with special scaling
@time X, clseig, Z, scalinfo = arec(A,G,Q; scaling = 'K', nrm = Inf)
rezk = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezk < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with special scaling
@time X, clseig, Z, scalinfo = nare(A',A,G,Q; scaling = 'K', nrm = Inf)
rezk = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezk < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with general scaling
@time X, clseig = arec(A,G,Q; scaling = 'G')
rezg = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezg < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with general scaling
@time X, clseig = nare(A',A,G,Q; scaling = 'G')
rezg = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezg < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# without scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'N')
rezn = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X*E,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X*E,E))))/norm(clseig)  < reltol)

# with block scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'B')
rezb = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X))
ev = eigvals(A-G*X*E,E)
@test rezb < reltol &&
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

@time X, clseig, Z, scalinfo = garec(BigFloat.(A),E,G,Q; scaling = 'B')
rezb = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X))
ev = schur(complex(A-G*X*E),complex(E)).values
@test rezb < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol &&
norm((scalinfo.Sx*(Z[5:8,:]/Z[1:4,:])*scalinfo.Sxi)/BigFloat.(E)-X)/norm(X) < reltol 

# with special scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'S');
rezs = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezs < reltol &&  
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

@time X, clseig = garec(A,E,G,Q; scaling = 'K');
rezk = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezk < reltol &&  
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

# with special scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'G')
rezg = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezg < reltol && 
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

# without scaling
@time X, clseig, F = arec(A, B, R, Q; scaling = 'N')
rezn = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test !( rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol)

# with scaling
@time X, clseig, F = arec(A,B,R,Q; scaling = 'B')
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(BigFloat.(A),B,R,Q; scaling = 'B')
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'B', orth = true)
rezb2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(BigFloat.(A),complex(B),R,Q; scaling = 'B', orth = true)
rezb2 = norm(A'*X+X*A-X*B*inv(BigFloat.(R))*B'*X+Q)/max(1,norm(X))
@test  rezb2 < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F, Z, scalinfo = arec(BigFloat.(A),B,R,Q; scaling = 'B', orth = true)
rezb2 = norm(A'*X+X*A-X*B*inv(BigFloat.(R))*B'*X+Q)/max(1,norm(X))
@test  rezb2 < 1.e-60*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm((scalinfo.Sx*(Z[5:8,:]/Z[1:4,:])*scalinfo.Sxi)-X)/norm(X) < reltol &&
norm(-(scalinfo.Sr*(Z[9:10,:]/Z[1:4,:])*scalinfo.Sxi)-F)/norm(F) < reltol 
 
@time X, clseig, F = arec(A,B,R,Q; scaling = 'S')
rezs1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'S', orth = true)
rezs2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', nrm = 1)
rezk1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezk1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', orth = true, nrm = 1)
rezk2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol


@time X, clseig, F = arec(A,B,R,Q; scaling = 'G')
rezg1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezg1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'G', orth = true)
rezg2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezg2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

# without scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'N')
rezn = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X))
@test !( rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol)

# with scaling  
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'B')
rezb  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezb < 10*reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'S')
rezs  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezs < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'K', nrm = 1)
rezk  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezs < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'K', pow2 = true)
rezk  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezk < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'G')
rezg  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezg < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'D')
rezd  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezd < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# with scaling
@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'T')
rezt  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
@test rezt < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# Example 2 from Petkov et al. 1998 (MATLAB scaling in icare fails to produce accurate result)
# k = 6
# A = BigFloat.(diagm([10^k; 2*10^k; 3*10^k])); Q = diagm([BigFloat(10.) ^(-k);1; 10^k])
# G = diagm([BigFloat(10.) ^(-k);BigFloat(10.) ^(-k); BigFloat(10.) ^(-k)]); B = BigFloat.(I(3)); R = BigFloat.(diagm([10^k; 10^k; 10^k]))

# x = [(A[i,i]+sqrt(A[i,i]^2+Q[i,i]*G[i,i]))/G[i,i] for i in 1:3]
# Xr = diagm(x)
# rez = norm(A'*Xr+Xr*A-Xr*G*Xr+Q)/max(1,norm(Xr))

reltol = sqrt(eps(1000.))
k = 6
A = diagm([10^k; 2*10^k; 3*10^k]); Q = diagm([10. ^(-k);1; 10^k])
G = diagm([10. ^(-k);10. ^(-k); 10. ^(-k)]); B = I(3); R = diagm([10^k; 10^k; 10^k])
E = [0.229087  0.569477  0.4308
0.187523  0.569036  0.342814
0.792381  0.690077  0.202204];

x = [(A[i,i]+sqrt(A[i,i]^2+Q[i,i]*G[i,i]))/G[i,i] for i in 1:3]
Xr = diagm(x)
rez = norm(A'*Xr+Xr*A-Xr*G*Xr+Q)/max(1,norm(Xr))

# without scaling
@time X, clseig = arec(A,G,Q; scaling = 'N')
rezn = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol)

# with scaling
@time X, clseig = arec(A,G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezb < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

@time X, clseig = arec(BigFloat.(A),G,Q; scaling = 'B')
rezb = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test  rezb < 1.e-60*reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

@time X, clseig = arec(A,G,Q; scaling = 'S')
rezs = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezs < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

@time X, clseig = arec(A,G,Q; scaling = 'K', nrm = 1)
rezk1 = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezk < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

@time X, clseig = arec(A,G,Q; scaling = 'G', pow2 = true)
rezg = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezg < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

@time X, clseig, Z, scalinfo = garec(A,E,G,Q; scaling = 'B')
rezb = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezb < 100*reltol &&
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'S')
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', nrm = 1)
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = garec(A, E, B, R, Q; scaling = 'N')
rezn  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X))

@time X, clseig, F, Z, scalinfo = garec(A, E, B, R, Q; scaling = 'K')
rezk  = norm(A'*X*E+E'*X*A-E'*X*B*inv(R)*B'*X*E+Q)/max(1,norm(X)) 
# @show rezk, rezn, norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig), norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)
@test rezk < 1.e-3*rezn 
@test norm(sort(real(clseig))-sort(real(eigvals(A-B*F,E))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F,E))))/norm(clseig)  < reltol

# E1 = diagm(diag(E)); A1 = E1*A; B1 = E1*B;
# @time X, clseig, F = garec(A1, E1, B1, R, Q; scaling = 'N')
# rezn  = norm(A1'*X*E1+E1'*X*A1-E1'*X*B1*inv(R)*B1'*X*E1+Q)/max(1,norm(X)) 
# @time X, clseig, F = garec(A1, E1, B1, R, Q; scaling = 'S')
# rezb  = norm(A1'*X*E1+E1'*X*A1-E1'*X*B1*inv(R)*B1'*X*E1+Q)/max(1,norm(X)) 
# @test rezb < 1.e-4*rezn &&
# norm(sort(real(clseig))-sort(real(eigvals(A1-B1*F,E1))))/norm(clseig)  < reltol &&
# norm(sort(imag(clseig))-sort(imag(eigvals(A1-B1*F,E1))))/norm(clseig)  < reltol

end
@testset "Scaling discrete-time Riccati equation" begin

# discrete-time
# 
n = 4
A = [ 0.0        0.0        -0.00135213   0.0
      0.0        0.0         0.0         -0.000755718
      0.0531806  0.0         0.0          0.0378156
      0.0        0.0531806  -0.0709927    0.0];
B = [ 1.0  0.0
 0.0  1.0
 0.0  0.0
 0.0  0.0]; 
Q = [ 0.0  0.0  0.0        0.0
 0.0  0.0  0.0        0.0
 0.0  0.0  1.17474e9  0.0
 0.0  0.0  0.0        4.14026e9]; 
R = [100.0    0.0
   0.0  100.0];
G = B*inv(R)*B'; G = (G+G)/2   
E = [0.3145695364503345 0.28299421375349765 0.7751430938038222 0.3817600131380937; 
0.6175205621578082 0.9791019859058574 0.6388662440424374 0.36327849747268603; 
0.13292183504367217 0.9319918486921431 0.15347895705946646 0.1378470943397626; 
0.1411783703336893 0.6496471027507487 0.7764461576953698 0.2687776918944005];
S = zeros(4,2); 
reltol = sqrt(eps(1000.))

M = abs.([zeros(4,4) A B; -A' -Q -S; B' S' R]) + 
    abs.([zeros(4,4)  E zeros(4,2) ; E' zeros(4,6); zeros(2,10) ])


@time X, clseig, F = ared(A,B,R,Q; scaling = 'N')
rezn = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol)

@time X, clseig, F = ared(A,B,R,Q; scaling = 'B')
rezb = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezb < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'S')
rezs = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezs < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'K', nrm = 1)
rezk = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezk < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'D')
rezd = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezd < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'T')
rezt = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezt < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'G')
rezg = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezg < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

# Example 1  Gudmundsson et al. 1992
k = 6; 
V = I-2*ones(3)*ones(3)'/3; A0 = diagm([0; 1; 3])
A = V*A0*V
B = I(3); R = 10. ^k*I(3)
Q = 10. ^6*I(3); Xr = V*(10. ^k*diagm([1;(1+sqrt(5))/2; (9+sqrt(85))/2]))*V
@time X, clseig, F = ared(A,B,R,Q; scaling = 'N')
rezn = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test !(rezn < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol && 
norm(X-Xr) < reltol)

@time X, clseig, F = ared(A,B,R,Q; scaling = 'B')
rezb = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezb < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'S')
rezs = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezs < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'K', nrm = 1)
rezk = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezk < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'D')
rezd = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezd < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'R')
rezd = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezd < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol

@time X, clseig, F = ared(A,B,R,Q; scaling = 'T')
rezt = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezt < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol


@time X, clseig, F = ared(A,B,R,Q; scaling = 'G')
rezg = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
@test rezb < reltol && norm(X-Xr)/norm(X) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))  < reltol


# Example 2  Gudmundsson et al. 1992 (modified)
A = [0 100; 0 0]; B = [0;1;;]; R = [1;;]; Q = 100*I(2); Xr = 100*[1 0; 0 10001]
@time X, clseig, F = ared(A,B,R,Q; scaling = 'N')
rezn = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xn = norm(X-Xr)/norm(X)

@time X, clseig, F = ared(A,B,R,Q; scaling = 'B')
rezb = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xb = norm(X-Xr)/norm(X)
@test rezb < rezn && xb < xn

@time X, clseig, F = ared(BigFloat.(A),B,R,Q; scaling = 'B')
rezb = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xb = norm(X-Xr)/norm(X)
@test rezb < rezn && xb < xn


@time X, clseig, F = ared(A,B,R,Q; scaling = 'S')
rezs = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xs = norm(X-Xr)/norm(X)
@test rezs < rezn && xs < xn

@time X, clseig, F = ared(A,B,R,Q; scaling = 'K')
rezk = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xk = norm(X-Xr)/norm(X)
@test rezk < rezn && xk < xn

@time X, clseig, F = ared(A,B,R,Q; scaling = 'D')
rezd = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xd = norm(X-Xr)/norm(X)
@test rezd < rezn && xd < xn

@time X, clseig, F = ared(A,B,R,Q; scaling = 'R')
rezd = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xd = norm(X-Xr)/norm(X)
@test rezd < rezn && xd < xn

@time X, clseig, F = ared(A,B,R,Q; scaling = 'T')
rezt = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xt = norm(X-Xr)/norm(X)
@test rezt < rezn && xt < xn

@time X, clseig, F = ared(A,B,R,Q; scaling = 'G')
rezg = norm(A'*X*A-X-A'*X*B*inv(R+B'*X*B)*B'*X*A+Q)/max(1,norm(X))
xg = norm(X-Xr)/norm(X)
@test rezg < rezn && xg < xn

end


end 

end
