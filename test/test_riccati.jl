module Test_riccati

using Random
using LinearAlgebra
using MatrixEquations
using Test
using GenericSchur
using GenericLinearAlgebra
using Polynomials
using Combinatorics

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

@testset "Testing algebraic Riccati/quadratic equation solvers" begin


@testset "Unilateral quadratic matrix equation" begin


# 2.3.1 Example 1: Choudhry, Serdica Math. Journal, 2019 
A = [ 1 1; 5 2 ]; B = [2 1;-1 2]; C = [7 -6; 2 -12];
@time x, ev, _, _, indeig, heigs = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 &&
      heigs == uqme(A,B,C; onlyeigvals = true) 
      
@time x1, ev1, _, _, indeig1 = uqme(A,B,C; as=false, disc=true, rev = true);      
@test norm(A*x1*x1+B*x1+C) < 1e-10 && x ≈ x1 &&
      norm(sort(real(ev1))-sort(real(eigvals(x1)))) < 1e-10 &&
      norm(sort(imag(ev1))-sort(imag(eigvals(x1)))) < 1e-10   


# determine all real solutions
sols = Vector{Matrix{Float64}}(undef, 0)     
inds = Vector{Vector{Int}}(undef, 0)  
norms = Vector{Float64}(undef, 0)  
vind = collect(combinations(vec(1:4),2))
for k in 1:length(vind)
    try x, ev, _, _, indeig = uqme(A,B,C; indeig = vind[k], rtol = 1.e-10);
        push!(sols,x)  
        push!(inds,indeig) 
        push!(norms,norm(A*x*x+B*x+C))
    catch
        continue
    end
end 
@test norm(norms,Inf) < 1.e-10

x, ev, _, _, indeig = uqme(I,A\B,A\C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10      

# determine all real solutions
sols = Vector{Matrix{Float64}}(undef, 0)     
inds = Vector{Vector{Int}}(undef, 0)  
norms = Vector{Float64}(undef, 0)  
vind = collect(combinations(vec(1:4),2))
for k in 1:length(vind)
    try x, ev, _, _, indeig = uqme(I,A\B,A\C; indeig = vind[k], rtol = 1.e-10);
        push!(sols,x)  
        push!(inds,indeig) 
        push!(norms,norm(A*x*x+B*x+C))
    catch
        continue
    end
end 
@test norm(norms,Inf) < 1.e-10



# 3.2.1 Example 1: Choudhry, Serdica Math. Journal, 2019 
A = [ 1 2 3; -1 0 1; 1 1 1]; B = [2 0 1;3 1 0; 7 2 1]; 
C = [7 7 -37;3 2 -7; -10 -7 -24];
@time x, ev, _, _, indeig = uqme(A,B,C; as=false, disc=false, scaling = 'N');
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

@time x1, ev1 = uqme(A,B,C; as=false, disc=false, indeig, scaling = 'B');
@test x ≈ x1 && ev ≈ ev1

@time x, ev, _, _, indeig = uqme(A,B,C; as=true, disc=false, scaling = 'K');
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

@time x, ev = uqme(A,B,C; as=false, disc=true, scaling = 'G');
@test norm(A*x*x+B*x+C) < 1e-8 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 

@time x, ev = uqme(A,B,C; as=true, disc=true, scaling = 'S');
@test norm(A*x*x+B*x+C) < 1e-8 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 


# determine all real solutions using random index permutations
sols = Vector{Matrix{Float64}}(undef, 0)     
inds = Vector{Vector{Int}}(undef, 0)  
norms = Vector{Float64}(undef, 0)  
for k in 1:100
    try x, ev, _, _, indeig = uqme(A,B,C; indeig = randperm(6));
        push!(sols,x)  
        push!(inds,indeig) 
        push!(norms,norm(A*x*x+B*x+C))
   catch
        continue
    end
end 
@test norm(norms,Inf) < 1.e-10

# determine all complex solutions
solsc = Vector{Matrix{Complex{Float64}}}(undef, 0)     
indsc = Vector{Vector{Int}}(undef, 0)  
normsc = Vector{Float64}(undef, 0)  

for k in 1:20
    try x, ev, _, _, indeig = uqme(Complex{Float64}.(A),B,C; indeig = randperm(6));
        push!(solsc,x)  
        push!(indsc,indeig) 
        push!(normsc,norm(A*x*x+B*x+C))
    catch
        continue
    end
end 
@test norm(normsc,Inf) < 1.e-10

# 3.2.2 Example 2: Choudhry, Serdica Math. Journal, 2019 
A = [ 1 2 3; -1 0 2; 1 1 1]; B = [2 0 1;3 1 0; 1 2 1]; 
C = [7 7 -37;9 7 -15; -4 -1 -18];
@time x, ev, _, _, indeig = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

# determine all real solutions
sols = Vector{Matrix{Float64}}(undef, 0)     
inds = Vector{Vector{Int}}(undef, 0)  
norms = Vector{Float64}(undef, 0)  
vind = collect(combinations(vec(1:6),3))
for k in 1:length(vind)
    try x, ev, _, _, indeig = uqme(A,B,C; indeig = vind[k], rtol = 1.e-10);
        push!(sols,x)  
        push!(inds,indeig) 
        push!(norms,norm(A*x*x+B*x+C))
    catch
        continue
    end
end 
@test norm(norms,Inf) < 1.e-10

# determine all complex solutions
solsc = Vector{Matrix{Complex{Float64}}}(undef, 0)     
indsc = Vector{Vector{Int}}(undef, 0)  
normsc = Vector{Float64}(undef, 0)  
vind = collect(combinations(vec(1:6),3))

vind = collect(combinations(vec(1:6),3))
for k in 1:length(vind)
    try x, ev, _, _, indeig = uqme(Complex{Float64}.(A),B,C; indeig = vind[k], rtol = 1.e-10);
        push!(solsc,x)  
        push!(indsc,indeig) 
        push!(normsc,norm(A*x*x+B*x+C))
    catch
        continue
    end
end 
@test norm(normsc,Inf) < 1.e-10


# Example Higham & Kim (SIMAX. 2001)
# no real solution exists for this example

A = Complex{Float64}.([17.6 1.28 2.89
1.28 0.824 0.413
2.89 0.413 0.725]);
B = Complex{Float64}.([7.66 2.45 2.1
0.23 1.04 0.223
0.6 0.756 0.658]);
C = Complex{Float64}.([
121 18.9 15.9
0 2.7 0.145
11.9 3.64 15.5]);
@test_throws "complex Jordan block" x, ev = uqme(real(A),real(B),real(C));

@time x, ev = uqme(A,B,C);
@test norm(A*x*x+B*x+C) < 1e-8 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 


# Example: quasi-birth–death process to solve R = β(A0+RA1+R^2A2): Higham & Kim (Math. Comp. 2000)
Q(g,x) = [g*x[1] (1-g)*x[1] 0 0 0;
          g*x[2] 0 (1-g)*x[2] 0 0;
          g*x[3] 0 0 (1-g)*x[3] 0;
          g*x[4] 0 0 0 (1-g)*x[4];
          g*x[5] 0 0 0 (1-g)*x[5]];
g = 0.2; b = [1, 0.4, 0.25, 0.1, 0.]; d = [0, 0.5, 0.55, 0.8, 1]; β = 0.5;
A0 = Q(g,b); A1 = Q(g,ones(5)-b-d); A2 = Q(g,d);
A = β*A2'; B = β*A1'-I; C = β*A0';
@time x, ev = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 


# Erfanifar & Hajarian Example 3, IET CTA, 2024

n = 100;
A = diagm(0=>1.5*ones(n),1=>7.5*ones(n-1),-1=>7.5*ones(n-1))+0.5*im*ones(n,n);
B = diagm(0=>30*ones(n),1=>-5.5*ones(n-1),
-1=>-(2.5*im+4.5)*ones(n-1));B[1,1] =21; B[n,n] = 21; B += 0.2*im*randn(n,n);
C = diagm(0=>2.5*im*ones(n),1=>1.5*im*ones(n-1),-1=>1.5*im*ones(n-1))-0.2*ones(n,n);
@time x, ev = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-8 && # norm(A*x*x+B*x+C) > 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 
# perform a Newton step to increase accuracy
E = gsylv(A,x,A*x+B,I(n),(A*x+B)*x+C)
x2 = x-E;
@test norm(A*x2*x2+B*x2+C) < norm(A*x*x+B*x+C)


# Erfanifar & Hajarian Example 2, IET CTA, 2024

n = 100;
A = diagm(0=>(-1.5*im+1)*ones(n),1=>-2*ones(n-1),-1=>-ones(n-1))+0.5*im*ones(n,n);
B = diagm(0=>-25*ones(n),1=>-5*ones(n-1),2=>(-im-0.5)*ones(n-2),
-1=>-6*ones(n-1),-2=>(1-2*im)*ones(n-2));B[1,1] = 4; B[n,n] = 4;
C = diagm(0=>-im*ones(n),1=>-0.5*ones(n-1),-1=>-1.5*ones(n-1))-0.2*ones(n,n);
@time x, ev = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

n = 10;
A = diagm(0=>(-1.5*im+1)*ones(n),1=>-2*ones(n-1),-1=>-ones(n-1))+0.5*im*ones(n,n);
B = diagm(0=>-25*ones(n),1=>-5*ones(n-1),2=>(-im-0.5)*ones(n-2),
-1=>-6*ones(n-1),-2=>(1-2*im)*ones(n-2));B[1,1] = 4; B[n,n] = 4;
C = diagm(0=>-im*ones(n),1=>-0.5*ones(n-1),-1=>-1.5*ones(n-1))-0.2*ones(n,n);
@time x, ev = uqme(Complex{BigFloat}.(A),Complex{BigFloat}.(B),Complex{BigFloat}.(C); as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-70 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-70 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-70 



# Erfanifar & Hajarian Example 1, IET CTA, 2024
n = 100;
A = diagm(0=>-1.5*ones(n),1=>-6*ones(n-1),-1=>-7*ones(n-1))-0.2*ones(n,n);
B = diagm(0=>45*ones(n),1=>-5*ones(n-1),-1=>-7*ones(n-1));B[1,1] = 15; B[n,n] = 15;
C = diagm(0=>-ones(n),1=>-0.4*ones(n-1),-1=>-0.5*ones(n-1))-0.2*ones(n,n);
@time x, ev = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-8 && norm(A*x*x+B*x+C) > 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 

# Newton step with exact line search
tmp = A*x+B
Qx = tmp*x+C
E = gsylv(A,x,tmp,I(n),Qx)
α = norm(Qx,2)^2
tmp = A*E*E
γ = norm(tmp,2)^2
β = tr(Qx'*tmp*tmp'*Qx)
p = Polynomial([α, -2α, α+β, -β, γ ])
pd = Polynomial([-2α,2(α+β),-3β,4γ])
r = roots(pd); 
tmin = real(r[findall(imag.(r) .== 0)])
x2 = x-tmin[1]*E;
@test norm(A*x2*x2+B*x2+C) < norm(A*x*x+B*x+C)


# p = 5;
# F1 = diagm(0=> 2*ones(p), -1 => ones(p-1), 1 => ones(p-1)); F1[1,1]=3; F1[p,p]=3; rmul!(F1,1/8)
# F2 = diagm(0=> 3*ones(p), -1 => ones(p-1), 1 => ones(p-1)); F2[1,1]=4; F2[p,p]=4; rmul!(F2,1/10)
# E0 = [zeros(p,p) F1; F2 zeros(p,p)];
# E1 = zeros(2p,2p);
# E2 = [zeros(p,p) F2; F1 zeros(p,p)];
# A = -E2; B = I-E1; C = -E0;
# x, ev = uqme(A,B,C; as=false, disc=true,stab = false);
# @test norm(A*x*x+B*x+C) < 1e-10 &&
#       norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
#       norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 

# Example 1: Li & Meini, arxiv:2026

E0 = [0 0 0 1/4;
33/160 0 0 0;
1/4 0 0 0;
0 1/4 0 0];
E1 = [0 0 0 0;
0 0 3/4 0;
0 3/4 0 0;
0 0 0 0];
E2 = [0 3/4 0 0;
0 0 0 7/160;
0 0 0 0;
3/4 0 0 0];
A = -E2; B = I-E1; C = -E0;
@time x, ev = uqme(A,B,C; as=false, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-6 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-6 


# Example 1: Tisseur, LAA 2000
A = [0.81 -0.38; -0.41 0.19];
B = [-0.24 0.97; -0.021 0.086];
C = [-0.11 -0.057; -0.88 -0.46]; 
@time x, ev = uqme(A,B,C; as=true, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

# unilateral quadratic matrix equation (UQME): Kressner et al. 2019
n = 50; 
A = I(n); 
B = diagm(0 => 30*ones(n),-1=>-10*ones(n-1),1=>-10*ones(n-1) ); B[1,1] = 20; B[n,n] = 20. ; 
C = diagm(0 => 15*ones(n),-1=>-5*ones(n-1),1=>-5*ones(n-1) );
@time x, ev = uqme(A,B,C; as=true, disc=true);
@test norm(A*x*x+B*x+C) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

end

@testset "Nonsymmetric/nonhermitian Riccati equation" begin

# only double precision tests are performed

# unilateral quadratic matrix equation (UQME): Kressner et al. 2019
n = 50; 
A1 = diagm(0 => 30*ones(n),-1=>-10*ones(n-1),1=>-10*ones(n-1) ); A1[1,1] = 20; A1[n,n] = 20. ; 
A2 = zeros(n,n); G = -Matrix{Float64}(I,n,n); 
Q = diagm(0 => 15*ones(n),-1=>-5*ones(n-1),1=>-5*ones(n-1) );
@time x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*x+A1*x+Q) < 1e-10 &&
      norm(sort(real(ev))-sort(real(eigvals(x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(x)))) < 1e-10 

# left and right stabilization example with Q = 0
A1 = [-2 0 0;0 1 0;0 0 4]; A2 = [5 0; 0 1]; G = [5 1 0; 1 2 1]; Q = zeros(3,2);
K = [0 0; 6 0; -30 5];
@time x, ev = nare(A1,A2,G,Q; as=false);
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
@time x, ev = nare(A1,A2,G,Q; as=false);
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
@time x, ev = nare(A1,A2,G,Q; as=false);
@test norm(x*C*x-x*D-A*x+B) < 1e-10 
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# # this example fails on Linux machines
# # doubly stochastic solution: Example 6.2 of Guo (SIMAX 2001)
# m = 100; 
# B = C = Matrix{Float64}(I,m,m);
# A = D = diagm(1 => -ones(m-1), 0 => 2*ones(m),-m+1 => [-1]);
# A1 = -A; A2 = -D; G = -C; Q = B;
# x, ev = nare(A1,A2,G,Q; as=false);
# @test norm(x*C*x-x*D-A*x+B) < 1e-10 
# @test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 && 
#       norm(x*ones(m)-ones(m),Inf) < 1e-7 && norm(x'*ones(m)-ones(m),Inf) < 1e-7 &&
#       norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
#       norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# positive solution:  
A1 = A2 = G = -Matrix{Float64}(I,2,2)
Q = [0 1;1 0]
@time x, ev = nare(A1,A2,G,Q; as=false);
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
@time x, ev = nare(A1,A2,G,Q; as=false);

@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && minimum(x) > 0.0 &&
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10

# positive solution: Example 5.1 SIMAX Guo & Laub (SIMAX 2000)
α = 4.27; α = 6.;
A1 = -[α -2;-1 α]; 
A2 = -[5 -1; -1 4];
G = -[3 4;2 1];
Q = [1 1;2 1]
@time x, ev = nare(A1,A2,G,Q; as=false);

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
@time x, ev = nare(A1,A2,G,Q; as=false);
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
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10 &&
      heigs == nare(A1,A2,G,Q; onlyeigvals = true)
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1

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
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1

@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=true, disc = false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1

@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=false, disc = true);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1

@test_throws "suitable" x, ev = nare(A1,A2,G,Q; as=true, disc = true, stab = true)
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=false, disc = true, stab = false);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1


A2 = -M[i1,i1]; G = -M[i1,i2]; Q = -M[i2,i1]; A1 = -M[i2,i2];      
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=true);
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1

A2 = 2M[i1,i1]; G = 2M[i1,i2]; Q = 2M[i2,i1]; A1 = 2M[i2,i2]; 
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=true,disc = true);  
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10   
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1


A2 = .2M[i1,i1]; G = .2M[i1,i2]; Q = .2M[i2,i1]; A1 = .2M[i2,i2]; 
@time x, ev, z, scalinfo, ind, heigs = nare(A1,A2,G,Q; as=false,disc = true);  
@test norm(-x*G*x+x*A2+A1*x+Q) < 1e-10 && 
      norm(sort(real(ev))-sort(real(eigvals(A2-G*x)))) < 1e-10 &&
      norm(sort(imag(ev))-sort(imag(eigvals(A2-G*x)))) < 1e-10   
@time x1, ev1, z1, scalinfo1, ind1, heigs1 = nare(A1,A2,G,Q; indeig = ind);
@test x ≈ x1 && ev ≈ ev1 && ind == ind1


# determine all real solutions
sols = Vector{Matrix{Float64}}(undef, 0)     
evs = Vector{Vector{ComplexF64}}(undef, 0)     
inds = Vector{Vector{Int}}(undef, 0)  
norms = Vector{Float64}(undef, 0)  
vind = collect(combinations(vec(1:nm),5))
for k in 1:length(vind)
    try x, ev, _, _, indeig = nare(A1,A2,G,Q; indeig = vind[k]);
        push!(sols,x)  
        push!(evs,ev)  
        push!(inds,indeig) 
        push!(norms,norm(-x*G*x+x*A2+A1*x+Q))
    catch
        continue
    end
end 
@test norm(norms,Inf) < 1.e-10      
end   

@testset "Continuous nonsymmetric/nonhermitian control Riccati equation" begin
      
(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, rrp) = ([0.5186557138979574 -0.7953357716286895 -0.13965163099627015; 1.0665449746409734 0.6088478332750862 -0.4354477612047091; 0.1346237794131495 -0.275096710601077 -0.8266369768501854], 
[0.439592295754915 1.6095385548830956 1.7450266867150095 0.9515226624325183 -0.45299430608369196; -1.1894770349215864 -1.833127306090445 -1.1647538251650997 -1.5539239104835676 0.499385904304347; -0.6942200110837333 0.34956600124480836 0.8296168635771622 0.3587636372051262 -0.41622240959483986; 0.265105372330221 0.22867388814844122 -1.6329916213697853 -1.4226690615687332 1.5368687390076006; -1.1757417631847757 0.03885753635651309 -0.446824369041719 0.5801034932261181 -0.03471186320555784], 
[0.5186348112743987 0.7474991470255862; 0.955498078107626 0.6830083502644749; 0.6611924906668183 0.6823004349691932], 
[0.12014651554785749 0.4453129602766438; 0.0004102814170844349 0.6839214359424151; 0.38047361736968777 0.551220186101371; 0.34537796132599685 0.5618344798799988; 0.42173503122390266 0.941521683484576], 
[0.05371792299042739 0.04619284070374271; 0.04619284070374271 0.6730150626829626], 
[0.21283054369611235 0.6265483296015649 0.4348130180922076; 0.30156829151199144 0.909711693125952 0.6116994206160582; 0.10360378494080247 0.28603624909442293 0.21547127081093073; 0.1737405801997451 0.5564421145975516 0.3459193005135064; 0.023361389826095407 0.09644201409099488 0.04216966977854824], 
[0.006107781332820873 0.009531391224963772; 0.0013325850570732102 0.0016990247858977737; 0.0022257836809778686 0.0004310378757048017; 0.00861942877183638 0.008869774550033002; 0.009363944250960697 0.008906153251790238], 
[0.005823564901536618 0.008126468121332591; 0.008481271880855925 0.0064250239975366785; 0.00015305440259740676 0.0052437417322659784], 
[1.048489154541614 0.3863247698690497; 0.386714282642807 0.16481751126973057])

(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2) = (ComplexF64[2.383423333382828 - 0.9674424607605681im 3.326060799683068 - 0.9149234177236291im -2.2020834879217923 - 1.796905889972681im; 1.5669645129804948 + 0.15568365597535902im 0.30103810642251533 + 1.5006175212828867im -0.4669229122037008 + 0.44028830564991994im; 2.503500531710142 + 1.5624996710640793im -1.6476850903029767 - 0.011300987559109209im -0.4386047567827293 + 0.7873291717592211im], 
ComplexF64[0.2829800186973279 + 0.7367545930222964im 1.5989316624240866 + 1.9415572790848212im 1.0104076981739254 - 0.28764414779827346im 0.11855275451671819 + 2.026539014120887im 2.9689972410730023 + 0.9379603278731301im; -1.690520890045875 - 0.8025144602588352im 0.3013906635821779 + 0.3972278938129992im -0.512046793400369 + 0.48720842067798265im -0.17702580734021373 + 0.6799960265046128im -0.531894975264036 - 0.55420215052255im; 1.5303446126266058 - 0.8951488094007409im -1.1831906823858331 - 1.0619494449102542im -0.31045991722111693 + 1.5459206557413376im -1.464955360447779 + 1.428340663245959im 0.4030345293693571 - 2.34031173488628im; -0.806885517681216 + 0.34010544445973184im 0.18854455300814824 - 0.3896419308924785im -0.16000137239613743 + 0.06672198340340292im -0.9792721360698929 - 1.1968951746042602im -2.7303586198247105 + 0.05366257814266763im; 0.29695377164963943 - 0.38582588553799685im 0.7398085557859827 + 0.6210268152688686im 0.5068363257001227 - 0.774272786241257im -0.41894207969039826 + 1.7546479595665716im -1.444949478392103 + 0.0819300049346403im], 
ComplexF64[0.9705270940196536 + 0.4189645598411159im 0.8457081248196681 + 0.37019813835650206im; 0.5861333986497245 + 0.4381536612745125im 0.7612896774715483 + 0.02634058705386122im; 0.016274524667106882 + 0.10847764068155064im 0.3376523372505591 + 0.7646916254690034im], 
ComplexF64[0.5194052188778323 + 0.23217463974012997im 0.576877325247173 + 0.5837239738093883im; 0.7151868373867054 + 0.03257014878610254im 0.6382429640403555 + 0.875550675260761im; 0.8190898623736202 + 0.4606037478714513im 0.48965008396243326 + 0.7119278940149301im; 0.812272783766871 + 0.9762296179476261im 0.6627341281894685 + 0.5185923055842024im; 0.30694215017808424 + 0.8143951179659183im 0.9998730565349984 + 0.6839551153903555im], 
ComplexF64[1.067011999183137 + 0.0im 0.4654143746957156 - 0.15709990966069343im; 0.4654143746957156 + 0.15709990966069343im 0.2574259293276042 + 0.0im], 
ComplexF64[0.5285396104970805 - 0.5032003533455865im 0.6737574122541499 - 0.0833540050037482im 1.021028481939776 - 0.5750579302228991im; 0.9954390792321932 - 0.450846759370976im 1.233227012862817 - 0.1369628298212674im 0.955789934055562 - 0.7912944081830555im; 0.9972523744470276 - 0.7424652403004819im 1.2108421213944736 - 0.406356995188239im 0.9773422138810146 - 0.8441031303360385im; 0.26869455187291363 - 0.17658265423737954im 0.3795926892049824 + 0.07742915095024652im 0.6103795530331712 - 0.3472220611267417im; 1.0404163476474775 + 0.6171422448568725im 0.8525153602852895 + 1.046108653424779im 1.3589918604323559 + 0.47932955407640854im], 
ComplexF64[0.002138798760413059 + 0.008713558802501947im 0.009178855263990003 + 0.007945170056984778im; 0.008271914914788877 + 0.004056861392440701im 0.0026564334964060375 + 0.0026819665645740898im; 0.009756301878539457 + 0.003388327294422431im 0.005049206080778017 + 0.0002363744942777579im; 0.006339603479513492 + 0.004224450339686472im 0.0087226962533692 + 0.005722653129533982im; 0.003627926982331645 + 0.004433845727082843im 0.005755074243651742 + 0.006022553884681203im], 
ComplexF64[0.004793993044016751 + 0.0007534632991737233im 0.0012314541782976751 + 0.006993569174823701im; 0.009603039873728335 + 0.004410253357550065im 0.0016015623713054072 + 0.007811037614929322im; 0.006827605372054339 + 0.007366175941323585im 0.007752574406482476 + 0.0036213139684334794im])

rtol = 20*sqrt(eps(1.)) 


@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6 &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6 &&
heigs == narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; onlyeigvals = true)

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, indeig = ind)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1


@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rrp, Qr, sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rrp)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6 &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6

@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2)
@test norm(ac2'*x+x*ac1-(x*bc1+sc1)*inv(rc)*(bc2'*x+sc2')+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2, indeig = ind)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1

@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2, orth = true)
@test norm(ac2'*x+x*ac1-(x*bc1+sc1)*inv(rc)*(bc2'*x+sc2')+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2, indeig = ind, orth = true)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1

@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr)
@test norm(ar2'*x+x*ar1-(x*br1)*inv(rr)*(br2'*x)+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
heigs == narec(ar1, ar2, br1, br2, rr, Qr; onlyeigvals = true)


@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr, sr1)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x)+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
heigs == narec(ar1, ar2, br1, br2, rr, Qr, sr1; onlyeigvals = true)


@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
heigs == narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, sr2; onlyeigvals = true)

@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'N',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
heigs == narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; onlyeigvals = true, scaling = 'N',orth = true)

nrez = norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))
@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'B',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'S',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'K',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'D',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, scaling = 'T',orth = true)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol





Ty = Float64

for (Ty,n,l, m, p) in ((Float64, 20, 20, 10, 10),)

#n = 3; l = 5; m = 2; p = m;
# n = 4; l = 4; m = 2; p = m;
# (Ty,n,l, m, p) = (Float64, 3, 5, 2, 2)
ar1 = randn(Ty,n,n)
ar2 = randn(Ty,l,l)
if m == 1
    br1 = rand(Ty,n)
    bc1 = rand(Ty,n)+im*rand(Ty,n)
    sc1 = rand(Complex{Ty},l)/100
    sr1 = rand(Ty,l)/100
    br2 = rand(Ty,l)
    bc2 = rand(Ty,l)+im*rand(Ty,l)
    sc2 = rand(Complex{Ty},n)/100
    sr2 = rand(Ty,n)/100
else
    br1 = rand(Ty,n,m)
    bc1 = rand(Ty,n,m)+im*rand(Ty,n,m)
    sc1 = rand(Complex{Ty},l,m)/100
    sr1 = rand(Ty,l,m)/100
    br2 = rand(Ty,l,m)
    bc2 = rand(Ty,l,m)+im*rand(Ty,l,m)
    sc2 = rand(Complex{Ty},n,m)/100
    sr2 = rand(Ty,n,m)/100
end
cr1 = rand(Ty,p,n)
cr2 = rand(Ty,p,l)
ac1 = randn(Ty,n,n) + im*randn(Ty,n,n)
ac2 = randn(Ty,l,l) + im*randn(Ty,l,l)
cc1 = rand(Ty,p,n)+im*rand(Ty,p,n)
cc2 = rand(Ty,p,l)+im*rand(Ty,p,l)
bc1 = rand(Ty,n,m)+im*rand(Ty,n,m)
bc2 = rand(Ty,l,m)+im*rand(Ty,l,m)
rr1 = rand(Ty,m,m)
rc1 = rand(Ty,m,m)+im*rand(Ty,m,m)
#Ty == Float64 ? rtol = n*n*sqrt(eps(1.)) : rtol = n*sqrt(eps(1.f0))
rtol = n*n*sqrt(eps(1.)) 

qc = cc2'*cc1
Qr = cr2'*cr1
gc = bc1*bc2'
gr = br1*br2'
rr = rr1*rr1'
rc = rc1*rc1'
rrp = rr+0.001*rand(m,m)

for orth in (false,true)
@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; orth)
catch
      narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; orth,as = true)
end
#@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6 &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; orth, indeig = ind)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1


@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ar1, ar2, br1, br2, rrp, Qr, sr1, sr2; orth)
catch
      narec(ar1, ar2, br1, br2, rrp, Qr, sr1, sr2; orth,as = true)
end
#@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rrp, Qr, sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rrp)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6 &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < 1.e-6


@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth)
catch
      narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth,as = true)
end
#@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2)
@test norm(ac2'*x+x*ac1-(x*bc1+sc1)*inv(rc)*(bc2'*x+sc2')+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth, indeig = ind)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1

@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth)
catch
      narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth, as = true)
end
#@time x, clseig, f, Z, scalinfo, ind, heigs = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2, orth = true)
@test norm(ac2'*x+x*ac1-(x*bc1+sc1)*inv(rc)*(bc2'*x+sc2')+qc)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac1-bc1*f))))/norm(clseig)  < rtol

@time x1, clseig1, f1, Z1, scalinfo1, ind1 = narec(ac1, ac2, bc1, bc2, rc, qc, sc1, sc2; orth, indeig = ind)
@test x ≈ x1 && clseig ≈ clseig1 && f ≈ f1 && ind == ind1

@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, 0*sr2; orth)
catch
      narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, 0*sr2; orth, as = true)
end
#@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, 0*sr2)
@test norm(ar2'*x+x*ar1-(x*br1+0*sr1)*inv(rr)*(br2'*x+0*sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ar1, ar2, br1, br2, rr, Qr, sr1, 0*sr2; orth)
catch
      narec(ar1, ar2, br1, br2, rr, Qr, sr1, 0*sr2; orth, as = true)
end
#@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, 0*sr2)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+0*sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo, ind, heigs = try 
      narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, sr2; orth)
catch
      narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, sr2; orth, as = true)
end
#@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, 0*sr1, sr2)
@test norm(ar2'*x+x*ar1-(x*br1+0*sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

end
end

for orth in (false,true)
(ar1, ar2, br1, br2, rr, Qr, sr1, sr2, rrp) = ([-0.980741321764109 -1.2313046167830222 -1.068417263874382 1.2769011627689626; -1.6718806837728895 -0.9693885786804483 0.8529720673935202 -0.6788720034899908; 0.5310891981616882 0.4587061390616016 0.5710682617358275 1.042709210759531; -0.6937836841084014 -0.5642664928884923 1.0554691106639253 1.067674880479393], 
[0.1919975794008983 0.32893417457914487 -0.3145536289623541 -1.292835065144256; 0.5796947512250342 -1.179979322317249 -1.142537147233044 -0.7882368183984778; -0.7029462016070656 -0.577193245566792 -0.8251918573848881 1.7146741700414359; 0.3344795192136782 0.7201973184512374 1.316726434001781 -0.5245709240573696], 
[0.23933110519932566, 0.5980797881262297, 0.6631396247563346, 0.7770411949284358], 
[0.08640351329091711, 0.9352474757316545, 0.29277448927966154, 0.20788086841045517], 
[0.1767555361279444;;], 
[1.169943003395885 0.7417490171460454 1.344185026851104 0.983097131890609; 1.1379979003425496 0.9859110922509912 1.6639020838707916 1.074521641099519; 0.8349144460968532 0.5556265023525219 0.9476198851875268 0.7242346857871664; 1.4353878733177885 0.8652895721672691 1.5353271335464302 1.1985257716600806], 
[0.007424767892969212, 0.005481936745675446, 0.004394467626371586, 0.0029778314949398476], 
[0.002128757234328724, 0.0076884248327638936, 0.004879408487667169, 0.008133291655792008], 
[0.17743162396987722;;])

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'N',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

nrez = norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))
@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'B',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'S',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'K',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'D',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

@time x, clseig, f, Z, scalinfo = narec(ar1, ar2, br1, br2, rr, Qr, sr1, sr2; scaling = 'T',orth)
@test norm(ar2'*x+x*ar1-(x*br1+sr1)*inv(rr)*(br2'*x+sr2')+Qr)/max(1,norm(x))  <= rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar1-br1*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar1-br1*f))))/norm(clseig)  < rtol

end


A = [1 0;0 5/22]; B11 = [1 0;0 1]; B12 = [1;0;;]; 
Q1 = [1 0;0 0]; Q2 = [1 1;1 2]; R11=[2 -1;-1 1]; R22 = [1;;];
# Nash-game
A1 = A; A2 = [A zeros(2,2); zeros(2,2) A]; B1 = [B11 B12]; B2 = [B11 zeros(2,1); zeros(2,2) B12];
Q = [Q1;Q2]; R = [R11 zeros(2,1); zeros(1,2) R22]; S1 = zeros(4,3); S2 = zeros(2,3);
x, clseig, F, Z, scalinfo = narec(A1, A2, B1, B2, R, Q, S1, S2) 
@test norm(A2'*x+x*A1-(x*B1+S1)*inv(R)*(B2'*x+S2')+Q)/max(1,norm(x)) < 1.e-10 &&
norm(sort(real(clseig))-sort(real(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10 &&
norm(sort(imag(clseig))-sort(imag(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10

# Nash-game example: Jank & Kremer, 2004
A = [-3 0 0 3.5; 0 -7 0 0; 0 .3 -7 1.3; 5 0 1 -16];
B11 = [.5;0;0;0;;]; B12 = [1 0 0 0;0 1 0 0;0 0 1 0; 0 0 0 2];
Q1 = [5 0 0 0;0 0 0 0;0 0 0 0;0 0 0 3]; R11 = [-1;;];
Q2 = [50 0 0 0;0 4 0 0;0 0 5 0;0 0 0 0]; R22 = [-40 0 0 0;0 -1 0 0;0 0 -1 0;0 0 0 -40];
R = [R11 zeros(1,4); zeros(4,1) R22];
A2 = [A' zeros(4,4); zeros(4,4) A']; A1 = A; Q = [Q1;Q2]; 
B1 = [B11 B12]; B2 = [B11 zeros(4,4); zeros(4,1) B12];
x, clseig, F, Z, scalinfo = narec(A1, A2, B1, B2, R, Q) 
@test norm(A2'*x+x*A1-(x*B1)*inv(R)*(B2'*x)+Q)/max(1,norm(x)) < 1.e-10 &&
norm(sort(real(clseig))-sort(real(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10 &&
norm(sort(imag(clseig))-sort(imag(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10


A = [-0.0366 0.0271 0.0188 -0.4555;
0.0482 -1.0100 0.0024 -4.0208;
0.1002 0.2855 -0.7070 1.3229;
0 0 1.0000 0];
B11 = [0.4422; 3.0447; -5.52; 0];
B12 = [0.1761; -7.5922; 4.99; 0];
Q1 = diagm([3.5, 2, 4, 5]);
Q2 = diagm([1.5, 6, 3, 1]);
R11 = 1;
R12 = 0.25;
R21 = 0.6;
R22 = 2;
R = [R11 R12; R21 R22];
A2 = [A' zeros(4,4); zeros(4,4) A']; A1 = A; Q = [Q1;Q2]; 
B1 = [B11 B12]; B2 = [B11 zeros(4,1); zeros(4,1) B12];
x, clseig, F, Z, scalinfo = narec(A1, A2, B1, B2, R, Q) 
@test norm(A2'*x+x*A1-(x*B1)*inv(R)*(B2'*x)+Q)/max(1,norm(x)) < 1.e-10 &&
norm(sort(real(clseig))-sort(real(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10 &&
norm(sort(imag(clseig))-sort(imag(eigvals(A1-B1*F))))/norm(clseig)  < 1.e-10

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

@time X, clseig, F = narec(A, A, B, B, R, Q; scaling = 'N')
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

@time X, clseig, F = narec(BigFloat.(A),A,B,complex(B),R,Q; scaling = 'B', orth = true)
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

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'S')
rezs1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'S', orth = true)
rezs2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'S', orth = true)
rezs2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', nrm = 1)
rezk1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezk1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'K', nrm = 1)
rezk1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezk1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', orth = true, nrm = 1)
rezk2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'K', orth = true, nrm = 1)
rezk2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezs2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol


@time X, clseig, F = arec(A,B,R,Q; scaling = 'G')
rezg1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezg1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'G')
rezg1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezg1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'G', orth = true)
rezg2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezg2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'G', orth = true)
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

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'S')
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = arec(A,B,R,Q; scaling = 'K', nrm = 1)
rezb1 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb1 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

@time X, clseig, F = narec(A,A,B,B,R,Q; scaling = 'K', nrm = 1)
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