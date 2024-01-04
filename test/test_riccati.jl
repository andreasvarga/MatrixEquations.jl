module Test_riccati

using Random
using LinearAlgebra
using MatrixEquations
using Test

println("Test_riccati")

@testset "Testing algebraic Riccati equation solvers" begin


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

@time x, clseig = arec(ar,gr,Qr,as = true)
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
    #
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
reltol = sqrt(eps())

# without scaling
@time X, clseig = arec(A,G,Q; scaling = 'N')
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

# with special scaling
@time X, clseig = arec(A,G,Q; scaling = 'S')
rezs = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezs < rezb && norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X)) < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-G*X))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-G*X))))/norm(clseig)  < reltol

# with general scaling
@time X, clseig = arec(A,G,Q; scaling = 'G')
rezg = norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X))
@test rezg < rezb && norm(A'*X+X*A-X*G*X+Q)/max(1,norm(X)) < reltol &&
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
rezb = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)); ev = eigvals(A-G*X*E,E)
@test rezb < reltol &&
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

# with special scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'S');
rezs = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezs < reltol && rezs < rezb && 
norm(sort(real(clseig))-sort(real(ev)))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(ev)))/norm(clseig)  < reltol

# with special scaling
@time X, clseig = garec(A,E,G,Q; scaling = 'G')
rezg = norm(A'*X*E+E'*X*A-E'*X*G*X*E+Q)/max(1,norm(X)) 
ev = eigvals(A-G*X*E,E)
@test rezg < reltol && rezg < rezb && 
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

@time X, clseig, F = arec(A,B,R,Q; scaling = 'B', orth = true)
rezb2 = norm(A'*X+X*A-X*B*inv(R)*B'*X+Q)/max(1,norm(X))
@test  rezb2 < reltol &&
norm(sort(real(clseig))-sort(real(eigvals(A-B*F))))/norm(clseig)  < reltol &&
norm(sort(imag(clseig))-sort(imag(eigvals(A-B*F))))/norm(clseig)  < reltol

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
@show rezb, reltol
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

end


end 

end
