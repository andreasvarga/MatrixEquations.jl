module Test_riccati

using Random
using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing algebraic Riccati equation solvers" begin

# only double precision tests are performed

#for (Ty,n,p,m) in ((Float64, 30, 10, 10), (Float32, 5, 3, 3))
for (Ty,n,p,m) in ((Float64, 20, 10, 10),  (Float64, 5, 3, 1))

Random.seed!(21235)

#(Ty,n,p,m) = (Float64, 30, 10, 10)
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
Ty == Float64 ? rtol = n*n*sqrt(eps(1.)) : rtol = n*sqrt(eps(1.f0))

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

end 

end
