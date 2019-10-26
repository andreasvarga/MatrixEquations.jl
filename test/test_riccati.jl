module Test_riccati

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing algebraic Riccati equation solvers" begin


#n, p, m = 300, 50, 50
n, p, m = 30, 10, 10
ar = randn(n,n)
er = rand(n,n)
br = rand(n,m)
cr = rand(p,n)
ac = randn(n,n) + im*randn(n,n)
ec = er+im*rand(n,n)
cc = rand(p,n)+im*rand(p,n)
bc = rand(n,m)+im*rand(n,m)
rr1 = rand(m,m)
rc1 = rand(m,m)+im*rand(m,m)
rtol = n*sqrt(eps(1.))

qc = cc'*cc
Qr = cr'*cr
gc = bc*bc'
gr = br*br'
rr = rr1*rr1'
rc = rc1*rc1'

sc = cc'/100
sr = cr'/100


@testset "Continuous Riccati equation" begin
@time x, clseig = arec(ar,gr,Qr)
@test norm(ar'*x+x*ar-x*gr*x+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr)
@test norm(ar'*x+x*ar-x*gr*x)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,2I,I)
@test norm(ar'*x+x*ar-x*2*x+I)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-2*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-2*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gr,2I)
@test norm(ar'*x+x*ar-x*gr*x+2I)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,2I,Qr)
@test norm(ar'*x+x*ar-x*2*x+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-2*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-2*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar',gr,Qr)
@test norm(ar*x+x*ar'-x*gr*x+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar'-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar'-gr*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac,gc,qc)
@test norm(ac'*x+x*ac-x*gc*x+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac',gc,qc)
@test norm(ac*x+x*ac'-x*gc*x+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac'-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac'-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ar,gc,qc)
@test norm(ar'*x+x*ar-x*gc*x+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-gc*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-gc*x))))/norm(clseig)  < rtol

@time x, clseig = arec(ac,gr,Qr)
@test norm(ac'*x+x*ac-x*gr*x+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-gr*x))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-gr*x))))/norm(clseig)  < rtol
end

@testset "Continuous control Riccati equation" begin


@time x, clseig, f = arec(ar,br,rr,Qr,sr)
@test norm(ar'*x+x*ar-(x*br+sr)*inv(rr)*(br'*x+sr')+Qr)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,rr,Qr)
@test norm(ar'*x+x*ar-x*br*inv(rr)*br'*x+Qr)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ar,br,rr,0I)
@test norm(ar'*x+x*ar-x*br*inv(rr)*br'*x)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rc,qc)
@test norm(ac'*x+x*ac-x*bc*inv(rc)*bc'*x+qc)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rc,qc,sc)
@test norm(ac'*x+x*ac-(x*bc+sc)*inv(rc)*(bc'*x+sc')+qc)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

@time x, clseig, f = arec(ac,bc,rr,Qr,sr)
@test norm(ac'*x+x*ac-(x*bc+sr)*inv(rr)*(bc'*x+sr')+Qr)/norm(x)  < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(clseig)  < rtol

end

@testset "Generalized continuous control Riccati equation" begin

@time x, clseig, f = garec(ar,er,br,rr,Qr,sr)
@test norm(ar'*x*er+er'*x*ar-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ar,er,br,rr,2I,sr)
@test norm(ar'*x*er+er'*x*ar-(er'x*br+sr)*inv(rr)*(br'*x*er+sr')+2I)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ac,ec,bc,rc,qc,sc)
@test norm(ac'*x*ec+ec'*x*ac-(ec'x*bc+sc)*inv(rc)*(bc'*x*ec+sc')+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

@time x, clseig, f = garec(ac,ec,bc,rr,Qr,sr)
@test norm(ac'*x*ec+ec'*x*ac-(ec'x*bc+sr)*inv(rr)*(bc'*x*ec+sr')+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

end


@testset "Discrete control Riccati equation" begin
@time x, clseig, f = ared(ar,br,rr,Qr)
@test norm(ar'*x*ar-x-ar'*x*br*inv(rr+br'*x*br)*br'*x*ar+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,0I)
@test norm(ar'*x*ar-x-ar'*x*br*inv(rr+br'*x*br)*br'*x*ar)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ar,br,rr,Qr,sr)
@test norm(ar'*x*ar-x-(ar'*x*br+sr)*inv(rr+br'*x*br)*(br'*x*ar+sr')+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f))))/norm(ar)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f))))/norm(ar)  < rtol

@time x, clseig, f = ared(ac,bc,rc,qc)
@test norm(ac'*x*ac-x-ac'*x*bc*inv(rc+bc'*x*bc)*bc'*x*ac+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(ac)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(ac)  < rtol

@time x, clseig, f = ared(ac,bc,rc,qc,sc)
@test norm(ac'*x*ac-x-(ac'*x*bc+sc)*inv(rc+bc'*x*bc)*(bc'*x*ac+sc')+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f))))/norm(ac)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f))))/norm(ac)  < rtol
end


@testset "Generalized discrete control Riccati equation" begin
@time x, clseig, f = gared(ar,er,br,rr,Qr,sr)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br+sr)*inv(rr+br'*x*br)*(br'*x*ar+sr')+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = gared(ar,er,br,rr,2I,sr)
@test norm(ar'*x*ar-er'*x*er-(ar'*x*br+sr)*inv(rr+br'*x*br)*(br'*x*ar+sr')+2I)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ar-br*f,er))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ar-br*f,er))))/norm(clseig)  < rtol

@time x, clseig, f = gared(ac,ec,bc,rc,qc,sc)
@test norm(ac'*x*ac-ec'*x*ec-(ac'*x*bc+sc)*inv(rc+bc'*x*bc)*(bc'*x*ac+sc')+qc)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

@time x, clseig, f = gared(ac,ec,bc,rr,Qr,sr)
@test norm(ac'*x*ac-ec'*x*ec-(ac'*x*bc+sr)*inv(rr+bc'*x*bc)*(bc'*x*ac+sr')+Qr)/norm(x) < rtol &&
norm(sort(real(clseig))-sort(real(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol &&
norm(sort(imag(clseig))-sort(imag(eigvals(ac-bc*f,ec))))/norm(clseig)  < rtol

end

end

end
