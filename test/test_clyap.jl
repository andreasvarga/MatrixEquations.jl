module Test_clyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing continuous Lyapunov equation solvers" begin

n = 100
Ty = Float64

@testset "Continuous Lyapunov equations" begin

reltol = eps(float(100))
a = -1; b = 2im; @time x = lyapc(a,b)
@test abs(a*x+x*a'+b) < reltol

reltol = eps(float(100f0))
a = 1f0-2f0im; b = 2f0; @time x = lyapc(a,b)
@test abs(a*x+x*a'+b) < reltol

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
c = rand(Ty,n,n)+im*rand(Ty,n,n)
qc = c'*c
Qr = real(qc)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))

@time x = lyapc(ac,qc);
@test norm(ac*x+x*ac'+qc)/norm(x)/norm(ac) < reltol

α = 3+im; # α = im  # SingularException
@time x = lyapc(α*I,qc);
@test norm(α*x+x*α'+qc)/norm(x) < reltol

α = 3+im; # α = im  # SingularException
@time x = lyapc(α,Qr);
@test norm(α*x+x*α'+Qr)/norm(x) < reltol

@time x = lyapc(ac',qc);
@test norm(ac'*x+x*ac+qc)/norm(x)/norm(ac) < reltol

@time x = lyapc(ar,Qr)
@test norm(ar*x+x*ar'+Qr)/norm(x)/norm(ar)  < reltol

@time x = lyapc(ar',Qr)
@test norm(ar'*x+x*ar+Qr)/norm(x)/norm(ar) < reltol

@time x = lyapc(ac,Qr);
@test norm(ac*x+x*ac'+Qr)/norm(x)/norm(ac) < reltol

@time x = lyapc(ac',Qr);
@test norm(ac'*x+x*ac+Qr)/norm(x)/norm(ac) < reltol

@time x = lyapc(ar,qc);
@test norm(ar*x+x*ar'+qc)/norm(x)/norm(ar) < reltol

@time x = lyapc(ar',qc);
@test norm(ar'*x+x*ar+qc)/norm(x)/norm(ar)  < reltol
end
end


@testset "Continuous generalized Lyapunov equations" begin

reltol = eps(float(100))
a = -1+im; ee = 3im; b = 2; @time x = lyapc(a,ee,b)
@test abs(a*x*ee'+ee*x*a'+b) < reltol

reltol = eps(float(100f0))
a = 1f0-2f0im; ee = 3f0im; b = 2f0; @time x = lyapc(a,ee,b)
@test abs(a*x*ee'+ee*x*a'+b) < reltol



for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
er = rand(Ty,n,n)
ec = er+im*rand(Ty,n,n)

c = rand(Ty,n,n)+im*rand(Ty,n,n)
qc = c'*c
Qr = real(qc)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time x = lyapc(ac,ec,qc);
@test norm(ac*x*ec'+ec*x*ac'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

β = 3
@time x = lyapc(ac,β*I,qc);
@test norm(ac*x*β'+β*x*ac'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

β = (1+im);
@time x = lyapc(ac,β,qc);
@test norm(ac*x*β'+β*x*ac'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

α = 2+3im
@time x = lyapc(α*I,ec,qc);
@test norm(α*x*ec'+ec*x*α'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

α = 2+3im; β = (1+im); # α = im; β = 1  # SingularException
@time x = lyapc(α,β,qc);
@test norm(α*x*β'+β*x*α'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

@time x = lyapc(ac',ec',qc);
@test norm(ac'*x*ec+ec'*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

β = (1+im);
@time x = lyapc(ac',β*I,qc);
@test norm(ac'*x*β+β'*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

α = 2+3im
@time x = lyapc(α*I,ec',qc);
@test norm(α'*x*ec+ec'*x*α+qc)/norm(x)/norm(ac)/norm(ec) < reltol

@time x = lyapc(ac',ec,qc);
@test norm(ac'*x*ec'+ec*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = lyapc(ac,ec',qc);
@test norm(ac*x*ec+ec'*x*ac'+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = lyapc(ar,er,Qr);
@test norm(ar*x*er'+er*x*ar'+Qr)/norm(x)/norm(ar)/norm(er)  < reltol

@time x = lyapc(ar,er,qc);
@test norm(ar*x*er'+er*x*ar'+qc)/norm(x)/norm(ar)/norm(er)  < reltol

@time x = lyapc(ar',er',Qr);
@test norm(ar'*x*er+er'*x*ar+Qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ar',er,Qr);
@test norm(ar'*x*er'+er*x*ar+Qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ar,er',Qr);
@test norm(ar*x*er+er'*x*ar'+Qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ac,ec,Qr);
@test norm(ac*x*ec'+ec*x*ac'+Qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = lyapc(ac',ec',Qr);
@test norm(ac'*x*ec+ec'*x*ac+Qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = lyapc(ac',ec,Qr);
@test norm(ac'*x*ec'+ec*x*ac+Qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = lyapc(ac,ec',Qr);
@test norm(ac*x*ec+ec'*x*ac'+Qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = lyapc(ar,er,qc);
@test norm(ar*x*er'+er*x*ar'+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ar',er',qc);
@test norm(ar'*x*er+er'*x*ar+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ar',er,qc);
@test norm(ar'*x*er'+er*x*ar+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = lyapc(ar,er',qc);
@test norm(ar*x*er+er'*x*ar'+qc)/norm(x)/norm(ar)/norm(er) < reltol
end
end

@testset "Continuous Lyapunov equations - Schur form" begin

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
er = rand(Ty,n,n)
ec = er+im*rand(Ty,n,n)
as, es = schur(ar,er)
acs, ecs = schur(ac,ec)

c = rand(Ty,n,n)+im*rand(Ty,n,n)
qc = c'*c
Qr = real(qc)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


x = copy(qc)
@time lyapcs!(acs,ecs,x);
@test norm(acs*x*ecs'+ecs*x*acs'+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(qc)
@time lyapcs!(acs,I,x);
@test norm(acs*x+x*acs'+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(qc)
@time lyapcs!(acs,ecs,x,adj=true);
@test norm(acs'*x*ecs+ecs'*x*acs+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(Qr)
@time lyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+qc)/norm(x)/norm(as)/norm(es) < reltol

x = copy(Qr)
@time lyapcs!(as,I,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

x = copy(Qr)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

x = copy(qc)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+qc)/norm(x)/norm(as)/norm(es) < reltol

if Ty == Float64
try
  x = convert(Matrix{Complex{Float32}},copy(qc))
  @time lyapcs!(as,x);
  @test false
catch
  @test true
end
end

x = copy(Qr)
@time lyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(acs,x);
@test norm(acs*x+x*acs'+qc)/norm(x)/norm(acs) < reltol


x = copy(qc)
@time lyapcs!(acs,x,adj=true);
@test norm(acs'*x+x*acs+qc)/norm(x)/norm(acs) < reltol

x = copy(Qr)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

x = copy(Qr)
@time lyapcs!(as,x,adj=true);
@test norm(as'*x+x*as+Qr)/norm(x)/norm(as) < reltol
end
end

end

end
