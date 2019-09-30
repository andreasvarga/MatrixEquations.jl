module Test_clyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing continuous Lyapunov equation solvers" begin

n = 100
ar = rand(n,n)
er = rand(n,n)
ac = ar+im*rand(n,n)
ec = er+im*rand(n,n)
as, es = schur(ar,er)
acs, ecs = schur(ac,ec)
c = rand(n,n)+im*rand(n,n)
qc = c'*c
Qr = real(qc)
reltol = eps(float(max(n,100)))

@testset "Continuous Lyapunov equations" begin
@time x = lyapc(ac,qc);
@test norm(ac*x+x*ac'+qc)/norm(x)/norm(ac) < reltol

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


@testset "Continuous generalized Lyapunov equations" begin
@time x = lyapc(ac,ec,qc);
@test norm(ac*x*ec'+ec*x*ac'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

@time x = lyapc(ac',ec',qc);
@test norm(ac'*x*ec+ec'*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = lyapc(ac',ec,qc);
@test norm(ac'*x*ec'+ec*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = lyapc(ac,ec',qc);
@test norm(ac*x*ec+ec'*x*ac'+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = lyapc(ar,er,Qr);
@test norm(ar*x*er'+er*x*ar'+Qr)/norm(x)/norm(ar)/norm(er)  < reltol

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

@testset "Continuous Lyapunov equations - Schur form" begin

x = copy(qc)
@time lyapcs!(acs,ecs,x);
@test norm(acs*x*ecs'+ecs*x*acs'+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(qc)
@time lyapcs!(acs,ecs,x,adj=true);
@test norm(acs'*x*ecs+ecs'*x*acs+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(Qr)
@time lyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(Qr)
@time lyapcs!(as,I,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

x = copy(Qr)
@time lyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(acs,x);
@test norm(acs*x+x*acs'+qc)/norm(x)/norm(acs) < reltol

x = copy(qc)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+qc)/norm(x)/norm(as) < reltol

x = copy(qc)
@time lyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+qc)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+qc)/norm(x)/norm(as)/norm(es) < reltol

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
