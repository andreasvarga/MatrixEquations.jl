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
qr = real(qc)
reltol = eps(float(max(n,100)))

@testset "Continuous Lyapunov equations" begin
@time x = lyapc(ac,qc);
@test norm(ac*x+x*ac'+qc)/norm(x)/norm(ac) < reltol

@time x = lyapc(ac',qc);
@test norm(ac'*x+x*ac+qc)/norm(x)/norm(ac) < reltol

@time x = lyapc(ac,qc,adj=true);
@test norm(ac'*x+x*ac+qc)/norm(x)/norm(ac)  < reltol

@time x = lyapc(ar,qr)
@test norm(ar*x+x*ar'+qr)/norm(x)/norm(ar)  < reltol

@time x = lyapc(ar',qr)
@test norm(ar'*x+x*ar+qr)/norm(x)/norm(ar) < reltol

@time x = lyapc(ar,qr,adj=true)
@test norm(ar'*x+x*ar+qr)/norm(x)/norm(ar) < reltol

@time x = lyapc(ac,qr);
@test norm(ac*x+x*ac'+qr)/norm(x)/norm(ac) < reltol

@time x = lyapc(ac',qr);
@test norm(ac'*x+x*ac+qr)/norm(x)/norm(ac) < reltol

@time x = lyapc(ac,qr,adj=true)
@test norm(ac'*x+x*ac+qr)/norm(x)/norm(ac) < reltol

@time x = lyapc(ar,qc);
@test norm(ar*x+x*ar'+qc)/norm(x)/norm(ar) < reltol

@time x = lyapc(ar',qc);
@test norm(ar'*x+x*ar+qc)/norm(x)/norm(ar)  < reltol

@time x = lyapc(ar,qc,adj = true);
@test norm(ar'*x+x*ar+qc)/norm(x)/norm(ar)  < reltol
end


@testset "Continuous generalized Lyapunov equations" begin
@time x = glyapc(ac,ec,qc);
@test norm(ac*x*ec'+ec*x*ac'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

@time x = glyapc(ac',ec',qc);
@test norm(ac'*x*ec+ec'*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = glyapc(ac,ec,qc,adj=true);
@test norm(ac'*x*ec+ec'*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = glyapc(ac',ec,qc);
@test norm(ac'*x*ec'+ec*x*ac+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = glyapc(ac,ec',qc);
@test norm(ac*x*ec+ec'*x*ac'+qc)/norm(x)/norm(ac)/norm(ec)  < reltol

@time x = glyapc(ar,er,qr);
@test norm(ar*x*er'+er*x*ar'+qr)/norm(x)/norm(ar)/norm(er)  < reltol

@time x = glyapc(ar',er',qr);
@test norm(ar'*x*er+er'*x*ar+qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar,er,qr,adj=true);
@test norm(ar'*x*er+er'*x*ar+qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar',er,qr);
@test norm(ar'*x*er'+er*x*ar+qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar,er',qr);
@test norm(ar*x*er+er'*x*ar'+qr)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ac,ec,qr);
@test norm(ac*x*ec'+ec*x*ac'+qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = glyapc(ac',ec',qr);
@test norm(ac'*x*ec+ec'*x*ac+qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = glyapc(ac,ec,qr,adj=true);
@test norm(ac'*x*ec+ec'*x*ac+qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = glyapc(ac',ec,qr);
@test norm(ac'*x*ec'+ec*x*ac+qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = glyapc(ac,ec',qr);
@test norm(ac*x*ec+ec'*x*ac'+qr)/norm(x)/norm(ac)/norm(ec)   < reltol

@time x = glyapc(ar,er,qc);
@test norm(ar*x*er'+er*x*ar'+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar',er',qc);
@test norm(ar'*x*er+er'*x*ar+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar,er,qc,adj=true);
@test norm(ar'*x*er+er'*x*ar+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar',er,qc);
@test norm(ar'*x*er'+er*x*ar+qc)/norm(x)/norm(ar)/norm(er) < reltol

@time x = glyapc(ar,er',qc);
@test norm(ar*x*er+er'*x*ar'+qc)/norm(x)/norm(ar)/norm(er) < reltol
end

@testset "Continuous Lyapunov equations - Schur form" begin

x = copy(qc)
@time glyapcs!(acs,ecs,x);
@test norm(acs*x*ecs'+ecs*x*acs'+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(qc)
@time glyapcs!(acs,ecs,x,adj=true);
@test norm(acs'*x*ecs+ecs'*x*acs+qc)/norm(x)/norm(acs)/norm(ecs) < reltol

x = copy(qr)
@time glyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qr)
@time glyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(acs,x);
@test norm(acs*x+x*acs'+qc)/norm(x)/norm(acs) < reltol

x = copy(qc)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+qc)/norm(x)/norm(as) < reltol

x = copy(qc)
@time glyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+qc)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time glyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+qc)/norm(x)/norm(as)/norm(es) < reltol

x = copy(qc)
@time lyapcs!(acs,x,adj=true);
@test norm(acs'*x+x*acs+qc)/norm(x)/norm(acs) < reltol

x = copy(qr)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+qr)/norm(x)/norm(as) < reltol

x = copy(qr)
@time lyapcs!(as,x,adj=true);
@test norm(as'*x+x*as+qr)/norm(x)/norm(as) < reltol
end
end

end
