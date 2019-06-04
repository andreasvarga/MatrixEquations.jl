module Test_dlyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing discrete Lyapunov equation solvers" begin

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

@testset "Discrete Lyapunov equations" begin
@time x = lyapd(ac,qc);
@test norm(ac*x*ac'-x+qc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time x = lyapd(ac',qc);
@test norm(ac'*x*ac-x+qc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time x = lyapd(ar,qr)
@test norm(ar*x*ar'-x+qr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ar',qr)
@test norm(ar'*x*ar-x+qr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ac,qr);
@test norm(ac*x*ac'-x+qr)/norm(x)/max(1.,norm(ac)^2)  < reltol

@time x = lyapd(ac',qr);
@test norm(ac'*x*ac-x+qr)/norm(x)/max(1.,norm(ac)^2)  < reltol

@time x = lyapd(ar,qc)
@test norm(ar*x*ar'-x+qc)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ar',qc)
@test norm(ar'*x*ar-x+qc)/norm(x)/max(1.,norm(ar)^2) < reltol
end

@testset "Discrete generalized Lyapunov equations" begin
@time x = lyapd(ac,ec,qc);
@test norm(ac*x*ac'-ec*x*ec'+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec',qc);
@test norm(ac'*x*ac-ec'*x*ec+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec,qc);
@test norm(ac'*x*ac-ec*x*ec'+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac,ec',qc);
@test norm(ac*x*ac'-ec'*x*ec+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ar,er,qr);
@test norm(ar*x*ar'-er*x*er'+qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er',qr);
@test norm(ar'*x*ar-er'*x*er+qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er,qr);
@test norm(ar'*x*ar-er*x*er'+qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er',qr);
@test norm(ar*x*ar'-er'*x*er+qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ac,ec,qr);
@test norm(ac*x*ac'-ec*x*ec'+qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec',qr);
@test norm(ac'*x*ac-ec'*x*ec+qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec,qr);
@test norm(ac'*x*ac-ec*x*ec'+qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac,ec',qr);
@test norm(ac*x*ac'-ec'*x*ec+qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er,qc);
@test norm(ar*x*ar'-er*x*er'+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er',qc);
@test norm(ar'*x*ar-er'*x*er+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er,qc);
@test norm(ar'*x*ar-er*x*er'+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er',qc);
@test norm(ar*x*ar'-er'*x*er+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol
end

@testset "Discrete Lyapunov equations - Schur form" begin

x = copy(qr)
@time lyapds!(as,x);
@test norm(as*x*as'+qr-x)/norm(x)/max(norm(as)^2,1.) < reltol

x = copy(qr)
@time lyapds!(as,x,adj=true);
@test norm(as'*x*as+qr-x)/norm(x)/max(norm(as)^2,1.) < reltol

x = copy(qc)
@time lyapds!(acs,ecs,x);
@test norm(acs*x*acs'+qc-ecs*x*ecs')/norm(x)/max(norm(acs)^2,norm(ecs)^2) < reltol

x = copy(qc)
@time lyapds!(acs,ecs,x,adj=true);
@test norm(acs'*x*acs+qc-ecs'*x*ecs)/norm(x)/max(norm(acs)^2,norm(ecs)^2) < reltol

x = copy(qr)
@time lyapds!(as,es,x,adj=true);
@test norm(as'*x*as+qr-es'*x*es)/norm(x)/max(norm(as)^2,norm(es)^2) < reltol

x = copy(qc)
@time lyapds!(as,es,x);
@test norm(as*x*as'+qc-es*x*es')/norm(x)/max(norm(as)^2,norm(es)^2) < reltol

x = copy(qc)
@time lyapds!(as,es,x,adj=true);
@test norm(as'*x*as+qc-es'*x*es)/norm(x)/max(norm(as)^2,norm(es)^2) < reltol

x = copy(qr)
@time lyapds!(as,es,x,adj=true);
@test norm(as'*x*as+qr-es'*x*es)/norm(x)/max(norm(as)^2,norm(es)^2) < reltol
end
end

end
