module Test_dlyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing discrete Lyapunov equation solvers" begin

n = 10
Ty = Float64

@testset "Discrete Lyapunov equations" begin

reltol = eps(float(1000))
a = -2+im; b = 2; @time x = lyapd(a,b)
@test abs(a*x*a'-x+b) < reltol

reltol = eps(float(1000f0))
a = 1f0-2f0im; b = 2f0; @time x = lyapd(a,b)
@test abs(a*x*a'-x+b)  < reltol

try
   x = lyapd(ones(1,1),ones(1,1))
   @test false
 catch
   @test true
 end  

 try
   x = lyapd([0 1;  -1 0 ],ones(2,2))
   @test false
 catch
   @test true
 end  
 
  try
   x = lyapd([0.5 1;  0 2],ones(2,2))
   @test false
 catch
   @test true
 end  
 
 try
   x = lyapd([1 1 1 1;  -1 1 0 1; 0 0 0.5 0.5; 0 0 -0.5 0.5],ones(4,4))
   @test false
 catch
   @test true
 end  
 

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
c = rand(Ty,n,n)+im*rand(Ty,n,n)
qc = c'*c
Qr = real(qc)
Ty == Float64 ? reltol = eps(float(10000)) : reltol = eps(10000*n*one(Ty))

@time x = lyapd(ac,qc);
@test norm(ac*x*ac'-x+qc)/norm(x)/max(1.,norm(ac)^2) < reltol

α = 3+im; # α = 1; #SingularException
@time x = lyapd(α*I,qc);
@test norm(α*x*α'-x+qc)/norm(x) < reltol

try
   α = 1; #SingularException
   @time x = lyapd(α,qc);
catch
   @test true
end

α = 3. +im;
@time x = lyapd(α,qc);
@test norm(α*x*α'-x+qc)/norm(x) < reltol

@time x = lyapd(ac',qc);
@test norm(ac'*x*ac-x+qc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time x = lyapd(ar,Qr)
@test norm(ar*x*ar'-x+Qr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ar',Qr)
@test norm(ar'*x*ar-x+Qr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ac,Qr);
@test norm(ac*x*ac'-x+Qr)/norm(x)/max(1.,norm(ac)^2)  < reltol

@time x = lyapd(ac',Qr);
@test norm(ac'*x*ac-x+Qr)/norm(x)/max(1.,norm(ac)^2)  < reltol

@time x = lyapd(ar,qc)
@test norm(ar*x*ar'-x+qc)/norm(x)/max(1.,norm(ar)^2) < reltol

@time x = lyapd(ar',qc)
@test norm(ar'*x*ar-x+qc)/norm(x)/max(1.,norm(ar)^2) < reltol
end
end

@testset "Discrete generalized Lyapunov equations" begin

reltol = eps(float(10000))
ac = -2+im; ec = 4+im; b = 2; @time x = lyapd(ac,ec,b)
@test abs(ac*x*ac'-ec*x*ec'+b) < reltol

try
   x = lyapd(ones(1,1),ones(1,1),ones(1,1))
   @test false
catch
   @test true
end  

try
   x = lyapd([0 2;  -2 0 ],[2 0; 0 2],ones(2,2))
   @test false
catch
   @test true
end  
 
try
   x = lyapd([2 1;  0 2],[2 0; 0 2],ones(2,2))
   @test false
catch
   @test true
end  
 
try
   x = lyapd(2*[1 1 1 1;  -1 1 0 1; 0 0 0.5 0.5; 0 0 -0.5 0.5],2*Matrix{Float64}(I,4,4),ones(4,4))
   @test false
catch
   @test true
end  

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
er = rand(Ty,n,n)
ec = er+im*rand(Ty,n,n)

c = rand(Ty,n,n)+im*rand(Ty,n,n)
qc = c'*c
Qr = real(qc)
Ty == Float64 ? reltol = eps(float(1000)) : reltol = eps(1000*n*one(Ty))

@time x = lyapd(ac,ec,qc);
@test norm(ac*x*ac'-ec*x*ec'+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

β = 3
@time x = lyapd(ac,β*I,qc);
@test norm(ac*x*ac'-β*x*β'+qc)/norm(x)/norm(ac) < reltol

β = (1+im);
@time x = lyapd(ac,β,qc);
@test norm(ac*x*ac'-β*x*β'+qc)/norm(x)/norm(ac) < reltol

α = 2+3im
@time x = lyapd(α*I,ec,qc);
@test norm(α*x*α'-ec*x*ec'+qc)/norm(x)/norm(ec) < reltol

α = 2+3im
@time x = lyapd(α,ec,qc);
@test norm(α*x*α'-ec*x*ec'+qc)/norm(x)/norm(ec) < reltol

try
   @time x = lyapd(0*ac,0*ec,qc);
   @test false
catch
   @test true
end

α = 2+3im; β = (1+im);
@time x = lyapd(α*I,β*I,qc);
@test norm(α*x*α'-β*x*β'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

try
   α = 1; β = 1; #SingularException
   @time x = lyapd(α,β,qc);
   @test false
catch
   @test true
end

α = 2+3im; β = (1+im);
@time x = lyapd(α,β,qc);
@test norm(α*x*α'-β*x*β'+qc)/norm(x)/norm(ac)/norm(ec) < reltol

@time x = lyapd(ac',ec',qc);
@test norm(ac'*x*ac-ec'*x*ec+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

β = (1+im);
@time x = lyapd(ac',β*I,qc);
@test norm(ac'*x*ac-β*x*β'+qc)/norm(x)/norm(ac) < reltol

α = 2+3im
@time x = lyapd(α*I,ec',qc);
@test norm(α*x*α'-ec'*x*ec+qc)/norm(x)/norm(ec) < reltol

@time x = lyapd(ac',ec,qc);
@test norm(ac'*x*ac-ec*x*ec'+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac,ec',qc);
@test norm(ac*x*ac'-ec'*x*ec+qc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ar,er,Qr);
@test norm(ar*x*ar'-er*x*er'+Qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er',Qr);
@test norm(ar'*x*ar-er'*x*er+Qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er,Qr);
@test norm(ar'*x*ar-er*x*er'+Qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er',Qr);
@test norm(ar*x*ar'-er'*x*er+Qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ac,ec,Qr);
@test norm(ac*x*ac'-ec*x*ec'+Qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec',Qr);
@test norm(ac'*x*ac-ec'*x*ec+Qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac',ec,Qr);
@test norm(ac'*x*ac-ec*x*ec'+Qr)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time x = lyapd(ac,ec',Qr);
@test norm(ac*x*ac'-ec'*x*ec+Qr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er,qc);
@test norm(ar*x*ar'-er*x*er'+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er',qc);
@test norm(ar'*x*ar-er'*x*er+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar',er,qc);
@test norm(ar'*x*ar-er*x*er'+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time x = lyapd(ar,er',qc);
@test norm(ar*x*ar'-er'*x*er+qc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol
end
end


@testset "Discrete Lyapunov equations - Schur form" begin

for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = rand(Ty,n,n)+im*rand(Ty,n,n);
er = rand(Ty,n,n);
ec = er+im*rand(Ty,n,n);
as, es = schur(ar,er);
acs, ecs = schur(ac,ec);

c = rand(Ty,n,n)+im*rand(Ty,n,n);
qc = c'*c;
Qr = real(qc);
Ty == Float64 ? reltol = eps(float(1000)) : reltol = eps(1000*n*one(Ty))

x = copy(Qr)
@time lyapds!(as,x);
@test norm(as*x*as'+Qr-x)/norm(x)/max(norm(as)^2,1.) < reltol

x = copy(Qr)
@time lyapds!(as,x,adj=true);
@test norm(as'*x*as+Qr-x)/norm(x)/max(norm(as)^2,1.) < reltol

x = copy(qc)
@time lyapds!(acs,ecs,x);
@test norm(acs*x*acs'+qc-ecs*x*ecs')/norm(x)/max(norm(acs)^2,norm(ecs)^2) < reltol

x = copy(qc)
@time lyapds!(acs,I,x);
@test norm(acs*x*acs'+qc-x)/norm(x)/max(norm(acs)^2,norm(ecs)^2) < reltol

x = copy(qc)
@time lyapds!(acs,ecs,x,adj=true);
@test norm(acs'*x*acs+qc-ecs'*x*ecs)/norm(x)/max(norm(acs)^2,norm(ecs)^2) < reltol

x = copy(Qr)
@time lyapds!(as,es,x);
@test norm(as*x*as'+Qr-es*x*es')/norm(x)/max(norm(as)^2,norm(es)^2) < reltol

x = copy(Qr)
@time lyapds!(as,es,x,adj=true);
@test norm(as'*x*as+Qr-es'*x*es)/norm(x)/max(norm(as)^2,norm(es)^2) < reltol

end
end

end

end
