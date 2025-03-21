module Test_clyap

using LinearAlgebra
using MatrixEquations
using GenericSchur
using GenericLinearAlgebra
using DoubleFloats
using Test


println("Test_clyap")
@testset "Testing continuous Lyapunov equation solvers" begin

n = 10; m = 7
Ty = Float64

@testset "Continuous Lyapunov equations" begin

reltol = eps(float(100))
a = -1; b = 2im; @time x = lyapc(a,b)
@test abs(a*x+x*a'+b) < reltol

reltol = eps(float(100f0))
a = 1f0-2f0im; b = 2f0; @time x = lyapc(a,b)
@test abs(a*x+x*a'+b) < reltol

try
  x = lyapc(zeros(1,1),ones(1,1))
  if norm(x,Inf) > 1.e10
    @test true
  else
    @test false
  end
catch
  @test true
end  

try
  x = lyapc([0 1;  -1 0 ],ones(2,2))
  if norm(x,Inf) > 1.e10
     @test true
  else
     @test false
  end
catch
  @test true
end  

try
  x = lyapc([1 1;  0 -1],ones(2,2))
  if norm(x,Inf) > 1.e10
     @test true
  else
     @test false
  end
catch
  @test true
end  

try
  x = lyapc([1 1 1 1;  -1 1 0 1; 0 0 -1 1; 0 0 -1 -1],ones(4,4))
  if norm(x,Inf) > 1.e10
     @test true
  else
     @test false
  end
catch
  @test true
end  

for Ty in (Float64, Float32, BigFloat, Double64)

ar = rand(Ty,n,n);
ac = rand(Ty,n,n)+im*rand(Ty,n,n);
c = rand(Ty,n,n)+im*rand(Ty,n,n);
qnh = rand(Ty,n,n)+im*rand(Ty,n,n);
#qc = c'*c;
qc = Matrix(Hermitian(c'*c));
Qr = real(qc);
Qnh = real(qnh);
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

if Ty != Float32
  # Fix for missing strsyl3 in OpenBLAS   
@time x = lyapc(ar,Qnh);
@test norm(ar*x+x*ar'+Qnh)/norm(x)/norm(ar)  < reltol

@time x = lyapc(ar',Qnh);
@test norm(ar'*x+x*ar+Qnh)/norm(x)/norm(ar)  < reltol
end

@time x = lyapc(ac,qnh);
@test norm(ac*x+x*ac'+qnh)/norm(x)/norm(ac)  < reltol

@time x = lyapc(ac',qnh);
@test norm(ac'*x+x*ac+qnh)/norm(x)/norm(ac)  < reltol


end
end


@testset "Continuous generalized Lyapunov equations" begin

reltol = eps(float(100))
a = -1+im; ee = 3im; b = 2; @time x = lyapc(a,ee,b)
@test abs(a*x*ee'+ee*x*a'+b) < reltol

reltol = eps(float(100f0))
a = 1f0-2f0im; ee = 3f0im; b = 2f0; @time x = lyapc(a,ee,b)
@test abs(a*x*ee'+ee*x*a'+b) < reltol

try
  x = lyapc(zeros(1,1),ones(1,1),ones(1,1))
  @test false
catch
  @test true
end  

try
  x = lyapc([0 2;  -2 0 ],[2 0; 0 2],ones(2,2))
  @test false
catch
  @test true
end  

try
  x = lyapc([2 2;  0 -2],[2 0; 0 2],ones(2,2))
  @test false
catch
  @test true
end  

try
  x = lyapc(2*[1 1 1 1;  -1 1 0 1; 0 0 -1 1; 0 0 -1 -1],2*Matrix{Float64}(I,4,4), ones(4,4))
  @test false
catch
  @test true
end  

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
er = rand(Ty,n,n)
ec = er+im*rand(Ty,n,n)

c = rand(Ty,n,n)+im*rand(Ty,n,n)
#qc = c'*c
qc = Matrix(Hermitian(c'*c));
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

α = 2+3im; β = (1+im); # α = im; β = 1  # SingularException
@time x = lyapc(α*I,β*I,qc);
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

for Ty in (Float64, Float32, BigFloat, Double64)
# for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = rand(Ty,n,n)+im*rand(Ty,n,n);
er = rand(Ty,n,n);
ec = er+im*rand(Ty,n,n);
es = triu(er);
as = es*schur(ar).T;
#as, es = schur(ar,er);
acs, ecs = schur(ac,ec);

c = rand(Ty,n,n)+im*rand(Ty,n,n);
#qc = c'*c;
qc = Matrix(Hermitian(c'*c));
Qr = real(qc);
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

# test
x = copy(Qr);
@time lyapcs!(as,es,x);
@test norm(as*x*es'+es*x*as'+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(Qr);
@time lyapcs!(as,es,x,adj=true);
@test norm(as'*x*es+es'*x*as+Qr)/norm(x)/norm(as)/norm(es) < reltol

x = copy(Qr)
@time lyapcs!(as,I,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

x = copy(Qr)
@time lyapcs!(as,x);
@test norm(as*x+x*as'+Qr)/norm(x)/norm(as) < reltol

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

@testset "Continuous Lyapunov-like equations" begin
for Ty in (Float64, Float32, BigFloat, Double64)

  ar = rand(Ty,n,n);
  er = rand(Ty,m,m)
  cr = rand(Ty,n,m)
  art = rand(Ty,m,n)
  crt = rand(Ty,m,m)
  ac = rand(Ty,n,n)+im*rand(Ty,n,n);
  cc = cr+im*rand(Ty,n,m)
  crt1 = rand(Ty,m,n)
  act = art+im*rand(Ty,m,n)
  cct = crt+im*rand(Ty,m,m)
  cct1 = crt1+im*rand(Ty,m,n)
  qnh = rand(Ty,n,n)+im*rand(Ty,n,n);
  Qc = Matrix(Hermitian(cc*cc'));
  Qr = real(Qc);
  Qcs = (Qc+transpose(Qc))/2
  Qcss = (Qc-transpose(Qc))/2
  Qrt1 = Matrix(Symmetric(crt1*crt1'))
  Qct1 = Matrix(Hermitian(cct1*cct1'))
  Ty == Float64 ? reltol = eps(float(100*n)) : reltol = eps(100*n*one(Ty))

  for fast in (true, false)

  @time x2 = tlyapc(ar,-Qr; fast)
  @test norm(ar*x2+transpose(x2)*ar'+Qr)/norm(x2) < reltol && x2 ≈ (-2ar)\Qr

  @time x2 = tlyapc(art,-Qrt1; fast)
  @test norm(art*x2+transpose(x2)*art'+Qrt1)/norm(x2) < reltol 

  @time x2 = tlyapc(art,-Qrt1; fast)
  @test norm(art*x2+transpose(x2)*art'+Qrt1)/norm(x2) < reltol 

  @time x2 = tlyapc(ac,-Qcs; fast)
  @test norm(ac*x2+transpose(x2)*transpose(ac)+Qcs)/norm(x2) < reltol 

  @time x2 = tlyapc(ac,-Qcss,-1; fast)
  @test norm(ac*x2-transpose(x2)*transpose(ac)+Qcss)/norm(x2) < reltol 

  @time x2 = tlyapc(ar,-ar+ar',-1; fast)
  @test norm(ar*x2-transpose(x2)*ar'+ar-ar')/norm(x2) < reltol && x2 ≈ (-2ar)\(ar-ar')

  @time x2 = tlyapc(art,-er+er',-1; fast)
  @test norm(art*x2-transpose(x2)*art'+er-er')/norm(x2) < reltol 
  
  @time x2 = tlyapc(transpose(ac),-ac+transpose(ac),-1; fast)
  @test norm(transpose(ac)*x2-transpose(x2)*ac+ac-transpose(ac))/norm(x2) < reltol && x2 ≈ (-2transpose(ac))\(ac-transpose(ac))

  @time x2 = hlyapc(ac',-Qc; fast)  
  @test norm(ac'*x2+adjoint(x2)*ac+Qc)/norm(x2) < reltol 

  @time x2 = hlyapc(act,-Qct1; fast)  
  @test norm(act*x2+adjoint(x2)*adjoint(act)+Qct1)/norm(x2) < reltol 

  @time x2 = hlyapc(ac',-ac+ac',-1; fast)  
  @test norm(ac'*x2-adjoint(x2)*ac+ac-ac')/norm(x2) < reltol 

end
end  

@testset "Continuous positive Lyapunov-like equations" begin
  n = 5
  Ty = Float64
  # nonsingular U
  for Ty in (Float64, Float32, BigFloat, Double64)
      Ty == Float64 ? reltol = eps(float(100*n)) : reltol = eps(100*n*one(Ty))
      U = triu(rand(Ty,n,n));
      X0 = triu(rand(Ty,n,n));
      Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
      @time X = tulyapc!(U, copy(Q); adj = false)  
      @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 
    
      Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
      @time X = tulyapc!(U, copy(Q); adj = true); 
      @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 

      U = triu(rand(Ty,n,n)+im*rand(Ty,n,n));
      X0 = triu(rand(Ty,n,n)+im*rand(Ty,n,n));
      Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
      @time X = tulyapc!(U, copy(Q); adj = false)  
      @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 

      Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
      @time X = tulyapc!(U, copy(Q); adj = true); 
      @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 

      Q = Matrix(Hermitian(U'*X0 + X0'*U))
      @time X = hulyapc!(U, copy(Q); adj = false)  
      @test norm(U'*X + X'*U - Q)/norm(X) < reltol 

      Q = Matrix(Hermitian(U*X0' + X0*U'))
      @time X = hulyapc!(U, copy(Q); adj = true); 
      @test norm(U*X' + X*U'- Q)/norm(X) < reltol 
 
  end

  # singular U
  n = 5
  Ty = Float64
  reltol = 1.e-7
  # real case
  #U = triu(rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; #U[7,7] = 0
  U = [ 0.0  0.446398  0.117541  0.108452   0.713935
  0.0  0.101524  0.167705  0.0307678  0.96426
  0.0  0.0       0.743782  0.75158    0.836415
  0.0  0.0       0.0       0.95128    0.0910885
  0.0  0.0       0.0       0.0        0.0];
  #X0 = triu(rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
  X0 = [0.0  0.516821  0.00618001  0.119169  0.85199
  0.0  0.344247  0.516058    0.271105  0.871912
  0.0  0.0       0.38342     0.154878  0.128648
  0.0  0.0       0.0         0.984649  0.459241
  0.0  0.0       0.0         0.0       0.0];
  Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
  @time X = tulyapc!(U, copy(Q); adj = false) 
  @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 

  LT = tulyaplikeop(U; adj = false)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(transpose(U)*X2 + transpose(X2)*U - Q)/norm(X2) < reltol 

  Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
  @time X = tulyapc!(U, copy(Q); adj = true) 
  @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 

  LT = tulyaplikeop(U; adj = true)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(U*transpose(X2) + X2*transpose(U)- Q)/norm(X2) < reltol 

  # complex case
  U = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; 
  X0 = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
  Q = Matrix(Symmetric(transpose(U)*X0 + transpose(X0)*U))
  @time X = tulyapc!(U, copy(Q); adj = false)  
  @test norm(transpose(U)*X + transpose(X)*U - Q)/norm(X) < reltol 

  LT = tulyaplikeop(U; adj = false)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(transpose(U)*X2 + transpose(X2)*U - Q)/norm(X2) < reltol 


  Q = Matrix(Symmetric(U*transpose(X0) + X0*transpose(U)))
  @time X = tulyapc!(U, copy(Q); adj = true); 
  @test norm(U*transpose(X) + X*transpose(U)- Q)/norm(X) < reltol 

  LT = tulyaplikeop(U; adj = true)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(U*transpose(X2) + X2*transpose(U)- Q)/norm(X) < reltol 

  # complex hermitian case
  U = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); U[1,1] = 0; U[n,n] = 0; 
  X0 = triu(rand(Ty,n,n)+im*rand(Ty,n,n)); X0[1,1] = 0; X0[n,n] = 0
  Q = Matrix(Hermitian(U'*X0 + X0'*U))
  @time X = hulyapc!(U, copy(Q); adj = false)  
  @test norm(U'*X + X'*U - Q)/norm(X) < reltol 

  LT = hulyaplikeop(U; adj = false)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(U'*X2 + X2'*U - Q)/norm(X2) < reltol 

  Q = Matrix(Hermitian(U*X0' + X0*U'))
  @time X = hulyapc!(U, copy(Q); adj = true); 
  @test norm(U*X' + X*U'- Q)/norm(X) < 10*reltol 

  LT = hulyaplikeop(U; adj = true)
  x2,info=MatrixEquations.cgls(LT,triu2vec(Q),reltol=1.e-14,maxiter=1000); X2 = vec2triu(x2);
  @test norm(U*X2' + X2*U'- Q)/norm(X) < reltol 

    
end

end

end
end
