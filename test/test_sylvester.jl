module Test_sylvester

using LinearAlgebra
using MatrixEquations
using GenericSchur
using DoubleFloats
using Test
using Random
using LinearAlgebra: BlasFloat

@testset "Testing Sylvester equation solvers" begin

Random.seed!(21235)
n = 15; m = 10; 
Ty = Float64
reltol = sqrt(eps(1.))

#  continuous Sylvester equations
@testset "Continuous Sylvester equations" begin

@time x = sylvc(1,2im,3.)
@test abs(x+x*2*im-3.) < reltol

try 
   @time x = sylvc(ones(1,1),-ones(1,1),ones(1,1)) 
   @test false
catch
   @test true
end

for Ty in (Float64, Float32, BigFloat, Double64)
#  for Ty in (Float64, Float32)


ar = rand(Ty,n,n)
ac = ar+im*rand(Ty,n,n)
br = rand(Ty,m,m)
bc = br+im*rand(Ty,m,m)
cr = rand(Ty,n,m)
cc = cr+im*rand(Ty,n,m)
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))


@time x = sylvc(ar,2I,cr)
@test norm(ar*x+x*2-cr)/norm(x) < reltol

@time x = sylvc(-3. *I,br,cr)
@test norm(-3. *x+x*br-cr)/norm(x) < reltol

@time x = sylvc(-3.,br,cr)
@test norm(-3. *x+x*br-cr)/norm(x) < reltol

@time x = sylvc(2I,3I,cr)
@test norm(2*x+x*3-cr)/norm(x) < reltol

@time x = sylvc(2,3,cr)
@test norm(2*x+x*3-cr)/norm(x) < reltol

@time x = sylvc(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvckr(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ac,bc,cc)
@test norm(ac*x+x*bc-cc)/norm(x) < reltol

if Ty <: LinearAlgebra.BlasFloat
      @time x = sylvester(ar,br,cr)
      @test norm(ar*x+x*br+cr)/norm(x) < reltol

      @time x = sylvester(ac,bc,cc)
      @test norm(ac*x+x*bc+cc)/norm(x) < reltol
end

@time x = sylvckr(ac,bc,cc)
@test norm(ac*x+x*bc-cc)/norm(x) < reltol

@time x = sylvc(ar,bc,cr)
@test norm(ar*x+x*bc-cr)/norm(x) < reltol

@time x = sylvc(ar',br,cr)
@test norm(ar'*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ar,br',cr)
@test norm(ar*x+x*br'-cr)/norm(x) < reltol

@time x = sylvc(ar',br',cr)
@test norm(ar'*x+x*br'-cr)/norm(x) < reltol

@time x = sylvc(ac,bc',cc)
@test norm(ac*x+x*bc'-cc)/norm(x) < reltol

@time x = sylvc(ac',bc,cc)
@test norm(ac'*x+x*bc-cc)/norm(x) < reltol

@time x = sylvc(ac',br,cc)
@test norm(ac'*x+x*br-cc)/norm(x) < reltol

@time x = sylvc(ac',bc',cc)
@test norm(ac'*x+x*bc'-cc)/norm(x) < reltol

@time x = sylvc(ac',br',cc)
@test norm(ac'*x+x*br'-cc)/norm(x) < reltol
end
end

@testset "Continuous Sylvester equations - Schur form" begin

try 
   sylvcs!(ones(1,1),-ones(1,1),ones(1,1)) 
   @test false
catch
   @test true
end

try 
   sylvcs!([1. -1; 1 1],-[1. -1; 1 1],ones(2,2)) 
   @test false
catch
   @test true
end

for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = ar+im*rand(Ty,n,n);
br = rand(Ty,m,m);
bc = br+im*rand(Ty,m,m);
cr = rand(Ty,n,m);
cc = cr+im*rand(Ty,n,m);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))
as, = schur(ar);
bs,  = schur(br);
acs, = schur(ac);
bcs, = schur(bc);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))

y = copy(cr); @time sylvcs!(as,bs,y)
@test norm(as*y+y*bs-cr)/norm(y) < reltol

y = copy(cr); @time sylvcs!(as,bs,y,adjA=true)
@test norm(as'*y+y*bs-cr)/norm(y) < reltol

y = copy(cr); @time sylvcs!(as,bs,y,adjB=true)
@test norm(as*y+y*bs'-cr)/norm(y) < reltol

y = copy(cr); @time sylvcs!(as,bs,y,adjA=true,adjB=true)
@test norm(as'*y+y*bs'-cr)/norm(y) < reltol

y = copy(cc); @time sylvcs!(acs,bcs,y)
@test norm(acs*y+y*bcs-cc)/norm(y) < reltol

y = copy(cc); @time sylvcs!(acs,bcs,y,adjA=true)
@test norm(acs'*y+y*bcs-cc)/norm(y) < reltol

y = copy(cc); @time sylvcs!(acs,bcs,y,adjB=true)
@test norm(acs*y+y*bcs'-cc)/norm(y) < reltol

y = copy(cc); @time sylvcs!(acs,bcs,y,adjA=true,adjB=true)
@test norm(acs'*y+y*bcs'-cc)/norm(y) < reltol

end
end


# discrete Sylvester equations
@testset "Discrete Sylvester equations" begin

try 
   @time x = sylvd(ones(1,1),-ones(1,1),ones(1,1)) 
   @test false
catch
   @test true
end

try 
   @time x = sylvd([1 -1;1 1],-0.5*[1 -1;1 1],ones(2,2)) 
   @test false
catch
   @test true
end


for Ty in (Float64, Float32, BigFloat, Double64)
#  for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = ar+im*rand(Ty,n,n)
br = -rand(Ty,m,m)
bc = br-im*rand(Ty,m,m)
cr = rand(Ty,n,m)
cc = cr+im*rand(Ty,n,m)
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))


@time x = sylvd(Ty(1),Ty(2)*im,Ty(3))
@test abs(x*2*im+x-3.) < reltol

@time x = sylvd(ar,2I,cr)
@test norm(2*ar*x+x-cr)/norm(x) < reltol

@time x = sylvd(-3. *I,br,cr)
@test norm(-3*x*br+x-cr)/norm(x) < reltol

@time x = sylvd(-3.,br,cr)
@test norm(-3*x*br+x-cr)/norm(x) < reltol

@time x = sylvd(2I,3I,cr)
@test norm(2*x*3+x-cr)/norm(x) < reltol

@time x = sylvd(2,3,cr)
@test norm(2*x*3+x-cr)/norm(x) < reltol

@time x = sylvd(ar,br,cr)
@test norm(ar*x*br+x-cr)/norm(x) < reltol

@time x = sylvd(ar',br,cr)
@test norm(ar'*x*br+x-cr)/norm(x) < reltol

@time x = sylvd(ar,br',cr)
@test norm(ar*x*br'+x-cr)/norm(x) < reltol

@time x = sylvd(ar',br',cr)
@test norm(ar'*x*br'+x-cr)/norm(x) < reltol

@time x = sylvd(ac,bc,cc)
@test norm(ac*x*bc+x-cc)/norm(x) < reltol

@time x = sylvd(ac,bc',cc)
@test norm(ac*x*bc'+x-cc)/norm(x) < reltol

@time x = sylvd(ac',bc,cc)
@test norm(ac'*x*bc+x-cc)/norm(x) < reltol

@time x = sylvd(ac',bc',cc)
@test norm(ac'*x*bc'+x-cc)/norm(x) < reltol

@time x = sylvdkr(ac,bc,cc)
@test norm(ac*x*bc+x-cc)/norm(x) < reltol
end
end

@testset "Discrete Sylvester equations - Schur form" begin

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = ar+im*rand(Ty,n,n);
br = rand(Ty,m,m)/10;
bc = br-im*rand(Ty,m,m);
cr = rand(Ty,n,m);
cc = cr+im*rand(Ty,n,m);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))
as, = schur(ar);
bs,  = schur(br);
acs, = schur(ac);
bcs, = schur(bc);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))


y = copy(cr); @time sylvds!(as,bs,y)
@test norm(as*y*bs+y-cr)/norm(y) < reltol

y = copy(cr); @time sylvds!(as,bs,y,adjA=true)
@test norm(as'*y*bs+y-cr)/norm(y) < reltol

y = copy(cr); @time sylvds!(as,bs,y,adjB=true)
@test norm(as*y*bs'+y-cr)/norm(y) < reltol

y = copy(cr); @time sylvds!(as,bs,y,adjA=true,adjB=true)
@test norm(as'*y*bs'+y-cr)/norm(y) < reltol

y = copy(cc); @time sylvds!(acs,bcs,y)
@test norm(acs*y*bcs+y-cc)/norm(y) < reltol

y = copy(cc); @time sylvds!(acs,bcs,y,adjA=true)
@test norm(acs'*y*bcs+y-cc)/norm(y) < reltol

y = copy(cc); @time sylvds!(acs,bcs,y,adjB=true)
@test norm(acs*y*bcs'+y-cc)/norm(y) < reltol

y = copy(cc); @time sylvds!(acs,bcs,y,adjA=true,adjB=true)
@test norm(acs'*y*bcs'+y-cc)/norm(y) < reltol

end
end

# generalized Sylvester equations
@testset "Generalized Sylvester equations" begin
try 
   gsylv(ones(1,1),-ones(1,1),ones(1,1),ones(1,1),ones(1,1)) 
   @test false
catch
   @test true
end

try 
   gsylv([1. -1; 1 1],-[1. -1; 1 1],[1. -1; 1 1],[1. -1; 1 1],ones(2,2)) 
   @test false
catch
   @test true
end

try 
   gsylv([0 -1; 1 0],[1. 0;0 1]',[1 0; 0 1],-[0 -1; 1 0]',ones(2,2))
   @test false
catch
   @test true
end


for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = ar+im*rand(Ty,n,n);
br = rand(Ty,m,m);
bc = br+im*rand(Ty,m,m);
cr = rand(Ty,n,m);
cc = cr+im*rand(Ty,n,m);
dr = rand(Ty,n,n);
er = rand(Ty,m,m);
dc = dr+im*rand(Ty,n,n);
ec = er+im*rand(Ty,m,m);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))

@time x = gsylv(ar,br,cr)
@test norm(ar*x*br-cr)/norm(x) < reltol

@time x = gsylv(2I,br,cr)
@test norm(2*x*br-cr)/norm(x) < reltol

@time x = gsylv(ar,2I,cr)
@test norm(ar*x*2-cr)/norm(x) < reltol

@time x = gsylv(2,br,cr)
@test norm(2*x*br-cr)/norm(x) < reltol

@time x = gsylv(2,1//2,cr)
@test norm(2*x/2-cr)/norm(x) < reltol

@time x = gsylv(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = gsylv(ar,br,2I,cr)
@test norm(ar*x*br+2*x-cr)/norm(x) < reltol

@time x = gsylv(ar,br,2im,cr)
@test norm(ar*x*br+2*im*x-cr)/norm(x) < reltol

@time x = gsylv(ar,br,2*im*I,im,cr)
@test norm(ar*x*br-2*x-cr)/norm(x) < reltol

@time x = gsylv(ar,br,2*im*I,er,cr)
@test norm(ar*x*br+2*im*x*er-cr)/norm(x) < reltol

@time x = gsylv(ar,br,dr,2*im*I,cr)
@test norm(ar*x*br+2*im*dr*x-cr)/norm(x) < reltol

@time x = gsylv(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = gsylv(ac,bc,dc,ec,cc)
@test norm(ac*x*bc+dc*x*ec-cc)/norm(x) < reltol

@time x = gsylv(ar',br,dr',er,cr)
@test norm(ar'*x*br+dr'*x*er-cr)/norm(x) < reltol

@time x = gsylv(ar,br',dr,er',cr)
@test norm(ar*x*br'+dr*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ar',br',dr',er',cr)
@test norm(ar'*x*br'+dr'*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ar,br',dr,er',cr)
@test norm(ar*x*br'+dr*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ar',br',dr',er',cr)
@test norm(ar'*x*br'+dr'*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ar',br,dr',er,cr)
@test norm(ar'*x*br+dr'*x*er-cr)/norm(x) < reltol

@time x = gsylv(ac',bc,-dc,ec',cc)
@test norm(ac'*x*bc-dc*x*ec'-cc)/norm(x) < reltol

@time x = gsylvkr(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = gsylvkr(ac,bc,dc,ec,cc)
@test norm(ac*x*bc+dc*x*ec-cc)/norm(x) < reltol

@time x = gsylvkr(ar,br',-dr,er',cr)
@test norm(ar*x*br'-dr*x*er'-cr)/norm(x) < reltol

@time x = gsylvkr(ac',bc,-dc,ec',cc)
@test norm(ac'*x*bc-dc*x*ec'-cc)/norm(x) < reltol
end
end

@testset "Generalized Sylvester equations - Schur form" begin

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ac = ar+im*rand(Ty,n,n);
br = rand(Ty,m,m);
bc = br+im*rand(Ty,m,m);
cr = rand(Ty,n,m);
cc = cr+im*rand(Ty,n,m);
dr = rand(Ty,n,n);
er = rand(Ty,m,m);
dc = dr+im*rand(Ty,n,n);
ec = er+im*rand(Ty,m,m);
as = schur(ar).T
ds = triu(dr)
as = ds*as
bs = schur(br).T
es = triu(er)
bs = es*bs
acs, dcs = schur(ac,dc);
bcs, ecs = schur(bc,ec);
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))

y = copy(cr); @time gsylvs!(as,bs,ds,es,y)
@test norm(as*y*bs+ds*y*es-cr)/norm(y) < reltol

y = copy(cr); @time gsylvs!(as,bs,ds,es,y,adjAC=true)
@test norm(as'*y*bs+ds'*y*es-cr)/norm(y) < reltol

y = copy(cr); @time gsylvs!(as,bs,ds,es,y,adjBD=true)
@test norm(as*y*bs'+ds*y*es'-cr)/norm(y) < reltol

y = copy(cr); @time gsylvs!(as,bs,ds,es,y,adjAC=true,adjBD=true)
@test norm(as'*y*bs'+ds'*y*es'-cr)/norm(y) < reltol

y = copy(cc); @time gsylvs!(acs,bcs,dcs,ecs,y)
@test norm(acs*y*bcs+dcs*y*ecs-cc)/norm(y) < reltol

y = copy(cc); @time gsylvs!(acs,bcs,dcs,ecs,y,adjBD=true)
@test norm(acs*y*bcs'+dcs*y*ecs'-cc)/norm(y) < reltol

y = copy(cc); @time gsylvs!(acs,bcs,dcs,ecs,y,adjAC=true)
@test norm(acs'*y*bcs+dcs'*y*ecs-cc)/norm(y) < reltol

y = copy(cc); @time gsylvs!(acs,bcs,dcs,ecs,y,adjAC=true,adjBD=true)
@test norm(acs'*y*bcs'+dcs'*y*ecs'-cc)/norm(y) < reltol
end
end

# Sylvester systems
@testset "Sylvester systems" begin

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = ar+im*rand(Ty,n,n)
br = rand(Ty,m,m)
bc = br+im*rand(Ty,m,m)
cr = rand(Ty,n,m)
cc = cr+im*rand(Ty,n,m)
dr = rand(Ty,n,n)
er = rand(Ty,m,m)
dc = dr+im*rand(Ty,n,n)
ec = er+im*rand(Ty,m,m)
fr = rand(Ty,n,m)
fc = fr+im*rand(Ty,n,m)
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))


@time x, y = sylvsys(ar,br,cr,dr,er,fr)
@test norm(ar*x+y*br-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr*x+y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsys(ar,-br,cr,dr,-er,fr)
@test norm(ar*x-y*br-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr*x-y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsys(ac,bc,cc,dc,ec,fc)
@test norm(ac*x+y*bc-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dc*x+y*ec-fc)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsys(ac,-bc,cc,dc,-ec,fc)
@test norm(ac*x-y*bc-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dc*x-y*ec-fc)/max(norm(x),norm(y)) < reltol


@time x, y = sylvsys(ac,bc,cc,dc,ec,fr)
@test norm(ac*x+y*bc-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dc*x+y*ec-fr)/max(norm(x),norm(y)) < reltol


@time x, y = sylvsys(ar,br,cr,dr,er,fc)
@test norm(ar*x+y*br-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr*x+y*er-fc)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsys(ar,br',cr,dr',er,fr)
@test norm(ar*x+y*br'-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr'*x+y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsyskr(ar,br,cr,dr,er,fr)
@test norm(ar*x+y*br-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr*x+y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsyskr(ar,-br,cr,dr,-er,fr)
@test norm(ar*x-y*br-cr)/max(norm(x),norm(y)) < reltol &&
      norm(dr*x-y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsyskr(ac,bc,cc,dc,ec,fc)
@test norm(ac*x+y*bc-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dc*x+y*ec-fc)/max(norm(x),norm(y)) < reltol

@time x, y = sylvsyskr(ac,-bc,cc,dc,-ec,fc)
@test norm(ac*x-y*bc-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dc*x-y*ec-fc)/max(norm(x),norm(y)) < reltol

end
end

# dual Sylvester systems
@testset "Dual Sylvester systems" begin

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = ar+im*rand(Ty,n,n)
br = rand(Ty,m,m)
bc = br+im*rand(Ty,m,m)
cr = rand(Ty,n,m)
cc = cr+im*rand(Ty,n,m)
dr = rand(Ty,n,n)
er = rand(Ty,m,m)
dc = dr+im*rand(Ty,n,n)
ec = er+im*rand(Ty,m,m)
fr = rand(Ty,n,m)
fc = fr+im*rand(Ty,n,m)
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))


@time x, y = dsylvsys(ar,br,cr,dr,er,fr)
@test norm(ar*x+dr*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br+y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar',br',cr,dr',er',fr)
@test norm(ar'*x+dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er'-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar',br',cr,-dr',-er',fr)
@test norm(ar'*x-dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'-y*er'-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar,br',cr,dr',er,fr)
@test norm(ar*x+dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ac,bc,cc,dc,ec,fc)
@test norm(ac*x+dc*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bc+y*ec-fc)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ac',bc',cc,dc',ec',fc)
@test norm(ac'*x+dc'*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bc'+y*ec'-fc)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ac',bc',cc,-dc',-ec',fc)
@test norm(ac'*x-dc'*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bc'-y*ec'-fc)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ac',bc,cc,dc,ec',fc)
@test norm(ac'*x+dc*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bc+y*ec'-fc)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar',bc,cc,dr,ec',fc)
@test norm(ar'*x+dr*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bc+y*ec'-fc)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsyskr(ar',br',cr,dr',er',fr)
@test norm(ar'*x+dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er'-fr)/max(norm(x),norm(y)) < reltol

end
end


# LAPACK wrappers of Sylvester system solvers
@testset "LAPACK wrappers of Sylvester system solvers" begin

for Ty in (Float64, Float32, BigFloat, Double64)
#for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ac = ar+im*rand(Ty,n,n)
br = rand(Ty,m,m)
bc = br+im*rand(Ty,m,m)
cr = rand(Ty,n,m)
cc = cr+im*rand(Ty,n,m)
dr = rand(Ty,n,n)
er = rand(Ty,m,m)
dc = dr+im*rand(Ty,n,n)
ec = er+im*rand(Ty,m,m)
fr = rand(Ty,n,m)
fc = fr+im*rand(Ty,n,m)
as = schur(ar).T
ds = triu(dr)
bs = schur(br).T
es = triu(er)
acs, dcs = schur(ac,dc)
bcs, ecs = schur(bc,ec)
Ty == Float64 ? reltol = eps(float(10*n*m)) : reltol = eps(10*n*m*one(Ty))

if Ty <: LinearAlgebra.BlasFloat

   x = copy(cr); y = copy(fr);
   @time x, y, scale =  tgsyl!(as, bs, x, ds, es, y)
   @test norm(as*x-y*bs-cr)/max(norm(x),norm(y)) < reltol &&
         norm(ds*x-y*es-fr)/max(norm(x),norm(y)) < reltol
   
   x = copy(cr); y = copy(fr);
   @time x, y, scale =  tgsyl!('T',as, bs, x, ds, es, y)
   @test norm(as'*x+ds'*y-cr)/max(norm(x),norm(y)) < reltol &&
         norm(x*bs'+y*es'+fr)/max(norm(x),norm(y)) < reltol  

   x = copy(cc); y = copy(fc);
   @time x, y, scale =  tgsyl!(acs, bcs, x, dcs, ecs, y)
   @test norm(acs*x-y*bcs-cc)/max(norm(x),norm(y)) < reltol &&
         norm(dcs*x-y*ecs-fc)/max(norm(x),norm(y)) < reltol
   
   x = copy(cc); y = copy(fc);
   @time x, y, scale =  tgsyl!('C',acs, bcs, x, dcs, ecs, y)
   @test norm(acs'*x+dcs'*y-cc)/max(norm(x),norm(y)) < reltol &&
         norm(x*bcs'+y*ecs'+fc)/max(norm(x),norm(y)) < reltol
   
   x = copy(cr); y = copy(fr);
   @time x, y, scale =  tgsyl!(as, bs, x, ds, es, y)
   @test norm(as*x-y*bs-cr)/max(norm(x),norm(y)) < reltol &&
         norm(ds*x-y*es-fr)/max(norm(x),norm(y)) < reltol
   
   x = copy(cr); y = copy(fr);
   @time x, y, scale =  tgsyl!('T',as, bs, x, ds, es, y)
   @test norm(as'*x+ds'*y-cr)/max(norm(x),norm(y)) < reltol &&
         norm(x*bs'+y*es'+fr)/max(norm(x),norm(y)) < reltol
   
   
   x = copy(cc); y = copy(fc);
   @time x, y, scale =  tgsyl!(acs, bcs, x, dcs, ecs, y)
   @test norm(acs*x-y*bcs-cc)/max(norm(x),norm(y)) < reltol &&
         norm(dcs*x-y*ecs-fc)/max(norm(x),norm(y)) < reltol
   
   x = copy(cc); y = copy(fc);
   @time x, y, scale =  tgsyl!('C',acs, bcs, x, dcs, ecs, y)
   @test norm(acs'*x+dcs'*y-cc)/max(norm(x),norm(y)) < reltol &&
         norm(x*bcs'+y*ecs'+fc)/max(norm(x),norm(y)) < reltol
   
end

x = copy(cr); y = copy(fr);
@time x, y =  sylvsyss!(as, bs, x, ds, es, y)
@test norm(as*x+y*bs-cr)/max(norm(x),norm(y)) < reltol &&
      norm(ds*x+y*es-fr)/max(norm(x),norm(y)) < reltol

x = copy(cr); y = copy(fr);
@time x, y =  dsylvsyss!(false,as, bs, x, ds, es, y)
@test norm(as*x+ds*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*bs+y*es-fr)/max(norm(x),norm(y)) < reltol

x = copy(cr); y = copy(fr);
@time x, y =  dsylvsyss!(true,as, bs, x, ds, es, y)
@test norm(as'*x+ds'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*bs'+y*es'-fr)/max(norm(x),norm(y)) < reltol

x = copy(cc); y = copy(fc);
@time x, y =  sylvsyss!(acs, bcs, x, dcs, ecs, y)
@test norm(acs*x+y*bcs-cc)/max(norm(x),norm(y)) < reltol &&
      norm(dcs*x+y*ecs-fc)/max(norm(x),norm(y)) < reltol

x = copy(cc); y = copy(fc);
@time x, y =  dsylvsyss!(false,acs, bcs, x, dcs, ecs, y)
@test norm(acs*x+dcs*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bcs+y*ecs-fc)/max(norm(x),norm(y)) < reltol

x = copy(cc); y = copy(fc);
@time x, y =  dsylvsyss!(true,acs, bcs, x, dcs, ecs, y)
@test norm(acs'*x+dcs'*y-cc)/max(norm(x),norm(y)) < reltol &&
      norm(x*bcs'+y*ecs'-fc)/max(norm(x),norm(y)) < reltol

end
end

end

end
