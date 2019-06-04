module Test_sylvester

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing Sylvester equation solvers" begin

n = 30; m = 20;
ar = rand(n,n)
br = rand(m,m)
cr = rand(n,m)
dr = rand(n,n)
er = rand(m,m)
fr = rand(n,m)
as, ds = schur(ar,dr)
bs, es = schur(br,er)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
cc = cr+im*rand(n,m)
dc = dr+im*rand(n,n)
ec = er+im*rand(m,m)
fc = fr+im*rand(n,m)
acs, dcs = schur(ac,dc)
bcs, ecs = schur(bc,ec)
reltol = sqrt(eps(1.))


#  continuous Sylvester equations
@testset "Continuous Sylvester equations" begin
@time x = sylvc(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvester(ar,br,cr)
@test norm(ar*x+x*br+cr)/norm(x) < reltol

@time x = sylvckr(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ac,bc,cc)
@test norm(ac*x+x*bc-cc)/norm(x) < reltol

@time x = sylvester(ac,bc,cc)
@test norm(ac*x+x*bc+cc)/norm(x) < reltol

@time x = sylvckr(ac,bc,cc)
@test norm(ac*x+x*bc-cc)/norm(x) < reltol

@time x = sylvc(ar,bc,cr)
@test norm(ar*x+x*bc-cr)/norm(x) < reltol

@time x = sylvc(ar',br,cr)
@test norm(ar'*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ar,-br',cr)
@test norm(ar*x-x*br'-cr)/norm(x) < reltol

@time x = sylvc(ac',bc,cc)
@test norm(ac'*x+x*bc-cc)/norm(x) < reltol

@time x = sylvc(ac',br,cc)
@test norm(ac'*x+x*br-cc)/norm(x) < reltol

@time x = sylvc(ac',br',cc)
@test norm(ac'*x+x*br'-cc)/norm(x) < reltol
end

# discrete Sylvester equations
@testset "Discrete Sylvester equations" begin
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


@testset "Discrete Sylvester equations - Schur form" begin

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

# generalized Sylvester equations
@testset "Generalized Sylvester equations" begin

@time x = gsylv(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = gsylv(ar',br,dr',er,cr)
@test norm(ar'*x*br+dr'*x*er-cr)/norm(x) < reltol

@time x = gsylv(ar,br',dr,er',cr)
@test norm(ar*x*br'+dr*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ar',br',dr',er',cr)
@test norm(ar'*x*br'+dr'*x*er'-cr)/norm(x) < reltol

@time x = gsylv(ac,bc,dc,ec,cc)
@test norm(ac*x*bc+dc*x*ec-cc)/norm(x) < reltol

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

@testset "Generalized Sylvester equations - Schur form" begin

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

# Sylvester systems
@testset "Sylvester systems" begin
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

# dual Sylvester systems
@testset "Dual Sylvester systems" begin
@time x, y = dsylvsys(ar',br',cr,dr',er',fr)
@test norm(ar'*x+dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er'-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar',br',cr,-dr',-er',fr)
@test norm(ar'*x-dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'-y*er'-fr)/max(norm(x),norm(y)) < reltol

@time x, y = dsylvsys(ar,br',cr,dr',er,fr)
@test norm(ar*x+dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er-fr)/max(norm(x),norm(y)) < reltol

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


# LAPACK wrappers of Sylvester system solvers
@testset "LAPACK wrappers of Sylvester system solvers" begin
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

end

end
