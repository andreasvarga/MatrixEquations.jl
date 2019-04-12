module Test_sylvester

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing Sylvester equation solvers" begin

n = 30; m = 40;
ar = rand(n,n)
br = rand(m,m)
cr = rand(n,m)
dr = rand(n,n)
er = rand(m,m)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
ars, ura = schur(ar)
brs, urb = schur(br)
acs, uca = schur(ac)
bcs, ucb = schur(bc)
c = cr+im*rand(n,m)
dc = dr+im*rand(n,n)
ec = er+im*rand(m,m)
reltol = sqrt(eps(1.))


@time x = sylvc(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvester(ar,br,cr)
@test norm(ar*x+x*br+cr)/norm(x) < reltol

@time x = sylvckr(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ac,bc,c)
@test norm(ac*x+x*bc-c)/norm(x) < reltol

@time x = sylvester(ac,bc,c)
@test norm(ac*x+x*bc+c)/norm(x) < reltol

@time x = sylvckr(ac,bc,c)
@test norm(ac*x+x*bc-c)/norm(x) < reltol

@time x = sylvc(ar,bc,cr)
@test norm(ar*x+x*bc-cr)/norm(x) < reltol

@time x = sylvc(ar',br,cr)
@test norm(ar'*x+x*br-cr)/norm(x) < reltol

@time x = sylvc(ac',bc,c)
@test norm(ac'*x+x*bc-c)/norm(x) < reltol

@time x = sylvc(ac',br,c)
@test norm(ac'*x+x*br-c)/norm(x) < reltol

@time x = sylvc(ac',br',c)
@test norm(ac'*x+x*br'-c)/norm(x) < reltol

@time x = sylvdkr(ar,br,cr)
@test norm(ar*x*br-x-cr)/norm(x) < reltol

@time x = sylvdkr(ac,bc,c)
@test norm(ac*x*bc-x-c)/norm(x) < reltol

@time x = sylvdkr(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = sylvdkr(ac,bc,dc,ec,c)
@test norm(ac*x*bc+dc*x*ec-c)/norm(x) < reltol

@time x = sylvdkr(ar,br',dr,er',cr,-1)
@test norm(ar*x*br'-dr*x*er'-cr)/norm(x) < reltol

@time x = sylvdkr(ac',bc,dc,ec',c,-1)
@test norm(ac'*x*bc-dc*x*ec'-c)/norm(x) < reltol

end

end
