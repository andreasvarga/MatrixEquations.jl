module Test_sylvkr

using LinearAlgebra
using MatrixEquations
using GenericSchur
using DoubleFloats
using Test

@testset "Testing Sylvester equation solvers based on Kronecker expansions" begin

n = 10; m = 7;

for Ty in (Float64, BigFloat, Double64)

ar = rand(Ty,n,n)
br = rand(Ty,m,m)
cr = rand(Ty,n,m)
dr = rand(Ty,n,n)
er = rand(Ty,m,m)
fr = rand(Ty,n,m)
ac = ar+im*rand(Ty,n,n)
bc = br+im*rand(Ty,m,m)
cc = cr+im*rand(Ty,n,m)
dc = dr+im*rand(Ty,n,n)
ec = er+im*rand(Ty,m,m)
fc = fr+im*rand(Ty,n,m)
qr = cr*cr'
qc = cc*cc'
reltol = sqrt(eps(one(Ty)))

# solving Sylvester equations
@time x = sylvckr(ar,br,cr)
@test norm(ar*x+x*br-cr)/norm(x) < reltol

@time x = sylvckr(ac,bc,cc)
@test norm(ac*x+x*bc-cc)/norm(x) < reltol

@time x = sylvdkr(ar,br,cr)
@test norm(ar*x*br+x-cr)/norm(x) < reltol

@time x = sylvdkr(ac,bc,cc)
@test norm(ac*x*bc+x-cc)/norm(x) < reltol

# solving Lyapunov equations
@time x = sylvckr(ar,ar',-qr)
@test norm(ar*x+x*ar'+qr)/norm(x) < reltol

@time x = sylvckr(ac',ac,-qc)
@test norm(ac'*x+x*ac+qc)/norm(x) < reltol

@time x = sylvdkr(-ar,ar',qr)
@test norm(ar*x*ar'-x+qr)/norm(x) < reltol

@time x = sylvdkr(-ac',ac,qc)
@test norm(ac'*x*ac-x+qc)/norm(x) < reltol


# solving generalized Sylvester equations
@time x = gsylvkr(ar,br,dr,er,cr)
@test norm(ar*x*br+dr*x*er-cr)/norm(x) < reltol

@time x = gsylvkr(ac,bc,dc,ec,cc)
@test norm(ac*x*bc+dc*x*ec-cc)/norm(x) < reltol

@time x = gsylvkr(ar,br',-dr,er',cr)
@test norm(ar*x*br'-dr*x*er'-cr)/norm(x) < reltol

@time x = gsylvkr(ac',bc,-dc,ec',cc)
@test norm(ac'*x*bc-dc*x*ec'-cc)/norm(x) < reltol

# solving generalized Lyapunov equations
@time x = gsylvkr(ar,dr',dr,ar',-qr)
@test norm(ar*x*dr'+dr*x*ar'+qr)/norm(x) < reltol

@time x = gsylvkr(ac',dc,dc',ac,-qc)
@test norm(ac'*x*dc+dc'*x*ac+qc)/norm(x) < reltol

@time x = gsylvkr(ar,ar',-dr,dr',-qr)
@test norm(ar*x*ar'-dr*x*dr'+qr)/norm(x) < reltol

@time x = gsylvkr(ac',ac,-dc',dc,-qc)
@test norm(ac'*x*ac-dc'*x*dc+qc)/norm(x) < reltol

# solving Sylvester systems
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

@time x, y = dsylvsyskr(ar',br',cr,-dr',er',-fr)
@test norm(ar'*x-dr'*y-cr)/max(norm(x),norm(y)) < reltol &&
      norm(x*br'+y*er'+fr)/max(norm(x),norm(y)) < reltol

end
end

end
