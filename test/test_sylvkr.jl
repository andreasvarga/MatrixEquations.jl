module Test_sylvkr

using LinearAlgebra
using MatrixEquations
using GenericLinearAlgebra
using DoubleFloats
using Test

@testset "Testing Sylvester equation solvers based on Kronecker expansions" begin

n = 5; m = 3;

for Ty in (Float64, BigFloat, Double64)

ar = rand(Ty,n,n)
br = rand(Ty,m,m)
cr = rand(Ty,n,m)
art = rand(Ty,m,n)
brt = rand(Ty,n,m)
crt = rand(Ty,m,m)
crt1 = rand(Ty,m,n)
dr = rand(Ty,n,n)
er = rand(Ty,m,m)
fr = rand(Ty,n,m)
ac = ar+im*rand(Ty,n,n)
bc = br+im*rand(Ty,m,m)
cc = cr+im*rand(Ty,n,m)
act = art+im*rand(Ty,m,n)
bct = brt+im*rand(Ty,n,m)
cct = crt+im*rand(Ty,m,m)
cct1 = crt1+im*rand(Ty,m,n)
dc = dr+im*rand(Ty,n,n)
ec = er+im*rand(Ty,m,m)
fc = fr+im*rand(Ty,n,m)
Qr = cr*cr'
Qr = (Qr+transpose(Qr))/2
Qrss = (Qr-transpose(Qr))/2
Qc = cc*cc'
Qc = (Qc+Qc')/2
Qrt1 = crt1*crt1'
Qrt1 = (Qrt1+transpose(Qrt1))/2
Qcs = (Qc+transpose(Qc))/2
Qcss = (Qc-transpose(Qc))/2
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

# solving Sylvester-like equations
@time x = tsylvckr(art,brt,crt)
@test norm(art*x+transpose(x)*brt-crt)/norm(x) < reltol

@time x = tsylvckr(act,bct,cct)
@test norm(act*x+transpose(x)*bct-cct)/norm(x) < reltol

@time x = hsylvckr(act,bct,cct)
@test norm(act*x+adjoint(x)*bct-cct)/norm(x) < reltol

@time x = csylvckr(ac,bc,cc)
@test norm(ac*x+conj(x)*bc-cc)/norm(x) < reltol

@time x = tsylvdkr(art,brt',crt1)
@test norm(art*transpose(x)*brt'+x-crt1)/norm(x) < reltol

@time x = tsylvdkr(act,bct',cct1)
@test norm(act*transpose(x)*bct'+x-cct1)/norm(x) < reltol

@time x = hsylvdkr(act,bct',cct1)
@test norm(act*adjoint(x)*bct'+x-cct1)/norm(x) < reltol

@time x = csylvdkr(ac,bc,cc)
@test norm(ac*conj(x)*bc+x-cc)/norm(x) < reltol

# solving Lyapunov equations
@time x = sylvckr(ar,ar',-Qr)
@test norm(ar*x+x*ar'+Qr)/norm(x) < reltol

@time x = sylvckr(ac',ac,-Qc)
@test norm(ac'*x+x*ac+Qc)/norm(x) < reltol

@time x = sylvdkr(-ar,ar',Qr)
@test norm(ar*x*ar'-x+Qr)/norm(x) < reltol

@time x = sylvdkr(-ac',ac,Qc)
@test norm(ac'*x*ac-x+Qc)/norm(x) < reltol

# solving Lyapunov-like equations
@time x = tsylvckr(ar,ar',-Qr)
@test norm(ar*x+transpose(x)*ar'+Qr)/norm(x) < reltol

@time x1 = tlyapckr(ar,-Qr)
@test norm(ar*x1+transpose(x1)*ar'+Qr)/norm(x1) < reltol && x ≈ x1

@time x = tsylvckr(art,art',-Qrt1)
@test norm(art*x+transpose(x)*art'+Qrt1)/norm(x) < reltol

@time x1 = tlyapckr(art,-Qrt1)
@test norm(art*x1+transpose(x1)*art'+Qrt1)/norm(x1) < reltol && x ≈ x1

@time x = tsylvckr(ac,transpose(ac),-Qcs)
@test norm(ac*x+transpose(x)*transpose(ac)+Qcs)/norm(x) < reltol

@time x1 = tlyapckr(ac,-Qcs)
@test norm(ac*x1+transpose(x1)*transpose(ac)+Qcs)/norm(x1) < reltol && x ≈ x1

@time x = tsylvckr(ar,-ar',-ar+ar')
@test norm(ar*x-transpose(x)*ar'+ar-ar')/norm(x) < reltol

@time x1 = tlyapckr(ar,-ar+ar',-1)
@test norm(ar*x1-transpose(x1)*ar'+ar-ar')/norm(x1) < reltol && x ≈ x1

@time x = tsylvckr(art,-art',-er+er')
@test norm(art*x-transpose(x)*art'+er-er')/norm(x) < reltol

@time x1 = tlyapckr(art,-er+er',-1)
@test norm(art*x1-transpose(x1)*art'+er-er')/norm(x1) < reltol && x ≈ x1

@time x = tsylvckr(transpose(ac),-ac,-ac+transpose(ac))
@test norm(transpose(ac)*x-transpose(x)*ac+ac-transpose(ac))/norm(x) < reltol

@time x1 = tlyapckr(transpose(ac),-ac+transpose(ac),-1)
@test norm(transpose(ac)*x1-transpose(x1)*ac+ac-transpose(ac))/norm(x1) < reltol && x ≈ x1

@time x = hsylvckr(ac',ac,-Qc)  
@test norm(ac'*x+adjoint(x)*ac+Qc)/norm(x) < reltol

@time x1 = hlyapckr(ac',-Qc)  
@test norm(ac'*x1+adjoint(x1)*ac+Qc)/norm(x1) < reltol && x ≈ x1

@time x = hsylvckr(ac',-ac,-ac+ac')  
@test norm(ac'*x-adjoint(x)*ac+ac-ac')/norm(x) < reltol

@time x1 = hlyapckr(ac',-ac+ac',-1)  
@test norm(ac'*x1-adjoint(x1)*ac+ac-ac')/norm(x1) < reltol && x ≈ x1

@time x = tsylvdkr(-ar,ar',Qr)
@test norm(ar*transpose(x)*ar'-x+Qr)/norm(x) < reltol

@time x = hsylvdkr(-ac',ac,Qc)
@test norm(ac'*adjoint(x)*ac-x+Qc)/norm(x) < reltol

@time x = csylvdkr(-ac',ac,Qc)
@test norm(ac'*conj(x)*ac-x+Qc)/norm(x) < reltol


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
@time x = gsylvkr(ar,dr',dr,ar',-Qr)
@test norm(ar*x*dr'+dr*x*ar'+Qr)/norm(x) < reltol

@time x = gsylvkr(ac',dc,dc',ac,-Qc)
@test norm(ac'*x*dc+dc'*x*ac+Qc)/norm(x) < reltol

@time x = gsylvkr(ar,ar',-dr,dr',-Qr)
@test norm(ar*x*ar'-dr*x*dr'+Qr)/norm(x) < reltol

@time x = gsylvkr(ac',ac,-dc',dc,-Qc)
@test norm(ac'*x*ac-dc'*x*dc+Qc)/norm(x) < reltol

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
