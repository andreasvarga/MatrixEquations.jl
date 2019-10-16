module Test_dplyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing positive discrete Lyapunov equation solvers" begin

n = 30
m = 5
p = 3
Ty = Float64


@testset "Positive discrete Lyapunov equations" begin

reltol = eps(float(100))
a = -.1+0.2im; b = 2im; @time u = plyapd(a,b)
@test abs(a*u*u'*a'-u*u'+b*b') < reltol

reltol = eps(float(100f0))
a = -.1f0+-.1f0*im; b = 2f0im; @time u = plyapd(a,b)
@test abs(a*u*u'*a'-u*u'+b*b')  < reltol

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar/(one(Ty) + norm(ar))
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac/(one(Ty) + norm(ac))
br = rand(Ty,n,m)
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapd(ar,br);
x = u*u'; @test norm(ar*x*ar'-x+br*br')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',cr');
x = u'*u; @test norm(ar'*x*ar-x+cr'*cr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ac,bc);
x = u*u'; @test norm(ac*x*ac'-x+bc*bc')/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cc');
x = u'*u; @test norm(ac'*x*ac-x+cc'*cc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cr');
x = u'*u; @test norm(ac'*x*ac-x+cr'*cr)/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ar,bc);
x = u*u'; @test norm(ar*x*ar'-x+bc*bc')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',cc');
x = u'*u; @test norm(ar'*x*ar-x+cc'*cc)/norm(x)/max(1.,norm(ar)^2) < reltol

end
end

@testset "Positive discrete generalized Lyapunov equations" begin

reltol = eps(float(100))
a = -1+im; ee = 4; b = 2im; @time u = plyapd(a,ee,b)
@test abs(a*u*u'*a'-ee*u*u'*ee'+b*b') < reltol

reltol = eps(float(100f0))
a = -1f0+1f0*im; ee = 4f0; b = 2f0im; @time u = plyapd(a,ee,b)
@test abs(a*u*u'*a'-ee*u*u'*ee'+b*b')  < reltol

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar/(one(Ty) + norm(ar))
er = rand(Ty,n,n)
ar = er*ar
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac/(one(Ty) + norm(ac))
br = rand(Ty,n,m)
ec = er+im*rand(n,n)
ac = ec*ac
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapd(ar,er,br);
x = u*u'; @test norm(ar*x*ar'-er*x*er'+br*br')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar,er,br);
x = u*u'; @test norm(ar*x*ar'-er*x*er'+br*br')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar',er',cr');
x = u'*u; @test norm(ar'*x*ar-er'*x*er+cr'*cr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ac,ec,bc);
x = u*u'; @test norm(ac*x*ac'-ec*x*ec'+bc*bc')/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time u = plyapd(ac',ec',cc');
x = u'*u; @test norm(ac'*x*ac-ec'*x*ec+cc'*cc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time u = plyapd(ar,er,bc);
x = u*u'; @test norm(ar*x*ar'-er*x*er'+bc*bc')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar',er',cc');
x = u'*u; @test norm(ar'*x*ar-er'*x*er+cc'*cc)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ac,ec,br);
x = u*u'; @test norm(ac*x*ac'-ec*x*ec'+br*br')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ac',ec',cr');
x = u'*u; @test norm(ac'*x*ac-ec'*x*ec+cr'*cr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol
end
end

@testset "Positive discrete 2x2 Lyapunov equations" begin

A = [-1.1 1.; -1. -1.]/10
E = [1. 1.; 0. 1.]
R = [1. 1.; 0. 1.]
reltol = eps(float(100))


U, scale, β, α = plyap2(A, R, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-X+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = plyap2(A, R, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-X+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-E'*X*E+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-E*X*E'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

A = [-1.1 1.; -1. -1.]/10; A = convert(Matrix{Float32},A)
E = [1. 1.; 0. 1.]; E = convert(Matrix{Float32},E)
R = [1. 1.; 0. 1.]; R = convert(Matrix{Float32},R)
reltol = eps(100f0)

U, scale, β, α = plyap2(A, R, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-X+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = plyap2(A, R, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-X+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-E'*X*E+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-E*X*E'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

end


@testset "Positive discrete Lyapunov equations - Schur form" begin


for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar/(one(Ty) + norm(ar))
as,  = schur(ar)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac/(one(Ty) + norm(ac))
acs,  = schur(ac)
br = rand(Ty,n,m)
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyaps(as,br,disc = true);
x = u*u'; @test norm(as*x*as'-x+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as',cr',disc = true);
x = u'*u; @test norm(as'*x*as-x+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,bc,disc = true);
x = u*u'; @test norm(acs*x*acs'-x+bc*bc')/norm(x)/norm(acs) < reltol

@time u = plyaps(acs',cc',disc = true);
x = u'*u; @test norm(acs'*x*acs-x+cc'*cc)/norm(x)/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n))
U = copy(R)
@time plyapds!(as, U, adj = false)
X = U*U'; @test norm(as*X*as'-X+R*R')/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n))
U = copy(R)
@time plyapds!(as,U,adj = true)
X = U'*U; @test norm(as'*X*as-X+R'*R)/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Complex{Ty},n,n))
U = copy(R)
@time plyapds!(acs,U,adj = false)
X = U*U'; @test norm(acs*X*acs'-X+R*R')/max(1,norm(X))/norm(acs) < reltol

R = UpperTriangular(complex(rand(Ty,n,n)))
U = copy(R)
@time plyapds!(acs,U,adj = true)
X = U'*U; @test norm(acs'*X*acs-X+R'*R)/max(1,norm(X))/norm(acs) < reltol



er = rand(Ty,n,n)
ar = er*ar
as, es = schur(ar,er)
ec = er+im*rand(Ty,n,n)
ac = ec*ac
acs, ecs = schur(ac,ec)

@time u = plyaps(as,es,br,disc = true);
x = u*u'; @test norm(as*x*as'-es*x*es'+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as',es',cr',disc = true);
x = u'*u; @test norm(as'*x*as-es'*x*es+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,ecs,bc,disc = true);
x = u*u'; @test norm(acs*x*acs'-ecs*x*ecs'+bc*bc')/norm(x)/norm(as) < reltol

@time u = plyaps(acs',ecs',cc',disc = true);
x = u'*u; @test norm(acs'*x*acs-ecs'*x*ecs+cc'*cc)/norm(x)/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n))
U = copy(R)
@time plyapds!(as,es,U,adj = false)
X = U*U'; @test norm(as*X*as'-es*X*es'+R*R')/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n))
U = copy(R)
@time plyapds!(as,es,U,adj = true)
X = U'*U; @test norm(as'*X*as-es'*X*es+R'*R)/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Complex{Ty},n,n)) # error
U = copy(R)
@time plyapds!(acs,ecs,U,adj = false)
X = U*U'; @test norm(acs*X*acs'-ecs*X*ecs'+R*R')/max(1,norm(X))/norm(acs) < reltol

R = UpperTriangular(rand(Complex{Ty},n,n)) # error
U = copy(R)
@time plyapds!(acs,ecs,U,adj = true)
X = U'*U; @test norm(acs'*X*acs-ecs'*X*ecs+R'*R)/max(1,norm(X))/norm(acs) < reltol
end
end
end

end
