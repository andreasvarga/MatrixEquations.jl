module Test_dplyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing discete positive Lyapunov equation solvers" begin

n = 30
m = 5
p = 3
br = rand(n,m)
bc = br+im*rand(n,m)
cr = rand(p,n)
cc = cr+im*rand(p,n)
reltol = eps(float(100))

@testset "Discrete positive Lyapunov equations" begin

ar = rand(n,n)
ar = ar/(1. + norm(ar))
ac = rand(n,n)+im*rand(n,n)
ac = ac/(1. + norm(ac))

@time u = plyapd(ar,br);
x = u*u'; @test norm(ar*x*ar'-x+br*br')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',cr');
x = u'*u; @test norm(ar'*x*ar-x+cr'*cr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ac,bc);
x = u*u'; @test norm(ac*x*ac'-x+bc*bc')/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cc');
x = u'*u; @test norm(ac'*x*ac-x+cc'*cc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac,br);
x = u*u'; @test norm(ac*x*ac'-x+br*br')/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cr');
x = u'*u; @test norm(ac'*x*ac-x+cr'*cr)/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ar,bc);
x = u*u'; @test norm(ar*x*ar'-x+bc*bc')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',cc');
x = u'*u; @test norm(ar'*x*ar-x+cc'*cc)/norm(x)/max(1.,norm(ar)^2) < reltol

end

@testset "Discrete generalized positive Lyapunov equations" begin

ar = rand(n,n)
ar = ar/(1. + norm(ar))
er = rand(n,n)
ar = er*ar
ac = rand(n,n)+im*rand(n,n)
ac = ac/(1. + norm(ac))
ec = er+im*rand(n,n)
ac = ec*ac

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

@testset "Positive 2x2 discrete Lyapunov equations" begin

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
end


@testset "discete Lyapunov equations - Schur form" begin

ar = rand(n,n)
ar = ar/(1. + norm(ar))
as,  = schur(ar)
ac = rand(n,n)+im*rand(n,n)
ac = ac/(1. + norm(ac))
acs,  = schur(ac)

@time u = plyaps(as,br,disc = true);
x = u*u'; @test norm(as*x*as'-x+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as',cr',disc = true);
x = u'*u; @test norm(as'*x*as-x+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,bc,disc = true);
x = u*u'; @test norm(acs*x*acs'-x+bc*bc')/norm(x)/norm(as) < reltol

@time u = plyaps(acs',cc',disc = true);
x = u'*u; @test norm(acs'*x*acs-x+cc'*cc)/norm(x)/norm(as) < reltol

R = UpperTriangular(rand(n,n))
U = copy(R)
@time plyapds!(as, U, adj = false)
X = U*U'; @test norm(as*X*as'-X+R*R')/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(n,n))
U = copy(R)
@time plyapds!(as,U,adj = true)
X = U'*U; @test norm(as'*X*as-X+R'*R)/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(complex(rand(n,n)))
U = copy(R)
@time plyapds!(acs,U,adj = false)
X = U*U'; @test norm(acs*X*acs'-X+R*R')/max(1,norm(X))/norm(acs) < reltol

R = UpperTriangular(complex(rand(n,n)))
U = copy(R)
@time plyapds!(acs,U,adj = true)
X = U'*U; @test norm(acs'*X*acs-X+R'*R)/max(1,norm(X))/norm(acs) < reltol

er = rand(n,n)
ar = er*ar
as, es = schur(ar,er)
ec = er+im*rand(n,n)
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

R = UpperTriangular(rand(n,n))
U = copy(R)
@time plyapds!(as,es,U,adj = false)
X = U*U'; @test norm(as*X*as'-es*X*es'+R*R')/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(n,n))
U = copy(R)
@time plyapds!(as,es,U,adj = true)
X = U'*U; @test norm(as'*X*as-es'*X*es+R'*R)/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(complex(rand(n,n)))
U = copy(R)
@time plyapds!(acs,ecs,U,adj = false)
X = U*U'; @test norm(acs*X*acs'-ecs*X*ecs'+R*R')/max(1,norm(X))/norm(acs) < reltol

R = UpperTriangular(complex(rand(n,n)))
U = copy(R)
@time plyapds!(acs,ecs,U,adj = true)
X = U'*U; @test norm(acs'*X*acs-ecs'*X*ecs+R'*R)/max(1,norm(X))/norm(acs) < reltol

end
end

end
