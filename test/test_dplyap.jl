module Test_dplyap

using LinearAlgebra
using MatrixEquations
using GenericSchur
using DoubleFloats
using Test

println("Test_dplyap")
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

for Ty in (Float64, Float32, BigFloat, Double64)
# for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar/(one(Ty) + norm(ar))
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac/(one(Ty) + norm(ac))
br = rand(Ty,n,m)
brw = rand(Ty,n,n+m)
bc = br+im*rand(Ty,n,m)
bcw = brw+im*rand(Ty,n,n+m)
cr = rand(Ty,p,n)
crt = rand(Ty,n+p,n)
cc = cr+im*rand(Ty,p,n)
cct = crt+im*rand(Ty,n+p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapd(ar,br);
x = u*u'; @test norm(ar*x*ar'-x+br*br')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar,0*br);
x = u*u'; @test norm(ar*x*ar'-x) < reltol

@time u = plyapd(ar,brw);
x = u*u'; @test norm(ar*x*ar'-x+brw*brw')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar,I,br);
x = u*u'; @test norm(ar*x*ar'-x+br*br')/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',cr');
x = u'*u; @test norm(ar'*x*ar-x+cr'*cr)/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ar',crt');
x = u'*u; @test norm(ar'*x*ar-x+crt'*crt)/norm(x)/max(1.,norm(ar)^2) < reltol

@time u = plyapd(ac,bc);
x = u*u'; @test norm(ac*x*ac'-x+bc*bc')/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac,0*bc);
x = u*u'; @test norm(ac*x*ac'-x) < reltol

@time u = plyapd(ac,bcw);
x = u*u'; @test norm(ac*x*ac'-x+bcw*bcw')/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cc');
x = u'*u; @test norm(ac'*x*ac-x+cc'*cc)/norm(x)/max(1.,norm(ac)^2) < reltol

@time u = plyapd(ac',cct');
x = u'*u; @test norm(ac'*x*ac-x+cct'*cct)/norm(x)/max(1.,norm(ac)^2) < reltol

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

for Ty in (Float64, Float32, BigFloat, Double64)
# for Ty in (Float64, Float32)
    
ar = rand(Ty,n,n)
ar = ar/(one(Ty) + norm(ar))
er = rand(Ty,n,n)
ar = er*ar
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac/(one(Ty) + norm(ac))
ec = er+im*rand(Ty,n,n)
ac = ec*ac
br = rand(Ty,n,m)
brw = rand(Ty,n,n+m)
bc = br+im*rand(Ty,n,m)
bcw = brw+im*rand(Ty,n,n+m)
cr = rand(Ty,p,n)
crt = rand(Ty,n+p,n)
cc = cr+im*rand(Ty,p,n)
cct = crt+im*rand(Ty,n+p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapd(ar',er',cr');
x = u'*u; @test norm(ar'*x*ar-er'*x*er+cr'*cr)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar',er',crt');
x = u'*u; @test norm(ar'*x*ar-er'*x*er+crt'*crt)/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar,er,br);
x = u*u'; @test norm(ar*x*ar'-er*x*er'+br*br')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ar,er,0*br);
x = u*u'; @test norm(ar*x*ar'-er*x*er') < reltol

@time u = plyapd(ar,er,brw);
x = u*u'; @test norm(ar*x*ar'-er*x*er'+brw*brw')/norm(x)/max(norm(ar)^2,norm(er)^2) < reltol

@time u = plyapd(ac,ec,bc);
x = u*u'; @test norm(ac*x*ac'-ec*x*ec'+bc*bc')/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time u = plyapd(ac,ec,0*bc);
x = u*u'; @test norm(ac*x*ac'-ec*x*ec') < reltol

@time u = plyapd(ac,ec,bcw);
x = u*u'; @test norm(ac*x*ac'-ec*x*ec'+bcw*bcw')/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time u = plyapd(ac',ec',cc');
x = u'*u; @test norm(ac'*x*ac-ec'*x*ec+cc'*cc)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

@time u = plyapd(ac',ec',cct');
x = u'*u; @test norm(ac'*x*ac-ec'*x*ec+cct'*cct)/norm(x)/max(norm(ac)^2,norm(ec)^2) < reltol

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


U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = true, disc = true)
#U, scale, β, α = plyap2(A, R, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-X+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-X+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-E'*X*E+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-E*X*E'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α - R)/max(1,norm(R)) < reltol

U = copy(R)
Q = qr(rand(2,2)).Q; A1 = Q*A; E1 = Q*E;
β, α = MatrixEquations.pglyap2!(A1, E1, U, adj = false, disc = true)
X = U*U'; @test norm(A1*X*A1'-E1*X*E1'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A1*U-E1*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E1*U*α - R)/max(1,norm(R)) < reltol

U = copy(0*R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = true)
X = U*U'; @test norm(A*X*E'+E*X*A')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α)/max(1,norm(R)) < reltol


A = [-1.1 1.; -1. -1.]/10; A = convert(Matrix{Float32},A)
E = [1. 1.; 0. 1.]; E = convert(Matrix{Float32},E)
R = [1. 1.; 0. 1.]; R = convert(Matrix{Float32},R)
reltol = eps(100f0)

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-X+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-X+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = true, disc = true)
X = U'*U; @test norm(A'*X*A-E'*X*E+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = true)
X = U*U'; @test norm(A*X*A'-E*X*E'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α - R)/max(1,norm(R)) < reltol

end


@testset "Positive discrete Lyapunov equations - Schur form" begin


for Ty in (Float64, Float32, BigFloat, Double64)
# for Ty in (Float64, Float32)

ar = rand(Ty,n,n);
ar = ar/(one(Ty) + norm(ar));
as,  = schur(ar);
ac = rand(Ty,n,n)+im*rand(Ty,n,n);
ac = ac/(one(Ty) + norm(ac));
acs,  = schur(ac);
br = rand(Ty,n,m);
bc = br+im*rand(Ty,n,m);
cr = rand(Ty,p,n);
cc = cr+im*rand(Ty,p,n);
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyaps(as,br,disc = true);
x = u*u'; @test norm(as*x*as'-x+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as,0*br,disc = true);
x = u*u'; @test norm(as*x*as'-x) < reltol

@time u = plyaps(as,ar,disc = true);
x = u*u'; @test norm(as*x*as'-x+ar*ar')/norm(x)/norm(as) < reltol

@time u = plyaps(as',cr',disc = true);
x = u'*u; @test norm(as'*x*as-x+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(as',ar',disc = true);
x = u'*u; @test norm(as'*x*as-x+ar'*ar)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,bc,disc = true);
x = u*u'; @test norm(acs*x*acs'-x+bc*bc')/norm(x)/norm(acs) < reltol

@time u = plyaps(acs,0*bc,disc = true);
x = u*u'; @test norm(acs*x*acs'-x) < reltol

@time u = plyaps(acs',cc',disc = true);
x = u'*u; @test norm(acs'*x*acs-x+cc'*cc)/norm(x)/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n));
U = copy(R);
@time plyapds!(as, U, adj = false);
X = U*U'; @test norm(as*X*as'-X+R*R')/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Ty,n,n));
U = copy(R);
@time plyapds!(as,U,adj = true);
X = U'*U; @test norm(as'*X*as-X+R'*R)/max(1,norm(X))/norm(as) < reltol

R = UpperTriangular(rand(Complex{Ty},n,n))
U = copy(R)
@time plyapds!(acs,U,adj = false)
X = U*U'; @test norm(acs*X*acs'-X+R*R')/max(1,norm(X))/norm(acs) < reltol

R = UpperTriangular(complex(rand(Ty,n,n)))
U = copy(R)
@time plyapds!(acs,U,adj = true)
X = U'*U; @test norm(acs'*X*acs-X+R'*R)/max(1,norm(X))/norm(acs) < reltol



as = schur(ar).T
er = rand(Ty,n,n)
ar = er*ar
es = triu(er)
as = es*as
#as, es = schur(ar,er)
ec = er+im*rand(Ty,n,n)
ac = ec*ac
acs, ecs = schur(ac,ec)


@time u = plyaps(as,es,br,disc = true);
x = u*u'; @test norm(as*x*as'-es*x*es'+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as,es,ar,disc = true);
x = u*u'; @test norm(as*x*as'-es*x*es'+ar*ar')/norm(x)/norm(as) < reltol

@time u = plyaps(as',es',cr',disc = true);
x = u'*u; @test norm(as'*x*as-es'*x*es+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(as',es',ar',disc = true);
x = u'*u; @test norm(as'*x*as-es'*x*es+ar'*ar)/norm(x)/norm(as) < reltol

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
