module Test_cplyap

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing positive continuous Lyapunov equation solvers" begin

n = 30
m = 5
p = 3
Ty = Float64

@testset "Positive continuous Lyapunov equations" begin

reltol = eps(float(100))
a = -1; b = 2im; @time u = plyapc(a,b)
@test abs(a*u*u'+u*u'*a'+b*b') < reltol

reltol = eps(float(100f0))
a = -1f0; b = 2f0im; @time u = plyapc(a,b)
@test abs(a*u*u'+u*u'*a'+b*b') < reltol


for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar-2*norm(ar)*Matrix(I,n,n)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac-2*norm(ac)*Matrix(I,n,n)
br = rand(Ty,n,m)
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))

@time u = plyapc(ar,br);
x = u*u'; @test norm(ar*x+x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,ar);
x = u*u'; @test norm(ar*x+x*ar'+ar*ar')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,I,br);
x = u*u'; @test norm(ar*x+x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',cr')
x = u'*u; @test norm(ar'*x+x*ar+cr'*cr)/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',ar')
x = u'*u; @test norm(ar'*x+x*ar+ar'*ar)/norm(x)/norm(ar) < reltol

@time u = plyapc(ac,bc);
x = u*u'; @test norm(ac*x+x*ac'+bc*bc')/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',cc');
x = u'*u; @test norm(ac'*x+x*ac+cc'*cc)/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',I,cc');
x = u'*u; @test norm(ac'*x+x*ac+cc'*cc)/norm(x)/norm(ac) < reltol

@time u = plyapc(ac,br);
x = u*u'; @test norm(ac*x+x*ac'+br*br')/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',cr');
x = u'*u; @test norm(ac'*x+x*ac+cr'*cr)/norm(x)/norm(ac) < reltol

@time u = plyapc(ar,bc);
x = u*u'; @test norm(ar*x+x*ar'+bc*bc')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',cc');
x = u'*u; @test norm(ar'*x+x*ar+cc'*cc)/norm(x)/norm(ar)  < reltol

end
end


@testset "Positive continuous generalized Lyapunov equations" begin

reltol = eps(float(100))
a = -1; ee = 4; b = 2im; @time u = plyapc(a,ee,b)
@test abs(a*u*u'*ee'+ee*u*u'*a'+b*b') < reltol

reltol = eps(float(100f0))
a = -1f0; ee = 4f0; b = 2f0im; @time u = plyapc(a,ee,b)
@test abs(a*u*u'*ee'+ee*u*u'*a'+b*b')  < reltol

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar-2*norm(ar)*Matrix(I,n,n)
er = rand(n,n)
ar = er*ar
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac-2*norm(ac)*Matrix(I,n,n)
ec = er+im*rand(Ty,n,n)
ac = ec*ac
br = rand(Ty,n,m)
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapc(ar,er,br);
x = u*u'; @test norm(ar*x*er'+er*x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,er,ar);
x = u*u'; @test norm(ar*x*er'+er*x*ar'+ar*ar')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',er',cr');
x = u'*u; @test norm(ar'*x*er+er'*x*ar+cr'*cr)/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',er',ar');
x = u'*u; @test norm(ar'*x*er+er'*x*ar+ar'*ar)/norm(x)/norm(ar) < reltol

@time u = plyapc(ac,ec,bc);
x = u*u'; @test norm(ac*x*ec'+ec*x*ac'+bc*bc')/norm(x)/norm(ac)/norm(ec) < reltol

@time u = plyapc(ac',ec',cc');
x = u'*u; @test norm(ac'*x*ec+ec'*x*ac+cc'*cc)/norm(x)/norm(ac) < reltol

@time u = plyapc(ar,er,bc);
x = u*u'; @test norm(ar*x*er'+er*x*ar'+bc*bc')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',er',cc');
x = u'*u; @test norm(ar'*x*er+er'*x*ar+cc'*cc)/norm(x)/norm(ar) < reltol

@time u = plyapc(ac,ec,br);
x = u*u'; @test norm(ac*x*ec'+ec*x*ac'+br*br')/norm(x)/norm(ac)/norm(ec) < reltol

@time u = plyapc(ac',ec',cr');
x = u'*u; @test norm(ac'*x*ec+ec'*x*ac+cr'*cr)/norm(x)/norm(ac) < reltol

end
end

@testset "Positive 2x2 continuous Lyapunov equations" begin

A = [-1.1 1.; -1. -1.]
E = [1. 1.; 0. 1.]
R = [1. 1.; 0. 1.]
reltol = eps(float(100))


U, scale, β, α = plyap2(A, R, adj = true, disc = false)
X = U'*U; @test norm(A'*X+X*A+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = plyap2(A, R, adj = false, disc = false)
X = U*U'; @test norm(A*X+X*A'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol


U, scale, β, α = pglyap2(A, E, R, adj = true, disc = false)
X = U'*U; @test norm(A'*X*E+E'*X*A+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

A = [-1.1 1.; -1. -1.]/10; A = convert(Matrix{Float32},A)
E = [1. 1.; 0. 1.]; E = convert(Matrix{Float32},E)
R = [1. 1.; 0. 1.]; R = convert(Matrix{Float32},R)
reltol = eps(100f0)

U, scale, β, α = plyap2(A, R, adj = true, disc = false)
X = U'*U; @test norm(A'*X+X*A+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = plyap2(A, R, adj = false, disc = false)
X = U*U'; @test norm(A*X+X*A'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol


U, scale, β, α = pglyap2(A, E, R, adj = true, disc = false)
X = U'*U; @test norm(A'*X*E+E'*X*A+scale^2*R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E/scale - scale*R)/max(1,norm(scale*R)) < reltol

U, scale, β, α = pglyap2(A, E, R, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A'+scale^2*R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α/scale - scale*R)/max(1,norm(scale*R)) < reltol

end

@testset "Continuous positive Lyapunov equations - Schur form" begin

for Ty in (Float64, Float32)

ar = rand(Ty,n,n)
ar = ar-2*norm(ar)*Matrix(I,n,n)
as,  = schur(ar)
ac = rand(Ty,n,n)+im*rand(Ty,n,n)
ac = ac-2*norm(ac)*Matrix(I,n,n)
acs,  = schur(ac)
br = rand(Ty,n,m)
bc = br+im*rand(Ty,n,m)
cr = rand(Ty,p,n)
cc = cr+im*rand(Ty,p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))

@time u = plyaps(as,br);
x = u*u'; @test norm(as*x+x*as'+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as,ar);
x = u*u'; @test norm(as*x+x*as'+ar*ar')/norm(x)/norm(as) < reltol

@time u = plyaps(as,I,br);
x = u*u'; @test norm(as*x+x*as'+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as',cr');
x = u'*u; @test norm(as'*x+x*as+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(as',ar');
x = u'*u; @test norm(as'*x+x*as+ar'*ar)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,bc);
x = u*u'; @test norm(acs*x+x*acs'+bc*bc')/norm(x)/norm(as) < reltol

@time u = plyaps(acs',cc');
x = u'*u; @test norm(acs'*x+x*acs+cc'*cc)/norm(x)/norm(as) < reltol

@time u = plyaps(acs',I,cc');
x = u'*u; @test norm(acs'*x+x*acs+cc'*cc)/norm(x)/norm(as) < reltol


F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,R,adj = true);
x = R'*R; @test norm(as'*x+x*as+F'*F)/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,I,R,adj = true);
x = R'*R; @test norm(as'*x+x*as+F'*F)/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,R,adj = false);
x = R*R'; @test norm(as*x+x*as'+F*F')/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,R,adj = true);
x = R'*R; @test norm(acs'*x+x*acs+F'*F)/norm(x)/norm(acs) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,R,adj = false);
x = R*R'; @test norm(acs*x+x*acs'+F*F')/norm(x)/norm(acs) < reltol

er = rand(Ty,n,n)
ar = er*ar
as, es = schur(ar,er)
ec = er+im*rand(Ty,n,n)
ac = ec*ac
acs, ecs = schur(ac,ec)

@time u = plyaps(as,es,br);
x = u*u'; @test norm(as*x*es'+es*x*as'+br*br')/norm(x)/norm(as) < reltol

@time u = plyaps(as,es,ar);
x = u*u'; @test norm(as*x*es'+es*x*as'+ar*ar')/norm(x)/norm(as) < reltol

@time u = plyaps(as',es',cr');
x = u'*u; @test norm(as'*x*es+es'*x*as+cr'*cr)/norm(x)/norm(as) < reltol

@time u = plyaps(as',es',ar');
x = u'*u; @test norm(as'*x*es+es'*x*as+ar'*ar)/norm(x)/norm(as) < reltol

@time u = plyaps(acs,ecs,bc);
x = u*u'; @test norm(acs*x*ecs'+ecs*x*acs'+bc*bc')/norm(x)/norm(as) < reltol

@time u = plyaps(acs',ecs',cc');
x = u'*u; @test norm(acs'*x*ecs+ecs'*x*acs+cc'*cc)/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,es,R,adj = true);
x = R'*R; @test norm(as'*x*es+es'*x*as+F'*F)/norm(x)/norm(as) < reltol


F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,es,R,adj = false);
x = R*R'; @test norm(as*x*es'+es*x*as'+F*F')/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,ecs,R,adj = false);
x = R*R'; @test norm(acs*x*ecs'+ecs*x*acs'+F*F')/norm(x)/norm(acs) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,ecs,R,adj = true);
x = R'*R; @test norm(acs'*x*ecs+ecs'*x*acs+F'*F)/norm(x)/norm(acs) < reltol

end

end
end

end
