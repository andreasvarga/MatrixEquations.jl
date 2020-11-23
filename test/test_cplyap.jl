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
brw = rand(Ty,n,n+m)
bc = br+im*rand(Ty,n,m)
bcw = brw+im*rand(Ty,n,n+m)
cr = rand(Ty,p,n)
crt = rand(Ty,n+p,n)
cc = cr+im*rand(Ty,p,n)
cct = crt+im*rand(Ty,n+p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))

@time u = plyapc(ar,br);
x = u*u'; @test norm(ar*x+x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,0*br);
x = u*u'; @test norm(ar*x+x*ar') < reltol

@time u = plyapc(ar,brw);
x = u*u'; @test norm(ar*x+x*ar'+brw*brw')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,I,br);
x = u*u'; @test norm(ar*x+x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',cr')
x = u'*u; @test norm(ar'*x+x*ar+cr'*cr)/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',crt')
x = u'*u; @test norm(ar'*x+x*ar+crt'*crt)/norm(x)/norm(ar) < reltol

@time u = plyapc(ac,bc);
x = u*u'; @test norm(ac*x+x*ac'+bc*bc')/norm(x)/norm(ac) < reltol

@time u = plyapc(ac,0*bc);
x = u*u'; @test norm(ac*x+x*ac') < reltol

@time u = plyapc(ac,bcw);
x = u*u'; @test norm(ac*x+x*ac'+bcw*bcw')/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',cc');
x = u'*u; @test norm(ac'*x+x*ac+cc'*cc)/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',cct');
x = u'*u; @test norm(ac'*x+x*ac+cct'*cct)/norm(x)/norm(ac) < reltol

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
brw = rand(Ty,n,n+m)
bc = br+im*rand(Ty,n,m)
bcw = brw+im*rand(Ty,n,n+m)
cr = rand(Ty,p,n)
crt = rand(Ty,n+p,n)
cc = cr+im*rand(Ty,p,n)
cct = crt+im*rand(Ty,n+p,n)
Ty == Float64 ? reltol = eps(float(100)) : reltol = eps(100*n*one(Ty))


@time u = plyapc(ar,er,br);
x = u*u'; @test norm(ar*x*er'+er*x*ar'+br*br')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar,er,0*br);
x = u*u'; @test norm(ar*x*er'+er*x*ar') < reltol

@time u = plyapc(ar,er,brw);
x = u*u'; @test norm(ar*x*er'+er*x*ar'+brw*brw')/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',er',cr');
x = u'*u; @test norm(ar'*x*er+er'*x*ar+cr'*cr)/norm(x)/norm(ar) < reltol

@time u = plyapc(ar',er',crt');
x = u'*u; @test norm(ar'*x*er+er'*x*ar+crt'*crt)/norm(x)/norm(ar) < reltol

@time u = plyapc(ac,ec,bc);
x = u*u'; @test norm(ac*x*ec'+ec*x*ac'+bc*bc')/norm(x)/norm(ac)/norm(ec) < reltol

@time u = plyapc(ac,ec,0*bc);
x = u*u'; @test norm(ac*x*ec'+ec*x*ac') < reltol

@time u = plyapc(ac,ec,bcw);
x = u*u'; @test norm(ac*x*ec'+ec*x*ac'+bcw*bcw')/norm(x)/norm(ac)/norm(ec) < reltol

@time u = plyapc(ac',ec',cc');
x = u'*u; @test norm(ac'*x*ec+ec'*x*ac+cc'*cc)/norm(x)/norm(ac) < reltol

@time u = plyapc(ac',ec',cct');
x = u'*u; @test norm(ac'*x*ec+ec'*x*ac+cct'*cct)/norm(x)/norm(ac) < reltol

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
R = UpperTriangular([1. 1.; 0. 1.])
reltol = eps(float(100))

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = true, disc = false)
X = U'*U; @test norm(A'*X+X*A+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = false, disc = false)
X = U*U'; @test norm(A*X+X*A'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α - R)/max(1,norm(R)) < reltol

U = copy(R)                
β, α = MatrixEquations.pglyap2!(A, E, U, adj = true, disc = false)
X = U'*U; @test norm(A'*X*E+E'*X*A+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α - R)/max(1,norm(R)) < reltol

U = copy(R)
Q = qr(rand(2,2)).Q; A1 = Q*A; E1 = Q*E;
β, α = MatrixEquations.pglyap2!(A1, E1, U, adj = false, disc = false)
X = U*U'; @test norm(A1*X*E1'+E1*X*A1'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A1*U-E1*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E1*U*α - R)/max(1,norm(R)) < reltol


U = copy(0*R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α - 0*R)/max(1,norm(R)) < reltol


A = [-1.1 1.; -1. -1.]/10; A = convert(Matrix{Float32},A)
E = [1. 1.; 0. 1.]; E = convert(Matrix{Float32},E)
R = [1. 1.; 0. 1.]; R = UpperTriangular(convert(Matrix{Float32},R))
reltol = eps(100f0)

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = true, disc = false)
X = U'*U; @test norm(A'*X+X*A+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.plyap2!(A, U, adj = false, disc = false)
X = U*U'; @test norm(A*X+X*A'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(U*β-A*U)/max(1,norm(U))/norm(A) < reltol &&
                norm(U*α - R)/max(1,norm(R)) < reltol

U = copy(R)                
β, α = MatrixEquations.pglyap2!(A, E, U, adj = true, disc = false)
X = U'*U; @test norm(A'*X*E+E'*X*A+R'*R)/max(1,norm(X))/norm(A) < reltol &&
                norm(β*U*E-U*A)/max(1,norm(U))/norm(A) < reltol &&
                norm(α*U*E - R)/max(1,norm(R)) < reltol

U = copy(R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A'+R*R')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α - R)/max(1,norm(R)) < reltol

U = copy(0*R)
β, α = MatrixEquations.pglyap2!(A, E, U, adj = false, disc = false)
X = U*U'; @test norm(A*X*E'+E*X*A')/max(1,norm(X))/norm(A) < reltol &&
                norm(A*U-E*U*β)/max(1,norm(U))/norm(A) < reltol &&
                norm(E*U*α)/max(1,norm(R)) < reltol

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

#test
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

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,I,R,adj = true);
x = R'*R; @test norm(acs'*x+x*acs+F'*F)/norm(x)/norm(acs) < reltol


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
R = copy(0*F)
@time plyapcs!(as,es,R,adj = true);
x = R'*R; @test norm(as'*x*es+es'*x*as) < reltol


F = UpperTriangular(rand(Ty,n,n))
R = copy(F)
@time plyapcs!(as,es,R,adj = false);
x = R*R'; @test norm(as*x*es'+es*x*as'+F*F')/norm(x)/norm(as) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,ecs,R,adj = false);
x = R*R'; @test norm(acs*x*ecs'+ecs*x*acs'+F*F')/norm(x)/norm(acs) < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(0*F)
@time plyapcs!(acs,ecs,R,adj = false);
x = R*R'; @test norm(acs*x*ecs'+ecs*x*acs') < reltol

F = UpperTriangular(rand(Ty,n,n)+im*rand(Ty,n,n))
R = copy(F)
@time plyapcs!(acs,ecs,R,adj = true);
x = R'*R; @test norm(acs'*x*ecs+ecs'*x*acs+F'*F)/norm(x)/norm(acs) < reltol

end

end
end

end
