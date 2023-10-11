module Test_MEcondest

using LinearAlgebra
using LinearMaps
using MatrixEquations
using Test

# function check_ctranspose(op::LinearMaps.LinearMap{T}) where {T <: Union{AbstractFloat, Complex}}
#    (m, n) = size(op)
#    x = rand(T,n)
#    y = rand(T,m)
#    yAx = dot(y, op * x)
#    xAty = dot(x, op' * y)
#    ε = eps(real(eltype(op)))
#    return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1 / 3)
# end
function check_ctranspose(op::LinearMaps.LinearMap) 
   (m, n) = size(op)
   T1 = promote_type(Float64,eltype(op))
   x = rand(T1,n)
   y = rand(T1,m)
   yAx = dot(y, op * x)
   xAty = dot(x, op' * y)
   ε = eps(real(T1))
   return abs(yAx - conj(xAty)) < (abs(yAx) + ε) * ε^(1 / 3)
end
   
    
println("Test_MEcondest")

#@testset "Testing Lyapunov and Sylvester operators" begin

n = 10; m = 5;
reltol = sqrt(eps(10.));
sc = 0.1;

@testset "Continuous Lyapunov operators" begin

#n = 3
ar = rand(n,n);
cr = rand(n,m);
cc = cr+im*rand(n,m);
cr = cr*cr';
as,  = schur(ar);
ac = ar+im*rand(n,n);
cc = cc*cc';
acs,  = schur(ac);
ur = triu(ar)
uc = triu(ac)

Tcr = lyapop(ar);
Tcrinv = invlyapop(ar);
Tcrs = lyapop(as);
Tcrsinv = invlyapop(as);
Tcrsym = lyapop(ar,her=true);
Tcrsyminv = invlyapop(ar,her=true);
Tcrssym = lyapop(as,her=true);
Tcrssyminv = invlyapop(as,her=true);

# define some T/H-Lyapunov operators for upper triangular arguments
Ttur = tulyapop(ur)
Ttaur = tulyapop(transpose(ur))
Ttuc = tulyapop(uc)
Ttauc = tulyapop(transpose(uc))
Thur = hulyapop(ur)
Thaur = hulyapop(ur')
Thuc = hulyapop(uc)
Thauc = hulyapop(uc')


Tcc = lyapop(ac);
Tccinv = invlyapop(ac);
Tccs = lyapop(acs);
Tccsinv = invlyapop(acs);
Tccsym = lyapop(ac,her=true);
Tccsyminv = invlyapop(ac,her=true);
Tccssym = lyapop(acs,her=true);
Tccssyminv = invlyapop(acs,her=true);
Π = trmatop(n); MPn = Matrix(Π)
@test issymmetric(Π) && ishermitian(Π)
@test size(Tcr) == size(Tcr') == size(Tcrinv)
Ln = eliminationop(n); MLn = Matrix(Ln)
Dn = duplicationop(n); MDn = Matrix(Dn)
@test MPn == MDn*MDn' + MLn'*MLn*MPn*MLn'*MLn - I
@test (Ln'*Ln*Π*Ln'*Ln)*vec(ar) == vec(Diagonal(ar))


@test check_ctranspose(Tcr) &&
      check_ctranspose(Tcrinv) &&
      check_ctranspose(Tcrs) &&
      check_ctranspose(Tcrsinv) &&
      check_ctranspose(Tcrsym) &&
      Matrix(Tcrsyminv') ≈  Matrix(Tcrsyminv)' &&
      Matrix(Tcrssyminv') ≈  Matrix(Tcrssyminv)' &&
      #check_ctranspose(Tcrsyminv) &&
      #check_ctranspose(Tcrssyminv) &&
      check_ctranspose(Tcc) &&
      check_ctranspose(Tccinv) &&
      check_ctranspose(Tccs) &&
      check_ctranspose(Tccsinv) &&
      Matrix(Tccsinv') ≈  Matrix(Tccsinv)'
      #check_ctranspose(Tccsym) &&
      #check_ctranspose(Tccsyminv) &&
      #check_ctranspose(Tccssyminv) &&
      #check_ctranspose(Tccssyminv) 

@test opnorm1(invlyapop([ 0 0;0 0])) == Inf

try
   T1 = invlyapop([1 0;0 1])
   T1 = invlyapop(convert(Matrix{Float32},as))
   T = invlyapop(as);
   T*rand(n*n);
   transpose(T)*rand(n*n);
   adjoint(T)*rand(n*n);
   T*rand(Float32,n*n);
   transpose(T)*rand(Float32,n*n);
   adjoint(T)*rand(Float32,n*n);
   T*ones(Int,n*n);
   transpose(T)*ones(Int,n*n);
   adjoint(T)*ones(Int,n*n);
   T*ones(Rational{Int},n*n);
   T1 = invlyapop(convert(Matrix{Complex{Float32}},acs));
   T = invlyapop(acs);
   T*rand(n*n);
   T*complex(rand(n*n));
   transpose(T)*rand(n*n);
   transpose(T)*complex(rand(n*n));
   adjoint(T)*rand(n*n);
   adjoint(T)*complex(rand(n*n));
   lyapop(schur(ar))
   lyapop(2)
   @test true
catch
   @test false
end 
 

@time x = Tcrinv*cr[:];
@test norm(Tcr*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrinv*cc[:];
@test norm(Tcr*x-cc[:])/norm(x[:]) < reltol

@time x = Tcrsyminv*triu2vec(cr);
@test norm(Tcrsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tcrinv)*cr[:];
@test norm(transpose(Tcr)*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tcrinv)*cc[:];
@test norm(transpose(Tcr)*x-cc[:])/norm(x[:]) < reltol

@time x = Tcrinv'*cc[:];
@test norm(Tcr'*x-cc[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsyminv)*triu2vec(cr);
@test norm(transpose(Tcrsym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tcrsinv*cr[:];
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cc[:];
@test norm(Tcrs*x-cc[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cr[:];
@test norm(transpose(Tcrs)*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cc[:];
@test norm(transpose(Tcrs)*x-cc[:])/norm(x[:]) < reltol


@time x = Tcrsinv'*cr[:];
@test norm(Tcrs'*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv'*cc[:];
@test norm(Tcrs'*x-cc[:])/norm(x[:]) < reltol


@time x = Tcrssyminv*triu2vec(cr);
@test norm(Tcrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tcrssyminv)*triu2vec(cr);
@test norm(transpose(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccinv*cc[:];
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol

@time x = Tccsyminv*triu2vec(cr);
@test norm(Tccsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tccinv'*cc[:];
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

# @time x = Tccsyminv'*triu2vec(cr);
# @test norm(Tccsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccsinv*cc[:];
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time x = Tccssyminv*triu2vec(cr);
@test norm(Tccssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:];
@test norm(transpose(Tcr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tcrsinv)*cr[:];
@test norm(transpose(Tcrs)*y-cr[:])/norm(y[:]) < reltol

@test abs(norm(Matrix(Tcr)'-Matrix(transpose(Tcr)))) < reltol 
@test abs(norm(Matrix(Tcc)'-Matrix(Tcc'))) < reltol 
@test norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol 
@test norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol 
@test norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol 
@test norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol  
@test opnorm1est(Π*Tcr-Tcr*Π) < reltol 
@test opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol 
@test opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol 
@test opnorm(Matrix(Tcr),1) ≈ opnorm1(Tcr) 
@test opnorm1(Tcrsinv-invlyapop(as')') == 0  
@test opnorm1(Tcrsinv-invlyapop(schur(ar))) == 0 

@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)


@test sc*oprcondest(Tcr,Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(Tcrs,Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(Tcc,Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv) &&
      oprcondest(Tcr) == oprcondest(Tcr,Tcrinv) == oprcondest(Tcrinv) 

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.]))) == 0.  &&
      opsepest(invlyapop([0. 1.; 0. 1.])) == 0.  &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.]))) == 0.


# some tests for T/H-Lyapunov operators with upper triangular arguments
@test transpose(Matrix(Ttur)) == Matrix(transpose(Ttur)) &&
      transpose(Matrix(Ttaur)) == Matrix(transpose(Ttaur)) &&
      transpose(Matrix(Ttuc)) == Matrix(transpose(Ttuc)) &&
      transpose(Matrix(Ttauc)) == Matrix(transpose(Ttauc))
@test adjoint(Matrix(Ttur)) == Matrix(adjoint(Ttur)) &&
      adjoint(Matrix(Ttaur)) == Matrix(adjoint(Ttaur)) &&
      adjoint(Matrix(Ttuc)) == Matrix(adjoint(Ttuc)) &&
      adjoint(Matrix(Ttauc)) == Matrix(adjoint(Ttauc))


end

@testset "Continuous generalized Lyapunov operators" begin

ar = rand(n,n)
er = rand(n,n)
cr = rand(n,m)
cc = cr+im*rand(n,m)
cr = cr*cr'
as,es  = schur(ar,er)
ac = ar+im*rand(n,n)
ec = er+im*rand(n,n)
cc = cc*cc'
acs, ecs  = schur(ac,ec)

Tcr = lyapop(ar,er)
Tcrinv = invlyapop(ar,er)
Tcrs = lyapop(as,es)
Tcrsinv = invlyapop(as,es)
Tcrsym = lyapop(ar,er,her=true);
Tcrsyminv = invlyapop(ar,er,her=true);
Tcrssym = lyapop(as,es,her=true);
Tcrssyminv = invlyapop(as,es,her=true);


Tcc = lyapop(ac,ec)
Tccinv = invlyapop(ac,ec)
Tccs = lyapop(acs,ecs)
Tccsinv = invlyapop(acs,ecs)
Tccsym = lyapop(ac,ec,her=true);
Tccsyminv = invlyapop(ac,ec,her=true);
Tccssym = lyapop(acs,ecs,her=true);
Tccssyminv = invlyapop(acs,ecs,her=true);
Π = trmatop(size(ar))
@test size(Tcr) == size(Tcr') == size(Tcrinv)

@test opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol*norm(ar)*norm(er) 


@test check_ctranspose(Tcr) &&
      check_ctranspose(Tcrinv) &&
      check_ctranspose(Tcrs) &&
      check_ctranspose(Tcrsinv) &&
      #check_ctranspose(Tcrsym) &&
      #check_ctranspose(Tcrsyminv) &&
      #check_ctranspose(Tcrssyminv) &&
      #check_ctranspose(Tcrssyminv) &&
      check_ctranspose(Tcc) &&
      check_ctranspose(Tccinv) &&
      check_ctranspose(Tccs) &&
      check_ctranspose(Tccsinv)  
      #check_ctranspose(Tccsym) &&
      #check_ctranspose(Tccsyminv) &&
      #check_ctranspose(Tccssyminv) &&
      #check_ctranspose(Tccssyminv) 


try
   T1 = invlyapop(convert(Matrix{Complex{Float32}},acs),ecs);
   T = invlyapop(acs,ecs);
   T*rand(n*n);
   transpose(T)*rand(n*n);
   adjoint(T)*rand(n*n);
   T*rand(Float32,n*n);
   transpose(T)*rand(Float32,n*n);
   adjoint(T)*rand(Float32,n*n);
   T*ones(Int,n*n);
   transpose(T)*ones(Int,n*n);
   adjoint(T)*ones(Int,n*n);
   T1 = invlyapop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},ecs));
   T = invlyapop(acs,ecs);
   T*rand(n*n);
   T*complex(rand(n*n));
   transpose(T)*rand(n*n);
   transpose(T)*complex(rand(n*n));
   T'*rand(n*n);
   T'*complex(rand(n*n));
   lyapop(schur(ar,er))
   lyapop(2,3)
   lyapop(rand(Int,2,2),rand(2,2))
   @test true
catch
   @test false
end 

@time x = Tcrinv*cr[:];
@test norm(Tcr*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrinv*cc[:];
@test norm(Tcr*x-cc[:])/norm(x[:]) < reltol

@time x = Tcrsyminv*triu2vec(cr);
@test norm(Tcrsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tcrsyminv*triu2vec(cc);
@test norm(Tcrsym*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = transpose(Tcrinv)*cr[:];
@test norm(transpose(Tcr)*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tcrinv)*cc[:];
@test norm(transpose(Tcr)*x-cc[:])/norm(x[:]) < reltol

@time x = adjoint(Tcrinv)*cc[:];
@test norm(adjoint(Tcr)*x-cc[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsyminv)*triu2vec(cr);
@test norm(transpose(Tcrsym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tcrsyminv)*triu2vec(cc); 
@test norm(transpose(Tcrsym)*x-triu2vec(cc))/norm(x[:]) < reltol   

@time x = Tcrsinv*cr[:];
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cr[:];
@test norm(transpose(Tcrs)*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrssyminv*triu2vec(cr);
@test norm(Tcrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tcrssyminv)*triu2vec(cr);
@test norm(transpose(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = adjoint(Tcrssyminv)*triu2vec(cr);
@test norm(adjoint(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccinv*cc[:];
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol

@time x = Tccsyminv*triu2vec(cr);
@test norm(Tccsym*x-triu2vec(cr))/norm(x[:]) < reltol  

@time y = Tccinv'*cc[:];
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

@time x = Tccsyminv'*triu2vec(cr);
@test norm(Tccsym'*x-triu2vec(cr))/norm(x[:]) < reltol 

@time x = Tccsinv*cc[:];
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time x = Tccssyminv*triu2vec(cr);
@test norm(Tccssym*x-triu2vec(cr))/norm(x[:]) < reltol 

@time y = Tcrinv'*cr[:];
@test norm(Tcr'*y-cr[:])/norm(y[:]) < reltol

@time y = Tcrsinv'*cr[:];
@test norm(Tcrs'*y-cr[:])/norm(y[:]) < reltol

@test abs(norm(Matrix(Tcr)'-Matrix(transpose(Tcr)))) < reltol
@test abs(norm(Matrix(Tcc)'-Matrix(Tcc'))) < reltol  
@test norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol 
@test norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol 
@test norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol 
@test norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol 
@test opnorm1est(Π*Tcr-Tcr*Π) < reltol 
@test opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol*norm(ar)*norm(er) 
@test opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol*norm(ar)*norm(er) 
@test opnorm(Matrix(Tcr),1) ≈ opnorm1(Tcr) 
@test opnorm1(Tcrsinv-invlyapop(as',es')') == 0 
@test opnorm1(Tcrsinv-invlyapop(schur(ar,er))) == 0

@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)


@test sc*oprcondest(Tcr,Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(Tcrs,Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(Tcc,Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv) &&
      oprcondest(Tcr) == oprcondest(Tcr,Tcrinv) == oprcondest(Tcrinv) 

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.])) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.])) == 0.  &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.  &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.

end


@testset "Discrete Lyapunov operators" begin

ar = rand(n,n);
cr = rand(n,m);
cc = cr+im*rand(n,m);
cr = cr*cr';
as,  = schur(ar);
ac = ar+im*rand(n,n);
cc = cc*cc';
acs,  = schur(ac);

Tdr = lyapop(ar,disc=true);
Tdrinv = invlyapop(ar,disc=true);
Tdrs = lyapop(as,disc=true);
Tdrsinv = invlyapop(as,disc=true);
Tdrsym = lyapop(ar,disc=true,her=true);
Tdrsyminv = invlyapop(ar,disc=true,her=true);
Tdrssym = lyapop(as,disc=true,her=true);
Tdrssyminv = invlyapop(as,disc=true,her=true);


Tdc = lyapop(ac,disc=true);
Tdcinv = invlyapop(ac,disc=true);
Tdcs = lyapop(acs,disc=true);
Tdcsinv = invlyapop(acs,disc=true);
Tdcsym = lyapop(ac,disc=true,her=true);
Tdcsyminv = invlyapop(ac,disc=true,her=true);
Tdcssym = lyapop(acs,disc=true,her=true);
Tdcssyminv = invlyapop(acs,disc=true,her=true);
Π = trmatop(ar)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 

try
   T1 = invlyapop(convert(Matrix{Float32},as))
   T = invlyapop(as,disc=true);
   T*rand(n*n);
   transpose(T)*rand(n*n);
   adjoint(T)*rand(n*n);
   T*rand(Float32,n*n);
   transpose(T)*rand(Float32,n*n);
   adjoint(T)*rand(Float32,n*n);
   T*ones(Int,n*n);
   transpose(T)*ones(Int,n*n);
   adjoint(T)*ones(Int,n*n);
   T1 = invlyapop(convert(Matrix{Complex{Float32}},acs));
   T = invlyapop(acs,disc=true);
    T*rand(n*n);
    T*complex(rand(n*n));
    transpose(T)*rand(n*n);
    transpose(T)*complex(rand(n*n));
    adjoint(T)*rand(n*n);
    adjoint(T)*complex(rand(n*n));
    @test true
catch
    @test false
end 


@time x = Tdrinv*cr[:];
@test norm(Tdr*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrinv*cc[:];
@test norm(Tdr*x-cc[:])/norm(x[:]) < reltol

@time x = Tdrsyminv*triu2vec(cr);
@test norm(Tdrsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tdrinv)*cr[:];
@test norm(transpose(Tdr)*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsyminv'*triu2vec(cr);
@test norm(Tdrsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrsyminv'*triu2vec(cc);
@test norm(Tdrsym'*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrssyminv*triu2vec(cr);
@test norm(Tdrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tdrsyminv)*triu2vec(cr);
@test norm(transpose(Tdrsym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrssyminv'*triu2vec(cr);
@test norm(Tdrssym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdcinv*cc[:]
@test norm(Tdc*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcsyminv*triu2vec(cr);
@test norm(Tdcsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y-cc[:])/norm(y[:]) < reltol

# @time x = Tdcsyminv'*triu2vec(cr);
# @test norm(Tdcsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcssyminv*triu2vec(cr);
@test norm(Tdcssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*y-cr[:])/norm(y[:]) < reltol


@test abs(norm(Matrix(Tdr)'-Matrix(transpose(Tdr)))) < reltol
@test abs(norm(Matrix(Tdc)'-Matrix(Tdc'))) < reltol 
@test norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol 
@test norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol 
@test norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol 
@test norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol 
@test opnorm1est(Π*Tdr-Tdr*Π) < reltol 
@test opnorm1est(Π*Tdrinv-Tdrinv*Π) < reltol 
@test opnorm1est(Π*Tdrsinv-Tdrsinv*Π) < reltol


@test sc*opnorm1(Tdr) < opnorm1est(Tdr)  &&
      sc*opnorm1(Tdrinv) < opnorm1est(Tdrinv)  &&
      sc*opnorm1(Tdrs) < opnorm1est(Tdrs)  &&
      sc*opnorm1(Tdrsinv) < opnorm1est(Tdrsinv)  &&
      sc*opnorm1(Tdc) < opnorm1est(Tdc)  &&
      sc*opnorm1(Tdcinv) < opnorm1est(Tdcinv)  &&
      sc*opnorm1(Tdcs) < opnorm1est(Tdcs)  &&
      sc*opnorm1(Tdcsinv) < opnorm1est(Tdcsinv)


@test sc*oprcondest(Tdr,Tdrinv) < 1/opnorm1(Tdr)/opnorm1(Tdrinv)  &&
      sc*oprcondest(Tdrs,Tdrsinv) < 1/opnorm1(Tdrs)/opnorm1(Tdrsinv)  &&
      sc*oprcondest(Tdc,Tdcinv) < 1/opnorm1(Tdc)/opnorm1(Tdcinv)  &&
      sc*oprcondest(Tdcs,Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv) &&
      oprcondest(Tdr) == oprcondest(Tdr,Tdrinv) == oprcondest(Tdrinv) 


@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],disc=true)) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],disc=true))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],disc=true))) == 0.  &&
      opsepest(invlyapop([0. 1.; 0. 1.],disc=true)) == 0.  &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],disc=true))) == 0.  &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],disc=true))) == 0.


end

@testset "Discrete generalized Lyapunov operators" begin

ar = rand(n,n)
er = rand(n,n)
cr = rand(n,m)
cc = cr+im*rand(n,m)
cr = cr*cr'
as,es  = schur(ar,er)
ac = ar+im*rand(n,n)
ec = er+im*rand(n,n)
cc = cc*cc'
acs, ecs  = schur(ac,ec)

Tdr = lyapop(ar,er,disc=true)
Tdrinv = invlyapop(ar,er,disc=true)
Tdrs = lyapop(as,es,disc=true)
Tdrsinv = invlyapop(as,es,disc=true)
Tdrsym = lyapop(ar,er,disc=true,her=true);
Tdrsyminv = invlyapop(ar,er,disc=true,her=true);
Tdrssym = lyapop(as,es,disc=true,her=true);
Tdrssyminv = invlyapop(as,es,disc=true,her=true);


Tdc = lyapop(ac,ec,disc=true)
Tdcinv = invlyapop(ac,ec,disc=true)
Tdcs = lyapop(acs,ecs,disc=true)
Tdcsinv = invlyapop(acs,ecs,disc=true)
Tdcsym = lyapop(ac,ec,disc=true,her=true);
Tdcsyminv = invlyapop(ac,ec,disc=true,her=true);
Tdcssym = lyapop(acs,ecs,disc=true,her=true);
Tdcssyminv = invlyapop(acs,ecs,disc=true,her=true);
Π = trmatop(n)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 

try
   #T1 = invlyapsop(convert(Matrix{Complex{Float32}},acs),ecs);
   T = invlyapop(acs,ecs,disc = true);
   T*rand(n*n);
   transpose(T)*rand(n*n);
   adjoint(T)*rand(n*n);
   T*rand(Float32,n*n);
   transpose(T)*rand(Float32,n*n);
   adjoint(T)*rand(Float32,n*n);
   T*ones(Int,n*n);
   transpose(T)*ones(Int,n*n);
   adjoint(T)*ones(Int,n*n);
   #T1 = invlyapsop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},ecs));
   T = invlyapop(acs,ecs,disc = true);
   T*rand(n*n);
   T*complex(rand(n*n));
   transpose(T)*rand(n*n);
   transpose(T)*complex(rand(n*n));
   T'*rand(n*n);
   T'*complex(rand(n*n));
   @test true
catch
   @test false
end 


@time x = Tdrinv*cr[:];
@test norm(Tdr*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsyminv*triu2vec(cr);
@test norm(Tdrsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tdrinv)*cr[:];
@test norm(transpose(Tdr)*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsyminv'*triu2vec(cr);
@test norm(Tdrsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:];
@test norm(Tdrs*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrssyminv*triu2vec(cr);
@test norm(Tdrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrssyminv*triu2vec(cc);
@test norm(Tdrssym*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = transpose(Tdrsyminv)*triu2vec(cr);
@test norm(transpose(Tdrsym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrssyminv'*triu2vec(cr);
@test norm(Tdrssym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrssyminv'*triu2vec(cc);
@test norm(Tdrssym'*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = Tdcinv*cc[:];
@test norm(Tdc*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcsyminv*triu2vec(cr);
@test norm(Tdcsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tdcinv'*cc[:];
@test norm(Tdc'*y-cc[:])/norm(y[:]) < reltol

@time x = Tdcsyminv'*triu2vec(cr);
@test norm(Tdcsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcssyminv*triu2vec(cr);
@test norm(Tdcssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:];
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tdrsinv)*cr[:];
@test norm(transpose(Tdrs)*y-cr[:])/norm(y[:]) < reltol

@test abs(norm(Matrix(Tdr)'-Matrix(transpose(Tdr)))) < reltol 
@test abs(norm(Matrix(Tdc)'-Matrix(Tdc'))) < reltol  
@test norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol 
@test norm(Matrix(Tdrsyminv)*Matrix(Tdrsym)-I) < reltol 
@test norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol 
@test norm(Matrix(Tdrssyminv)*Matrix(Tdrssym)-I) < reltol 
@test norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol 
@test norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol 
@test opnorm1est(Π*Tdr-Tdr*Π) < reltol 
@test opnorm1est(Π*Tdrinv-Tdrinv*Π) < reltol 
@test opnorm1est(Π*Tdrsinv-Tdrsinv*Π) < reltol


@test sc*opnorm1(Tdr) < opnorm1est(Tdr)  &&
      sc*opnorm1(Tdrinv) < opnorm1est(Tdrinv)  &&
      sc*opnorm1(Tdrs) < opnorm1est(Tdrs)  &&
      sc*opnorm1(Tdrsinv) < opnorm1est(Tdrsinv)  &&
      sc*opnorm1(Tdc) < opnorm1est(Tdc)  &&
      sc*opnorm1(Tdcinv) < opnorm1est(Tdcinv)  &&
      sc*opnorm1(Tdcs) < opnorm1est(Tdcs)  &&
      sc*opnorm1(Tdcsinv) < opnorm1est(Tdcsinv)


@test sc*oprcondest(Tdr,Tdrinv) < 1/opnorm1(Tdr)/opnorm1(Tdrinv)  &&
      sc*oprcondest(Tdrs,Tdrsinv) < 1/opnorm1(Tdrs)/opnorm1(Tdrsinv)  &&
      sc*oprcondest(Tdc,Tdcinv) < 1/opnorm1(Tdc)/opnorm1(Tdcinv)  &&
      sc*oprcondest(Tdcs,Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv) &&
      oprcondest(Tdr) == oprcondest(Tdr,Tdrinv) == oprcondest(Tdrinv) 

@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0. &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0.  &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.  &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.


end



#  continuous and discrete Sylvester equations
@testset "Continuous Sylvester operators" begin

# n = 3; m = 2; 
ar = rand(n,n)
br = rand(m,m)
cr = rand(n,m)
as,  = schur(ar)
bs,  = schur(br)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
cc = cr+im*rand(n,m)
acs,  = schur(ac)
bcs,  = schur(bc)

Tcr = sylvop(ar, br)
Tcrinv = invsylvop(ar,br)
Tcrs = sylvop(as, bs)
Tcrsinv = invsylvop(as,bs)


Tcc = sylvop(ac, bc)
Tccinv = invsylvop(ac,bc)
Tccs = sylvop(acs, bcs)
Tccsinv = invsylvop(acs,bcs)

@test size(Tcr) == size(Tcr')

@test check_ctranspose(Tcr) &&
      check_ctranspose(Tcrinv) &&
      check_ctranspose(Tcrs) &&
      check_ctranspose(Tcrsinv) && 
      check_ctranspose(Tcc) &&
      check_ctranspose(Tccinv) &&
      check_ctranspose(Tccs) &&
      check_ctranspose(Tccsinv) 


try
    T = invsylvop(as,bs);
    T*rand(n*m);
    transpose(T)*rand(n*m);
    adjoint(T)*rand(n*m);
    T*rand(Float32,n*m);
    transpose(T)*rand(Float32,n*m);
    adjoint(T)*rand(Float32,n*m);
    T*ones(Int,n*m);
    transpose(T)*ones(Int,n*m);
    adjoint(T)*ones(Int,n*m);
    T = invsylvop(acs,bcs);
    T*rand(n*m);
    T*complex(rand(n*m));
    transpose(T)*rand(n*m);
    transpose(T)*complex(rand(n*m));
    T'*rand(n*m);
    T'*complex(rand(n*m));
    sylvop(schur(rand(3,3)),schur(rand(3,3)))
    sylvop(1,im)
    invsylvop(1,im)
    sylvop(rand(Int,2,2),rand(2,2))
    invsylvop(rand(Int,2,2),rand(2,2))
    @test true
catch
    @test false
end 
 
@time x = Tcrinv*cr[:];
@test norm(Tcr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cr[:];
@test norm(Tcrs*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as,bs)*cr[:];
@test norm(sylvop(as, bs)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as,bs')*cr[:];
@test norm(sylvop(as, bs')*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as',bs)*cr[:];
@test norm(sylvop(as', bs)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as',bs')*cr[:];
@test norm(sylvop(as', bs')*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as,bs))*cr[:];
@test norm(transpose(sylvop(as, bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as,bs'))*cr[:];
@test norm(transpose(sylvop(as, bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as',bs))*cr[:];
@test norm(transpose(sylvop(as', bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as',bs'))*cr[:];
@test norm(transpose(sylvop(as', bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as,bs))*cr[:];
@test norm(adjoint(sylvop(as, bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as,bs'))*cr[:];
@test norm(adjoint(sylvop(as, bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as',bs))*cr[:];
@test norm(adjoint(sylvop(as', bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as',bs'))*cr[:];
@test norm(adjoint(sylvop(as', bs'))*x[:]-cr[:])/norm(x[:]) < reltol


@time x = Tccinv*cc[:];
@test norm(Tcc*x[:]-cc[:])/norm(x[:]) < reltol

@time x = Tccsinv*cc[:];
@test norm(Tccs*x[:]-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:];
@test norm(transpose(Tcr)*y[:]-cr[:])/norm(y[:]) < reltol

@time y = Tccinv'*cc[:];
@test norm(Tcc'*y[:]-cc[:])/norm(y[:]) < reltol

x = rand(n*m);
@test abs(norm(Matrix(Tcr)'-Matrix(transpose(Tcr)))) < reltol 
@test abs(norm(Matrix(Tcc)'-Matrix(Tcc'))) < reltol 
@test norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol 
@test norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol 
@test norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol 
@test norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol 
@test norm(Tcrsinv'*x-invsylvop(as',bs')*x) == 0  
@test opnorm1(Tcrsinv-invsylvop(schur(ar),schur(br))) == 0 


@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)

@test sc*oprcondest(Tcr,Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(Tcrs,Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(Tcc,Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv) &&
      oprcondest(Tcr) == oprcondest(Tcr,Tcrinv) == oprcondest(Tcrinv) 

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0.

end

# discrete Sylvester equations
@testset "Discrete Sylvester operators" begin

ar = rand(n,n)
br = rand(m,m)
cr = rand(n,m)
as,  = schur(ar)
bs,  = schur(br)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
cc = cr+im*rand(n,m)
acs,  = schur(ac)
bcs,  = schur(bc)

Tdr = sylvop(ar, br, disc = true)
Tdrinv = invsylvop(ar,br, disc = true)
Tdrs = sylvop(as, bs, disc = true)
Tdrsinv = invsylvop(as,bs, disc = true)

Tdc = sylvop(ac, bc, disc = true)
Tdcinv = invsylvop(ac,bc, disc = true)
Tdcs = sylvop(acs, bcs, disc = true)
Tdcsinv = invsylvop(acs,bcs, disc = true)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 

try
    T = invsylvop(as,bs,disc = true);
    T*rand(n*m);
    transpose(T)*rand(n*m);
    adjoint(T)*rand(n*m);
    T*rand(Float32,n*m);
    transpose(T)*rand(Float32,n*m);
    adjoint(T)*rand(Float32,n*m);
    T*ones(Int,n*m);
    transpose(T)*ones(Int,n*m);
    adjoint(T)*ones(Int,n*m);
    T = invsylvop(acs,bcs,disc = true);
    T*rand(n*m);
    T*complex(rand(n*m));
    transpose(T)*rand(n*m);
    transpose(T)*complex(rand(n*m));
    T'*rand(n*m);
    T'*complex(rand(n*m));
    @test true
catch
    @test false
end 

@time x = Tdrinv*cr[:]
@test norm(Tdr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as,bs, disc = true)*cr[:]
@test norm(sylvop(as, bs, disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as,bs', disc = true)*cr[:]
@test norm(sylvop(as, bs', disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as',bs, disc = true)*cr[:]
@test norm(sylvop(as', bs, disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvop(as',bs', disc = true)*cr[:]
@test norm(sylvop(as', bs', disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as,bs, disc = true))*cr[:]
@test norm(transpose(sylvop(as, bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as,bs', disc = true))*cr[:]
@test norm(transpose(sylvop(as, bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as',bs, disc = true))*cr[:]
@test norm(transpose(sylvop(as', bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as',bs', disc = true))*cr[:]
@test norm(transpose(sylvop(as', bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as,bs, disc = true))*cr[:]
@test norm(adjoint(sylvop(as, bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as,bs', disc = true))*cr[:]
@test norm(adjoint(sylvop(as, bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as',bs, disc = true))*cr[:]
@test norm(adjoint(sylvop(as', bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as',bs', disc = true))*cr[:]
@test norm(adjoint(sylvop(as', bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tdcinv*cc[:]
@test norm(Tdc*x[:]-cc[:])/norm(x[:]) < reltol

@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x[:]-cc[:])/norm(x[:]) < reltol

@time x = Tdcsinv'*cr[:]
@test norm(Tdcs'*x[:]-cr[:])/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y[:]-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*y[:]-cr[:])/norm(y[:]) < reltol

@time x = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*x[:]-cr[:])/norm(x[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y[:]-cc[:])/norm(y[:]) < reltol

@time y = Tdcsinv'*cc[:]
@test norm(Tdcs'*y[:]-cc[:])/norm(y[:]) < reltol


x = rand(n*m);
@test abs(norm(Matrix(Tdr)'-Matrix(transpose(Tdr)))) < reltol 
@test abs(norm(Matrix(Tdc)'-Matrix(Tdc'))) < reltol 
@test norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol 
@test norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol 
@test norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol 
@test norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol 
@test norm(Tdrsinv'*x-invsylvop(as',bs', disc = true)*x) == 0 
@test opnorm1(Tdrsinv-invsylvop(schur(ar),schur(br), disc = true)) == 0



@test sc*opnorm1(Tdr) < opnorm1est(Tdr)  &&
      sc*opnorm1(Tdrinv) < opnorm1est(Tdrinv)  &&
      sc*opnorm1(Tdrs) < opnorm1est(Tdrs)  &&
      sc*opnorm1(Tdrsinv) < opnorm1est(Tdrsinv)  &&
      sc*opnorm1(Tdc) < opnorm1est(Tdc)  &&
      sc*opnorm1(Tdcinv) < opnorm1est(Tdcinv)  &&
      sc*opnorm1(Tdcs) < opnorm1est(Tdcs)  &&
      sc*opnorm1(Tdcsinv) < opnorm1est(Tdcsinv)

@test sc*oprcondest(Tdr,Tdrinv) < 1/opnorm1(Tdr)/opnorm1(Tdrinv)  &&
      sc*oprcondest(Tdrs,Tdrsinv) < 1/opnorm1(Tdrs)/opnorm1(Tdrsinv)  &&
      sc*oprcondest(Tdc,Tdcinv) < 1/opnorm1(Tdc)/opnorm1(Tdcinv)  &&
      sc*oprcondest(Tdcs,Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv) &&
      oprcondest(Tdr) == oprcondest(Tdr,Tdrinv) == oprcondest(Tdrinv) 



@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.],disc=true)) == 0. &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.],disc=true)) == 0.

end


# generalized Sylvester equations
@testset "Generalized Sylvester operators" begin

ar = rand(n,n)
br = rand(m,m)
cr = rand(n,n)
dr = rand(m,m)
er = rand(n,m)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
cc = cr+im*rand(n,n)
dc = dr+im*rand(m,m)
ec = er+im*rand(n,m)
as, cs = schur(ar,cr)
bs, ds = schur(br,dr)
acs, ccs = schur(ac,cc)
bcs, dcs = schur(bc,dc)

Tr = sylvop(ar, br, cr, dr)
Trinv = invsylvop(ar, br, cr, dr)
Trs = sylvop(as, bs, cs, ds)
Trsinv = invsylvop(as, bs, cs, ds)
Trs1 = sylvop(as, ds, cs, bs)
Trs1inv = invsylvop(as, ds, cs, bs)

Tc = sylvop(ac, bc, cc, dc)
Tcinv = invsylvop(ac, bc, cc, dc)
Tcs = sylvop(acs, bcs, ccs, dcs)
Tcsinv = invsylvop(acs, bcs, ccs, dcs)
@test size(Tr) == size(Tr') == size(Trinv') 
   
@test check_ctranspose(Tr) &&   
      check_ctranspose(Trinv) &&  
      check_ctranspose(Trs) &&
      check_ctranspose(Trsinv) && 
      check_ctranspose(Tc) &&
      check_ctranspose(Tcinv) &&
      check_ctranspose(Tcs) &&
      check_ctranspose(Tcsinv) 


try
    T = invsylvop(as,bs,cs,ds);
    T*rand(n*m);
    transpose(T)*rand(n*m);
    adjoint(T)*rand(n*m);
    T*rand(Float32,n*m);
    transpose(T)*rand(Float32,n*m);
    adjoint(T)*rand(Float32,n*m);
    T*ones(Int,n*m);
    transpose(T)*ones(Int,n*m);
    adjoint(T)*ones(Int,n*m);
    T = invsylvop(acs,bcs,ccs,dcs);
    T*rand(n*m);
    T*complex(rand(n*m));
    transpose(T)*rand(n*m);
    transpose(T)*complex(rand(n*m));
    T'*rand(n*m);
    T'*complex(rand(n*m));
    sylvop(ar,complex(br),cr,dr);
    invsylvop(ar,complex(br),cr,dr);
    @test true
 catch
    @test false
 end 


@time x = Trinv*er[:];
@test norm(Tr*x-er[:])/norm(x[:]) < reltol 

@time x = Trsinv*er[:]
@test norm(Trs*x[:]-er[:])/norm(x[:]) < reltol

@time x = Trs1inv*er[:]
@test norm(Trs1*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvop(as, bs, cs, ds)*er[:]
@test norm(sylvop(as, bs, cs, ds)*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvop(as', bs, cs', ds)*er[:]
@test norm(sylvop(as', bs, cs', ds)*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvop(as, bs', cs, ds')*er[:]
@test norm(sylvop(as, bs', cs, ds')*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvop(as', bs', cs', ds')*er[:]
@test norm(sylvop(as', bs', cs', ds')*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as, bs, cs, ds))*er[:]
@test norm(transpose(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as, bs, cs, ds))*er[:]
@test norm(transpose(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as', bs, cs', ds))*er[:]
@test norm(transpose(sylvop(as', bs, cs', ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as, bs', cs, ds'))*er[:]
@test norm(transpose(sylvop(as, bs', cs, ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvop(as', bs', cs', ds'))*er[:]
@test norm(transpose(sylvop(as', bs', cs', ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as, bs, cs, ds))*er[:]
@test norm(adjoint(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as', bs, cs', ds))*er[:]
@test norm(adjoint(sylvop(as', bs, cs', ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as, bs', cs, ds'))*er[:]
@test norm(adjoint(sylvop(as, bs', cs, ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvop(as', bs', cs', ds'))*er[:]
@test norm(adjoint(sylvop(as', bs', cs', ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = Tcinv*ec[:]
@test norm(Tc*x[:]-ec[:])/norm(x[:]) < reltol

@time x = Tcsinv*ec[:]
@test norm(Tcs*x[:]-ec[:])/norm(x[:]) < reltol

@time x = Tcsinv'*er[:]
@test norm(Tcs'*x[:]-er[:])/norm(x[:]) < reltol

@time y = transpose(Trinv)*er[:]
@test norm(transpose(Tr)*y[:]-er[:])/norm(y[:]) < reltol

@time y = transpose(Trsinv)*er[:]
@test norm(transpose(Trs)*y[:]-er[:])/norm(y[:]) < reltol

@time x = transpose(Trsinv)*er[:]
@test norm(transpose(Trs)*x[:]-er[:])/norm(x[:]) < reltol

@time x = Trs1inv'*er[:]
@test norm(Trs1'*x[:]-er[:])/norm(x[:]) < reltol


@time y = Tcinv'*ec[:]
@test norm(Tc'*y[:]-ec[:])/norm(y[:]) < reltol

@time y = Tcsinv'*ec[:]
@test norm(Tcs'*y[:]-ec[:])/norm(y[:]) < reltol

x = rand(n*m);
@test abs(norm(Matrix(Tr)'-Matrix(transpose(Tr)))) < reltol  
@test abs(norm(Matrix(Tc)'-Matrix(Tc'))) < reltol 
@test norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol 
@test norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol 
@test norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol 
@test norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol 
@test norm(Trsinv'*x-invsylvop(as',bs', cs',ds')*x) == 0 
@test opnorm1(Trsinv-invsylvop(schur(ar,cr),schur(br,dr))) == 0


@test sc*opnorm1(Tr) < opnorm1est(Tr)  &&
      sc*opnorm1(Trinv) < opnorm1est(Trinv)  &&
      sc*opnorm1(Trs) < opnorm1est(Trs)  &&
      sc*opnorm1(Trsinv) < opnorm1est(Trsinv)  &&
      sc*opnorm1(Tc) < opnorm1est(Tc)  &&
      sc*opnorm1(Tcinv) < opnorm1est(Tcinv)  &&
      sc*opnorm1(Tcs) < opnorm1est(Tcs)  &&
      sc*opnorm1(Tcsinv) < opnorm1est(Tcsinv)

@test sc*oprcondest(Tr,Trinv) < 1/opnorm1(Tr)/opnorm1(Trinv)  &&
      sc*oprcondest(Trs,Trsinv) < 1/opnorm1(Trs)/opnorm1(Trsinv)  &&
      sc*oprcondest(Tc,Tcinv) < 1/opnorm1(Tc)/opnorm1(Tcinv)  &&
      sc*oprcondest(Tcs,Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv) &&
      oprcondest(Tr) == oprcondest(Tr,Trinv) == oprcondest(Trinv) 

@test opsepest(Trsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2)*n*opsepest(Trsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. 

end

@testset "Sylvester system operators" begin

ar = rand(n,n)
br = rand(m,m)
cr = rand(n,n)
dr = rand(m,m)
er = rand(n,m)
fr = rand(n,m)
ac = ar+im*rand(n,n)
bc = br+im*rand(m,m)
cc = cr+im*rand(n,n)
dc = dr+im*rand(m,m)
ec = er+im*rand(n,m)
fc = fr+im*rand(n,m)
as, cs = schur(ar,cr)
bs, ds = schur(br,dr)
acs, ccs = schur(ac,cc)
bcs, dcs = schur(bc,dc)

Tr = sylvsysop(ar, br, cr, dr)
Trinv = invsylvsysop(ar, br, cr, dr)
Trs = sylvsysop(as, bs, cs, ds)
Trsinv = invsylvsysop(as, bs, cs, ds)

Tc = sylvsysop(ac, bc, cc, dc)
Tcinv = invsylvsysop(ac, bc, cc, dc)
Tcs = sylvsysop(acs, bcs, ccs, dcs)
Tcsinv = invsylvsysop(acs, bcs, ccs, dcs)
@test size(Tr) == size(Tr') == size(Trinv') 


@test check_ctranspose(Tr) &&
      check_ctranspose(Trinv) &&
      check_ctranspose(Trs) &&
      check_ctranspose(Trsinv) &&
      check_ctranspose(Tc) &&
      check_ctranspose(Tcinv) &&
      check_ctranspose(Tcs) &&
      check_ctranspose(Tcsinv) 


try
    T = invsylvsysop(as,bs,cs,ds);
    T*rand(2n*m);
    transpose(T)*rand(2n*m);
    adjoint(T)*rand(2n*m);
    T*rand(Float32,2n*m);
    transpose(T)*rand(Float32,2n*m);
    adjoint(T)*rand(Float32,2n*m);
    T*ones(Int,2n*m);
    transpose(T)*ones(Int,2n*m);
    adjoint(T)*ones(Int,2n*m);
    T = invsylvsysop(acs,bcs,ccs,dcs);
    T*rand(2n*m);
    T*complex(rand(2n*m));
    transpose(T)*rand(2n*m);
    transpose(T)*complex(rand(2n*m));
    T'*rand(2n*m);
    T'*complex(rand(2n*m));
    sylvsysop(ar,complex(br),cr,dr);
    invsylvsysop(ar,complex(br),cr,dr);
    @test true
 catch
    @test false
 end 




@time xy = Trinv*[er[:];fr[:]]
@test norm(Tr*xy-[er[:];fr[:]])/norm(xy[:]) < reltol
 
@time xy = Trinv*[ec[:];fc[:]]
@test norm(Tr*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trinv)*[er[:];fr[:]]
@test norm(transpose(Tr)*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trinv)*[ec[:];fc[:]]
@test norm(transpose(Tr)*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Tcinv*[ec[:];fc[:]]
@test norm(Tc*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Tcinv'*[ec[:];fc[:]]
@test norm(Tc'*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Trsinv*[er[:];fr[:]]
@test norm(Trs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trsinv)*[er[:];fr[:]]
@test norm(transpose(Trs)*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv'*[er[:];fr[:]]
@test norm(Tcs'*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@test abs(norm(Matrix(Tr)'-Matrix(transpose(Tr)))) < reltol 
@test abs(norm(Matrix(Tc)'-Matrix(Tc'))) < reltol 
@test norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol 
@test norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol 
@test norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol 
@test norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol 
@test opnorm1(Trsinv-invsylvsysop(schur(ar,cr),schur(br,dr))) == 0


@test sc*opnorm1(Tr) < opnorm1est(Tr)  &&
      sc*opnorm1(Trinv) < opnorm1est(Trinv)  &&
      sc*opnorm1(Trs) < opnorm1est(Trs)  &&
      sc*opnorm1(Trsinv) < opnorm1est(Trsinv)  &&
      sc*opnorm1(Tc) < opnorm1est(Tc)  &&
      sc*opnorm1(Tcinv) < opnorm1est(Tcinv)  &&
      sc*opnorm1(Tcs) < opnorm1est(Tcs)  &&
      sc*opnorm1(Tcsinv) < opnorm1est(Tcsinv)

@test sc*oprcondest(Tr,Trinv) < 1/opnorm1(Tr)/opnorm1(Trinv)  &&
      sc*oprcondest(Trs,Trsinv) < 1/opnorm1(Trs)/opnorm1(Trsinv)  &&
      sc*oprcondest(Tc,Tcinv) < 1/opnorm1(Tc)/opnorm1(Tcinv)  &&
      sc*oprcondest(Tcs,Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv) &&
      oprcondest(Tr) == oprcondest(Tr,Trinv) == oprcondest(Trinv) 


@test opsepest(Trsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2)*n*opsepest(Trsinv)  &&
      opsepest(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0.   &&
      opsepest(transpose(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0.  &&
      opsepest(transpose(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. 


end

#end

end
