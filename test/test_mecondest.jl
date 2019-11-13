module Test_MEcondest

using LinearAlgebra
using LinearOperators
using MatrixEquations
using Test

@testset "Testing Lyapunov and Sylvester operators" begin

n = 10; m = 5;
reltol = sqrt(eps(1.));
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

Tcr = lyapop(ar);
Tcrinv = invlyapop(ar);
Tcrs = lyapop(as);
Tcrsinv = invlyapsop(as);
Tcrsym = lyapop(ar,her=true);
Tcrsyminv = invlyapop(ar,her=true);
Tcrssym = lyapop(as,her=true);
Tcrssyminv = invlyapsop(as,her=true);


Tcc = lyapop(ac);
Tccinv = invlyapop(ac);
Tccs = lyapop(acs);
Tccsinv = invlyapsop(acs);
Tccsym = lyapop(ac,her=true);
Tccsyminv = invlyapop(ac,her=true);
Tccssym = lyapop(acs,her=true);
Tccssyminv = invlyapsop(acs,her=true);
Π = trmatop(n); 

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
   T = invlyapsop(convert(Matrix{Float32},as));
   T*rand(n*n);
   T*complex(rand(n*n));
   transpose(T)*rand(n*n);
   transpose(T)*complex(rand(n*n));
   adjoint(T)*rand(n*n);
   adjoint(T)*complex(rand(n*n));
   T = invlyapsop(convert(Matrix{Complex{Float32}},acs));
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

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cc[:]
@test norm(Tcrs*x-cc[:])/norm(x[:]) < reltol

@time x = Tcrsinv*real(cc[:])+im*Tcrsinv*imag(cc[:])
@test norm(Tcrs*x-cc[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cr[:]
@test norm(transpose(Tcrs)*x-cr[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cc[:]
@test norm(transpose(Tcrs)*x-cc[:])/norm(x[:]) < reltol

@time x = Tcrssyminv*triu2vec(cr);
@test norm(Tcrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tcrssyminv*triu2vec(cc);
@test norm(Tcrssym*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = transpose(Tcrssyminv)*triu2vec(cr);
@test norm(transpose(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccinv*cc[:]
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol

@time x = Tccsyminv*triu2vec(cr);
@test norm(Tccsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

@time x = Tccsyminv'*triu2vec(cr);
@test norm(Tccsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccsinv*cc[:]
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time x = Tccssyminv*triu2vec(cr);
@test norm(Tccssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:]
@test norm(transpose(Tcr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tcrsinv)*cr[:]
@test norm(transpose(Tcrs)*y-cr[:])/norm(y[:]) < reltol

@test norm(Matrix(Tcr)'-Matrix(transpose(Tcr))) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol  &&
      opnorm1est(Π*Tcr-Tcr*Π) < reltol &&
      opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol &&
      opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol &&
      opnorm(Matrix(Tcr),1) ≈ opnorm1(Tcr) &&
      opnorm1(Tcrsinv-invlyapsop(as')') == 0 &&
      opnorm1(Tcrsinv-invlyapsop(schur(ar))) == 0

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
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.]))) == 0.  &&
      opsepest(invlyapsop([0. 1.; 0. 1.])) == 0.  &&
      opsepest(transpose(invlyapsop([0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.]))) == 0.

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
Tcrsinv = invlyapsop(as,es)
Tcrs = lyapop(as,es)
Tcrsinv = invlyapsop(as,es)
Tcrsym = lyapop(ar,er,her=true);
Tcrsyminv = invlyapop(ar,er,her=true);
Tcrssym = lyapop(as,es,her=true);
Tcrssyminv = invlyapsop(as,es,her=true);


Tcc = lyapop(ac,ec)
Tccinv = invlyapop(ac,ec)
Tccs = lyapop(acs,ecs)
Tccsinv = invlyapsop(acs,ecs)
Tccsym = lyapop(ac,ec,her=true);
Tccsyminv = invlyapop(ac,ec,her=true);
Tccssym = lyapop(acs,ecs,her=true);
Tccssyminv = invlyapsop(acs,ecs,her=true);
Π = trmatop(size(ar))


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
   T = invlyapsop(triu(as),ecs);
   T = invlyapsop(acs,es);
   T = invlyapsop(convert(Matrix{Complex{Float32}},acs),ecs);
   T*rand(n*n);
   T*complex(rand(n*n));
   T = invlyapsop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},ecs));
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

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cc[:]
@test norm(Tcrs*x-cc[:])/norm(x[:]) < reltol

@time x = transpose(Tcrsinv)*cr[:]
@test norm(transpose(Tcrs)*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrssyminv*triu2vec(cr);
@test norm(Tcrssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tcrssyminv*triu2vec(cc);
@test norm(Tcrssym*x-triu2vec(cc))/norm(x[:]) < reltol

@time x = transpose(Tcrssyminv)*triu2vec(cr);
@test norm(transpose(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = adjoint(Tcrssyminv)*triu2vec(cr);
@test norm(adjoint(Tcrssym)*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccinv*cc[:]
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol

@time x = Tccsyminv*triu2vec(cr);
@test norm(Tccsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

@time x = Tccsyminv'*triu2vec(cr);
@test norm(Tccsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tccsinv*cc[:]
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time x = Tccssyminv*triu2vec(cr);
@test norm(Tccssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tcrinv'*cr[:]
@test norm(Tcr'*y-cr[:])/norm(y[:]) < reltol

@time y = Tcrsinv'*cr[:]
@test norm(Tcrs'*y-cr[:])/norm(y[:]) < reltol


@test norm(Matrix(Tcr)'-Matrix(transpose(Tcr))) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol &&
      opnorm1est(Π*Tcr-Tcr*Π) < reltol &&
      opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol*norm(ar)*norm(er) &&
      opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol*norm(ar)*norm(er) &&
      opnorm(Matrix(Tcr),1) ≈ opnorm1(Tcr) &&
      opnorm1(Tcrsinv-invlyapsop(as',es')') == 0 &&
      opnorm1(Tcrsinv-invlyapsop(schur(ar,er))) == 0

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
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.])) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0. &&
      opsepest(adjoint(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.  &&
      opsepest(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.])) == 0.  &&
      opsepest(transpose(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.  &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.]))) == 0.

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
Tdrsinv = invlyapsop(as,disc=true);
Tdrsym = lyapop(ar,disc=true,her=true);
Tdrsyminv = invlyapop(ar,disc=true,her=true);
Tdrssym = lyapop(as,disc=true,her=true);
Tdrssyminv = invlyapsop(as,disc=true,her=true);


Tdc = lyapop(ac,disc=true);
Tdcinv = invlyapop(ac,disc=true);
Tdcs = lyapop(acs,disc=true);
Tdcsinv = invlyapsop(acs,disc=true);
Tdcsym = lyapop(ac,disc=true,her=true);
Tdcsyminv = invlyapop(ac,disc=true,her=true);
Tdcssym = lyapop(acs,disc=true,her=true);
Tdcssyminv = invlyapsop(acs,disc=true,her=true);
Π = trmatop(ar)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 


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

@time x = Tdrssyminv*triu2vec(cc);
@test norm(Tdrssym*x-triu2vec(cc))/norm(x[:]) < reltol

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

@time x = Tdcsyminv'*triu2vec(cr);
@test norm(Tdcsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcssyminv*triu2vec(cr);
@test norm(Tdcssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*y-cr[:])/norm(y[:]) < reltol


@test norm(Matrix(Tdr)'-Matrix(transpose(Tdr))) == 0. &&
      norm(Matrix(Tdc)'-Matrix(Tdc')) == 0. &&
      norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol &&
      norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol &&
      norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol &&
      norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol &&
      opnorm1est(Π*Tdr-Tdr*Π) < reltol &&
      opnorm1est(Π*Tdrinv-Tdrinv*Π) < reltol &&
      opnorm1est(Π*Tdrsinv-Tdrsinv*Π) < reltol


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
      sc*oprcondest(Tdcs,Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv)


@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],disc=true)) == 0. &&
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],disc=true))) == 0. &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.],disc=true))) == 0.  &&
      opsepest(invlyapsop([0. 1.; 0. 1.],disc=true)) == 0.  &&
      opsepest(transpose(invlyapsop([0. 1.; 0. 1.],disc=true))) == 0.  &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.],disc=true))) == 0.


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
Tdrsinv = invlyapsop(as,es,disc=true)
Tdrsym = lyapop(ar,er,disc=true,her=true);
Tdrsyminv = invlyapop(ar,er,disc=true,her=true);
Tdrssym = lyapop(as,es,disc=true,her=true);
Tdrssyminv = invlyapop(as,es,disc=true,her=true);


Tdc = lyapop(ac,ec,disc=true)
Tdcinv = invlyapop(ac,ec,disc=true)
Tdcs = lyapop(acs,ecs,disc=true)
Tdcsinv = invlyapsop(acs,ecs,disc=true)
Tdcsym = lyapop(ac,ec,disc=true,her=true);
Tdcsyminv = invlyapop(ac,ec,disc=true,her=true);
Tdcssym = lyapop(acs,ecs,disc=true,her=true);
Tdcssyminv = invlyapsop(acs,ecs,disc=true,her=true);
Π = trmatop(n)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 


@time x = Tdrinv*cr[:];
@test norm(Tdr*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsyminv*triu2vec(cr);
@test norm(Tdrsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = transpose(Tdrinv)*cr[:];
@test norm(transpose(Tdr)*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsyminv'*triu2vec(cr);
@test norm(Tdrsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cc[:]
@test norm(Tdrs*x-cc[:])/norm(x[:]) < reltol

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

@time x = Tdcinv*cc[:]
@test norm(Tdc*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcsyminv*triu2vec(cr);
@test norm(Tdcsym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y-cc[:])/norm(y[:]) < reltol

@time x = Tdcsyminv'*triu2vec(cr);
@test norm(Tdcsym'*x-triu2vec(cr))/norm(x[:]) < reltol

@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time x = Tdcssyminv*triu2vec(cr);
@test norm(Tdcssym*x-triu2vec(cr))/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = transpose(Tdrsinv)*cr[:]
@test norm(transpose(Tdrs)*y-cr[:])/norm(y[:]) < reltol

@test norm(Matrix(Tdr)'-Matrix(transpose(Tdr))) == 0. &&
      norm(Matrix(Tdc)'-Matrix(Tdc')) == 0. &&
      norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol &&
      norm(Matrix(Tdrsyminv)*Matrix(Tdrsym)-I) < reltol &&
      norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol &&
      norm(Matrix(Tdrssyminv)*Matrix(Tdrssym)-I) < reltol &&
      norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol &&
      norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol &&
      opnorm1est(Π*Tdr-Tdr*Π) < reltol &&
      opnorm1est(Π*Tdrinv-Tdrinv*Π) < reltol &&
      opnorm1est(Π*Tdrsinv-Tdrsinv*Π) < reltol


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
      sc*oprcondest(Tdcs,Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv)

@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0. &&
      opsepest(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0.
      opsepest(transpose(invlyapop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0. &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.  &&
      opsepest(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true)) == 0.  &&
      opsepest(transpose(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.  &&
      opsepest(adjoint(invlyapsop([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true))) == 0.


end



#  continuous and discrete Sylvester equations
@testset "Continuous Sylvester operators" begin

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
Tcrsinv = invsylvsop(as,bs)


Tcc = sylvop(ac, bc)
Tccinv = invsylvop(ac,bc)
Tccs = sylvop(acs, bcs)
Tccsinv = invsylvsop(acs,bcs)

@test check_ctranspose(Tcr) &&
      check_ctranspose(Tcrinv) &&
      check_ctranspose(Tcrs) &&
      check_ctranspose(Tcrsinv) &&
      check_ctranspose(Tcc) &&
      check_ctranspose(Tccinv) &&
      check_ctranspose(Tccs) &&
      check_ctranspose(Tccsinv) 


try
    T = invsylvsop(triu(as),bcs);
    T = invsylvsop(acs,triu(bs));
    T*rand(n*m);
    T*complex(rand(n*m));
    T = invsylvsop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},bcs));
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
 
@time x = Tcrinv*cr[:]
@test norm(Tcr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs)*cr[:]
@test norm(sylvop(as, bs)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs)*cc[:]
@test norm(sylvop(as, bs)*x[:]-cc[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs')*cr[:]
@test norm(sylvop(as, bs')*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as',bs)*cr[:]
@test norm(sylvop(as', bs)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as',bs')*cr[:]
@test norm(sylvop(as', bs')*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as,bs))*cr[:]
@test norm(transpose(sylvop(as, bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as,bs))*cc[:]
@test norm(transpose(sylvop(as, bs))*x[:]-cc[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as,bs'))*cr[:]
@test norm(transpose(sylvop(as, bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as',bs))*cr[:]
@test norm(transpose(sylvop(as', bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as',bs'))*cr[:]
@test norm(transpose(sylvop(as', bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as,bs))*cr[:]
@test norm(adjoint(sylvop(as, bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as,bs'))*cr[:]
@test norm(adjoint(sylvop(as, bs'))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as',bs))*cr[:]
@test norm(adjoint(sylvop(as', bs))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as',bs'))*cr[:]
@test norm(adjoint(sylvop(as', bs'))*x[:]-cr[:])/norm(x[:]) < reltol


@time x = Tccinv*cc[:]
@test norm(Tcc*x[:]-cc[:])/norm(x[:]) < reltol

@time x = Tccsinv*cc[:]
@test norm(Tccs*x[:]-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:]
@test norm(transpose(Tcr)*y[:]-cr[:])/norm(y[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y[:]-cc[:])/norm(y[:]) < reltol

x = rand(n*m);
@test norm(Matrix(Tcr)'-Matrix(transpose(Tcr))) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol &&
      norm(Tcrsinv'*x-invsylvsop(as',bs')*x) == 0 &&
      opnorm1(Tcrsinv-invsylvsop(schur(ar),schur(br))) == 0


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
      sc*oprcondest(Tccs,Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test opsepest(Tcrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*opsepest(Tcrsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(invsylvsop([0. 1.; 0. 1.],-[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvsop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvsop([0. 1.; 0. 1.],-[0. 1.; 0. 1.]))) == 0.

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
Tdrsinv = invsylvsop(as,bs, disc = true)

Tdc = sylvop(ac, bc, disc = true)
Tdcinv = invsylvop(ac,bc, disc = true)
Tdcs = sylvop(acs, bcs, disc = true)
Tdcsinv = invsylvsop(acs,bcs, disc = true)

@test check_ctranspose(Tdr) &&
      check_ctranspose(Tdrinv) &&
      check_ctranspose(Tdrs) &&
      check_ctranspose(Tdrsinv) &&
      check_ctranspose(Tdc) &&
      check_ctranspose(Tdcinv) &&
      check_ctranspose(Tdcs) &&
      check_ctranspose(Tdcsinv) 


@time x = Tdrinv*cr[:]
@test norm(Tdr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs, disc = true)*cr[:]
@test norm(sylvop(as, bs, disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs, disc = true)*cc[:]
@test norm(sylvop(as, bs, disc = true)*x[:]-cc[:])/norm(x[:]) < reltol

@time x = invsylvsop(as,bs', disc = true)*cr[:]
@test norm(sylvop(as, bs', disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as',bs, disc = true)*cr[:]
@test norm(sylvop(as', bs, disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = invsylvsop(as',bs', disc = true)*cr[:]
@test norm(sylvop(as', bs', disc = true)*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as,bs, disc = true))*cr[:]
@test norm(transpose(sylvop(as, bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as,bs', disc = true))*cr[:]
@test norm(transpose(sylvop(as, bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as',bs, disc = true))*cr[:]
@test norm(transpose(sylvop(as', bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as',bs', disc = true))*cr[:]
@test norm(transpose(sylvop(as', bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as,bs, disc = true))*cr[:]
@test norm(adjoint(sylvop(as, bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as,bs', disc = true))*cr[:]
@test norm(adjoint(sylvop(as, bs', disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as',bs, disc = true))*cr[:]
@test norm(adjoint(sylvop(as', bs, disc = true))*x[:]-cr[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as',bs', disc = true))*cr[:]
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
@test norm(Matrix(Tdr)'-Matrix(transpose(Tdr))) == 0. &&
      norm(Matrix(Tdc)'-Matrix(Tdc')) == 0. &&
      norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol &&
      norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol &&
      norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol &&
      norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol &&
      norm(Tdrsinv'*x-invsylvsop(as',bs', disc = true)*x) == 0 &&
      opnorm1(Tdrsinv-invsylvsop(schur(ar),schur(br), disc = true)) == 0



@test sc*opnorm1(Tdr) < opnorm1est(Tdr)  &&
      sc*opnorm1(Tdrinv) < opnorm1est(Tdrinv)  &&
      sc*opnorm1(Tdrs) < opnorm1est(Tdrs)  &&
      sc*opnorm1(Tdrsinv) < opnorm1est(Tdrsinv)  &&
      sc*opnorm1(Tdc) < opnorm1est(Tdc)  &&
      sc*opnorm1(Tdcinv) < opnorm1est(Tdcinv)  &&
      sc*opnorm1(Tdcs) < opnorm1est(Tdcs)  &&
      sc*opnorm1(Tdcsinv) < opnorm1est(Tdcsinv)


@test opsepest(Tdrsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*opsepest(Tdrsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],-[0. 1.; 0. 1.],disc=true)) == 0. &&
      opsepest(invsylvsop([0. 1.; 0. 1.],-[0. 1.; 0. 1.],disc=true)) == 0.

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
Trsinv = invsylvsop(as, bs, cs, ds)
Trs1 = sylvop(as, ds, cs, bs)
Trs1inv = invsylvsop(as, ds, cs, bs, DBSchur=true)

Tc = sylvop(ac, bc, cc, dc)
Tcinv = invsylvop(ac, bc, cc, dc)
Tcs = sylvop(acs, bcs, ccs, dcs)
Tcsinv = invsylvsop(acs, bcs, ccs, dcs)

@test check_ctranspose(Tr) &&
      check_ctranspose(Trinv) &&
      check_ctranspose(Trs) &&
      check_ctranspose(Trsinv) &&
      check_ctranspose(Tc) &&
      check_ctranspose(Tcinv) &&
      check_ctranspose(Tcs) &&
      check_ctranspose(Tcsinv) 


try
    T = invsylvsop(triu(as),triu(bs),ccs,dcs);
    T = invsylvsop(acs,bcs,cs,ds);
    T*rand(n*m);
    T*complex(rand(n*m));
    T = invsylvsop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},bcs),
                   convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},dcs));
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


@time x = Trinv*er[:]
@test norm(Tr*x-er[:])/norm(x[:]) < reltol

@time x = Trsinv*er[:]
@test norm(Trs*x[:]-er[:])/norm(x[:]) < reltol

@time x = Trs1inv*er[:]
@test norm(Trs1*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvsop(as, bs, cs, ds)*er[:]
@test norm(sylvop(as, bs, cs, ds)*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvsop(as', bs, cs', ds)*er[:]
@test norm(sylvop(as', bs, cs', ds)*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvsop(as, bs', cs, ds')*er[:]
@test norm(sylvop(as, bs', cs, ds')*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvsop(as', bs', cs', ds')*er[:]
@test norm(sylvop(as', bs', cs', ds')*x[:]-er[:])/norm(x[:]) < reltol

@time x = invsylvsop(as, bs, cs, ds)*ec[:]
@test norm(sylvop(as, bs, cs, ds)*x[:]-ec[:])/norm(x[:]) < reltol

@time x = invsylvsop(as', bs, cs', ds)*ec[:]
@test norm(sylvop(as', bs, cs', ds)*x[:]-ec[:])/norm(x[:]) < reltol

@time x = invsylvsop(as, bs', cs, ds')*ec[:]
@test norm(sylvop(as, bs', cs, ds')*x[:]-ec[:])/norm(x[:]) < reltol

@time x = invsylvsop(as', bs', cs', ds')*ec[:]
@test norm(sylvop(as', bs', cs', ds')*x[:]-ec[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as, bs, cs, ds))*er[:]
@test norm(transpose(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as, bs, cs, ds))*er[:]
@test norm(transpose(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as', bs, cs', ds))*er[:]
@test norm(transpose(sylvop(as', bs, cs', ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as, bs', cs, ds'))*er[:]
@test norm(transpose(sylvop(as, bs', cs, ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = transpose(invsylvsop(as', bs', cs', ds'))*er[:]
@test norm(transpose(sylvop(as', bs', cs', ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as, bs, cs, ds))*er[:]
@test norm(adjoint(sylvop(as, bs, cs, ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as', bs, cs', ds))*er[:]
@test norm(adjoint(sylvop(as', bs, cs', ds))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as, bs', cs, ds'))*er[:]
@test norm(adjoint(sylvop(as, bs', cs, ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as', bs', cs', ds'))*er[:]
@test norm(adjoint(sylvop(as', bs', cs', ds'))*x[:]-er[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as, bs, cs, ds))*ec[:]
@test norm(adjoint(sylvop(as, bs, cs, ds))*x[:]-ec[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as', bs, cs', ds))*ec[:]
@test norm(adjoint(sylvop(as', bs, cs', ds))*x[:]-ec[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as, bs', cs, ds'))*ec[:]
@test norm(adjoint(sylvop(as, bs', cs, ds'))*x[:]-ec[:])/norm(x[:]) < reltol

@time x = adjoint(invsylvsop(as', bs', cs', ds'))*ec[:]
@test norm(adjoint(sylvop(as', bs', cs', ds'))*x[:]-ec[:])/norm(x[:]) < reltol

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
@test norm(Matrix(Tr)'-Matrix(transpose(Tr))) == 0. &&
      norm(Matrix(Tc)'-Matrix(Tc')) == 0. &&
      norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol &&
      norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol &&
      norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol &&
      norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol &&
      norm(Trsinv'*x-invsylvsop(as',bs', cs',ds')*x) == 0 &&
      opnorm1(Trsinv-invsylvsop(schur(ar,cr),schur(br,dr))) == 0


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
      sc*oprcondest(Tcs,Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv)

@test opsepest(Trsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2)*n*opsepest(Trsinv)  &&
      opsepest(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(invsylvsop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0. &&
      opsepest(transpose(invsylvsop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. &&
      opsepest(adjoint(invsylvsop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. 

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
Trsinv = invsylvsyssop(as, bs, cs, ds)

Tc = sylvsysop(ac, bc, cc, dc)
Tcinv = invsylvsysop(ac, bc, cc, dc)
Tcs = sylvsysop(acs, bcs, ccs, dcs)
Tcsinv = invsylvsyssop(acs, bcs, ccs, dcs)


@test check_ctranspose(Tr) &&
      check_ctranspose(Trinv) &&
      check_ctranspose(Trs) &&
      check_ctranspose(Trsinv) &&
      check_ctranspose(Tc) &&
      check_ctranspose(Tcinv) &&
      check_ctranspose(Tcs) &&
      check_ctranspose(Tcsinv) 


try
    T = invsylvsyssop(triu(as),triu(bs),ccs,dcs);
    T = invsylvsyssop(acs,bcs,cs,ds);
    T*rand(2n*m);
    T*complex(rand(2n*m));
    T = invsylvsyssop(convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},bcs),
                   convert(Matrix{Complex{Float32}},acs),convert(Matrix{Complex{Float32}},dcs));
    T*rand(2n*m);
    T*complex(rand(2n*m));
    transpose(T)*rand(2n*m);
    transpose(T)*complex(rand(2n*m));
    T'*rand(2n*m);
    T'*complex(rand(2n*m));
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

@time xy = Trsinv*[ec[:];fc[:]]
@test norm(Trs*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trsinv)*[er[:];fr[:]]
@test norm(transpose(Trs)*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trsinv)*[ec[:];fc[:]]
@test norm(transpose(Trs)*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv'*[er[:];fr[:]]
@test norm(Tcs'*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@test norm(Matrix(Tr)'-Matrix(transpose(Tr))) == 0. &&
      norm(Matrix(Tc)'-Matrix(Tc')) == 0. &&
      norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol &&
      norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol &&
      norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol &&
      norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol &&
      opnorm1(Trsinv-invsylvsyssop(schur(ar,cr),schur(br,dr))) == 0


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
      sc*oprcondest(Tcs,Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv)


@test opsepest(Trsinv)/n/sqrt(2) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2)*n*opsepest(Trsinv)  &&
      opsepest(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0.   &&
      opsepest(transpose(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invsylvsysop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(invsylvsyssop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.])) == 0.  &&
      opsepest(transpose(invsylvsyssop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0.  &&
      opsepest(adjoint(invsylvsyssop([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]))) == 0. 


end

end

end
