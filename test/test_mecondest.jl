module Test_MEcondest

using LinearAlgebra
using MatrixEquations
using Test

@testset "Testing Lyapunov and Sylvester operators" begin

n = 10; m = 5;
reltol = sqrt(eps(1.))
sc = 0.1


@testset "Continuous Lyapunov operators" begin

ar = rand(n,n)
cr = rand(n,m)
cc = cr+im*rand(n,m)
cr = cr*cr'
as,  = schur(ar)
ac = ar+im*rand(n,n)
cc = cc*cc'
acs,  = schur(ac)

Tcr = lyapcop(ar)
Tcrinv = invlyapcop(ar)
Tcrs = lyapcop(as)
Tcrsinv = invlyapcsop(as)


Tcc = lyapcop(ac)
Tccinv = invlyapcop(ac)
Tccs = lyapcop(acs)
Tccsinv = invlyapcsop(acs)
Π = trmat(n)


@time x = Tcrinv*cr[:]
@test norm(Tcr*x-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tccinv*cc[:]
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol


@time x = Tccsinv*cc[:]
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:]
@test norm(transpose(Tcr)*y-cr[:])/norm(y[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

@test norm(Matrix(Tcr)'-Matrix(Tcr')) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol  &&
      opnorm1est(Π*Tcr-Tcr*Π) < reltol &&
      opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol &&
      opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol &&
      opnorm(Matrix(Tcr),1) ≈ opnorm1(Tcr)

@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)


@test sc*oprcondest(opnorm1est(Tcr),Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(opnorm1est(Tcrs),Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(opnorm1est(Tcc),Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(opnorm1est(Tccs),Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test lyapsepest(ar) == opsepest(Tcrsinv) &&
      lyapsepest(ar') == opsepest(Tcrsinv') &&
      lyapsepest(as) == opsepest(Tcrsinv) &&
      lyapsepest(as') == opsepest(Tcrsinv')  &&
      lyapsepest(ac) == opsepest(Tccsinv) &&
      lyapsepest(ac') == opsepest(Tccsinv') &&
      lyapsepest(acs) == opsepest(Tccsinv) &&
      lyapsepest(acs') == opsepest(Tccsinv') &&
      lyapsepest(as)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*lyapsepest(as)  &&
      lyapsepest([0. 1.; 0. 1.]) == 0.

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

Tcr = lyapcop(ar,er)
Tcrinv = invlyapcop(ar,er)
Tcrs = lyapcop(as,es)
Tcrsinv = invlyapcsop(as,es)


Tcc = lyapcop(ac,ec)
Tccinv = invlyapcop(ac,ec)
Tccs = lyapcop(acs,ecs)
Tccsinv = invlyapcsop(acs,ecs)
Π = trmat(n)


@time x = Tcrinv*cr[:]
@test norm(Tcr*x-cr[:])/norm(x) < reltol

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tccinv*cc[:]
@test norm(Tcc*x-cc[:])/norm(x[:]) < reltol


@time x = Tccsinv*cc[:]
@test norm(Tccs*x-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:]
@test norm(transpose(Tcr)*y-cr[:])/norm(y[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y-cc[:])/norm(y[:]) < reltol

@test norm(Matrix(Tcr)'-Matrix(Tcr')) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol &&
      opnorm1est(Π*Tcr-Tcr*Π) < reltol &&
      opnorm1est(Π*Tcrinv-Tcrinv*Π) < reltol*norm(ar)*norm(er) &&
      opnorm1est(Π*Tcrsinv-Tcrsinv*Π) < reltol*norm(ar)*norm(er)

@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)


@test sc*oprcondest(opnorm1est(Tcr),Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(opnorm1est(Tcrs),Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(opnorm1est(Tcc),Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(opnorm1est(Tccs),Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test lyapsepest(ar,er) == opsepest(Tcrsinv) &&
      lyapsepest(ar',er') == opsepest(Tcrsinv') &&
      lyapsepest(as,es) == opsepest(Tcrsinv) &&
      lyapsepest(as',es') == opsepest(Tcrsinv')  &&
      lyapsepest(ac,ec) == opsepest(Tccsinv) &&
      lyapsepest(ac',ec') == opsepest(Tccsinv') &&
      lyapsepest(acs,ecs) == opsepest(Tccsinv) &&
      lyapsepest(acs',ecs') == opsepest(Tccsinv') &&
      lyapsepest(as,es)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*lyapsepest(as,es)  &&
      lyapsepest([0. 1.; 0. 1.],[1. 1.;0. 1.]) == 0.

end


@testset "Discrete Lyapunov operators" begin

ar = rand(n,n)
cr = rand(n,m)
cc = cr+im*rand(n,m)
cr = cr*cr'
as,  = schur(ar)
ac = ar+im*rand(n,n)
cc = cc*cc'
acs,  = schur(ac)

Tdr = lyapdop(ar)
Tdrinv = invlyapdop(ar)
Tdrs = lyapdop(as)
Tdrsinv = invlyapdsop(as)

Tdc = lyapdop(ac)
Tdcinv = invlyapdop(ac)
Tdcs = lyapdop(acs)
Tdcsinv = invlyapdsop(acs)
Π = trmat(n)


@time x = Tdrinv*cr[:]
@test norm(Tdr*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tdcinv*cc[:]
@test norm(Tdc*x-cc[:])/norm(x[:]) < reltol


@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y-cc[:])/norm(y[:]) < reltol

@test norm(Matrix(Tdr)'-Matrix(Tdr')) == 0. &&
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


@test sc*oprcondest(opnorm1est(Tdr),Tdrinv) < 1/opnorm1(Tdr)/opnorm1(Tdrinv)  &&
      sc*oprcondest(opnorm1est(Tdrs),Tdrsinv) < 1/opnorm1(Tdrs)/opnorm1(Tdrsinv)  &&
      sc*oprcondest(opnorm1est(Tdc),Tdcinv) < 1/opnorm1(Tdc)/opnorm1(Tdcinv)  &&
      sc*oprcondest(opnorm1est(Tdcs),Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv)


@test lyapsepest(ar,disc=true) == opsepest(Tdrsinv) &&
      lyapsepest(ar',disc=true) == opsepest(Tdrsinv') &&
      lyapsepest(as,disc=true) == opsepest(Tdrsinv) &&
      lyapsepest(as',disc=true) == opsepest(Tdrsinv')  &&
      lyapsepest(ac,disc=true) == opsepest(Tdcsinv) &&
      lyapsepest(ac',disc=true) == opsepest(Tdcsinv') &&
      lyapsepest(acs,disc=true) == opsepest(Tdcsinv) &&
      lyapsepest(acs',disc=true) == opsepest(Tdcsinv') &&
      lyapsepest(as,disc=true)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*lyapsepest(as,disc=true)  &&
      lyapsepest([0. 1.; 0. 1.],disc=true) == 0.


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

Tdr = lyapdop(ar,er)
Tdrinv = invlyapdop(ar,er)
Tdrs = lyapdop(as,es)
Tdrsinv = invlyapdsop(as,es)


Tdc = lyapdop(ac,ec)
Tdcinv = invlyapdop(ac,ec)
Tdcs = lyapdop(acs,ecs)
Tdcsinv = invlyapdsop(acs,ecs)
Π = trmat(n)



@time x = Tdrinv*cr[:]
@test norm(Tdr*x-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x-cr[:])/norm(x[:]) < reltol

@time x = Tdcinv*cc[:]
@test norm(Tdc*x-cc[:])/norm(x[:]) < reltol


@time x = Tdcsinv*cc[:]
@test norm(Tdcs*x-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tdrinv)*cr[:]
@test norm(transpose(Tdr)*y-cr[:])/norm(y[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y-cc[:])/norm(y[:]) < reltol

@test norm(Matrix(Tdr)'-Matrix(Tdr')) == 0. &&
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


@test sc*oprcondest(opnorm1est(Tdr),Tdrinv) < 1/opnorm1(Tdr)/opnorm1(Tdrinv)  &&
      sc*oprcondest(opnorm1est(Tdrs),Tdrsinv) < 1/opnorm1(Tdrs)/opnorm1(Tdrsinv)  &&
      sc*oprcondest(opnorm1est(Tdc),Tdcinv) < 1/opnorm1(Tdc)/opnorm1(Tdcinv)  &&
      sc*oprcondest(opnorm1est(Tdcs),Tdcsinv) < 1/opnorm1(Tdcs)/opnorm1(Tdcsinv)

@test lyapsepest(ar,er,disc=true) == opsepest(Tdrsinv) &&
      lyapsepest(ar',er',disc=true) == opsepest(Tdrsinv') &&
      lyapsepest(as,es,disc=true) == opsepest(Tdrsinv) &&
      lyapsepest(as',es',disc=true) == opsepest(Tdrsinv')  &&
      lyapsepest(ac,ec,disc=true) == opsepest(Tdcsinv) &&
      lyapsepest(ac',ec',disc=true) == opsepest(Tdcsinv') &&
      lyapsepest(acs,ecs,disc=true) == opsepest(Tdcsinv) &&
      lyapsepest(acs',ecs',disc=true) == opsepest(Tdcsinv') &&
      lyapsepest(as,es,disc=true)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*lyapsepest(as,es,disc=true)  &&
      lyapsepest([0. 1.; 0. 1.],[1. 1.;0. 1.],disc=true) == 0.

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

Tcr = sylvcop(ar, br)
Tcrinv = invsylvcop(ar,br)
Tcrs = sylvcop(as, bs)
Tcrsinv = invsylvcsop(as,bs)


Tcc = sylvcop(ac, bc)
Tccinv = invsylvcop(ac,bc)
Tccs = sylvcop(acs, bcs)
Tccsinv = invsylvcsop(acs,bcs)


@time x = Tcrinv*cr[:]
@test norm(Tcr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tcrsinv*cr[:]
@test norm(Tcrs*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tccinv*cc[:]
@test norm(Tcc*x[:]-cc[:])/norm(x[:]) < reltol

@time x = Tccsinv*cc[:]
@test norm(Tccs*x[:]-cc[:])/norm(x[:]) < reltol

@time y = transpose(Tcrinv)*cr[:]
@test norm(transpose(Tcr)*y[:]-cr[:])/norm(y[:]) < reltol

@time y = Tccinv'*cc[:]
@test norm(Tcc'*y[:]-cc[:])/norm(y[:]) < reltol

@test norm(Matrix(Tcr)'-Matrix(Tcr')) == 0. &&
      norm(Matrix(Tcc)'-Matrix(Tcc')) == 0. &&
      norm(Matrix(Tcrinv)*Matrix(Tcr)-I) < reltol &&
      norm(Matrix(Tcrsinv)*Matrix(Tcrs)-I) < reltol &&
      norm(Matrix(Tccinv)*Matrix(Tcc)-I) < reltol &&
      norm(Matrix(Tccsinv)*Matrix(Tccs)-I) < reltol


@test sc*opnorm1(Tcr) < opnorm1est(Tcr)  &&
      sc*opnorm1(Tcrinv) < opnorm1est(Tcrinv)  &&
      sc*opnorm1(Tcrs) < opnorm1est(Tcrs)  &&
      sc*opnorm1(Tcrsinv) < opnorm1est(Tcrsinv)  &&
      sc*opnorm1(Tcc) < opnorm1est(Tcc)  &&
      sc*opnorm1(Tccinv) < opnorm1est(Tccinv)  &&
      sc*opnorm1(Tccs) < opnorm1est(Tccs)  &&
      sc*opnorm1(Tccsinv) < opnorm1est(Tccsinv)

@test sc*oprcondest(opnorm1est(Tcr),Tcrinv) < 1/opnorm1(Tcr)/opnorm1(Tcrinv)  &&
      sc*oprcondest(opnorm1est(Tcrs),Tcrsinv) < 1/opnorm1(Tcrs)/opnorm1(Tcrsinv)  &&
      sc*oprcondest(opnorm1est(Tcc),Tccinv) < 1/opnorm1(Tcc)/opnorm1(Tccinv)  &&
      sc*oprcondest(opnorm1est(Tccs),Tccsinv) < 1/opnorm1(Tccs)/opnorm1(Tccsinv)

@test sylvsepest(ar,br) == opsepest(Tcrsinv) &&
      sylvsepest(ar',br') == opsepest(Tcrsinv') &&
      sylvsepest(as,bs) == opsepest(Tcrsinv) &&
      sylvsepest(as',bs') == opsepest(Tcrsinv')  &&
      sylvsepest(ac,bc) == opsepest(Tccsinv) &&
      sylvsepest(ac',bc') == opsepest(Tccsinv') &&
      sylvsepest(acs,bcs) == opsepest(Tccsinv) &&
      sylvsepest(acs',bcs') == opsepest(Tccsinv') &&
      sylvsepest(as,bs)/n/sqrt(2) <= minimum(svdvals(Matrix(Tcrs)))  &&
      minimum(svdvals(Matrix(Tcrs))) <= sqrt(2)*n*sylvsepest(as,bs)  &&
      sylvsepest([0. 1.; 0. 1.],-[0. 1.; 0. 1.]) == 0.

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

Tdr = sylvdop(ar, br)
Tdrinv = invsylvdop(ar,br)
Tdrs = sylvdop(as, bs)
Tdrsinv = invsylvdsop(as,bs)

Tdc = sylvdop(ac, bc)
Tdcinv = invsylvdop(ac,bc)
Tdcs = sylvdop(acs, bcs)
Tdcsinv = invsylvdsop(acs,bcs)

@time x = Tdrinv*cr[:]
@test norm(Tdr*x[:]-cr[:])/norm(x[:]) < reltol

@time x = Tdrsinv*cr[:]
@test norm(Tdrs*x[:]-cr[:])/norm(x[:]) < reltol

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

@time x = Tdrsinv'*cr[:]
@test norm(Tdrs'*x[:]-cr[:])/norm(x[:]) < reltol

@time y = Tdcinv'*cc[:]
@test norm(Tdc'*y[:]-cc[:])/norm(y[:]) < reltol

@time y = Tdcsinv'*cc[:]
@test norm(Tdcs'*y[:]-cc[:])/norm(y[:]) < reltol


@test norm(Matrix(Tdr)'-Matrix(Tdr')) == 0. &&
      norm(Matrix(Tdc)'-Matrix(Tdc')) == 0. &&
      norm(Matrix(Tdrinv)*Matrix(Tdr)-I) < reltol &&
      norm(Matrix(Tdrsinv)*Matrix(Tdrs)-I) < reltol &&
      norm(Matrix(Tdcinv)*Matrix(Tdc)-I) < reltol &&
      norm(Matrix(Tdcsinv)*Matrix(Tdcs)-I) < reltol



@test sc*opnorm1(Tdr) < opnorm1est(Tdr)  &&
      sc*opnorm1(Tdrinv) < opnorm1est(Tdrinv)  &&
      sc*opnorm1(Tdrs) < opnorm1est(Tdrs)  &&
      sc*opnorm1(Tdrsinv) < opnorm1est(Tdrsinv)  &&
      sc*opnorm1(Tdc) < opnorm1est(Tdc)  &&
      sc*opnorm1(Tdcinv) < opnorm1est(Tdcinv)  &&
      sc*opnorm1(Tdcs) < opnorm1est(Tdcs)  &&
      sc*opnorm1(Tdcsinv) < opnorm1est(Tdcsinv)


@test sylvsepest(ar,br,disc=true) == opsepest(Tdrsinv) &&
      sylvsepest(ar',br',disc=true) == opsepest(Tdrsinv') &&
      sylvsepest(as,bs,disc=true) == opsepest(Tdrsinv) &&
      sylvsepest(as',bs',disc=true) == opsepest(Tdrsinv')  &&
      sylvsepest(ac,bc,disc=true) == opsepest(Tdcsinv) &&
      sylvsepest(ac',bc',disc=true) == opsepest(Tdcsinv') &&
      sylvsepest(acs,bcs,disc=true) == opsepest(Tdcsinv) &&
      sylvsepest(acs',bcs',disc=true) == opsepest(Tdcsinv') &&
      sylvsepest(as,bs,disc=true)/n/sqrt(2) <= minimum(svdvals(Matrix(Tdrs)))  &&
      minimum(svdvals(Matrix(Tdrs))) <= sqrt(2)*n*sylvsepest(as,bs,disc=true)  &&
      sylvsepest([0. 1.; 0. 1.],-[0. 1.; 0. 1.],disc=true) == 0.


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

Tr = gsylvop(ar, br, cr, dr)
Trinv = invgsylvop(ar, br, cr, dr)
Trs = gsylvop(as, bs, cs, ds)
Trsinv = invgsylvsop(as, bs, cs, ds)
Trs1 = gsylvop(as, ds, cs, bs)
Trs1inv = invgsylvsop(as, ds, cs, bs, DBSchur=true)

Tc = gsylvop(ac, bc, cc, dc)
Tcinv = invgsylvop(ac, bc, cc, dc)
Tcs = gsylvop(acs, bcs, ccs, dcs)
Tcsinv = invgsylvsop(acs, bcs, ccs, dcs)

@time x = Trinv*er[:]
@test norm(Tr*x-er[:])/norm(x[:]) < reltol

@time x = Trsinv*er[:]
@test norm(Trs*x[:]-er[:])/norm(x[:]) < reltol

@time x = Trs1inv*er[:]
@test norm(Trs1*x[:]-er[:])/norm(x[:]) < reltol

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

@time x = Trsinv'*er[:]
@test norm(Trs'*x[:]-er[:])/norm(x[:]) < reltol

@time x = Trs1inv'*er[:]
@test norm(Trs1'*x[:]-er[:])/norm(x[:]) < reltol


@time y = Tcinv'*ec[:]
@test norm(Tc'*y[:]-ec[:])/norm(y[:]) < reltol

@time y = Tcsinv'*ec[:]
@test norm(Tcs'*y[:]-ec[:])/norm(y[:]) < reltol

@test norm(Matrix(Tr)'-Matrix(Tr')) == 0. &&
      norm(Matrix(Tc)'-Matrix(Tc')) == 0. &&
      norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol &&
      norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol &&
      norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol &&
      norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol


@test sc*opnorm1(Tr) < opnorm1est(Tr)  &&
      sc*opnorm1(Trinv) < opnorm1est(Trinv)  &&
      sc*opnorm1(Trs) < opnorm1est(Trs)  &&
      sc*opnorm1(Trsinv) < opnorm1est(Trsinv)  &&
      sc*opnorm1(Tc) < opnorm1est(Tc)  &&
      sc*opnorm1(Tcinv) < opnorm1est(Tcinv)  &&
      sc*opnorm1(Tcs) < opnorm1est(Tcs)  &&
      sc*opnorm1(Tcsinv) < opnorm1est(Tcsinv)

@test sc*oprcondest(opnorm1est(Tr),Trinv) < 1/opnorm1(Tr)/opnorm1(Trinv)  &&
      sc*oprcondest(opnorm1est(Trs),Trsinv) < 1/opnorm1(Trs)/opnorm1(Trsinv)  &&
      sc*oprcondest(opnorm1est(Tc),Tcinv) < 1/opnorm1(Tc)/opnorm1(Tcinv)  &&
      sc*oprcondest(opnorm1est(Tcs),Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv)

@test sylvsepest(ar,br,cr,dr) == opsepest(Trsinv) &&
      sylvsepest(ar',br',cr',dr') == opsepest(Trsinv') &&
      sylvsepest(as,bs,cs,ds) == opsepest(Trsinv) &&
      sylvsepest(as',bs',cs',ds') == opsepest(Trsinv')  &&
      sylvsepest(ac,bc,cc,dc) == opsepest(Tcsinv) &&
      sylvsepest(ac',bc',cc',dc') == opsepest(Tcsinv') &&
      sylvsepest(acs,bcs,ccs,dcs) == opsepest(Tcsinv) &&
      sylvsepest(acs',bcs',ccs',dcs') == opsepest(Tcsinv') &&
      sylvsepest(as,bs,cs,ds)/sqrt(2*n*m) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2*n*m)*sylvsepest(as,bs,cs,ds)  &&
      sylvsepest([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]) == 0.

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

@time xy = Trinv*[er[:];fr[:]]
@test norm(Tr*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Trinv'*[er[:];fr[:]]
@test norm(Tr'*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcinv*[ec[:];fc[:]]
@test norm(Tc*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Tcinv'*[ec[:];fc[:]]
@test norm(Tc'*xy-[ec[:];fc[:]])/norm(xy[:]) < reltol

@time xy = Trsinv*[er[:];fr[:]]
@test norm(Trs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Trsinv'*[er[:];fr[:]]
@test norm(Trs'*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = transpose(Trsinv)*[er[:];fr[:]]
@test norm(transpose(Trs)*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv*[er[:];fr[:]]
@test norm(Tcs*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@time xy = Tcsinv'*[er[:];fr[:]]
@test norm(Tcs'*xy-[er[:];fr[:]])/norm(xy[:]) < reltol

@test norm(Matrix(Tr)'-Matrix(Tr')) == 0. &&
      norm(Matrix(Tc)'-Matrix(Tc')) == 0. &&
      norm(Matrix(Trinv)*Matrix(Tr)-I) < reltol &&
      norm(Matrix(Trsinv)*Matrix(Trs)-I) < reltol &&
      norm(Matrix(Tcinv)*Matrix(Tc)-I) < reltol &&
      norm(Matrix(Tcsinv)*Matrix(Tcs)-I) < reltol


@test sc*opnorm1(Tr) < opnorm1est(Tr)  &&
      sc*opnorm1(Trinv) < opnorm1est(Trinv)  &&
      sc*opnorm1(Trs) < opnorm1est(Trs)  &&
      sc*opnorm1(Trsinv) < opnorm1est(Trsinv)  &&
      sc*opnorm1(Tc) < opnorm1est(Tc)  &&
      sc*opnorm1(Tcinv) < opnorm1est(Tcinv)  &&
      sc*opnorm1(Tcs) < opnorm1est(Tcs)  &&
      sc*opnorm1(Tcsinv) < opnorm1est(Tcsinv)

@test sc*oprcondest(opnorm1est(Tr),Trinv) < 1/opnorm1(Tr)/opnorm1(Trinv)  &&
      sc*oprcondest(opnorm1est(Trs),Trsinv) < 1/opnorm1(Trs)/opnorm1(Trsinv)  &&
      sc*oprcondest(opnorm1est(Tc),Tcinv) < 1/opnorm1(Tc)/opnorm1(Tcinv)  &&
      sc*oprcondest(opnorm1est(Tcs),Tcsinv) < 1/opnorm1(Tcs)/opnorm1(Tcsinv)


@test sylvsyssepest(ar,br,cr,dr) == opsepest(Trsinv) &&
      sylvsyssepest(as,bs,cs,ds) == opsepest(Trsinv) &&
      sylvsyssepest(ac,bc,cc,dc) == opsepest(Tcsinv) &&
      sylvsyssepest(acs,bcs,ccs,dcs) == opsepest(Tcsinv) &&
      sylvsyssepest(as,bs,cs,ds)/sqrt(2*n*m) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2*n*m)*sylvsyssepest(as,bs,cs,ds)  &&
      sylvsyssepest([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]) == 0.

end

end

end
