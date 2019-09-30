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
      norm(Matrix(Tcrinv)-inv(Matrix(Tcr))) < reltol*cond(Matrix(Tcr),1) &&
      norm(Matrix(Tcrsinv)-inv(Matrix(Tcrs))) < reltol*cond(Matrix(Tcrs),1) &&
      norm(Matrix(Tccinv)-inv(Matrix(Tcc))) < reltol*cond(Matrix(Tcc),1) &&
      norm(Matrix(Tccsinv)-inv(Matrix(Tccs))) < reltol*cond(Matrix(Tccs),1)  &&
      opnormest(Π*Tcr-Tcr*Π) < reltol &&
      opnormest(Π*Tcrinv-Tcrinv*Π) < reltol &&
      opnormest(Π*Tcrsinv-Tcrsinv*Π) < reltol

@test sc*opnorm(Matrix(Tcr),1) < opnormest(Tcr)  &&
      sc*opnorm(Matrix(Tcrinv),1) < opnormest(Tcrinv)  &&
      sc*opnorm(Matrix(Tcrs),1) < opnormest(Tcrs)  &&
      sc*opnorm(Matrix(Tcrsinv),1) < opnormest(Tcrsinv)  &&
      sc*opnorm(Matrix(Tcc),1) < opnormest(Tcc)  &&
      sc*opnorm(Matrix(Tccinv),1) < opnormest(Tccinv)  &&
      sc*opnorm(Matrix(Tccs),1) < opnormest(Tccs)  &&
      sc*opnorm(Matrix(Tccsinv),1) < opnormest(Tccsinv)


@test sc*oprcondest(opnormest(Tcr),Tcrinv) < 1/opnorm(Matrix(Tcr),1)/opnorm(Matrix(Tcrinv),1)  &&
      sc*oprcondest(opnormest(Tcrs),Tcrsinv) < 1/opnorm(Matrix(Tcrs),1)/opnorm(Matrix(Tcrsinv),1)  &&
      sc*oprcondest(opnormest(Tcc),Tccinv) < 1/opnorm(Matrix(Tcc),1)/opnorm(Matrix(Tccinv),1)  &&
      sc*oprcondest(opnormest(Tccs),Tccsinv) < 1/opnorm(Matrix(Tccs),1)/opnorm(Matrix(Tccsinv),1)

@test lyapsepest(ar) == oprcondest(1,Tcrsinv) &&
      lyapsepest(ar') == oprcondest(1,Tcrsinv') &&
      lyapsepest(as) == oprcondest(1,Tcrsinv) &&
      lyapsepest(as') == oprcondest(1,Tcrsinv')  &&
      lyapsepest(ac) == oprcondest(1,Tccsinv) &&
      lyapsepest(ac') == oprcondest(1,Tccsinv') &&
      lyapsepest(acs) == oprcondest(1,Tccsinv) &&
      lyapsepest(acs') == oprcondest(1,Tccsinv') &&
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
      norm(Matrix(Tcrinv)-inv(Matrix(Tcr))) < reltol*cond(Matrix(Tcr),1) &&
      norm(Matrix(Tcrsinv)-inv(Matrix(Tcrs))) < reltol*cond(Matrix(Tcrs),1) &&
      norm(Matrix(Tccinv)-inv(Matrix(Tcc))) < reltol*cond(Matrix(Tcc),1) &&
      norm(Matrix(Tccsinv)-inv(Matrix(Tccs))) < reltol*cond(Matrix(Tccs),1) &&
      opnormest(Π*Tcr-Tcr*Π) < reltol &&
      opnormest(Π*Tcrinv-Tcrinv*Π) < reltol*norm(ar)*norm(er) &&
      opnormest(Π*Tcrsinv-Tcrsinv*Π) < reltol*norm(ar)*norm(er)

@test sc*opnorm(Matrix(Tcr),1) < opnormest(Tcr)  &&
      sc*opnorm(Matrix(Tcrinv),1) < opnormest(Tcrinv)  &&
      sc*opnorm(Matrix(Tcrs),1) < opnormest(Tcrs)  &&
      sc*opnorm(Matrix(Tcrsinv),1) < opnormest(Tcrsinv)  &&
      sc*opnorm(Matrix(Tcc),1) < opnormest(Tcc)  &&
      sc*opnorm(Matrix(Tccinv),1) < opnormest(Tccinv)  &&
      sc*opnorm(Matrix(Tccs),1) < opnormest(Tccs)  &&
      sc*opnorm(Matrix(Tccsinv),1) < opnormest(Tccsinv)


@test sc*oprcondest(opnormest(Tcr),Tcrinv) < 1/opnorm(Matrix(Tcr),1)/opnorm(Matrix(Tcrinv),1)  &&
      sc*oprcondest(opnormest(Tcrs),Tcrsinv) < 1/opnorm(Matrix(Tcrs),1)/opnorm(Matrix(Tcrsinv),1)  &&
      sc*oprcondest(opnormest(Tcc),Tccinv) < 1/opnorm(Matrix(Tcc),1)/opnorm(Matrix(Tccinv),1)  &&
      sc*oprcondest(opnormest(Tccs),Tccsinv) < 1/opnorm(Matrix(Tccs),1)/opnorm(Matrix(Tccsinv),1)

@test lyapsepest(ar,er) == oprcondest(1,Tcrsinv) &&
      lyapsepest(ar',er') == oprcondest(1,Tcrsinv') &&
      lyapsepest(as,es) == oprcondest(1,Tcrsinv) &&
      lyapsepest(as',es') == oprcondest(1,Tcrsinv')  &&
      lyapsepest(ac,ec) == oprcondest(1,Tccsinv) &&
      lyapsepest(ac',ec') == oprcondest(1,Tccsinv') &&
      lyapsepest(acs,ecs) == oprcondest(1,Tccsinv) &&
      lyapsepest(acs',ecs') == oprcondest(1,Tccsinv') &&
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
      norm(Matrix(Tdrinv)-inv(Matrix(Tdr))) < reltol &&
      norm(Matrix(Tdrsinv)-inv(Matrix(Tdrs))) < reltol &&
      norm(Matrix(Tdcinv)-inv(Matrix(Tdc))) < reltol &&
      norm(Matrix(Tdcsinv)-inv(Matrix(Tdcs))) < reltol &&
      opnormest(Π*Tdr-Tdr*Π) < reltol &&
      opnormest(Π*Tdrinv-Tdrinv*Π) < reltol &&
      opnormest(Π*Tdrsinv-Tdrsinv*Π) < reltol


@test sc*opnorm(Matrix(Tdr),1) < opnormest(Tdr)  &&
      sc*opnorm(Matrix(Tdrinv),1) < opnormest(Tdrinv)  &&
      sc*opnorm(Matrix(Tdrs),1) < opnormest(Tdrs)  &&
      sc*opnorm(Matrix(Tdrsinv),1) < opnormest(Tdrsinv)  &&
      sc*opnorm(Matrix(Tdc),1) < opnormest(Tdc)  &&
      sc*opnorm(Matrix(Tdcinv),1) < opnormest(Tdcinv)  &&
      sc*opnorm(Matrix(Tdcs),1) < opnormest(Tdcs)  &&
      sc*opnorm(Matrix(Tdcsinv),1) < opnormest(Tdcsinv)


@test sc*oprcondest(opnormest(Tdr),Tdrinv) < 1/opnorm(Matrix(Tdr),1)/opnorm(Matrix(Tdrinv),1)  &&
      sc*oprcondest(opnormest(Tdrs),Tdrsinv) < 1/opnorm(Matrix(Tdrs),1)/opnorm(Matrix(Tdrsinv),1)  &&
      sc*oprcondest(opnormest(Tdc),Tdcinv) < 1/opnorm(Matrix(Tdc),1)/opnorm(Matrix(Tdcinv),1)  &&
      sc*oprcondest(opnormest(Tdcs),Tdcsinv) < 1/opnorm(Matrix(Tdcs),1)/opnorm(Matrix(Tdcsinv),1)


@test lyapsepest(ar,disc=true) == oprcondest(1,Tdrsinv) &&
      lyapsepest(ar',disc=true) == oprcondest(1,Tdrsinv') &&
      lyapsepest(as,disc=true) == oprcondest(1,Tdrsinv) &&
      lyapsepest(as',disc=true) == oprcondest(1,Tdrsinv')  &&
      lyapsepest(ac,disc=true) == oprcondest(1,Tdcsinv) &&
      lyapsepest(ac',disc=true) == oprcondest(1,Tdcsinv') &&
      lyapsepest(acs,disc=true) == oprcondest(1,Tdcsinv) &&
      lyapsepest(acs',disc=true) == oprcondest(1,Tdcsinv') &&
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
      norm(Matrix(Tdrinv)-inv(Matrix(Tdr))) < reltol &&
      norm(Matrix(Tdrsinv)-inv(Matrix(Tdrs))) < reltol &&
      norm(Matrix(Tdcinv)-inv(Matrix(Tdc))) < reltol &&
      norm(Matrix(Tdcsinv)-inv(Matrix(Tdcs))) < reltol &&
      opnormest(Π*Tdr-Tdr*Π) < reltol &&
      opnormest(Π*Tdrinv-Tdrinv*Π) < reltol &&
      opnormest(Π*Tdrsinv-Tdrsinv*Π) < reltol


@test sc*opnorm(Matrix(Tdr),1) < opnormest(Tdr)  &&
      sc*opnorm(Matrix(Tdrinv),1) < opnormest(Tdrinv)  &&
      sc*opnorm(Matrix(Tdrs),1) < opnormest(Tdrs)  &&
      sc*opnorm(Matrix(Tdrsinv),1) < opnormest(Tdrsinv)  &&
      sc*opnorm(Matrix(Tdc),1) < opnormest(Tdc)  &&
      sc*opnorm(Matrix(Tdcinv),1) < opnormest(Tdcinv)  &&
      sc*opnorm(Matrix(Tdcs),1) < opnormest(Tdcs)  &&
      sc*opnorm(Matrix(Tdcsinv),1) < opnormest(Tdcsinv)


@test sc*oprcondest(opnormest(Tdr),Tdrinv) < 1/opnorm(Matrix(Tdr),1)/opnorm(Matrix(Tdrinv),1)  &&
      sc*oprcondest(opnormest(Tdrs),Tdrsinv) < 1/opnorm(Matrix(Tdrs),1)/opnorm(Matrix(Tdrsinv),1)  &&
      sc*oprcondest(opnormest(Tdc),Tdcinv) < 1/opnorm(Matrix(Tdc),1)/opnorm(Matrix(Tdcinv),1)  &&
      sc*oprcondest(opnormest(Tdcs),Tdcsinv) < 1/opnorm(Matrix(Tdcs),1)/opnorm(Matrix(Tdcsinv),1)

@test lyapsepest(ar,er,disc=true) == oprcondest(1,Tdrsinv) &&
      lyapsepest(ar',er',disc=true) == oprcondest(1,Tdrsinv') &&
      lyapsepest(as,es,disc=true) == oprcondest(1,Tdrsinv) &&
      lyapsepest(as',es',disc=true) == oprcondest(1,Tdrsinv')  &&
      lyapsepest(ac,ec,disc=true) == oprcondest(1,Tdcsinv) &&
      lyapsepest(ac',ec',disc=true) == oprcondest(1,Tdcsinv') &&
      lyapsepest(acs,ecs,disc=true) == oprcondest(1,Tdcsinv) &&
      lyapsepest(acs',ecs',disc=true) == oprcondest(1,Tdcsinv') &&
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
      norm(Matrix(Tcrinv)-inv(Matrix(Tcr))) < reltol &&
      norm(Matrix(Tcrsinv)-inv(Matrix(Tcrs))) < reltol &&
      norm(Matrix(Tccinv)-inv(Matrix(Tcc))) < reltol &&
      norm(Matrix(Tccsinv)-inv(Matrix(Tccs))) < reltol


@test sc*opnorm(Matrix(Tcr),1) < opnormest(Tcr)  &&
      sc*opnorm(Matrix(Tcrinv),1) < opnormest(Tcrinv)  &&
      sc*opnorm(Matrix(Tcrs),1) < opnormest(Tcrs)  &&
      sc*opnorm(Matrix(Tcrsinv),1) < opnormest(Tcrsinv)  &&
      sc*opnorm(Matrix(Tcc),1) < opnormest(Tcc)  &&
      sc*opnorm(Matrix(Tccinv),1) < opnormest(Tccinv)  &&
      sc*opnorm(Matrix(Tccs),1) < opnormest(Tccs)  &&
      sc*opnorm(Matrix(Tccsinv),1) < opnormest(Tccsinv)

@test sc*oprcondest(opnormest(Tcr),Tcrinv) < 1/opnorm(Matrix(Tcr),1)/opnorm(Matrix(Tcrinv),1)  &&
      sc*oprcondest(opnormest(Tcrs),Tcrsinv) < 1/opnorm(Matrix(Tcrs),1)/opnorm(Matrix(Tcrsinv),1)  &&
      sc*oprcondest(opnormest(Tcc),Tccinv) < 1/opnorm(Matrix(Tcc),1)/opnorm(Matrix(Tccinv),1)  &&
      sc*oprcondest(opnormest(Tccs),Tccsinv) < 1/opnorm(Matrix(Tccs),1)/opnorm(Matrix(Tccsinv),1)

@test sylvsepest(ar,br) == oprcondest(1,Tcrsinv) &&
      sylvsepest(ar',br') == oprcondest(1,Tcrsinv') &&
      sylvsepest(as,bs) == oprcondest(1,Tcrsinv) &&
      sylvsepest(as',bs') == oprcondest(1,Tcrsinv')  &&
      sylvsepest(ac,bc) == oprcondest(1,Tccsinv) &&
      sylvsepest(ac',bc') == oprcondest(1,Tccsinv') &&
      sylvsepest(acs,bcs) == oprcondest(1,Tccsinv) &&
      sylvsepest(acs',bcs') == oprcondest(1,Tccsinv') &&
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
      norm(Matrix(Tdrinv)-inv(Matrix(Tdr))) < reltol &&
      norm(Matrix(Tdrsinv)-inv(Matrix(Tdrs))) < reltol &&
      norm(Matrix(Tdcinv)-inv(Matrix(Tdc))) < reltol &&
      norm(Matrix(Tdcsinv)-inv(Matrix(Tdcs))) < reltol



@test sc*opnorm(Matrix(Tdr),1) < opnormest(Tdr)  &&
      sc*opnorm(Matrix(Tdrinv),1) < opnormest(Tdrinv)  &&
      sc*opnorm(Matrix(Tdrs),1) < opnormest(Tdrs)  &&
      sc*opnorm(Matrix(Tdrsinv),1) < opnormest(Tdrsinv)  &&
      sc*opnorm(Matrix(Tdc),1) < opnormest(Tdc)  &&
      sc*opnorm(Matrix(Tdcinv),1) < opnormest(Tdcinv)  &&
      sc*opnorm(Matrix(Tdcs),1) < opnormest(Tdcs)  &&
      sc*opnorm(Matrix(Tdcsinv),1) < opnormest(Tdcsinv)


@test sylvsepest(ar,br,disc=true) == oprcondest(1,Tdrsinv) &&
      sylvsepest(ar',br',disc=true) == oprcondest(1,Tdrsinv') &&
      sylvsepest(as,bs,disc=true) == oprcondest(1,Tdrsinv) &&
      sylvsepest(as',bs',disc=true) == oprcondest(1,Tdrsinv')  &&
      sylvsepest(ac,bc,disc=true) == oprcondest(1,Tdcsinv) &&
      sylvsepest(ac',bc',disc=true) == oprcondest(1,Tdcsinv') &&
      sylvsepest(acs,bcs,disc=true) == oprcondest(1,Tdcsinv) &&
      sylvsepest(acs',bcs',disc=true) == oprcondest(1,Tdcsinv') &&
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
      norm(Matrix(Trinv)-inv(Matrix(Tr))) < reltol &&
      norm(Matrix(Trsinv)-inv(Matrix(Trs))) < reltol &&
      norm(Matrix(Tcinv)-inv(Matrix(Tc))) < reltol &&
      norm(Matrix(Tcsinv)-inv(Matrix(Tcs))) < reltol


@test sc*opnorm(Matrix(Tr),1) < opnormest(Tr)  &&
      sc*opnorm(Matrix(Trinv),1) < opnormest(Trinv)  &&
      sc*opnorm(Matrix(Trs),1) < opnormest(Trs)  &&
      sc*opnorm(Matrix(Trsinv),1) < opnormest(Trsinv)  &&
      sc*opnorm(Matrix(Tc),1) < opnormest(Tc)  &&
      sc*opnorm(Matrix(Tcinv),1) < opnormest(Tcinv)  &&
      sc*opnorm(Matrix(Tcs),1) < opnormest(Tcs)  &&
      sc*opnorm(Matrix(Tcsinv),1) < opnormest(Tcsinv)

@test sc*oprcondest(opnormest(Tr),Trinv) < 1/opnorm(Matrix(Tr),1)/opnorm(Matrix(Trinv),1)  &&
      sc*oprcondest(opnormest(Trs),Trsinv) < 1/opnorm(Matrix(Trs),1)/opnorm(Matrix(Trsinv),1)  &&
      sc*oprcondest(opnormest(Tc),Tcinv) < 1/opnorm(Matrix(Tc),1)/opnorm(Matrix(Tcinv),1)  &&
      sc*oprcondest(opnormest(Tcs),Tcsinv) < 1/opnorm(Matrix(Tcs),1)/opnorm(Matrix(Tcsinv),1)

@test sylvsepest(ar,br,cr,dr) == oprcondest(1,Trsinv) &&
      sylvsepest(ar',br',cr',dr') == oprcondest(1,Trsinv') &&
      sylvsepest(as,bs,cs,ds) == oprcondest(1,Trsinv) &&
      sylvsepest(as',bs',cs',ds') == oprcondest(1,Trsinv')  &&
      sylvsepest(ac,bc,cc,dc) == oprcondest(1,Tcsinv) &&
      sylvsepest(ac',bc',cc',dc') == oprcondest(1,Tcsinv') &&
      sylvsepest(acs,bcs,ccs,dcs) == oprcondest(1,Tcsinv) &&
      sylvsepest(acs',bcs',ccs',dcs') == oprcondest(1,Tcsinv') &&
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
      norm(Matrix(Trinv)-inv(Matrix(Tr))) < reltol &&
      norm(Matrix(Trsinv)-inv(Matrix(Trs))) < reltol &&
      norm(Matrix(Tcinv)-inv(Matrix(Tc))) < reltol &&
      norm(Matrix(Tcsinv)-inv(Matrix(Tcs))) < reltol


@test sc*opnorm(Matrix(Tr),1) < opnormest(Tr)  &&
      sc*opnorm(Matrix(Trinv),1) < opnormest(Trinv)  &&
      sc*opnorm(Matrix(Trs),1) < opnormest(Trs)  &&
      sc*opnorm(Matrix(Trsinv),1) < opnormest(Trsinv)  &&
      sc*opnorm(Matrix(Tc),1) < opnormest(Tc)  &&
      sc*opnorm(Matrix(Tcinv),1) < opnormest(Tcinv)  &&
      sc*opnorm(Matrix(Tcs),1) < opnormest(Tcs)  &&
      sc*opnorm(Matrix(Tcsinv),1) < opnormest(Tcsinv)

@test sc*oprcondest(opnormest(Tr),Trinv) < 1/opnorm(Matrix(Tr),1)/opnorm(Matrix(Trinv),1)  &&
      sc*oprcondest(opnormest(Trs),Trsinv) < 1/opnorm(Matrix(Trs),1)/opnorm(Matrix(Trsinv),1)  &&
      sc*oprcondest(opnormest(Tc),Tcinv) < 1/opnorm(Matrix(Tc),1)/opnorm(Matrix(Tcinv),1)  &&
      sc*oprcondest(opnormest(Tcs),Tcsinv) < 1/opnorm(Matrix(Tcs),1)/opnorm(Matrix(Tcsinv),1)


@test sylvsyssepest(ar,br,cr,dr) == oprcondest(1,Trsinv) &&
      sylvsyssepest(as,bs,cs,ds) == oprcondest(1,Trsinv) &&
      sylvsyssepest(ac,bc,cc,dc) == oprcondest(1,Tcsinv) &&
      sylvsyssepest(acs,bcs,ccs,dcs) == oprcondest(1,Tcsinv) &&
      sylvsyssepest(as,bs,cs,ds)/sqrt(2*n*m) <= minimum(svdvals(Matrix(Trs)))  &&
      minimum(svdvals(Matrix(Trs))) <= sqrt(2*n*m)*sylvsyssepest(as,bs,cs,ds)  &&
      sylvsyssepest([0. 1.; 0. 1.],[0. 1.; 0. 1.],-[0. 1.; 0. 1.],[0. 1.; 0. 1.]) == 0.

end

end

end
