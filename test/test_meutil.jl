using LinearAlgebra
using MatrixEquations
using Test

n = 500
ar, ur, = schur(rand(n,n))
cr = rand(n,n)
qr = cr'*cr

@testset "Various usages of utqu! and utqu" begin
x = copy(qr); @time utqu!(x,ur);
@time y = ur'*qr*ur;
@test norm(x-y) < sqrt(eps(1.))

x = copy(qr); @time utqu!(x,ur,adj=true);
@time y = ur*qr*ur';
@test norm(x-y) < sqrt(eps(1.))

@time x = utqu(qr,ur);
@time y = ur'*qr*ur;
@test norm(x-y) < sqrt(eps(1.))

@time x = utqu(qr,ur,adj=true);
@time y = ur*qr*ur';
@test norm(x-y) < sqrt(eps(1.))
end
