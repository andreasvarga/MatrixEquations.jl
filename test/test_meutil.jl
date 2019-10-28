module Test_meutil

using LinearAlgebra
using MatrixEquations
using Test

n = 500
ar, ur, = schur(rand(n,n))
cr = rand(n,n)
Qr = cr'*cr

@testset "Matrix Equations Utilities" begin
x = copy(Qr); @time utqu!(x,ur);
@time y = ur'*Qr*ur;
@test norm(x-y) < sqrt(eps(1.))

x = copy(Qr); @time utqu!(x,ur');
@time y = ur*Qr*ur';
@test norm(x-y) < sqrt(eps(1.))

@time x = utqu(Qr,ur);
@time y = ur'*Qr*ur;
@test norm(x-y) < sqrt(eps(1.))

@time x = utqu(Qr,ur');
@time y = ur*Qr*ur';
@test norm(x-y) < sqrt(eps(1.))

@time x = vec2her(her2vec(Qr, rowwise = false, check = true),rowwise = false)
@test x == Qr

@time x = vec2her(her2vec(Qr, rowwise = true, check = true),rowwise = true)
@test x == Qr

@time x = vec2triu(triu2vec(Qr, rowwise = false, her = false),rowwise = false, her = false)
@test x == triu(Qr)

@time x = vec2triu(triu2vec(Qr, rowwise = false, her = true),rowwise = false, her = true)
@test x == Qr

@time x = vec2triu(triu2vec(Qr, rowwise = true, her = false),rowwise = true, her = false)
@test x == triu(Qr)

@time x = vec2triu(triu2vec(Qr, rowwise = true, her = true),rowwise = true, her = true)
@test x == Qr

end

end
