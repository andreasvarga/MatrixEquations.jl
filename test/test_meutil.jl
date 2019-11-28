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

@time x = vec2triu(triu2vec(Qr, rowwise = false, her = false),rowwise = false, her = false)
@test x == triu(Qr)

@time x = vec2triu(triu2vec(Qr, rowwise = false, her = true),rowwise = false, her = true)
@test x == Qr

@time x = vec2triu(triu2vec(Qr, rowwise = true, her = false),rowwise = true, her = false)
@test x == triu(Qr)

@time x = vec2triu(triu2vec(Qr, rowwise = true, her = true),rowwise = true, her = true)
@test x == Qr

@test isschur(rand(2,3)) == false &&
      isschur(rand(1,1)) == true &&
      isschur(rand(3,3)) == false &&
      isschur(rand(3,3),rand(2,2)) == false

 u = [1 2;3 4]
 q = [1 3; 3 1]

try
    utqu(q,u) 
    utqu(q//2,u) 
    utqu(q//2,u//2) 
    utqu!(q,u) 
    utqu!(q//2,u) 
    utqu(q//2,u//2) 
   @test true   
catch
    @test false   
end     

try
    utqu!(q,u//2) 
    @test false   
catch
    @test true   
end     
     
try
    utqu!(rand(3,3),rand(3,3)) 
    @test false   
catch
    @test true   
end     

try
    utqu!(Symmetric(rand(3,3)),rand(2,2))
    @test false   
catch
    @test true   
end     

try
    utqu(rand(3,3),rand(3,3)) 
    @test false   
catch
    @test true   
end     

try
    utqu(Symmetric(rand(3,3)),rand(2,2))
    @test false   
catch
    @test true   
end     

try
    utqu(Symmetric(rand(3,3)),rand(2,2)')
    @test false   
catch
    @test true   
end     

try
    qrupdate!(triu(rand(4,4)), rand(2,3))
    @test false   
catch
    @test true   
end     

try
    rqupdate!(triu(rand(4,4)), rand(2,3))
    @test false   
catch
    @test true   
end     

try
    triu2vec(rand(2,2), her = true)
    @test false   
catch
    @test true   
end     

end

end
