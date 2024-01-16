module Runtests

using Test, MatrixEquations

@testset "Test MatrixEquations.jl" begin
    include("test_clyap.jl")
    include("test_dlyap.jl")
    include("test_meutil.jl")
    include("test_riccati.jl")
    include("test_sylvester.jl")
    include("test_sylvkr.jl")
    include("test_cplyap.jl")
    include("test_dplyap.jl")
    include("test_mecondest.jl")
    include("test_iterative.jl")
end

end
