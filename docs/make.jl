using Documenter, MatrixEquations
DocMeta.setdocmeta!(MatrixEquations, :DocTestSetup, :(using MatrixEquations); recursive=true)

makedocs(
  modules  = [MatrixEquations],
  sitename = "MatrixEquations.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
     "Home"   => "index.md",
     "Library" => [
        "lyapunov.md",
        "plyapunov.md",
        "riccati.md",
        "sylvester.md",
        "sylvkr.md",
        "condest.md",
        "meoperators.md"
     ],
     "Internal" => [
        "lapackutil.md",
        "meutil.md"
     ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(deps = nothing, make = nothing,
  repo = "github.com/andreasvarga/MatrixEquations.jl.git",
  target = "build",
  devbranch = "master"
)
