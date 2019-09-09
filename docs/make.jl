using Documenter, MatrixEquations

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
        "sylvkr.md"
     ],
     "Internal" => [
        "lapackutil.md",
        "meutil.md"
     ],
     "Index" => "makeindex.md"
  ]
)
