using Documenter, MatrixEquations
DocMeta.setdocmeta!(MatrixEquations, :DocTestSetup, :(using MatrixEquations); recursive=true)

makedocs(warnonly = true, 
  modules  = [MatrixEquations],
  sitename = "MatrixEquations.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
     "Overview"   => "index.md",
     "Library" => [
        "lyapunov.md",
        "plyapunov.md",
        "riccati.md",
        "sylvester.md",
        "sylvkr.md",
        "condest.md",
        "meoperators.md",
        "iterative.md"
     ],
     "Internal" => [
        "lapackutil.md",
        "meutil.md"
     ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(
  repo = "github.com/andreasvarga/MatrixEquations.jl.git",
  push_preview = true,
  target = "build",
  )
