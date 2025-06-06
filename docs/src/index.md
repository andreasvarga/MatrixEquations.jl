```@meta
CurrentModule = MatrixEquations
DocTestSetup = quote
    using MatrixEquations
end
```

# MatrixEquations.jl

[![DocBuild](https://github.com/andreasvarga/MatrixEquations.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/MatrixEquations.jl/actions)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/MatrixEquations.jl)

This collection of Julia functions is an attemp to implement high performance
numerical software to solve classes of Lyapunov, Sylvester and Riccati matrix equations
at a performance level comparable with efficient structure exploiting Fortran implementations, as those available in the Systems and Control Library [SLICOT](https://github.com/SLICOT).
This goal has been fully achieved for Lyapunov and Sylvester equation solvers, for which the
codes for both real and complex data perform at practically same performance level as similar functions available in
the MATLAB Control System Toolbox (which rely on SLICOT).

The available functions in the `MatrixEquations.jl` package cover both standard
and generalized continuous and discrete Lyapunov, Sylvester and Riccati equations for both real and complex data. The functions for the solution of Lyapunov and Sylvester equations rely on efficient structure exploiting solvers for which the input data are in Schur or generalized Schur forms. A comprehensive set of Lyapunov and Sylvester operators has been implemented, which allow the estimation of condition numbers of these operators and the iterative solution of various Lyapunov and Sylvester matrix equations using the conjugate gradient method. The implementation of Riccati equation solvers employ orthogonal Schur vectors
based methods and their extensions to linear matrix pencil based reduction approaches. The calls of all functions with adjoint (in complex case) or transposed (in real case) arguments are fully supported by appropriate computational algorithms, thus the matrix copying operations are mostly avoided.

The current version of the package includes the following functions:

**Solution of Lyapunov equations**

| Function | Description |
| :--- | :--- |
| **[`lyapc`](@ref)**  | Solution of the continuous Lyapunov equations |
| **[`lyapd`](@ref)**  | Solution of the discrete Lyapunov equations |
| **[`tlyapc`](@ref)** | Solution of the continuous T-Lyapunov equations |
| **[`hlyapc`](@ref)** | Solution of the continuous H-Lyapunov equations |
| **[`tulyapc!`](@ref)** | Computation of the upper triangular solution of the continuous T-Lyapunov equation |
| **[`hulyapc!`](@ref)** | Computation of the upper triangular solution of the continuous H-Lyapunov equation |
| **[`plyapc`](@ref)** | Solution of the positive continuous Lyapunov equations|
| **[`plyapd`](@ref)** | Solution of the positive discrete Lyapunov equations|

 **Solution of algebraic  Riccati equations**

| Function | Description |
| :--- | :--- |
| **[`arec`](@ref)**  |  Solution of the continuous Riccati equations|
| **[`garec`](@ref)** |  Solution of the generalized continuous Riccati equation|
| **[`ared`](@ref)**  |  Solution of the discrete Riccati equation|
| **[`gared`](@ref)** |  Solution of the generalized discrete Riccati equation|

 **Solution of Sylvester equations and systems**

| Function | Description |
| :--- | :--- |
| **[`sylvc`](@ref)** | Solution of the (continuous) Sylvester equations|
| **[`sylvd`](@ref)** | Solution of the (discrete) Sylvester equations |
| **[`gsylv`](@ref)** | Solution of the generalized Sylvester equations |
| **[`sylvsys`](@ref)** | Solution of the Sylvester system of matrix equations |
| **[`dsylvsys`](@ref)** | Solution of the dual Sylvester system of matrix equations |


**Iterative solution of linear matrix equations**

| Function | Description |
| :--- | :--- |
| **[`lyapci`](@ref)** |Iterative solution of continuous Lyapunov equations|
| **[`plyapci`](@ref)** |Iterative solution of low-rank non-negative continuous Lyapunov equations|
| **[`lyapdi`](@ref)** |Iterative solution of discrete Lyapunov equations|
| **[`plyapdi`](@ref)** |Iterative solution of low-rank non-negative discrete Lyapunov equations|
| **[`tlyapci`](@ref)** |Iterative solution of the continuous T-Lyapunov equations|
| **[`hlyapci`](@ref)** |Iterative solution of the continuous H-Lyapunov equations|
| **[`tulyapci`](@ref)** |Iterative solution of the continuous T-Lyapunov equations with upper triangular solutions|
| **[`hulyapci`](@ref)** |Iterative solution of the continuous H-Lyapunov equations with upper triangular solutions|
| **[`sylvci`](@ref)** |Iterative solution of the (continuous) Sylvester equations|
| **[`sylvdi`](@ref)** |Iterative solution of the (discrete) Sylvester equations |
| **[`gsylvi`](@ref)** |Iterative solution of the generalized Sylvester equations|
| **[`gtsylvi`](@ref)** |Iterative solution of the generalized T-Sylvester equations|
| **[`ghsylvi`](@ref)** |Iterative solution of the generalized H-Sylvester equations|
| **[`cgls`](@ref)** |The conjugate gradient method for nonsymmetric linear equations and least squares problems|


**Norm, condition and separation estimation of linear operators**

| Function | Description |
| :--- | :--- |
| **[`opnorm1`](@ref)** | Computation of the 1-norm of a linear operator|
| **[`opnorm1est`](@ref)** | Estimation of the 1-norm of a linear operator|
| **[`oprcondest`](@ref)** | Estimation of the reciprocal 1-norm condition number of an operator|
| **[`opsepest`](@ref)** | Estimation of the separation of a linear operator|

The general solvers of Lyapunov and Sylvester equations rely on a set of specialized solvers for real or complex matrices in appropriate Schur forms. For testing purposes, a set of solvers for Sylvester equations has been implemented, which employ the Kronecker-product expansion of the equations. These solvers are not recommended for large order matrices. Based on the conjugate gradient method to solve linear systems or least-squares problems, several iterative solvers have been implemented, which are potentially applicable to solve linear matrix equations with large order dense and sparse matrices. The norms, reciprocal condition numbers and separations can be estimated for a comprehensive set of predefined Lyapunov and Sylvester operators. A complete list of implemented functions is available [here](https://sites.google.com/view/andreasvarga/home/software/matrix-equations-in-julia).

## Future plans

The collection of tools can be extended by adding new functionality, such as expert solvers, which additionally compute error bounds and condition estimates, or solvers for new classes of Sylvester-like equations or Riccati equations (as those arising in game-theoretic optimization problems). Further performance improvements are still possible by employing blocking based variants of solvers for Lyapunov and Sylvester equations.

## [Release Notes](https://github.com/andreasvarga/MatrixEquations.jl/blob/master/ReleaseNotes.md)

## Main developer

[Andreas Varga](https://sites.google.com/view/andreasvarga/home)

License: MIT (expat)
