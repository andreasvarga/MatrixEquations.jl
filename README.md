# MatrixEquations.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3556867.svg)](https://doi.org/10.5281/zenodo.3556867)
[![DocBuild](https://github.com/andreasvarga/MatrixEquations.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/MatrixEquations.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixEquations.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixEquations.jl?branch=master)
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreasvarga.github.io/MatrixEquations.jl/stable/)
[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andreasvarga.github.io/MatrixEquations.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) 
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FMatrixEquations&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/MatrixEquations)

**Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia**

## Compatibility

Julia 1.10 and higher.

## How to Install

````Julia
pkg> add MatrixEquations
pkg> test MatrixEquations
````

## About

This collection of Julia functions is an attemp to implement high performance
numerical software to solve several classes of Lyapunov, Sylvester and Riccati matrix equations
at a performance level comparable with efficient structure exploiting Fortran implementations, as those available in the Systems and Control Library [SLICOT](https://github.com/SLICOT).
This goal has been fully achieved for Lyapunov and Sylvester equation solvers, for which the
codes for both real and complex data perform at practically same performance level as similar functions available in
the MATLAB Control System Toolbox (which rely on SLICOT).

The available functions in the `MatrixEquation.jl` package cover both standard
and generalized continuous and discrete Lyapunov, Sylvester and Riccati equations for both real and complex data. The functions for the solution of Lyapunov and Sylvester equations rely on efficient structure
exploiting solvers for which the input data are in Schur or generalized Schur forms. A comprehensive set of Lyapunov and Sylvester operators has been implemented, which allow the estimation of condition numbers of these operators. The implementation of Riccati equation solvers employ orthogonal Schur vectors
based methods and their extensions to linear matrix pencil based reduction approaches. The calls of all functions with adjoint (in complex case) or transposed (in real case) arguments are fully supported by appropriate computational algorithms, thus the matrix copying operations are mostly avoided.

The current version of the package includes the following functions:

**Solution of Lyapunov equations**

* **lyapc**   Solution of the continuous Lyapunov equations `AX+XA'+C = 0` and `AXE'+EXA'+C = 0`.
* **lyapd**  Solution of discrete Lyapunov equations `AXA'-X +C = 0` and `AXA'-EXE'+C = 0`.
* **tlyapc** Solution of the continuous T-Lyapunov equations `A*X±X^⊤*A^⊤+C = 0`.
* **hlyapc** Solution of the continuous H-Lyapunov equations `A*X±X'*A'+C = 0`.
* **tulyapc!** Solution of the continuous T-Lyapunov equations `U*X^⊤+X*U^⊤ = C` and `U^⊤*X+X^⊤*U = C` for `U` and `X` upper triangular.
* **hulyapc!** Solution of the continuous H-Lyapunov equations `U*X'+X*U' = C` and `U'*X+X'*U = C` for `U` and `X` upper triangular.
* **plyapc**  Solution of the positive continuous Lyapunov equations `AX+XA'+BB' = 0` and `AXE'+EXA'+BB' = 0`.
* **plyapd**  Solution of the positive discrete Lyapunov equations `AXA'-X +C = 0` and `AXA'-EXE'+C = 0`.

 **Solution of algebraic  Riccati equations**

* **arec**  Solution of the continuous Riccati equations `A'X+XA-XRX+Q = 0` and
 `A'X+XA-(XB+S)R^(-1)(B'X+S')+Q = 0`.
* **garec** Solution of the generalized continuous Riccati equation
 `A'XE+E'XA-(A'XB+S)R^(-1)(B'XA+S')+Q = 0`.
* **ared** Solution of the discrete Riccati equation
 `A'XA - X - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0`.
* **gared**  Solution of the generalized discrete Riccati equation
 `A'XA - E'XE - (A'XB+S)(R+B'XB)^(-1)(B'XA+S') + Q = 0`.

 **Solution of Sylvester equations and systems**

* **sylvc** Solution of the (continuous) Sylvester equation `AX+XB = C`.
* **sylvd** Solution of the (discrete) Sylvester equation `AXB+X = C`.
* **gsylv** Solution of the generalized Sylvester equation `AXB+CXD = E`.
* **sylvsys** Solution of the Sylvester system of matrix equations `AX+YB = C, DX+YE = F`.
* **dsylvsys** Solution of the dual Sylvester system of matrix equations `AX+DY = C, XB+YE = F`.

**Iterative solution of linear matrix equations**

* **lyapci** Iterative solution of the continuous Lyapunov equations `AX+XA'+C = 0` and `AXE'+EXA'+C = 0`.
* **lyapdi**  Iterative solution of the discrete Lyapunov equations `AXA'-X+C = 0` and `AXA'-EXE'+C = 0`.
* **tlyapci** Iterative solution of the continuous T-Lyapunov equations `A*X+X^⊤*A^⊤ = C` and `A*X^⊤+X*A^⊤ = C`.
* **hlyapci** Iterative solution of the continuous H-Lyapunov equations `A*X+X'*A' = C` and `A*X'+X*A' = C`.
* **tulyapci** Iterative solution of the continuous T-Lyapunov equations `U*X^⊤+X*U^⊤ = C` and `U^⊤*X+X^⊤*U = C` for `U` and `X` upper triangular.
* **hulyapci** Iterative solution of the continuous H-Lyapunov equations `U*X'+X*U' = C` and `U'*X+X'*U = C` for `U` and `X` upper triangular.
* **sylvci** Iterative solution of the (continuous) Sylvester equation `AX+XB = C`.
* **sylvdi** Iterative solution of the (discrete) Sylvester equation `AXB+X = C`.
* **gsylvi** Iterative solution of the generalized Sylvester equation `AXB+CXD = E`.
* **gtsylvi** Iterative solution of the generalized T-Sylvester equations `∑ A_i*X*B_i + ∑ C_j*X^⊤*D_j = E`.
* **ghsylvi** Iterative solution of the generalized T-Sylvester equations `∑ A_i*X*B_i + ∑ C_j*X'*D_j = E`.
* **cgls** The conjugate gradient method for nonsymmetric linear equations and least squares problems.

**Norm, condition and separation estimation**

* **opnorm1**  Computation of the 1-norm of a linear operator.
* **opnorm1est** Estimation of 1-norm of a linear operator.
* **oprcondest** Estimation of the reciprocal 1-norm condition number of a linear operator.
* **opsepest** Estimation of the 1-norm separation a linear operator.

The general solvers of Lyapunov and Sylvester equations rely on a set of specialized solvers for real or complex matrices in appropriate Schur forms. For testing purposes, a set of solvers for various linear matrix equations has been implemented, which employ the Kronecker-product expansion of the equations. These solvers are not recommended for large order matrices. Based on the conjugate gradient method to solve linear systems or least-squares problems, several iterative solvers have been implemented, which potentially are applicable to solve linear matrix equations with large order dense and sparse matrices. The norms, reciprocal condition numbers and separations can be estimated for a comprehensive set of predefined Lyapunov and Sylvester operators.  
A complete list of implemented functions is available [here](https://sites.google.com/view/andreasvarga/home/software/matrix-equations-in-julia).

## Future plans

The collection of tools can be extended by adding new functionality, such as expert solvers, which additionally compute error bounds and condition estimates, or solvers for new classes of Riccati equations, as those arising in game-theoretic optimization problems. Further performance improvements are still possible (e.g., in some positive Lyapunov solvers by employing specially taylored solvers for the underlying particular Sylvester equations) or by employing blocking based variants of solvers for Lyapunov and Sylvester equations.
