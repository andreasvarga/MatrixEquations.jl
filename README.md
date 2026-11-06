# MatrixEquations.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3556867.svg)](https://doi.org/10.5281/zenodo.3556867)
[![DocBuild](https://github.com/andreasvarga/MatrixEquations.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/MatrixEquations.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixEquations.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixEquations.jl?branch=master)
[![stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreasvarga.github.io/MatrixEquations.jl/stable/)
[![dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://andreasvarga.github.io/MatrixEquations.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) 
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FMatrixEquations&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/MatrixEquations)
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-Article-informational?style=flat&logo=linkedin&logoColor=white&color=0D76A8)](https://www.linkedin.com/pulse/matrixequations-solving-matrix-equations-julia-andreas-varga-wauue/?trackingId=bSzyqKO0Rl2faMhNEX%2BMeQ%3D%3D)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/DescriptorSystems.jl/blob/main/LICENSE.md)


**Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia**

## Compatibility

Julia 1.10 and higher.

## How to Install

````Julia
pkg> add MatrixEquations
pkg> test MatrixEquations
````

## About

This collection of Julia functions implements high performance
numerical software to solve several classes of Lyapunov, Sylvester and Riccati matrix equations at a performance level similar with that of efficient structure exploiting Fortran implementations available in the Systems and Control Library [SLICOT](https://github.com/SLICOT) and the Linear Algebra Package [LAPACK](https://github.com/Reference-LAPACK/lapack). 
The available functions in the `MatrixEquation.jl` package cover both standard
and generalized continuous and discrete Lyapunov, Sylvester and Riccati equations for both real and complex data. The functions for the solution of Lyapunov and Sylvester equations rely on efficient structure exploiting solvers for which the input data are in Schur or generalized Schur forms. A comprehensive set of Lyapunov and Sylvester operators has been implemented, which allow the estimation of condition numbers of these operators. The implementation of Riccati equation solvers employ orthogonal Schur vectors based methods and their extensions to linear matrix pencil based reduction approaches. 

The current version of the package includes the following functions:

**Solution of Lyapunov equations**

* **lyapc**   Solution of the continuous Lyapunov equations `AX+XA'+C = 0` and `AXE'+EXA'+C = 0`.
* **lyapd**  Solution of discrete Lyapunov equations `AXA'-X +C = 0` and `AXA'-EXE'+C = 0`.
* **tlyapc** Solution of the continuous T-Lyapunov equations `AX±XᵀAᵀ+C = 0`.
* **hlyapc** Solution of the continuous H-Lyapunov equations `AX±X'A'+C = 0`.
* **tulyapc!** Solution of the continuous T-Lyapunov equations `UXᵀ+XUᵀ = C` and `UᵀX+XᵀU = C` for `U` and `X` upper triangular.
* **hulyapc!** Solution of the continuous H-Lyapunov equations `UX'+XU' = C` and `U'X+X'U = C` for `U` and `X` upper triangular.
* **plyapc**  Solution of the positive continuous Lyapunov equations `AX+XA'+BB' = 0` and `AXE'+EXA'+BB' = 0`.
* **plyapd**  Solution of the positive discrete Lyapunov equations `AXA'-X +C = 0` and `AXA'-EXE'+C = 0`.

 **Solution of algebraic  Riccati equations**

* **arec**  Solution of the continuous Riccati equations `A'X+XA-XRX+Q = 0` and
 `A'X+XA-(XB+S)R⁻¹(B'X+S')+Q = 0`.
* **narec**  Solution of the nonsymmetric/nonhermitian continuous Riccati equation 
`A₂'X+XA₁-(XB₁+S₁)R⁻¹(B₂'X+S₂')+Q = 0`.
* **garec** Solution of the generalized continuous Riccati equation
 `A'XE+E'XA-(E'XB+S)R⁻¹(B'XE+S')+Q = 0`.
* **ared** Solution of the discrete Riccati equation
 `A'XA - X - (A'XB+S)(R+B'XB)⁻¹(B'XA+S') + Q = 0`.
* **nared** Solution of the nonsymmetric/nonhermitian discrete Riccati equation
 `A₂'XA₁ - X - (A₂'XB₁+S₁)(R+B₂'XB₁)⁻¹(B₂'XA₁+S₂') + Q = 0`.
* **gared**  Solution of the generalized discrete Riccati equation
 `A'XA - E'XE - (A'XB+S)(R+B'XB)⁻¹(B'XA+S') + Q = 0`.
* **nare**  Solution of the nonsymmetric/nonhermitian Riccati equation `A₁X+XA₂-XRX+Q = 0`.
* **uqme**  Solution of the unilateral quadratic matrix equation `AX² + BX + C = 0`.
 

 **Solution of Sylvester equations and systems**

* **sylvc** Solution of the (continuous) Sylvester equation `AX+XB = C`.
* **sylvd** Solution of the (discrete) Sylvester equation `AXB+X = C`.
* **gsylv** Solution of the generalized Sylvester equation `AXB+CXD = E`.
* **sylvsys** Solution of the Sylvester system of matrix equations `AX+YB = C, DX+YE = F`.
* **dsylvsys** Solution of the dual Sylvester system of matrix equations `AX+DY = C, XB+YE = F`.

**Iterative solution of linear matrix equations**

* **lyapci** Iterative solution of the continuous Lyapunov equations `AX+XA'+C = 0` and `AXE'+EXA'+C = 0`.
* **plyapci** Iterative solution of the low-rank positive continuous Lyapunov equations `AX+XA'+BB' = 0` and `AXE'+EXA'+BB' = 0`.
* **lyapdi**  Iterative solution of the discrete Lyapunov equations `AXA'-X+C = 0` and `AXA'-EXE'+C = 0`.
* **plyapdi** Iterative solution of the low-rank positive discrete Lyapunov equations `AXA'-X+BB' = 0` and `AXA'-EXE'+BB' = 0`.
* **tlyapci** Iterative solution of the continuous T-Lyapunov equations `AX+XᵀAᵀ = C` and `AXᵀ+XAᵀ = C`.
* **hlyapci** Iterative solution of the continuous H-Lyapunov equations `AX+X'A' = C` and `AX'+XA' = C`.
* **tulyapci** Iterative solution of the continuous T-Lyapunov equations `UXᵀ+XUᵀ = C` and `UᵀX+XᵀU = C` for `U` and `X` upper triangular.
* **hulyapci** Iterative solution of the continuous H-Lyapunov equations `UX'+XU' = C` and `U'X+X'U = C` for `U` and `X` upper triangular.
* **sylvci** Iterative solution of the (continuous) Sylvester equation `AX+XB = C`.
* **sylvdi** Iterative solution of the (discrete) Sylvester equation `AXB+X = C`.
* **gsylvi** Iterative solution of the generalized Sylvester equation `AXB+CXD = E`.
* **gtsylvi** Iterative solution of the generalized T-Sylvester equations `∑AᵢXBᵢ + ∑CⱼXᵀDⱼ = E`.
* **ghsylvi** Iterative solution of the generalized H-Sylvester equations `∑AᵢXBᵢ + ∑CⱼX'Dⱼ = E`.
* **cgls** The conjugate gradient method for nonsymmetric linear equations and least squares problems.

**Norm, condition and separation estimation**

* **opnorm1**  Computation of the 1-norm of a linear operator.
* **opnorm1est** Estimation of 1-norm of a linear operator.
* **oprcondest** Estimation of the reciprocal 1-norm condition number of a linear operator.
* **opsepest** Estimation of the 1-norm separation a linear operator.

The general solvers of Lyapunov and Sylvester equations rely on a set of specialized solvers for real or complex matrices in appropriate Schur forms. For large dimensions these solvers employ recursive-blocking based algorithms which ensure substantial performance boosts. For testing purposes, a set of solvers for various linear matrix equations has been implemented, which employ the Kronecker-product expansion of the equations. These solvers are not recommended for large order matrices. Based on the conjugate gradient method to solve linear systems or least-squares problems, several iterative solvers have been implemented, which potentially are applicable to solve linear matrix equations with large order dense and sparse matrices. The norms, reciprocal condition numbers and separations can be estimated for a comprehensive set of predefined Lyapunov and Sylvester operators. The last additions include solvers for nonsymmetric/nonhermitian algebraic Riccati equations and quadratic matrix equations.  
