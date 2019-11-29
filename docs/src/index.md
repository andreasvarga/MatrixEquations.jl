```@meta
CurrentModule = MatrixEquations
DocTestSetup = quote
    using MatrixEquations
end
```

# MatrixEquations.jl

[![Build Status](https://travis-ci.com/andreasvarga/MatrixEquations.jl.svg?branch=master)](https://travis-ci.com/andreasvarga/MatrixEquations.jl)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/MatrixEquations.jl)

This collection of Julia functions is an attemp to implement high performance
numerical software to solve classes of Lyapunov, Sylvester and Riccati matrix equations
at a performance level comparable with efficient structure exploiting Fortran implementations, as those available in the Systems and Control Library [SLICOT](http://slicot.org/).
This goal has been fully achieved for Lyapunov and Sylvester equation solvers, for which the
codes for both real and complex data perform at practically same performance level as similar functions available in
the MATLAB Control System Toolbox (which rely on SLICOT).

The available functions in the `MatrixEquation.jl` package cover both standard
and generalized continuous and discrete Lyapunov, Sylvester and Riccati equations for both real and complex data. The functions for the solution of Lyapunov and Sylvester equations rely on efficient structure exploiting solvers for which the input data are in Schur or generalized Schur forms. A comprehensive set of Lyapunov and Sylvester operators has been implemented, which allow the estimation of condition numbers of these operators. The implementation of Riccati equation solvers employ orthogonal Schur vectors
based methods and their extensions to linear matrix pencil based reduction approaches. The calls of all functions with adjoint (in complex case) or transposed (in real case) arguments are fully supported by appropriate computational algorithms, thus the matrix copying operations are mostly avoided.

The current version of the package includes the following functions:

**Solution of Lyapunov equations**

| Function | Description |
| :--- | :--- |
| **lyapc**  | Solution of the continuous Lyapunov equations |
| **lyapd**  | Solution of the discrete Lyapunov equations |
| **plyapc** | Solution of the positive continuous Lyapunov equations|
| **plyapd** | Solution of the positive discrete Lyapunov equations|

 **Solution of algebraic  Riccati equations**

| Function | Description |
| :--- | :--- |
| **arec**  |  Solution of the continuous Riccati equations|
| **garec** |  Solution of the generalized continuous Riccati equation|
| **ared**  |  Solution of the discrete Riccati equation|
| **gared** |  Solution of the generalized discrete Riccati equation|

 **Solution of Sylvester equations and systems**

| Function | Description |
| :--- | :--- |
|  **sylvc** | Solution of the (continuous) Sylvester equations|
|  **sylvd** | Solution of the (discrete) Sylvester equations |
|  **gsylv** | Solution of the generalized Sylvester equations |
|  **sylvsys** | Solution of the Sylvester system of matrix equations |
|  **dsylvsys** | Solution of the dual Sylvester system of matrix equations |

**Norm, condition and separation estimation of linear operators**

| Function | Description |
| :--- | :--- |
|  **opnorm1** | Computation of the 1-norm of a linear operator|
|  **opnorm1est** | Estimation of the 1-norm of a linear operator|
|  **oprcondest** | Estimation of the reciprocal 1-norm condition number of an operator|
|  **opsepest** | Estimation of the separation of a linear operator|

The general solvers of Lyapunov and Sylvester equations rely on a set of specialized solvers for real or complex matrices in appropriate Schur forms. For testing purposes, a set of solvers for Sylvester equations has been implemented, which employ the Kronecker-product expansion of the equations. These solvers are not recommended for large order matrices. The norms, reciprocal condition numbers and separations can be estimated for a comprehensive set of predefined Lyapunov and Sylvester operators. A complete list of implemented functions is available [here](https://sites.google.com/site/andreasvargacontact/home/software/matrix-equations-in-julia).

## Future plans

The collection of tools will be extended by adding new functionality, such as expert solvers which additionally compute error bounds and condition estimates. Furthermore, performance improvements are planned to be implemented employing more efficient and accurate low dimensional linear system solvers available in LAPACK, using static arrays for manipulation of small order matrices, and exploring block variant solvers for Lyapunov and Sylvester equations.

## Release Notes

### Version 1.1.0

This release includes several enhancements of the Riccati equation solvers:

- Enhanced functionality to determine anti-stabilizing solutions
- Enhanced user interface to allow simpler specification of weighting matrices
- Enhanced parameter and error checks  

### Version 1.0.0

This release is intended to be the first registered version for the public. The latest additions include:

- New functions for estimation of norms, reciprocal condition numbers and separations of linear operators.
- New funtions defining a comprehensive set of Lyapunov and Sylvester operators.
- Updated documentation, with examples for the main functions
- Enhancements of all functions to cover all numerical data types
- Full coverage of all basic floating point types by the solvers

### Version 0.8

This release covers the planned main classes of solvers for Lyapunov, Riccati and Sylvester matrix equations. A preliminary version of documentation has been setup. The main addition consists of :

- New solvers for non-negative stable standard and generalized Lyapunov equations, for both continuous and discrete cases.

### Version 0.2

This release is the first functionally complete collection of solvers. It includes new functions and several enhancements:

- New solvers for Sylvester matrix equations and Sylvester systems of matrix equations.
- Simplification of user interfaces for Lyapunov solvers

### Version 0.1.0

This is the initial release covering prototype implementations of several solvers for Lyapunov and Riccati matrix equations and some solvers for Sylvester matrix equations.

## Main developer

[Andreas Varga](https://sites.google.com/site/andreasvargacontact/home)

License: MIT (expat)
