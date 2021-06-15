# Release Notes

## Version 1.4

This is a minor release intended to increase the speed of solvers for Lyapunov equations and minimize the memory allocation burden. The achieved spectacular performance improvements can be illustrated in the case of function `lyapd`, where for an 500-th order example, the performance determined executing `@btime lyap(a,q);` 
was improved from `449.960 ms (734641 allocations: 600.41 MiB)` to `240.251 ms (27 allocations: 11.60 MiB)` (i.e., a 2 times reduction of execution speed and about 50 (*fifty*) times reduction of the allocated memory). 

## Version 1.3

This is a minor release solely intended to update the package to perform CI with Github Actions instead Travis-CI.

## Version 1.2.1

Patch release to address fallback issues to ensure compatibility to versions prior 1.3 of Julia,
some enhancements of the 2x2 positive generalized Lyapunov equation solver, explicit handling of null dimension case in Riccati solvers.

## Version 1.2.0

Minor release targeting sensible (up to 50%) speed increase of various lower level solvers for Lyapunov and Sylvester equations. This goal has been achieved by the reduction of allocation burden using preallocation of small size work arrays, explicit forming of small order Kronecker product based coefficient matrices, performing updating operations with the 5-term `mul!` function introduced in `Julia 1.3` (compatibility with prior Julia versions ensured using calls to BLAS `gemm!`).  The functionality of lower level solvers has been strictly restricted to the basic real and complex data of types `BlasReal` and `BlasComplex`.

## Versions 1.1.1-1.1.4

Patch releases to fix upgrading problems to version v0.7.1 of LinearOperators.jl, compatibility problems with Julia 1.0 - 1.3, and updating problems of the online documentation on the gh-pages.

## Version 1.1.0

This release includes several enhancements of the Riccati equation solvers:

- Enhanced functionality to determine anti-stabilizing solutions
- Enhanced user interface to allow simpler specification of weighting matrices
- Enhanced parameter and error checks  

## Version 1.0.0

This release is intended to be the first registered version for the public. The latest additions include:

- New functions for estimation of norms, reciprocal condition numbers and separations of linear operators.
- New funtions defining a comprehensive set of Lyapunov and Sylvester operators.
- Updated documentation, with examples for the main functions
- Enhancements of all functions to cover all numerical data types
- Full coverage of all basic floating point types by the solvers

## Version 0.8

This release covers the planned main classes of solvers for Lyapunov, Riccati and Sylvester matrix equations. A preliminary version of documentation has been setup. The main addition consists of :

- New solvers for non-negative stable standard and generalized Lyapunov equations, for both continuous and discrete cases.

## Version 0.2

This release is the first functionally complete collection of solvers. It includes new functions and several enhancements:

- New solvers for Sylvester matrix equations and Sylvester systems of matrix equations.
- Simplification of user interfaces for Lyapunov solvers

## Version 0.1.0

This is the initial release covering prototype implementations of several solvers for Lyapunov and Riccati matrix equations and some solvers for Sylvester matrix equations.

