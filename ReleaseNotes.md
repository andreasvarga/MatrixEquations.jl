# Release Notes

## Version 2.5.4
Enhanced functionality for `sylvcs!`. 

## Version 2.5.3
Fix a bug in `ghsylvi`. 

## Version 2.5.2
Fix improper error message for some singular Lyapunov equations. 

## Version 2.5.1
Version bump to enforce compatibility with last version of `GenericSchur`.

## Version 2.5.0
Version bump to enforce compatibility with Julia's LTS version.
New iterative solvers for positive Lyapunov equations based on low-rank ADI methods. 

## Version 2.4.5
Enhanced Lyapunov solvers `lyapc` and `lyapd` and Sylvester solvers `sylvc` and `sylvd` to efficiently handle symmetric/Hermitian/diagonal inputs.  

## Version 2.4.4
Back to using the standard Sylvester solver `sylvc` with the old wrappers (the `*trsyl` family of solvers), until issue [#150](https://github.com/JuliaLinearAlgebra/libblastrampoline/issues/150) will be fixed.

## Version 2.4.3
New wrappers for BLAS Level 3 based LAPACK family `*trsylv3` of solvers for Sylvester equation. Updating the standard Sylvester solver `sylvc` to use the new wrappers (instead the `*trsyl` family of solvers).    

## Version 2.4.2
Version bump to fix type piracy detected by Aqua.  

## Version 2.4.1
Patch release which implements a new structure exploiting scaling option for all Riccati equation solvers. 
This scaling preserves the Hamiltonian/symplectic matrix structures and is based on a symmetric matrix
equilibration technique. 

## Version 2.4.0 

Minor release containing the following enhancements:
* implementation of several scaling options for all Riccati equation solvers;
* support for arbitrary floating-point types in all Riccati equation solvers. 

## Version 2.3.2
Version bump to correct setting of `liblapack`.  

## Version 2.3.1
Patch release to optimize array allocations in the function `cgls`.  

## Version 2.3.0

Minor release containing the following changes:
* renamed functions `tulyapc!` and `hulyapc!` to cover singular input matrices;
* new functions to define some  _T/H-Lyapunov_ and  _generalized T/H-Sylvester_ operators;
* a new function `cgls` which implements the conjugate gradient method [`CGLS`](https://web.stanford.edu/group/SOL/software/cgls/) to solve linear equations and linear least-squares problems with matrix and linear operator arguments;
* new functions to solve Lyapunov, Lyapunov-like, Sylvester and Sylvester-like matrix equations using conjugate gradient based iterative techniques;
* new operators to handle half-vector operations, such as, the _elimination_ and _duplication_ operators;  
* enhanced _transpose_ (_commutation_) operator;
* explicit definitions of 3-term `mul!` operations for transpose/adjoint of Lyapunov and Sylvester operators.

## Version 2.2.11

Patch release to fix bugs in `tlyapc` and `hlyapc`. New functions tlyapcu! and hlyapcu! have been implemented to solve continuous T/H-Lyapunov equations for the upper triangular solution. 

## Version 2.2.10

Patch release to fix issue [MatrixPencils#11](https://github.com/andreasvarga/MatrixPencils.jl/issues/11).

## Version 2.2.9

Patch release with generic functions to solve various Lyapunov-like and Sylvester-like equations using Kronecker product based expansions (not suited for large problems).  

## Version 2.2.8

Patch release with two new functions to solve Lyapunov-like equations and several experimental functions to solve various Lyapunov-like and Sylvester-like equations using Kronecker product based expansions (not suited for large problems).  

## Version 2.2.7

Patch release to allow arbitrary floating-point types in all Sylvester system of equations solvers. 

## Version 2.2.6

Patch release to allow arbitrary floating-point types in all Sylvester equation solvers. 

## Version 2.2.5

Patch release to fix a bug in `sylvds!`. 

## Version 2.2.4

Patch release to allow arbitrary floating-point types in all positive Lyapunov solvers. 

## Version 2.2.3

Patch release to allow arbitrary floating-point types in all Lyapunov solvers. 

## Version 2.2.2

Patch release to enforce AbstractMatrix type in all Schur form based solvers. 

## Version 2.2.1

Patch release to address hidden character length arguments issue discussed in [JuliaLang/julia#32870](https://github.com/JuliaLang/julia/issues/32870). 

## Version 2.2 

This is a minor release to use Julia 1.6 and higher.  

## Version 2.1 

This is a minor release which enhances the definitions of Lyapunov and Sylvester operators, by introducing types for `Adjoint`, `Discrete` and `Continuous` Lyapunov/Sylvester maps.   

## Version 2.0 

This is a major release which concludes the efforts to increase the speed of all solvers and minimize the memory allocation burden. The Lyapunov solvers have been functionally extended to cover the case of non-symmmetric/non-hermitian solutions. The implementations of linear Lyapunov and Sylvester operators now rely on the `LinearMaps.jl` (instead of `LinearOperators.jl` in previous releases).   

## Version 1.5

This is a minor release intended to increase the speed of solvers for Sylvester matrix equations and minimize the memory allocation burden. Similar spectacular improvements have been achieved as for the Lyapunov solvers. 

## Version 1.4

This is a minor release intended to increase the speed of solvers for Lyapunov matrix equations and minimize the memory allocation burden. The achieved spectacular performance improvements can be illustrated in the case of function `lyapd`, where for an 500-th order example, the performance determined executing `@btime lyapd(a,q);` was improved from `449.960 ms (734641 allocations: 600.41 MiB)` to `240.251 ms (27 allocations: 11.60 MiB)` (i.e., a 2 times reduction of execution speed and about 50 (*fifty*) times reduction of the allocated memory). 

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

