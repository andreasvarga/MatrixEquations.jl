# MatrixEquations.jl
**Solution of some control system related matrix equations using Julia**

## About
This collection of Julia functions is an attemp to implement high performance
numerical software to solve classes of Lyapunov, Sylvester and Riccati matrix equations
at a performance level comparable with efficient structure exploiting Fortran implementations, as those available in the Systems and Control Library [SLICOT](http://slicot.org/).
This goal has been fully achieved for Lyapunov equation solvers, for which the
codes for complex data significantly outperform similar functions available in
the MATLAB Control System Toolbox (which rely on SLICOT), while performing at
practically same performance level for real data.

The available functions in the `MatrixEquation` collection cover both standard
and generalized continuous and discrete Lyapunov, Sylvester and Riccati equations.
The functions for the solution of Lyapunov equations rely on efficient structure
exploiting solvers for which the input data are in Schur or generalized Schur forms.
The implementation of Riccati equation solvers employ orthogonal Schur vectors
based methods and their extensions to linear matrix pencil based reduction approaches.
The functions provided for the solution of several classes of Sylvester equations
and Sylvester systems of equations represent partly prototype implementations suitable
for small order matrices, and will be replaced in a future version with structure
exploiting solvers.    

## Future plans
The collection of tools will be extended by adding new functionality, such as the solution
of all classes of Sylvester equations and Sylvester systems of equations using structure exploiting methods, the solution of stable Lyapunov equations directly for the Cholesky factors of the solutions, computation of condition number estimators for Lyapunov and Sylvester equations, etc.
