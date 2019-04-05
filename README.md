# MatrixEquations
Solution of some control system related matrix equations using Julia

## About
This collection of Julia functions is an attemp to implement high performance
numerical software to solve classes of Lyapunov and Riccati equations, which
is able to compute the symmetric/hermitian solutions at a performance level
which is typical for efficient structure exploiting Fortran implementations as those
available in the Systems and Control Library [SLICOT](http://slicot.org/).
This goal has been fully achieved for complex data, for which the codes significantly outperform the implementations relying on SLICOT available in the MATLAB Cotrol System Toolbox, or in Julia (function `LYAP`).
For real data, the same performance level has been practically achieved as in MATLAB and
better performance obtained as for the existing implementation in Julia. We note in passing
that the existing Julia implementation `LYAP` has some notable weaknesses, as for example,
the resulting solution is usually not hermitian in the complex case or not symmetric
in the real case, and the function fails when calling it with transposed arguments.

The available functions in the `MatrixEquation` collection cover all classes of
standard and generalized continuous and discrete Lyapunov and Riccati equations.
The Lyapunov equation solvers rely on efficient structure exploiting
solvers  for the input data in Schur or generalized Schur forms.
The implementation of Riccati equation solvers employ orthogonal Schur vectors
based methods and their extensions to linear matrix pencil based reduction approaches.   

## Future plans
The collection of tools will be extended by adding new functionality, such as the solution
of classes of Sylvester equations and Sylvester systems of equations, the solution of stable Lyapunov
equations directly for the Cholesky factors of the solutions, computation of condition
number estimators for Lyapunov equations, etc.
