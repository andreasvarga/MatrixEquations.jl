module LapackUtil


import LinearAlgebra.BLAS.@blasfunc

using LinearAlgebra
import LinearAlgebra: BlasFloat, BlasInt, BlasReal, BlasComplex, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

using Base: iszero, has_offset_axes

export tgsyl!, lanv2, ladiv, lag2, lacn2!

@static if VERSION < v"1.7"
    using LinearAlgebra.LAPACK: liblapack
elseif VERSION < v"1.9"
    const liblapack = "libblastrampoline"
else
    const liblapack = LinearAlgebra.libblastrampoline
end


function chklapackerror(ret::BlasInt)
   ret == 0 && return
   ret < 0 ? throw(ArgumentError("invalid argument #$(-ret) to LAPACK call")) : throw(LAPACKException(ret))
end


for (fn, elty, relty) in ((:dtgsyl_, :Float64, :Float64),
                   (:stgsyl_, :Float32, :Float32),
                   (:ztgsyl_, :ComplexF64, :Float64),
                   (:ctgsyl_, :ComplexF32, :Float32))
    @eval begin
        function tgsyl!(trans::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, C::AbstractMatrix{$elty},
                        D::AbstractMatrix{$elty}, E::AbstractMatrix{$elty}, F::AbstractMatrix{$elty})
            @assert !has_offset_axes(A, B, C, D, E, F)
            chkstride1(A, B, C, D, E, F)
            m, n = checksquare(A, B)
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))
            m1, n1 = size(C)
            (m == m1 && n == n1) ||
                throw(DimensionMismatch("dimensions of A($m,$m),  B($n,$n), and C($m1,$n1) must match"))
            m2, n2 = checksquare(D, E)
            m == m2 || throw(DimensionMismatch("dimensions of A($m,$m) and D($m2,$m2) must match"))
            n == n2 || throw(DimensionMismatch("dimensions of B($n,$n) and E($n2,$n2), must match"))
            ldd = max(1, stride(D, 2))
            lde = max(1, stride(E, 2))
            ldf = max(1, stride(F, 2))
            m3, n3 = size(F)
            (m2 == m3 && n2 == n3) ||
                throw(DimensionMismatch("dimensions of D($m,$m),  E($n,$n), and F($m3,$n3) must match"))
            dif = Vector{$relty}(undef, 1)
            scale = Vector{$relty}(undef, 1)
            info  = Ref{BlasInt}()
            ijob = 0
            work = Vector{$elty}(undef, 1)
            lwork = 1
            iwork = Vector{BlasInt}(undef,m+n+6)
            # SUBROUTINE DTGSYL( TRANS, IJOB, M, N, A, LDA, B, LDB, C, LDC, D,
            #       LDD, E, LDE, F, LDF, SCALE, DIF, WORK, LWORK,
            #       IWORK, INFO )
            ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$relty}, Ptr{$relty}, Ptr{$relty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                 trans, ijob, m, n,
                 A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf,
                 scale, dif, work, lwork, iwork, info, 1)
            chklapackerror(info[])
            C, F, scale[1]
        end
    end
end

"""
    tgsyl!(A, B, C, D, E, F) -> (C, F, scale)

Solve the Sylvester system of matrix equations

      AX - YB = scale*C
      DX - YE = scale*F ,

where `X` and `Y` are unknown matrices, the pairs `(A, D)`, `(B, E)` and  `(C, F)`
have the same sizes, and the pairs `(A, D)` and `(B, E)` are in
generalized (real) Schur canonical form, i.e. `A`, `B` are upper quasi
triangular and `D`, `E` are upper triangular.
Returns `X` (overwriting `C`), `Y` (overwriting `F`) and `scale`.

    tgsyl!(trans, A, B, C, D, E, F) -> (C, F, scale)

Solve for `trans = 'T'` and real matrices or for `trans = 'C'` and complex
matrices,  the (adjoint) Sylvester system of matrix equations

      A'X + D'Y = scale*C
      XB' + YE' = scale*(-F) .

`tgsyl!('N', A, B, C, D, E, F)` corresponds to the call `tgsyl!(A, B, C, D, E, F)`.

Interface to the LAPACK subroutines DTGSYL/STGSYL/ZTGSYL/CTGSYL.
"""
tgsyl!(trans::AbstractChar,A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

tgsyl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix) =
tgsyl!('N',A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

for (fn, elty) in ((:dlanv2_, :Float64),
                   (:slanv2_, :Float32))
    @eval begin
        function lanv2(A::$elty, B::$elty, C::$elty, D::$elty)
           """
           SUBROUTINE DLANV2( A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN )

           DOUBLE PRECISION A, B, C, CS, D, RT1I, RT1R, RT2I, RT2R, SN
           """
           RT1R = Ref{$elty}(1.0)
           RT1I = Ref{$elty}(1.0)
           RT2R = Ref{$elty}(1.0)
           RT2I = Ref{$elty}(1.0)
           CS = Ref{$elty}(1.0)
           SN = Ref{$elty}(1.0)
           ccall((@blasfunc($fn), liblapack), Cvoid,
                 (Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},
                 Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty}),
                 A, B, C, D,
                 RT1R, RT1I, RT2R, RT2I, CS, SN)
           return RT1R[], RT1I[], RT2R[], RT2I[], CS[], SN[]
        end
    end
end
"""
    lanv2(A, B, C, D) -> (RT1R, RT1I, RT2R, RT2I, CS, SN)

Compute the Schur factorization of a real 2-by-2 nonsymmetric matrix `[A,B;C,D]` in
standard form. `A`, `B`, `C`, `D` are overwritten on output by the corresponding elements of the
standardised Schur form. `RT1R+im*RT1I` and `RT2R+im*RT2I` are the resulting eigenvalues.
`CS` and `SN` are the parameters of the rotation matrix.
Interface to the LAPACK subroutines DLANV2/SLANV2.
"""
lanv2(A::BlasReal, B::BlasReal, C::BlasReal, D::BlasReal)


for (fn, elty) in ((:dlag2_, :Float64),
                   (:slag2_, :Float32))
    @eval begin
        function lag2(A::StridedMatrix{$elty}, B::StridedMatrix{$elty}, SAFMIN::$elty)
           """
           SUBROUTINE DLAG2( A, LDA, B, LDB, SAFMIN, SCALE1, SCALE2, WR1, WR2, WI )

           INTEGER            LDA, LDB
           DOUBLE PRECISION   SAFMIN, SCALE1, SCALE2, WI, WR1, WR2
           DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
           """
           LDA = stride(A,2)
           LDB = stride(B,2)
           SCALE1 = Ref{$elty}(1.0)
           SCALE2 = Ref{$elty}(1.0)
           WR1 = Ref{$elty}(1.0)
           WR2 = Ref{$elty}(1.0)
           WI = Ref{$elty}(1.0)
           ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty},
                Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty}),
                A, LDA, B, LDB, SAFMIN,
                SCALE1, SCALE2, WR1, WR2, WI)
           return SCALE1[], SCALE2[], WR1[], WR2[], WI[]
        end
    end
end
"""
    lag2(A, B, SAFMIN) -> (SCALE1, SCALE2, WR1, WR2, WI)

Compute the eigenvalues of a 2-by-2 generalized real eigenvalue problem for
the matrix pair `(A,B)`, with scaling as necessary to avoid over-/underflow.
`SAFMIN` is the smallest positive number s.t. `1/SAFMIN` does not overflow.
If `WI = 0`, `WR1/SCALE1` and `WR2/SCALE2` are the resulting real eigenvalues, while
if `WI <> 0`, then `(WR1+/-im*WI)/SCALE1` are the resulting complex eigenvalues.
Interface to the LAPACK subroutines DLAG2/SLAG2.
"""
lag2(A::StridedMatrix{BlasReal}, B::StridedMatrix{BlasReal}, SAFMIN::BlasReal)

# lag2(A::StridedMatrix{T}, B::StridedMatrix{T}) where T <: BlasReal = lag2(A,B,safemin(T))

# function safemin(::Type{T}) where T <: BlasReal
#     SMLNUM = (T == Float64) ? reinterpret(Float64, 0x2000000000000000) : reinterpret(Float32, 0x20000000)
#     return SMLNUM * 2/ eps(T)
# end
function smlnum(::Type{T}) where T <: BlasReal
    (T == Float64) ? reinterpret(Float64, 0x2000000000000000) : reinterpret(Float32, 0x20000000)
end
function safemin(::Type{T}) where T <: BlasReal
    smlnum(T) * 2/ eps(T)
end
for (fn, elty) in ((:dladiv_, :Float64),
                   (:sladiv_, :Float32))
    @eval begin
        function ladiv(A::$elty, B::$elty, C::$elty, D::$elty)
           """
           SUBROUTINE DLADIV( A, B, C, D, P, Q )

           DOUBLE PRECISION   A, B, C, D, P, Q
           """
           P = Ref{$elty}(1.0)
           Q = Ref{$elty}(1.0)
           ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},
                 Ref{$elty},Ref{$elty}),
                 A, B, C, D, P, Q)
           return P[], Q[]
        end
    end
end
"""
    ladiv(A, B, C, D) -> (P, Q)

Perform the complex division in real arithmetic

  ``P + iQ = \\displaystyle\\frac{A+iB}{C+iD}``

by avoiding unnecessary overflow.
Interface to the LAPACK subroutines DLADIV/SLADIV.
"""
ladiv(A::BlasReal, B::BlasReal, C::BlasReal, D::BlasReal)

for (fn, elty) in ((:dlacn2_, :Float64),
                   (:slacn2_, :Float32))
    @eval begin
        function lacn2!(V::AbstractVector{$elty}, X::AbstractVector{$elty}, ISGN::AbstractVector{BlasInt},
                        EST::$elty, KASE::BlasInt, ISAVE::AbstractVector{BlasInt})
            """
            *       SUBROUTINE DLACN2( N, V, X, ISGN, EST, KASE, ISAVE )
            *
            *       .. Scalar Arguments ..
            *       INTEGER            KASE, N
            *       DOUBLE PRECISION   EST
            *       ..
            *       .. Array Arguments ..
            *       INTEGER            ISGN( * ), ISAVE( 3 )
            *       DOUBLE PRECISION   V( * ), X( * )
            """
            @assert !has_offset_axes(V, X)
            chkstride1(V,X)
            n = length(V)
            (n != length(X) || n != length(ISGN)) && throw(DimensionMismatch("dimensions of V,  X, and ISIGN must be equal"))
            KASE1 = Array{BlasInt,1}(undef,1)
            KASE1[1] = KASE
            EST1 = Vector{$elty}(undef, 1)
            EST1[1] = EST
            ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}),
                n, V, X, ISGN, EST1, KASE1, ISAVE)
            return EST1[1], KASE1[1]
        end
    end
end
"""
    lacn2!(V, X, ISGN, EST, KASE, ISAVE ) -> (EST, KASE )
Estimate the 1-norm of a `real` linear operator `A`, using reverse communication
by applying the operator or its transpose/adjoint to a vector.
`KASE` is a parameter to control the norm evaluation process as follows.
On the initial call, `KASE` should be `0`. On an intermediate return,
`KASE` will be `1` or `2`, indicating whether the `real` vector `X` should be overwritten
by `A * X`  or `A' * X` at the next call.
On the final return, `KASE` will again be `0` and `EST` is an estimate (a lower bound)
for the 1-norm of `A`. `V` is a real work vector, `ISGN` is an integer work
vector and `ISAVE` is a 3-dimensional integer vector used to save information
between the calls.
Interface to the LAPACK subroutines DLACN2/SLACN2.
"""
lacn2!(V::AbstractVector{BlasReal}, X::AbstractVector{BlasReal}, ISGN::AbstractVector{BlasInt}, EST::BlasReal, KASE::BlasInt, ISAVE::AbstractVector{BlasInt})


for (fn, elty, relty) in ((:zlacn2_, :ComplexF64, :Float64),
                          (:clacn2_, :ComplexF32, :Float32))
    @eval begin
        function lacn2!(V::AbstractVector{$elty}, X::AbstractVector{$elty},
                        EST::$relty, KASE::BlasInt, ISAVE::AbstractVector{BlasInt})
            """
            *       SUBROUTINE ZLACN2( N, V, X, EST, KASE, ISAVE )
            *
            *       .. Scalar Arguments ..
            *       INTEGER            KASE, N
            *       DOUBLE PRECISION   EST
            *       ..
            *       .. Array Arguments ..
            *       INTEGER            ISGN( * ), ISAVE( 3 )
            *       COMPLEX*16         V( * ), X( * )
            """
            @assert !has_offset_axes(V, X)
            chkstride1(V,X)
            n = length(V)
            n == length(X) || throw(DimensionMismatch("dimensions of V and X must be equal"))
            KASE1 = Array{BlasInt,1}(undef,1)
            KASE1[1] = KASE
            EST1 = Vector{$relty}(undef, 1)
            EST1[1] = EST
            ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}),
                n, V, X, EST1, KASE1, ISAVE)
            return EST1[1], KASE1[1]
        end
    end
end
"""
    lacn2!(V, X, EST, KASE, ISAVE ) -> (EST, KASE )
Estimate the 1-norm of a `complex` linear operator `A`, using reverse communication
by applying the operator or its adjoint to a vector.
`KASE` is a parameter to control the norm evaluation process as follows.
On the initial call, `KASE` should be `0`. On an intermediate return,
`KASE` will be `1` or `2`, indicating whether the `complex` vector `X` should be overwritten
by `A * X`  or `A' * X` at the next call.
On the final return, `KASE` will again be `0` and `EST` is an estimate (a lower bound)
for the 1-norm of `A`. `V` is a complex work vector and `ISAVE` is a 3-dimensional
integer vector used to save information between the calls.
Interface to the LAPACK subroutines ZLACN2/CLACN2.
"""
lacn2!(V::AbstractVector{BlasComplex}, X::AbstractVector{BlasComplex}, EST::BlasReal, KASE::BlasInt, ISAVE::AbstractVector{BlasInt})

end
