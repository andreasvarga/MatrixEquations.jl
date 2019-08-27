module LapackUtil

const liblapack = Base.liblapack_name

import LinearAlgebra.BLAS.@blasfunc

import LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

using Base: iszero, has_offset_axes

export tgsyl!

function chklapackerror(ret::BlasInt)
    if ret == 0
        return
    elseif ret < 0
        throw(ArgumentError("invalid argument #$(-ret) to LAPACK call"))
    else # ret > 0
        throw(LAPACKException(ret))
    end
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
            if m != m1 || n != n1
                throw(DimensionMismatch("dimensions of A($m,$m),  B($n,$n), and C($m1,$n1) must match"))
            end
            m2, n2 = checksquare(D, E)
            if m != m2
                throw(DimensionMismatch("dimensions of A($m,$m) and D($m2,$m2) must match"))
            end
            if n != n2
                throw(DimensionMismatch("dimensions of B($n,$n) and E($n2,$n2), must match"))
            end
            ldd = max(1, stride(D, 2))
            lde = max(1, stride(E, 2))
            ldf = max(1, stride(F, 2))
            m3, n3 = size(F)
            if m2 != m3 || n2 != n3
                throw(DimensionMismatch("dimensions of D($m,$m),  E($n,$n), and F($m3,$n3) must match"))
            end
            dif = Vector{$relty}(undef, 1)
            scale = Vector{$relty}(undef, 1)
            info  = Ref{BlasInt}()
            #trans = AbstractChar('N')
            ijob = 0
            work = Vector{$elty}(undef, 1)
            lwork = 1
            iwork = Vector{BlasInt}(undef,m+n+6)
            #SUBROUTINE DTGSYL( TRANS, IJOB, M, N, A, LDA, B, LDB, C, LDC, D,
            #       LDD, E, LDE, F, LDF, SCALE, DIF, WORK, LWORK,
            #       IWORK, INFO )
            ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$relty}, Ptr{$relty}, Ptr{$relty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                 trans, ijob, m, n,
                 A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf,
                 scale, dif, work, lwork, iwork, info)
            chklapackerror(info[])
            C, F, scale[1]
        end
    end
end

"""
    tgsyl!(A, B, C, D, E, F) -> (C, F, scale)

Solve the Sylvester system of
matrix equations

      AX - YB = scale*C
      DX - YE = scale*F ,

where `X` and `Y` are unknown matrices, the pairs `(A, D)`, `(B, E)` and  `(C, F)`
have the same sizes, and the pairs `(A, D)` and `(B, E)` are in
generalized (real) Schur canonical form, i.e. `A`, `B` are upper quasi
triangular and `D`, `E` are upper triangular.
Returns `X` (overwriting `C`), `Y` (overwriting `F`) and `scale`.

    tgsyl!(trans, A, B, C, D, E, F) -> (C, F, scale)

Solve for `trans = 'T'` and
real matrices or for `trans = 'C'` and complex matrices,  the (transposed) Sylvester
system of matrix equations

      A'X + D'Y = scale*C
      XB' + YE' = scale*(-F) .

`tgsyl!('N', A, B, C, D, E, F)` corresponds to the call `tgsyl!(A, B, C, D, E, F)`.
"""
tgsyl!(trans::AbstractChar,A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

tgsyl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix) =
tgsyl!('N',A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

"""
    lanv2(A, B, C, D) -> (RT1R, RT1I, RT2R, RT2I, CS, SN)

Compute the Schur factorization of a real 2-by-2 nonsymmetric matrix in
standard form. Interface to the LAPACK subroutine DLANV2.
"""
function lanv2(A::Float64, B::Float64, C::Float64, D::Float64)
    """
    SUBROUTINE DLANV2( A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN )

    DOUBLE PRECISION A, B, C, CS, D, RT1I, RT1R, RT2I, RT2R, SN
    """
    RT1R = Ref{Float64}(1.0)
    RT1I = Ref{Float64}(1.0)
    RT2R = Ref{Float64}(1.0)
    RT2I = Ref{Float64}(1.0)
    CS = Ref{Float64}(1.0)
    SN = Ref{Float64}(1.0)
    ccall((@blasfunc("dlanv2_"), liblapack), Cvoid,
          (Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},
           Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64}),
           A, B, C, D,
           RT1R, RT1I, RT2R, RT2I, CS, SN)
    return RT1R[], RT1I[], RT2R[], RT2I[], CS[], SN[]
end

"""
    lag2(A, B, SAFMIN) -> (SCALE1, SCALE2, WR1, WR2, WI)


Compute the eigenvalues of a 2-by-2 generalized eigenvalue problem, with scaling
 as necessary to avoid over-/underflow. Interface to the LAPACK subroutine DLAG2.

"""
function lag2(A::StridedMatrix{Float64}, B::StridedMatrix{Float64}, SAFMIN::Float64)
    """
    SUBROUTINE DLAG2( A, LDA, B, LDB, SAFMIN, SCALE1, SCALE2, WR1, WR2, WI )

    INTEGER            LDA, LDB
    DOUBLE PRECISION   SAFMIN, SCALE1, SCALE2, WI, WR1, WR2
    DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
    """
    LDA = stride(A,2)
    LDB = stride(B,2)
    SCALE1 = Ref{Float64}(1.0)
    SCALE2 = Ref{Float64}(1.0)
    WR1 = Ref{Float64}(1.0)
    WR2 = Ref{Float64}(1.0)
    WI = Ref{Float64}(1.0)
    ccall((@blasfunc("dlag2_"), liblapack), Cvoid,
          (Ptr{Float64}, Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ref{Float64},
           Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64}),
           A, LDA, B, LDB, SAFMIN,
           SCALE1, SCALE2, WR1, WR2, WI)
    return SCALE1[], SCALE2[], WR1[], WR2[], WI[]
end

"""
   ladiv(A, B, C, D) -> (P, Q)

Performs complex division in real arithmetic, avoiding unnecessary overflow.
Interface to the LAPACK subroutine DLADIV.
"""
function ladiv(A::Float64, B::Float64, C::Float64, D::Float64)
    """
    SUBROUTINE DLADIV( A, B, C, D, P, Q )

    DOUBLE PRECISION   A, B, C, D, P, Q
    """
    P = Ref{Float64}(1.0)
    Q = Ref{Float64}(1.0)
    ccall((@blasfunc("dladiv_"), liblapack), Cvoid,
          (Ref{Float64},Ref{Float64},Ref{Float64},Ref{Float64},
           Ref{Float64},Ref{Float64}),
           A, B, C, D,
           P, Q)
    return P[], Q[]
end

end
