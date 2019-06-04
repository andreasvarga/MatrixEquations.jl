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
`tgsyl!(A, B, C, D, E, F) -> (C, F, scale)` solves the Sylvester system of
matrix equations

      AX - YB = scale*C
      DX - YE = scale*F ,

where `X` and `Y` are unknown matrices, the pairs `(A, D)`, `(B, E)` and  `(C, F)`
have the same sizes, and the pairs `(A, D)` and `(B, E)` are in
generalized (real) Schur canonical form, i.e. `A`, `B` are upper quasi
triangular and `D`, `E` are upper triangular.
Returns `X` (overwriting `C`), `Y` (overwriting `F`) and `scale`.

tgsyl!(trans, A, B, C, D, E, F) -> (C, F, scale)` solves for trans = 'T' and
real matrices or for trans = 'C' and complex matrices,  the (transposed) Sylvester
system of matrix equations

      A'X + D'Y = scale*C
      XB' + YE' = scale*(-F) .

`tgsyl!('N', A, B, C, D, E, F)` corresponds to the call `tgsyl!(A, B, C, D, E, F)`.
"""
tgsyl!(trans::AbstractChar,A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

tgsyl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix) =
tgsyl!('N',A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)


end
