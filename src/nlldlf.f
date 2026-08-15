C ======================================================================
C
C     NLLDLF : Factorization  C = L D L^T  of the quasi-Newton matrix.
C              The strictly lower triangle of C is overwritten by the
C              unit lower triangular factor L, the diagonal matrix is
C              returned in D.  The routine is used to provide the
C              output format requested for LQL = .FALSE.
C
C     NLLDLC : Inverse operation, i.e., the full symmetric matrix
C              C = L D L^T is reconstructed from a factorization
C              provided by the calling program.
C
C ======================================================================
C
      SUBROUTINE NLLDLF (N, NMAX, C, D)
C
      IMPLICIT NONE
      INTEGER N, NMAX
      DOUBLE PRECISION C(NMAX,*), D(*)
      INTEGER I, J, K
      DOUBLE PRECISION S, ZERO, ONE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
      DO 40 J = 1, N
         S = C(J,J)
         DO 10 K = 1, J-1
            S = S - C(J,K)*C(J,K)*D(K)
   10    CONTINUE
         D(J) = S
         IF (ABS(S).LE.ZERO) THEN
            D(J) = ONE
            S    = ONE
         ENDIF
         DO 30 I = J+1, N
            S = C(I,J)
            DO 20 K = 1, J-1
               S = S - C(I,K)*C(J,K)*D(K)
   20       CONTINUE
            C(I,J) = S/D(J)
   30    CONTINUE
   40 CONTINUE
      DO 60 J = 1, N
         C(J,J) = ONE
         DO 50 I = 1, J-1
            C(I,J) = ZERO
   50    CONTINUE
   60 CONTINUE
      RETURN
      END
