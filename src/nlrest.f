C ======================================================================
C
C     NLREST : Internal restart in an error situation.  The quasi-Newton
C              matrix is replaced by RHO times the identity matrix, the
C              multiplier approximation is set to zero and the penalty
C              parameters of the merit function are reset.  The BFGS
C              update procedure starts again from this point.
C
C ======================================================================
C
      SUBROUTINE NLREST (N, NMAX, M, MNN2, C, U, V, R, RHO)
C
      IMPLICIT NONE
      INTEGER N, NMAX, M, MNN2
      DOUBLE PRECISION C(NMAX,*), U(*), V(*), R(*), RHO
      INTEGER I, J
      DOUBLE PRECISION ZERO, ONE, RR
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
      RR = RHO
      IF (RR.LE.ZERO) RR = ONE
      DO 20 J = 1, N
         DO 10 I = 1, N
            C(I,J) = ZERO
   10    CONTINUE
         C(J,J) = RR
   20 CONTINUE
      DO 30 J = 1, MNN2
         U(J) = ZERO
         V(J) = ZERO
   30 CONTINUE
      DO 40 J = 1, M
         R(J) = ONE
   40 CONTINUE
      RETURN
      END
