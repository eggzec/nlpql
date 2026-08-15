C ======================================================================
C
C     NLPEN : Update of the penalty parameters of the augmented
C             Lagrangian merit function.  The parameters must be chosen
C             in such a way that the solution of the quadratic
C             programming subproblem becomes a descent direction of the
C             merit function.  Implemented is the rule
C
C        r_j := max ( rreq_j , 1/2 ( r_j + rreq_j ) ) ,
C
C        rreq_j := 2 m ( u_j - v_j )^2 / ( (1-delta) d^T C d ) ,
C
C     which never lets a penalty parameter fall below the value that is
C     required for a sufficient descent property, but allows a slow
C     decrease if the required value becomes small again.
C
C ======================================================================
C
      SUBROUTINE NLPEN (M, ME, MMAX, N, G, U, V, R, DBD, DELTA, EPSM)
C
      IMPLICIT NONE
      INTEGER M, ME, MMAX, N
      DOUBLE PRECISION G(MMAX,*), U(*), V(*), R(*), DBD, DELTA, EPSM
      INTEGER J
      DOUBLE PRECISION RREQ, DEN, ZERO, ONE, HALF, TWO
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, HALF=0.5D0, TWO=2.0D0)
C
      DEN = (ONE - DELTA)*DBD
      IF (DEN.LE.EPSM) RETURN
      DO 10 J = 1, M
         RREQ = TWO*DBLE(MAX(M,1))*(U(J)-V(J))*(U(J)-V(J))/DEN
         R(J) = MAX(RREQ,HALF*(R(J)+RREQ))
         IF (R(J).LE.ZERO) R(J) = ONE
   10 CONTINUE
      RETURN
      END
