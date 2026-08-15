C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLLDLC (N, NMAX, C, D)
C
      IMPLICIT NONE
      INTEGER N, NMAX
      DOUBLE PRECISION C(NMAX,*), D(*)
      INTEGER I, J, K
      DOUBLE PRECISION S, ZERO, ONE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
C   The rows are processed from the last one to the first one, so that
C   the factor L is still available whenever it is needed.
C
      DO 30 I = N, 1, -1
         DO 20 J = I, 1, -1
            S = ZERO
            DO 10 K = 1, J-1
               S = S + C(I,K)*D(K)*C(J,K)
   10       CONTINUE
            IF (I.EQ.J) THEN
               S = S + D(J)
            ELSE
               S = S + C(I,J)*D(J)
            ENDIF
            C(I,J) = S
   20    CONTINUE
   30 CONTINUE
      DO 50 I = 1, N
         DO 40 J = I+1, N
            C(I,J) = C(J,I)
   40    CONTINUE
   50 CONTINUE
      RETURN
      END
