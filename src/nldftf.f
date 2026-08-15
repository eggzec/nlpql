C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLDFTF (MODEL, L, M, ME, LMMAX, N, LNMAX, X, G, FV)
C
C   Objective function value and constraint values of the transformed
C   program.  On entry G contains the M constraint values followed by
C   the L individual function values, on return the constraints of the
C   transformed program.
C
      IMPLICIT NONE
      INTEGER MODEL, L, M, ME, LMMAX, N, LNMAX
      DOUBLE PRECISION X(LNMAX), G(LMMAX), FV(1)
      INTEGER I, J
      DOUBLE PRECISION S, ZERO
      PARAMETER (ZERO=0.0D0)
C
      IF (MODEL.EQ.1) THEN
C
C   Least squares.  The individual functions are moved to the front,
C   the constraints are shifted by L positions.  Positions M+L+1 to
C   M+2*L are used as scratch space.
C
         DO 10 I = 1, L
            G(M+L+I) = G(M+I)
   10    CONTINUE
         DO 20 J = M, 1, -1
            G(J+L) = G(J)
   20    CONTINUE
         S = ZERO
         DO 30 I = 1, L
            G(I) = G(M+L+I) - X(N+I)
            S    = S + X(N+I)*X(N+I)
   30    CONTINUE
         FV(1) = S
      ELSE IF (MODEL.EQ.2) THEN
         S = ZERO
         DO 40 I = 1, L
            G(M+L+I) = G(M+I) + X(N+I)
            S        = S + X(N+I)
   40    CONTINUE
         DO 50 I = 1, L
            G(M+I) = -G(M+I) + X(N+I)
   50    CONTINUE
         FV(1) = S
      ELSE IF (MODEL.EQ.3) THEN
         DO 60 I = 1, L
            G(M+L+I) = G(M+I) + X(N+1)
   60    CONTINUE
         DO 70 I = 1, L
            G(M+I) = -G(M+I) + X(N+1)
   70    CONTINUE
         FV(1) = X(N+1)
      ELSE IF (MODEL.EQ.4) THEN
         DO 80 I = 1, L
            G(M+I) = -G(M+I) + X(N+1)
   80    CONTINUE
         FV(1) = X(N+1)
      ELSE IF (MODEL.EQ.5) THEN
         S = ZERO
         DO 90 I = 1, L
            S = S + G(M+I)*G(M+I)
   90    CONTINUE
         FV(1) = S
      ELSE
         S = ZERO
         DO 95 I = 1, L
            S = S + G(M+I)
   95    CONTINUE
         FV(1) = S
      ENDIF
      RETURN
      END
