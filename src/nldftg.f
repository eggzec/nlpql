C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLDFTG (MODEL, L, M, ME, LMMAX, N, LNMAX, NT, X, G,
     /                   DF, DG)
C
C   Gradient of the objective function and Jacobian of the transformed
C   program.  On entry the first M rows of DG contain the constraint
C   gradients and the subsequent L rows the gradients of the individual
C   functions.
C
      IMPLICIT NONE
      INTEGER MODEL, L, M, ME, LMMAX, N, LNMAX, NT
      DOUBLE PRECISION X(LNMAX), G(LMMAX), DF(LNMAX), DG(LMMAX,LNMAX)
      INTEGER I, J, K
      DOUBLE PRECISION S, ZERO, ONE, TWO
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0)
C
      IF (MODEL.EQ.5 .OR. MODEL.EQ.6) THEN
C
C   No additional variables, the objective function and its gradient
C   are assembled from the individual terms directly.
C
         DO 20 K = 1, N
            S = ZERO
            DO 10 I = 1, L
               IF (MODEL.EQ.5) THEN
                  S = S + G(M+I)*DG(M+I,K)
               ELSE
                  S = S + DG(M+I,K)
               ENDIF
   10       CONTINUE
            IF (MODEL.EQ.5) THEN
               DF(K) = TWO*S
            ELSE
               DF(K) = S
            ENDIF
   20    CONTINUE
         RETURN
      ENDIF
C
      DO 30 K = 1, NT
         DF(K) = ZERO
   30 CONTINUE
      IF (MODEL.EQ.1) THEN
         DO 40 I = 1, L
            DF(N+I) = TWO*X(N+I)
   40    CONTINUE
      ELSE IF (MODEL.EQ.2) THEN
         DO 50 I = 1, L
            DF(N+I) = ONE
   50    CONTINUE
      ELSE
         DF(N+1) = ONE
      ENDIF
C
      IF (MODEL.EQ.1) THEN
C
C   Move the L gradients of the individual functions to the front and
C   shift the constraint gradients by L rows, using rows M+L+1 to
C   M+2*L as scratch space.
C
         DO 70 I = 1, L
            DO 60 K = 1, N
               DG(M+L+I,K) = DG(M+I,K)
   60       CONTINUE
   70    CONTINUE
         DO 90 J = M, 1, -1
            DO 80 K = 1, N
               DG(J+L,K) = DG(J,K)
   80       CONTINUE
   90    CONTINUE
         DO 120 I = 1, L
            DO 100 K = 1, N
               DG(I,K) = DG(M+L+I,K)
  100       CONTINUE
            DO 110 K = 1, L
               DG(I,N+K) = ZERO
  110       CONTINUE
            DG(I,N+I) = -ONE
  120    CONTINUE
         DO 140 J = 1, M
            DO 130 K = 1, L
               DG(J+L,N+K) = ZERO
  130       CONTINUE
  140    CONTINUE
         RETURN
      ENDIF
C
C   The remaining models append new rows, so that the constraint
C   gradients keep their positions.
C
      DO 160 J = 1, M
         DO 150 K = N+1, NT
            DG(J,K) = ZERO
  150    CONTINUE
  160 CONTINUE
      IF (MODEL.EQ.2 .OR. MODEL.EQ.3) THEN
         DO 180 I = 1, L
            DO 170 K = 1, N
               DG(M+L+I,K) = DG(M+I,K)
  170       CONTINUE
  180    CONTINUE
      ENDIF
      DO 200 I = 1, L
         DO 190 K = 1, N
            DG(M+I,K) = -DG(M+I,K)
  190    CONTINUE
  200 CONTINUE
      IF (MODEL.EQ.2) THEN
         DO 230 I = 1, L
            DO 210 K = 1, L
               DG(M+I,N+K)   = ZERO
               DG(M+L+I,N+K) = ZERO
  210       CONTINUE
            DG(M+I,N+I)   = ONE
            DG(M+L+I,N+I) = ONE
  230    CONTINUE
      ELSE IF (MODEL.EQ.3) THEN
         DO 240 I = 1, L
            DG(M+I,N+1)   = ONE
            DG(M+L+I,N+1) = ONE
  240    CONTINUE
      ELSE
         DO 250 I = 1, L
            DG(M+I,N+1) = ONE
  250    CONTINUE
      ENDIF
      RETURN
      END
