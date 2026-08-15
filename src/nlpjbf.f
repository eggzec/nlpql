C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLPJBF (L, M, LMMAX, NEX, MODEL, IMIN, A, B, W, FK, G,
     /                   FW)
C
C   Insert the additional constraints and the residuals into the array
C   of constraint values.  Positions M+L+1 to M+2*L serve as scratch
C   space for the objective function values.
C
      IMPLICIT NONE
      INTEGER L, M, LMMAX, NEX, MODEL, IMIN
      DOUBLE PRECISION A(L), B(L), W(L), FK(L), G(LMMAX), FW(L)
      INTEGER I, J, K
      DOUBLE PRECISION ONE, HUND
      PARAMETER (ONE=1.0D0, HUND=1.0D+2)
C
      DO 10 I = 1, L
         G(M+L+I) = G(M+I)
         FW(I)    = G(M+I)
   10 CONTINUE
      K = 0
      IF (MODEL.EQ.2) THEN
         DO 20 J = 1, IMIN-1
            K = K + 1
            G(M+K) = (ONE + W(J)/HUND)*FK(J) - G(M+L+J)
   20    CONTINUE
      ELSE IF (MODEL.EQ.3) THEN
         DO 30 J = 1, L
            IF (J.NE.IMIN) THEN
               K = K + 1
               G(M+K) = W(J) - G(M+L+J)
            ENDIF
   30    CONTINUE
      ENDIF
      DO 40 I = 1, L
         G(M+NEX+I) = A(I)*(G(M+L+I) - B(I))
   40 CONTINUE
      RETURN
      END
