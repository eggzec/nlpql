C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLPJBG (L, M, LMMAX, N, LNMAX, NEX, MODEL, IMIN, A,
     /                   DG)
C
C   Insert the gradients of the additional constraints and of the
C   residuals into the Jacobian.  Rows M+L+1 to M+2*L serve as scratch
C   space for the gradients of the objective functions.
C
      IMPLICIT NONE
      INTEGER L, M, LMMAX, N, LNMAX, NEX, MODEL, IMIN
      DOUBLE PRECISION A(L), DG(LMMAX,LNMAX)
      INTEGER I, J, K
C
      DO 20 I = 1, L
         DO 10 K = 1, N
            DG(M+L+I,K) = DG(M+I,K)
   10    CONTINUE
   20 CONTINUE
      J = 0
      IF (MODEL.EQ.2) THEN
         DO 40 I = 1, IMIN-1
            J = J + 1
            DO 30 K = 1, N
               DG(M+J,K) = -DG(M+L+I,K)
   30       CONTINUE
   40    CONTINUE
      ELSE IF (MODEL.EQ.3) THEN
         DO 60 I = 1, L
            IF (I.NE.IMIN) THEN
               J = J + 1
               DO 50 K = 1, N
                  DG(M+J,K) = -DG(M+L+I,K)
   50          CONTINUE
            ENDIF
   60    CONTINUE
      ENDIF
      DO 80 I = 1, L
         DO 70 K = 1, N
            DG(M+NEX+I,K) = A(I)*DG(M+L+I,K)
   70    CONTINUE
   80 CONTINUE
      RETURN
      END
