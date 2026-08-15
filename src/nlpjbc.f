C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLPJBC (L, MODEL, IMIN, W, FK, A, B, KIND, NEX, IERR)
C
C   Coefficients of the residual vector, kind of the norm and number of
C   additional constraints for the desired transformation.  KIND is the
C   model number of NLDFTW, i.e., 1 = least squares, 2 = L1, 3 = maximum
C   of absolute values, 4 = maximum, 6 = plain sum.
C
      IMPLICIT NONE
      INTEGER L, MODEL, IMIN, KIND, NEX, IERR
      DOUBLE PRECISION W(L), FK(L), A(L), B(L)
      INTEGER I
      DOUBLE PRECISION ZERO, ONE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
      LOGICAL NEEDFK
C
      IERR = 0
      NEX  = 0
      NEEDFK = MODEL.EQ.6 .OR. MODEL.EQ.7 .OR. MODEL.EQ.11
     /         .OR. MODEL.EQ.12 .OR. MODEL.EQ.14 .OR. MODEL.EQ.15
      IF (NEEDFK) THEN
         DO 10 I = 1, L
            IF (FK(I).EQ.ZERO) THEN
               IERR = 11
               RETURN
            ENDIF
   10    CONTINUE
      ENDIF
      IF ((MODEL.EQ.0 .OR. MODEL.EQ.2 .OR. MODEL.EQ.3)
     /    .AND. (IMIN.LT.1 .OR. IMIN.GT.L)) THEN
         IERR = 9
         RETURN
      ENDIF
C
      DO 20 I = 1, L
         A(I) = ONE
         B(I) = ZERO
   20 CONTINUE
C
      IF (MODEL.EQ.0 .OR. MODEL.EQ.2 .OR. MODEL.EQ.3) THEN
         KIND = 6
         DO 30 I = 1, L
            A(I) = ZERO
   30    CONTINUE
         A(IMIN) = ONE
         IF (MODEL.EQ.2) NEX = IMIN - 1
         IF (MODEL.EQ.3) NEX = L - 1
      ELSE IF (MODEL.EQ.1) THEN
         KIND = 6
         DO 40 I = 1, L
            A(I) = W(I)
   40    CONTINUE
      ELSE IF (MODEL.EQ.4) THEN
         KIND = 2
         DO 50 I = 1, L
            B(I) = W(I)
   50    CONTINUE
      ELSE IF (MODEL.EQ.5) THEN
         KIND = 1
         DO 60 I = 1, L
            B(I) = W(I)
   60    CONTINUE
      ELSE IF (MODEL.EQ.6) THEN
         KIND = 6
         DO 70 I = 1, L
            A(I) = ONE/ABS(FK(I))
            B(I) = FK(I)
   70    CONTINUE
      ELSE IF (MODEL.EQ.7) THEN
         KIND = 1
         DO 80 I = 1, L
            A(I) = ONE/FK(I)
            B(I) = FK(I)
   80    CONTINUE
      ELSE IF (MODEL.EQ.8) THEN
         KIND = 3
      ELSE IF (MODEL.EQ.9) THEN
         KIND = 4
      ELSE IF (MODEL.EQ.10) THEN
         KIND = 3
         DO 90 I = 1, L
            B(I) = W(I)
   90    CONTINUE
      ELSE IF (MODEL.EQ.11) THEN
         KIND = 4
         DO 100 I = 1, L
            A(I) = ONE/ABS(FK(I))
            B(I) = FK(I)
  100    CONTINUE
      ELSE IF (MODEL.EQ.12) THEN
         KIND = 4
         DO 110 I = 1, L
            A(I) = W(I)/ABS(FK(I))
            B(I) = FK(I)
  110    CONTINUE
      ELSE IF (MODEL.EQ.13) THEN
         KIND = 4
         DO 120 I = 1, L
            A(I) = W(I)
  120    CONTINUE
      ELSE IF (MODEL.EQ.14) THEN
         KIND = 6
         DO 130 I = 1, L
            A(I) = W(I)/FK(I)
            B(I) = FK(I)
  130    CONTINUE
      ELSE
         KIND = 1
         DO 140 I = 1, L
            IF (W(I).LT.ZERO) THEN
               IERR = 9
               RETURN
            ENDIF
            A(I) = SQRT(W(I))/FK(I)
            B(I) = FK(I)
  140    CONTINUE
      ENDIF
      RETURN
      END
