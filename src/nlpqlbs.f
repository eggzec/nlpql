C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLPQLBS (M, ME, MW, ACC, G, KW, GS, IFAIL)
C
C   Determination of a new working set.  All equality constraints are
C   inserted first, the remaining positions are filled with the indices
C   of those inequality constraints that possess the smallest function
C   values, i.e., which are active or closest to being active.  The
C   selection is performed by one single pass over all constraints and
C   an insertion into a sorted list of length MW, so that no sorting of
C   the whole constraint vector is required.  IFAIL = 11 is returned if
C   more than MW constraints are active.
C
      IMPLICIT NONE
      INTEGER M, ME, MW, IFAIL
      INTEGER KW(*)
      DOUBLE PRECISION ACC, G(*), GS(*)
      INTEGER I, J, NACT, NREST, NSEL, NA
      DOUBLE PRECISION GJ
C
      IFAIL = 0
      NACT  = MIN(ME,MW)
      DO 10 J = 1, NACT
         KW(J) = J
   10 CONTINUE
      IF (ME.GT.MW) THEN
         IFAIL = 11
         RETURN
      ENDIF
      NREST = MW - NACT
      NSEL  = 0
      NA    = 0
      DO 40 J = ME+1, M
         GJ = G(J)
         IF (GJ.LE.ACC) NA = NA + 1
         IF (NSEL.EQ.NREST) THEN
            IF (NREST.LE.0) GOTO 40
            IF (GJ.GE.GS(NSEL)) GOTO 40
            NSEL = NSEL - 1
         ENDIF
         I = NSEL
   20    CONTINUE
         IF (I.GE.1) THEN
            IF (GS(I).GT.GJ) THEN
               GS(I+1)      = GS(I)
               KW(NACT+I+1) = KW(NACT+I)
               I = I - 1
               GOTO 20
            ENDIF
         ENDIF
         GS(I+1)      = GJ
         KW(NACT+I+1) = J
         NSEL = NSEL + 1
   40 CONTINUE
      IF (NA+NACT.GT.MW) IFAIL = 11
      RETURN
      END
