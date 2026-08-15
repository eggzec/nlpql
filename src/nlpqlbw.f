C ======================================================================
C
C     NLPQLBW : Working routine of NLPQLB.  Maintains the working set
C               and drives the SQP code NLPQLP with the reduced problem
C               consisting of MW constraints only.
C
C        ST(1) - phase, 1 = function values requested,
C                       2 = gradient values requested
C        ST(2) - termination flag of NLPQLP
C        ST(3) - 1, if the working set has been rearranged
C        ST(4) - number of rearrangements after a successful return
C
C ======================================================================
C
      SUBROUTINE NLPQLBW (M, ME, MW, MWMAX, MWG, N, NMAX, MNN2, X, F,
     /                    G, DF, DG, U, XL, XU, C, D, ACC, ACCQP,
     /                    MAXFUN, MAXIT, MAXNM, RHOB, IPRINT, IOUT,
     /                    IFAIL, GW, FV, ST, WA, LWA, KW, KP, LKP,
     /                    ACT, AW, LAW, QPSLVE)
C
      IMPLICIT NONE
      INTEGER M, ME, MW, MWMAX, MWG, N, NMAX, MNN2, MAXFUN, MAXIT,
     /        MAXNM, IPRINT, IOUT, IFAIL, LWA, LKP, LAW
      INTEGER KW(*), KP(LKP)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MWMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(LWA), ACC, ACCQP,
     /                 RHOB
      DOUBLE PRECISION GW(MWG), FV(1), ST(6)
      LOGICAL ACT(*), AW(LAW)
      EXTERNAL QPSLVE
C
      INTEGER I, J, K, IPH, IFP, MODE, NEWSET
      DOUBLE PRECISION ZERO, ONE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
      IF (IFAIL.EQ.0) GOTO 100
      IPH = INT(ST(1))
      IF (IPH.EQ.1) GOTO 200
      IF (IPH.EQ.2) GOTO 300
      IFAIL = 9
      RETURN
C
C   Initialization.  The calling program has provided a first working
C   set which must contain all equality constraints and all constraints
C   that are active or violated at the starting point.
C
  100 CONTINUE
      ST(2) = ZERO
      ST(3) = ZERO
      ST(4) = ZERO
      DO 110 J = 1, M
         ACT(J) = .FALSE.
  110 CONTINUE
      DO 120 I = 1, MW
         J = KW(I)
         IF (J.LT.1 .OR. J.GT.M) THEN
            IFAIL = 6
            RETURN
         ENDIF
         ACT(J) = .TRUE.
  120 CONTINUE
      DO 130 J = 1, ME
         IF (.NOT.ACT(J)) THEN
            IFAIL = 11
            IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
            RETURN
         ENDIF
  130 CONTINUE
      DO 140 J = ME+1, M
         IF (G(J).LE.ACC .AND. .NOT.ACT(J)) THEN
            IFAIL = 11
            IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
            RETURN
         ENDIF
  140 CONTINUE
      GOTO 400
C
C   Re-entry with new function values.  At each intermediate step of the
C   line search at most MW constraints may be violated.  If the actual
C   test point does not satisfy this condition, the steplength is
C   reduced by the factor 0.5 until the number of violated constraints
C   becomes sufficiently small.
C
  200 CONTINUE
      K = 0
      DO 210 J = ME+1, M
         IF (G(J).LE.ACC) K = K + 1
  210 CONTINUE
      IF (K+ME.GT.MW) THEN
         ST(2) = -1.0D+1
      ENDIF
      GOTO 400
C
C   Re-entry with new gradient values of the constraints of the actual
C   working set.
C
  300 CONTINUE
      IF (INT(ST(3)).EQ.1) THEN
C
C   The working set has been rearranged, so that the SQP algorithm has
C   to be started again from the actual iterate.  The quasi-Newton
C   matrix accumulated so far is retained, the multiplier approximation
C   is reset since the index sets are no longer comparable.
C
         ST(3) = ZERO
         ST(2) = ZERO
         DO 310 I = 1, MNN2
            U(I) = ZERO
  310    CONTINUE
      ENDIF
      GOTO 400
C
C   One call of the SQP algorithm with the reduced set of constraints.
C
  400 CONTINUE
      DO 410 I = 1, MW
         GW(I) = G(KW(I))
  410 CONTINUE
      FV(1) = F
      IFP   = INT(ST(2))
      MODE  = 0
      IF (IFP.EQ.0 .AND. IFAIL.NE.0) MODE = 1
      CALL NLPQLP (1, MW, ME, MWMAX, N, NMAX, MNN2, X, FV, GW, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, ZERO, MAXFUN, MAXIT,
     /             MAXNM, RHOB, IPRINT, MODE, IOUT, IFP, WA, LWA,
     /             KP, LKP, AW, LAW, .TRUE., QPSLVE)
      ST(2) = DBLE(IFP)
      F     = FV(1)
      DO 420 I = 1, MW
         G(KW(I)) = GW(I)
  420 CONTINUE
C
      IF (IFP.EQ.-1) THEN
         ST(1) = ONE
         IFAIL = -1
         RETURN
      ENDIF
      IF (IFP.EQ.-2) THEN
C
C   A new iterate has been accepted.  Check whether a constraint that
C   does not belong to the working set has become active or violated.
C
         DO 430 J = 1, M
            ACT(J) = .FALSE.
  430    CONTINUE
         DO 440 I = 1, MW
            ACT(KW(I)) = .TRUE.
  440    CONTINUE
C
C   Only a constraint that is really violated forces a rearrangement of
C   the working set during the iteration.  Constraints that are active
C   but satisfied are taken into account by the final check below, so
C   that the SQP iteration is not interrupted unnecessarily often.
C
         NEWSET = 0
         DO 450 J = ME+1, M
            IF (G(J).LT.-ACC .AND. .NOT.ACT(J)) NEWSET = 1
  450    CONTINUE
         IF (NEWSET.EQ.1) THEN
            CALL NLPQLBS (M, ME, MW, ACC, G, KW, GW, IFAIL)
            IF (IFAIL.NE.0) THEN
               IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
               RETURN
            ENDIF
            ST(3) = ONE
            DO 460 I = 1, MW
               ACT(KW(I)) = .TRUE.
  460       CONTINUE
         ELSE
C
C   The working set is unchanged, only the gradients of the constraints
C   that the SQP algorithm considers to be active are required.
C
            DO 470 J = 1, M
               ACT(J) = .FALSE.
  470       CONTINUE
            DO 480 I = 1, MW
               IF (AW(I)) ACT(KW(I)) = .TRUE.
  480       CONTINUE
         ENDIF
         ST(1) = 2.0D0
         IFAIL = -2
         RETURN
      ENDIF
C
C
C   The SQP algorithm has terminated.  The convergence conditions of the
C   reduced problem are applicable for the original one only if all
C   constraints outside the working set are inactive.  Otherwise the
C   working set is rearranged and the iteration is continued.
C
      IF (IFP.EQ.0) THEN
         DO 500 J = 1, M
            ACT(J) = .FALSE.
  500    CONTINUE
         DO 510 I = 1, MW
            ACT(KW(I)) = .TRUE.
  510    CONTINUE
         NEWSET = 0
         DO 520 J = ME+1, M
            IF (G(J).LE.ACC .AND. .NOT.ACT(J)) NEWSET = 1
  520    CONTINUE
         IF (NEWSET.EQ.1 .AND. INT(ST(4)).LT.MAXIT) THEN
            ST(4) = ST(4) + ONE
            CALL NLPQLBS (M, ME, MW, ACC, G, KW, GW, IFAIL)
            IF (IFAIL.NE.0) THEN
               IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
               RETURN
            ENDIF
            ST(3) = ONE
            DO 530 I = 1, MW
               ACT(KW(I)) = .TRUE.
  530       CONTINUE
            ST(1) = 2.0D0
            IFAIL = -2
            RETURN
         ENDIF
      ENDIF
C
      IFAIL = IFP
      RETURN
 1000 FORMAT (' *** ERROR IN NLPQLB: IFAIL = ',I5,
     /        ', too many active constraints, increase MW')
      END
