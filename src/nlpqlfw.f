C ======================================================================
C
C     NLPQLFW : Working routine of NLPQLF.  Drives the SQP code NLPQLP
C               and inserts an evaluation of the feasibility
C               constraints in front of every evaluation of the
C               objective function and of the remaining constraints.
C
C        ST(1) - phase, 1 = feasibility constraints requested,
C                       2 = objective and constraints requested,
C                       3 = gradients requested
C        ST(2) - termination flag of NLPQLP
C
C ======================================================================
C
      SUBROUTINE NLPQLFW (MF, MEF, M, MMAX, N, NMAX, MNN2, X, F, G, DF,
     /                    DG, U, XL, XU, C, D, ACC, ACCF, ACCQP,
     /                    MAXFUN, MAXIT, MAXNM, RHO, IPRINT, IOUT,
     /                    IFAIL, FV, ST, WA, LWA, KWA, LKWA, ACT, LACT,
     /                    QPSLVE)
C
      IMPLICIT NONE
      INTEGER MF, MEF, M, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(NMAX), F, G(MMAX), DF(NMAX), DG(MMAX,NMAX),
     /                 U(MNN2), XL(N), XU(N), C(NMAX,NMAX), D(NMAX),
     /                 WA(LWA), ACC, ACCF, ACCQP, RHO
      DOUBLE PRECISION FV(1), ST(6)
      LOGICAL ACT(LACT)
      EXTERNAL QPSLVE
C
      INTEGER I, J, IPH, IFP, MT, MODEQ
      DOUBLE PRECISION S, TOL, ZERO, ONE, TWO, THREE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0, THREE=3.0D0)
C
      MT  = MF + M
      TOL = ACCF
      IF (TOL.LT.ZERO) TOL = ZERO
C
      IF (IFAIL.EQ.0)  GOTO 100
      IPH = INT(ST(1))
      IF (IPH.EQ.1) GOTO 200
      IF (IPH.EQ.2) GOTO 300
      IF (IPH.EQ.3) GOTO 300
      IFAIL = 9
      RETURN
C
C   Initialization.  The starting point must satisfy all feasibility
C   constraints, otherwise the objective function cannot be evaluated
C   at all.
C
  100 CONTINUE
      ST(2) = ZERO
      DO 110 I = 1, M
         IF (G(MF+I).LT.-TOL) THEN
            IFAIL = 8
            IF (IPRINT.GT.0) WRITE (IOUT,1000)
            RETURN
         ENDIF
  110 CONTINUE
      GOTO 300
C
C   Re-entry with the values of the feasibility constraints only.  If
C   one of them is violated, the steplength of the line search is
C   reduced by the factor one half, which is requested by returning
C   IFAIL = -10 to the SQP algorithm.  Otherwise the objective function
C   and the remaining constraints may safely be evaluated.
C
  200 CONTINUE
      S = ZERO
      DO 210 I = 1, M
         S = MIN(S,G(MF+I))
  210 CONTINUE
      IF (S.LT.-TOL) THEN
         ST(2) = -1.0D+1
         GOTO 300
      ENDIF
      ST(1) = TWO
      IFAIL = -1
      RETURN
C
C   One call of the SQP algorithm.
C
  300 CONTINUE
      FV(1) = F
      IFP   = INT(ST(2))
      MODEQ = 0
      CALL NLPQLP (1, MT, MEF, MMAX, N, NMAX, MNN2, X, FV, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, ZERO, MAXFUN, MAXIT,
     /             MAXNM, RHO, IPRINT, MODEQ, IOUT, IFP, WA, LWA, KWA,
     /             LKWA, ACT, LACT, .TRUE., QPSLVE)
      ST(2) = DBLE(IFP)
      F     = FV(1)
      IF (IFP.EQ.-1) THEN
C
C   New function values are needed.  Ask for the feasibility
C   constraints first, they are much cheaper and decide whether the
C   test point may be used at all.
C
         ST(1) = ONE
         IFAIL = -3
         RETURN
      ENDIF
      IF (IFP.EQ.-2) THEN
         ST(1) = THREE
         IFAIL = -2
         RETURN
      ENDIF
      IFAIL = IFP
      RETURN
 1000 FORMAT (' *** ERROR IN NLPQLF: the starting point violates a',
     /        ' feasibility constraint')
      END
