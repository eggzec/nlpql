C ======================================================================
C
C     NLPQLYW : Working routine of NLPQLY.  The routine drives NLPQLP
C               and inserts the forward difference approximation of the
C               gradients into the reverse communication protocol.  The
C               complete state is kept in the working arrays, so that
C               the routine remains re-entrant and thread-safe.
C
C        ST(1) - phase, 1 = function value requested by the line search
C                       2 = function value of a difference formula
C        ST(2) - index of the actual partial derivative
C        ST(3) - actual increment eta_i
C        ST(4) - termination flag of NLPQLP
C        ST(5) - objective function value handed over to NLPQLP
C
C ======================================================================
C
      SUBROUTINE NLPQLYW (M, ME, N, NQ, MG, MNN2, X, F, G, XL, XU,
     /                    ACC, MAXIT, IPRINT, IOUT, IFAIL, XW, DF, DG,
     /                    U, C, D, GW, XB, FB, GB, ST, WA, LWA, KWA,
     /                    LKWA, ACT, LACT)
C
      IMPLICIT NONE
      INTEGER M, ME, N, NQ, MG, MNN2, MAXIT, IPRINT, IOUT, IFAIL,
     /        LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), XL(*), XU(*), ACC
      DOUBLE PRECISION XW(NQ), DF(NQ), DG(MG,NQ), U(*), C(NQ,NQ),
     /                 D(NQ), GW(MG), XB(*), FB, GB(*), ST(6), WA(LWA)
      LOGICAL ACT(LACT)
C
      INTEGER I, J, IPH, IFP
      DOUBLE PRECISION ETA, ETAI, ZERO, ONE, HALF, TWO, EPSM, RHO
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, HALF=0.5D0, TWO=2.0D0)
      EXTERNAL QL
C
      EPSM = ONE
   10 CONTINUE
      EPSM = HALF*EPSM
      IF (ONE + EPSM/TWO .GT. ONE) GOTO 10
      EPSM = TWO*EPSM
      ETA  = SQRT(EPSM)
      RHO  = 1.0D+2
C
      IF (IFAIL.EQ.0) GOTO 100
      IPH = INT(ST(1))
      IF (IPH.EQ.1) GOTO 300
      IF (IPH.EQ.2) GOTO 400
      IFAIL = 9
      RETURN
C
C   Initialization.  Objective and constraint function values at the
C   starting point are already provided by the calling program.
C
  100 CONTINUE
      DO 110 I = 1, N
         XW(I) = X(I)
  110 CONTINUE
      XW(NQ) = ZERO
      ST(5)  = F
      DO 120 J = 1, M
         GW(J) = G(J)
  120 CONTINUE
      ST(4) = ZERO
      GOTO 500
C
C   Re-entry with a function value requested by the line search of the
C   SQP algorithm.
C
  300 CONTINUE
      ST(5) = F
      DO 310 J = 1, M
         GW(J) = G(J)
  310 CONTINUE
      GOTO 600
C
C   Re-entry with a function value of the forward difference formula.
C
  400 CONTINUE
      I     = INT(ST(2))
      ETAI  = ST(3)
      DF(I) = (F - FB)/ETAI
      DO 410 J = 1, M
         DG(J,I) = (G(J) - GB(J))/ETAI
  410 CONTINUE
      I = I + 1
      IF (I.LE.N) GOTO 520
C
C   All partial derivatives are available, restore the actual iterate.
C
      DO 420 I = 1, N
         X(I) = XB(I)
  420 CONTINUE
      F = FB
      DO 430 J = 1, M
         G(J) = GB(J)
  430 CONTINUE
      GOTO 600
C
C   Start of a new gradient approximation at the actual iterate XW.
C
  500 CONTINUE
      DO 505 I = 1, N
         XB(I) = XW(I)
  505 CONTINUE
      FB = ST(5)
      DO 510 J = 1, M
         GB(J) = GW(J)
  510 CONTINUE
      I = 1
  520 CONTINUE
      ETAI = ETA*MAX(1.0D-5,ABS(XB(I)))
      IF (XB(I)+ETAI .GT. XU(I)) ETAI = -ETAI
      IF (XB(I)+ETAI .LT. XL(I)) ETAI = ABS(ETAI)
      ST(1) = TWO
      ST(2) = DBLE(I)
      ST(3) = ETAI
      DO 530 J = 1, N
         X(J) = XB(J)
  530 CONTINUE
      X(I)  = XB(I) + ETAI
      IFAIL = -1
      RETURN
C
C   One call of the SQP algorithm.
C
  600 CONTINUE
      IFP = INT(ST(4))
      CALL NLPQLP (1, M, ME, MG, N, NQ, MNN2, XW, ST(5), GW, DF, DG,
     /             U, XL, XU, C, D, ACC, ZERO, ZERO, 20, MAXIT, 10,
     /             RHO, IPRINT, 0, IOUT, IFP, WA, LWA, KWA, LKWA,
     /             ACT, LACT, .TRUE., QL)
      ST(4) = DBLE(IFP)
      IF (IFP.EQ.-1) THEN
         ST(1) = ONE
         DO 610 I = 1, N
            X(I) = XW(I)
  610    CONTINUE
         IFAIL = -1
         RETURN
      ENDIF
      IF (IFP.EQ.-2) GOTO 500
C
C   Termination.
C
      DO 620 I = 1, N
         X(I) = XW(I)
  620 CONTINUE
      F = ST(5)
      DO 630 J = 1, M
         G(J) = GW(J)
  630 CONTINUE
      IFAIL = IFP
      RETURN
      END
