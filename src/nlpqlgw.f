C ======================================================================
C
C
C ======================================================================
C
      SUBROUTINE NLPQLGW (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G,
     /                    DF, DG, U, XL, XU, C, D, ACC, ACCQP, STPMIN,
     /                    MAXFUN, MAXIT, MAXNM, RHO, NCYCLE, IPRINT,
     /                    MODE, IOUT, IFAIL, XB, FB, GB, ST, WA, LWA,
     /                    KWA, LKWA, ACT, LACT, LQL, QPSLVE)
C
C   ST(1) - phase, 1 = inside a cycle, 2 = function values at a new
C           starting point, 3 = gradient values at a new starting point
C   ST(2) - termination flag of NLPQLP
C   ST(3) - number of the actual cycle
C   ST(4) - 1, if a feasible solution has been stored
C
      IMPLICIT NONE
      INTEGER NP, M, ME, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        NCYCLE, IPRINT, MODE, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), DF(*), DG(MMAX,*),
     /                 U(*), XL(*), XU(*), C(NMAX,*), D(*), WA(LWA),
     /                 ACC, ACCQP, STPMIN, RHO
      DOUBLE PRECISION XB(*), FB, GB(*), ST(8)
      LOGICAL ACT(LACT), LQL
      EXTERNAL QPSLVE
C
      INTEGER I, J, IPH, IFP, ICYC
      DOUBLE PRECISION SCV, T, ZERO, ONE, TWO, BIG, HUGEB
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0, BIG=1.0D+72,
     /           HUGEB=1.0D+30)
C
      IF (IFAIL.EQ.0) GOTO 100
      IPH = INT(ST(1))
      IF (IPH.EQ.1) GOTO 200
      IF (IPH.EQ.2) GOTO 700
      IF (IPH.EQ.3) GOTO 200
      IFAIL = 9
      RETURN
C
  100 CONTINUE
      ST(2) = ZERO
      ST(3) = ONE
      ST(4) = ZERO
      FB    = BIG
      GOTO 200
C
C   One call of the SQP algorithm.
C
  200 CONTINUE
      IFP = INT(ST(2))
      CALL NLPQLP (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, STPMIN, MAXFUN,
     /             MAXIT, MAXNM, RHO, IPRINT, MODE, IOUT, IFP,
     /             WA, LWA, KWA, LKWA, ACT, LACT, LQL, QPSLVE)
      ST(2) = DBLE(IFP)
      IF (IFP.LT.0) THEN
         ST(1) = ONE
         IFAIL = IFP
         RETURN
      ENDIF
C
C   The actual cycle is finished, store the solution if it is feasible
C   and better than the best one obtained so far.
C
      SCV = ZERO
      DO 210 J = 1, ME
         SCV = SCV + ABS(G(J,1))
  210 CONTINUE
      DO 220 J = ME+1, M
         SCV = SCV + MAX(ZERO,-G(J,1))
  220 CONTINUE
      IF (SCV.LE.SQRT(ACC) .AND. F(1).LT.FB) THEN
         DO 230 I = 1, N
            XB(I) = X(I,1)
  230    CONTINUE
         FB = F(1)
         DO 240 J = 1, M
            GB(J) = G(J,1)
  240    CONTINUE
         ST(4) = ONE
      ENDIF
C
      ICYC = INT(ST(3)) + 1
      ST(3) = DBLE(ICYC)
      IF (ICYC.GT.NCYCLE) GOTO 800
C
C   Generation of the next starting point.
C
      DO 260 I = 1, N
         T = DBLE(ICYC-1)*SQRT(DBLE(2*I-1) + ONE)
         T = T - DBLE(INT(T))
         IF (XL(I).GT.-HUGEB .AND. XU(I).LT.HUGEB) THEN
            X(I,1) = XL(I) + T*(XU(I)-XL(I))
         ELSE
            IF (INT(ST(4)).EQ.1) THEN
               X(I,1) = XB(I) + (TWO*T-ONE)*(ONE+ABS(XB(I)))
            ELSE
               X(I,1) = X(I,1) + (TWO*T-ONE)*(ONE+ABS(X(I,1)))
            ENDIF
            X(I,1) = MIN(MAX(X(I,1),XL(I)),XU(I))
         ENDIF
  260 CONTINUE
      ST(2) = ZERO
      ST(1) = TWO
      IFAIL = -1
      RETURN
C
C   Function values at the new starting point are available, request
C   the gradients.
C
  700 CONTINUE
      ST(1) = 3.0D0
      IFAIL = -2
      RETURN
C
C   All cycles are finished, return the best solution.
C
  800 CONTINUE
      IF (INT(ST(4)).EQ.1) THEN
         IF (FB.LT.F(1) .OR. IFP.NE.0) THEN
            DO 810 I = 1, N
               X(I,1) = XB(I)
  810       CONTINUE
            F(1) = FB
            DO 820 J = 1, M
               G(J,1) = GB(J)
  820       CONTINUE
            IFP = 0
         ENDIF
      ENDIF
      IFAIL = IFP
      RETURN
      END
