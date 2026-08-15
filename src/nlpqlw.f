C ======================================================================
C
C     NLPQLW : Working routine of NLPQLP.  The complete internal state
C              of the SQP iteration is kept in the working arrays, so
C              that the routine is re-entrant and thread-safe and can
C              be interrupted after each function or gradient request
C              (reverse communication).
C
C     Meaning of the internal state variables:
C
C        KWA(1) - number of function evaluations
C        KWA(2) - number of gradient evaluations
C        KWA(3) - number of iterations
C        KWA(4) - number of QP solutions
C        KWA(5) - flag for a better feasible, but non-stationary iterate
C        KWA(6) - re-entry branch, 1 = line search, 2 = gradients
C        KWA(7) - number of function calls in the actual line search
C        KWA(8) - number of restarts performed so far
C        KWA(9) - number of merit function values on the stack
C        KWA(10)- actual position in the circular stack
C        KWA(11)- number of active constraints
C        KWA(12)- number of steps since the last reset of the BFGS matrix
C        KWA(13)- 1, if the additional variable delta had to be released
C        KWA(14)- number of successive non-evaluable function calls
C        KWA(15)- index of the accepted test point of the line search
C
C        SC(1)  - actual steplength alpha
C        SC(2)  - merit function value at alpha = 0
C        SC(3)  - directional derivative of the merit function
C        SC(4)  - d^T C d
C        SC(5)  - objective function value at the actual iterate
C        SC(6)  - sum of constraint violations at the actual iterate
C        SC(7)  - value of the additional variable delta
C        SC(8)  - penalty parameter sigma of the additional variable
C        SC(9)  - reduction factor of the parallel line search
C        SC(10) - reference value of the non-monotone line search
C        SC(11) - Karush-Kuhn-Tucker optimality criterion
C        SC(12) - machine precision
C        SC(13) - tolerance of the QP solver
C
C ======================================================================
C
      SUBROUTINE NLPQLW (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G,
     /                   DF, DG, U, XL, XU, C, D, ACC, ACCQP, STPMIN,
     /                   MAXFUN, MAXIT, MAXNM, RHO, IPRINT, MODE, IOUT,
     /                   IFAIL, BEST, XOLD, QK, GOLD, DD, V, R, PNM,
     /                   ALP, PHI, QD, QXL, QXU, CP, QV, SC, QLWA,
     /                   LQLWA, KWA, IQLWA, LIQLWA, ACT, LACT, LQL,
     /                   QPSLVE)
C
      IMPLICIT NONE
      INTEGER NP, M, ME, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, MODE, IOUT, IFAIL, LQLWA, LIQLWA, LACT
      INTEGER KWA(25), IQLWA(LIQLWA)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), DF(*), DG(MMAX,*),
     /                 U(*), XL(*), XU(*), C(NMAX,*), D(*),
     /                 ACC, ACCQP, STPMIN, RHO
      DOUBLE PRECISION BEST(*), XOLD(*), QK(*), GOLD(*), DD(*), V(*),
     /                 R(*), PNM(50), ALP(*), PHI(*), QD(*), QXL(*),
     /                 QXU(*), CP(*), QV(*), SC(20), QLWA(LQLWA)
      LOGICAL ACT(LACT), LQL
      EXTERNAL QPSLVE
C
      INTEGER I, J, K, NQ, IFQP, MODEQP, IACC, NRES, NA, ITER, IDUM
      DOUBLE PRECISION ZERO, ONE, HALF, TWO, BIG, MU, BETA
      DOUBLE PRECISION S, T, ALPHA, PHIA, PHI0, DPHI0, DBD, SCV, SCVN,
     /                 DELTA, SIGMA, EPSM, EPSQP, GAM, PTQ, PTP, PTCP,
     /                 TH, DNRM, ABAR, PREF, SUM, FK
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, HALF=0.5D0, TWO=2.0D0,
     /           BIG=1.0D+72, MU=1.0D-1, BETA=1.0D-1)
      DOUBLE PRECISION NLMERT
      EXTERNAL NLMERT
C
      NQ = N + 1
C
C   Machine precision.
C
      EPSM = ONE
   10 CONTINUE
      EPSM = HALF*EPSM
      IF (ONE + EPSM/TWO .GT. ONE) GOTO 10
      EPSM = TWO*EPSM
      EPSQP = ACCQP
      IF (EPSQP.LE.ZERO) EPSQP = 1.0D+1*EPSM
      SC(12) = EPSM
      SC(13) = EPSQP
C
C   Re-entry.
C
      IF (IFAIL.EQ.0)   GOTO 1000
      IF (IFAIL.EQ.-1)  GOTO 3000
      IF (IFAIL.EQ.-2)  GOTO 4000
      IF (IFAIL.EQ.-10) GOTO 2900
      IFAIL = 9
      GOTO 9000
C
C   ------------------------------------------------------------------
C   Initialization.
C   ------------------------------------------------------------------
C
 1000 CONTINUE
      DO 1010 I = 1, 25
         KWA(I) = 0
 1010 CONTINUE
      DO 1020 I = 1, 20
         IF (I.NE.12 .AND. I.NE.13) SC(I) = ZERO
 1020 CONTINUE
      DO 1030 I = 1, N
         IF (XL(I).GT.XU(I)) THEN
            IFAIL = 6
            GOTO 9000
         ENDIF
         IF (X(I,1).LT.XL(I)-EPSM .OR. X(I,1).GT.XU(I)+EPSM) THEN
            IFAIL = 8
            GOTO 9000
         ENDIF
         X(I,1) = MIN(MAX(X(I,1),XL(I)),XU(I))
 1030 CONTINUE
C
      IF (MODE.EQ.1) THEN
         IF (.NOT.LQL) CALL NLLDLC (N, NMAX, C, D)
      ELSE
         DO 1050 J = 1, N
            DO 1040 I = 1, N
               C(I,J) = ZERO
 1040       CONTINUE
            C(J,J) = ONE
 1050    CONTINUE
         DO 1060 J = 1, MNN2
            U(J) = ZERO
 1060    CONTINUE
      ENDIF
      DO 1070 J = 1, MNN2
         V(J) = U(J)
 1070 CONTINUE
      DO 1080 J = 1, M
         R(J)   = ONE
         ACT(J) = .TRUE.
 1080 CONTINUE
      DO 1090 I = 1, N
         XOLD(I) = X(I,1)
 1090 CONTINUE
      SIGMA   = 1.0D+4
      SC(8)   = SIGMA
      BEST(N+1) = BIG
      KWA(1)  = 1
      KWA(2)  = 1
      KWA(11) = M
      IF (IPRINT.GT.0) CALL NLPRNT (0, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
      GOTO 2000
C
C   ------------------------------------------------------------------
C   Formulation and solution of the quadratic programming subproblem.
C   ------------------------------------------------------------------
C
 2000 CONTINUE
      ITER = KWA(3) + 1
      KWA(3) = ITER
C
C   Sum of constraint violations at the actual iterate.
C
      SCV = ZERO
      DO 2010 J = 1, ME
         SCV = SCV + ABS(G(J,1))
 2010 CONTINUE
      DO 2020 J = ME+1, M
         SCV = SCV + MAX(ZERO,-G(J,1))
 2020 CONTINUE
      SC(6) = SCV
      SC(5) = F(1)
C
C   Store the best feasible iterate obtained so far.
C
      IF (SCV.LE.SQRT(ACC) .AND. F(1).LT.BEST(N+1)) THEN
         DO 2030 I = 1, N
            BEST(I) = X(I,1)
 2030    CONTINUE
         BEST(N+1) = F(1)
         DO 2040 J = 1, M
            BEST(N+1+J) = G(J,1)
 2040    CONTINUE
         IF (ITER.LE.5 .AND. ITER.GT.1) KWA(5) = 1
      ENDIF
C
C   Right hand side, linear term, bounds and the additional column of
C   the Jacobian belonging to the variable delta.
C
 2050 CONTINUE
      DO 2060 I = 1, N
         QD(I)  = DF(I)
         QXL(I) = XL(I) - X(I,1)
         QXU(I) = XU(I) - X(I,1)
 2060 CONTINUE
      QD(NQ)  = ZERO
      QXL(NQ) = ZERO
      QXU(NQ) = ZERO
      IF (KWA(13).EQ.1) QXU(NQ) = ONE
      DO 2070 J = 1, M
         DG(J,NQ) = -G(J,1)
 2070 CONTINUE
      T = ZERO
      DO 2080 I = 1, N
         T = T + ABS(C(I,I))
 2080 CONTINUE
      SIGMA = MAX(ONE,T/DBLE(N))*SC(8)
      DO 2090 I = 1, NQ
         C(I,NQ) = ZERO
         C(NQ,I) = ZERO
 2090 CONTINUE
      C(NQ,NQ) = SIGMA
C
      MODEQP = 1
      CALL QPSLVE (M, ME, MMAX, NQ, NMAX, MNN2, C, QD, DG, G, QXL,
     /             QXU, DD, U, EPSQP, MODEQP, IOUT, IFQP, 0,
     /             QLWA, LQLWA, IQLWA, LIQLWA)
      KWA(4) = KWA(4) + 1
      IF (IFQP.NE.0 .AND. KWA(13).EQ.0) THEN
C
C   Inconsistent linearization, release the additional variable delta.
C
         KWA(13) = 1
         GOTO 2050
      ENDIF
      IF (IFQP.NE.0) THEN
         IF (IFQP.GT.100) THEN
            IFAIL = 100 + IFQP
         ELSE
            IFAIL = 10
         ENDIF
         GOTO 8000
      ENDIF
      DELTA = DD(NQ)
      IF (ABS(DELTA).LT.SQRT(EPSM)) DELTA = ZERO
      DD(NQ) = DELTA
      SC(7) = DELTA
      IF (KWA(13).EQ.1 .AND. DELTA.LT.SQRT(EPSM)) KWA(13) = 0
C
C   Karush-Kuhn-Tucker optimality criterion.
C
      SUM = ZERO
      DO 2100 I = 1, N
         SUM = SUM + DF(I)*DD(I)
 2100 CONTINUE
      SC(11) = ABS(SUM)
      DO 2110 J = 1, M
         SC(11) = SC(11) + ABS(U(J)*G(J,1))
 2110 CONTINUE
      DO 2120 I = 1, N
         SC(11) = SC(11) + ABS(U(M+I)*(X(I,1)-XL(I)))
     /                   + ABS(U(M+NQ+I)*(XU(I)-X(I,1)))
 2120 CONTINUE
C
      DNRM = ZERO
      DO 2130 I = 1, N
         DNRM = DNRM + DD(I)*DD(I)
 2130 CONTINUE
      DNRM = SQRT(DNRM)
      DBD = ZERO
      DO 2150 I = 1, N
         S = ZERO
         DO 2140 J = 1, N
            S = S + C(I,J)*DD(J)
 2140    CONTINUE
         DBD = DBD + DD(I)*S
 2150 CONTINUE
      SC(4) = DBD
C
C
C   Norm of the gradient of the Lagrangian function.
C
      SUM = ZERO
      DO 2158 I = 1, N
         S = DF(I) - U(M+I) + U(M+NQ+I)
         DO 2157 J = 1, M
            S = S - U(J)*DG(J,I)
 2157    CONTINUE
         SUM = SUM + S*S
 2158 CONTINUE
      SC(15) = SQRT(SUM)
C
      IF (IPRINT.EQ.2) CALL NLPRNT (1, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
      IF (IPRINT.GT.2) CALL NLPRNT (3, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
C
C   Determination of the active constraints.
C
      NA = 0
      DO 2160 J = 1, M
         ACT(J) = .FALSE.
         IF (J.LE.ME) ACT(J) = .TRUE.
         IF (G(J,1).LT.SQRT(ACC)) ACT(J) = .TRUE.
         IF (U(J).GT.ZERO) ACT(J) = .TRUE.
         IF (ACT(J)) NA = NA + 1
 2160 CONTINUE
      KWA(11) = NA
C
C   Termination tests.
C
      IF (SC(11).LT.ACC .AND. SCV.LT.SQRT(ACC)) THEN
         IFAIL = 0
         GOTO 8000
      ENDIF
C
C   A search direction of length zero means that the actual iterate
C   satisfies the optimality conditions of the linearized problem.  If
C   in addition the iterate is feasible, it is a Karush-Kuhn-Tucker
C   point of the nonlinear program within the attainable accuracy.
C
      T = ZERO
      DO 2155 I = 1, N
         T = MAX(T,ABS(X(I,1)))
 2155 CONTINUE
      IF (DNRM.LE.ACC*(ONE+T)) THEN
         IF (SCV.LT.SQRT(ACC)) THEN
            IFAIL = 0
         ELSE
            IFAIL = 7
         ENDIF
         GOTO 8000
      ENDIF
      IF (ITER.GE.MAXIT) THEN
         IFAIL = 1
         GOTO 8000
      ENDIF
C
C   Update of the penalty parameters of the merit function.
C
      CALL NLPEN (M, ME, MMAX, N, G, U, V, R, DBD, DELTA, EPSM)
C
C   Merit function value and directional derivative at alpha = 0.
C
      PHI0  = NLMERT (M, ME, F(1), G, V, U, R, ZERO)
      CALL NLDPHI (M, ME, MMAX, N, DF, DG, G, U, V, R, DD, DPHI0)
      NRES = 0
 2200 CONTINUE
      IF (DPHI0.GT.-0.1D0*DBD .AND. NRES.LT.20 .AND. DBD.GT.ZERO) THEN
         DO 2210 J = 1, M
            R(J) = 1.0D+1*R(J)
 2210    CONTINUE
         PHI0 = NLMERT (M, ME, F(1), G, V, U, R, ZERO)
         CALL NLDPHI (M, ME, MMAX, N, DF, DG, G, U, V, R, DD, DPHI0)
         NRES = NRES + 1
         GOTO 2200
      ENDIF
      IF (DPHI0.GE.ZERO) THEN
C
C   Uphill search direction, restart with a scaled unit matrix.
C
         IF (RHO.GT.ZERO .AND. KWA(8).LT.MAXFUN) THEN
            CALL NLREST (N, NMAX, M, MNN2, C, U, V, R, RHO)
            KWA(8)  = KWA(8) + 1
            KWA(12) = 0
            KWA(3)  = KWA(3) - 1
            GOTO 2000
         ENDIF
         IFAIL = 2
         GOTO 8000
      ENDIF
C
C   If the predicted decrease of the merit function is below the
C   accuracy by which the merit function itself can be computed, no
C   further progress is possible and the algorithm has to be stopped.
C
      IF (ABS(DPHI0) .LE. 1.0D+1*EPSM*(ONE + ABS(PHI0))) THEN
         IF (SCV.LT.SQRT(ACC) .AND.
     /       SC(11).LT.SQRT(ACC)*(ONE+ABS(F(1)))) THEN
            IFAIL = 0
         ELSE
            IFAIL = 4
         ENDIF
         GOTO 8000
      ENDIF
      SC(2)  = PHI0
      SC(3)  = DPHI0
      SC(16) = DPHI0
      IF (IPRINT.GT.2) CALL NLPRNT (6, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
      SC(5) = F(1)
      FK    = F(1)
      DO 2220 J = 1, M
         GOLD(J) = G(J,1)
 2220 CONTINUE
C
C   Reference value of the non-monotone line search.
C
      IF (MAXNM.GT.0) THEN
         K = KWA(10) + 1
         IF (K.GT.MAXNM) K = 1
         KWA(10) = K
         PNM(K)  = PHI0
         IF (KWA(9).LT.MAXNM) KWA(9) = KWA(9) + 1
         PREF = PNM(1)
         DO 2230 I = 2, KWA(9)
            PREF = MAX(PREF,PNM(I))
 2230    CONTINUE
      ELSE
         PREF = PHI0
      ENDIF
      SC(10) = PREF
C
C   ------------------------------------------------------------------
C   Set up of the test points of the line search.
C   ------------------------------------------------------------------
C
      KWA(7)  = 0
      KWA(14) = 0
      IF (NP.EQ.1) THEN
         SC(1) = ONE
      ELSE
         T = STPMIN
         IF (T.LE.ZERO) T = 1.0D-10
         SC(9) = T**(ONE/DBLE(NP-1))
         SC(1) = ONE
      ENDIF
 2800 CONTINUE
      ALP(1) = SC(1)
      IF (NP.GT.1) THEN
         DO 2810 I = 2, NP
            ALP(I) = ALP(I-1)*SC(9)
 2810    CONTINUE
      ENDIF
      DO 2830 K = 1, NP
         DO 2820 I = 1, N
            X(I,K) = XOLD(I) + ALP(K)*DD(I)
            X(I,K) = MIN(MAX(X(I,K),XL(I)),XU(I))
 2820    CONTINUE
 2830 CONTINUE
      KWA(1) = KWA(1) + 1
      KWA(7) = KWA(7) + 1
      IFAIL  = -1
      RETURN
C
C   Function values could not be computed, reduce the steplength.
C
 2900 CONTINUE
      KWA(14) = KWA(14) + 1
      IF (KWA(14).GT.MAXFUN) THEN
         IFAIL = 11
         GOTO 8000
      ENDIF
      SC(1) = HALF*SC(1)
      GOTO 2800
C
C   ------------------------------------------------------------------
C   Line search.
C   ------------------------------------------------------------------
C
 3000 CONTINUE
      KWA(14) = 0
      IACC = 0
      DO 3020 K = 1, NP
         PHI(K) = NLMERT (M, ME, F(K), G(1,K), V, U, R, ALP(K))
         IF (IPRINT.GT.3) THEN
            SC(1)  = ALP(K)
            SC(17) = PHI(K)
            CALL NLPRNT (4, N, M, ME, MMAX, NMAX, MNN2, MODE, ACC,
     /           ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT, IPRINT,
     /           IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
         ENDIF
         IF (IACC.EQ.0) THEN
            IF (PHI(K) .LT. SC(2) + MU*ALP(K)*SC(3)) IACC = K
         ENDIF
 3020 CONTINUE
      IF (IACC.GT.0) GOTO 3600
C
C   No sufficient decrease.  In the serial case a new steplength is
C   computed by a quadratic interpolation combined with an Armijo type
C   reduction, in the parallel case the whole set of test values is
C   moved towards zero.
C
      IF (KWA(7).GE.MAXFUN) THEN
C
C   Non-monotone stopping criterion in the error situation.
C
         IF (MAXNM.GT.0) THEN
            DO 3030 K = 1, NP
               IF (IACC.EQ.0) THEN
                  IF (PHI(K) .LE. SC(10) + MU*ALP(K)*SC(3)) IACC = K
               ENDIF
 3030       CONTINUE
         ENDIF
         IF (IACC.GT.0) GOTO 3600
         IF (RHO.GT.ZERO .AND. KWA(8).LT.MAXFUN) THEN
            CALL NLREST (N, NMAX, M, MNN2, C, U, V, R, RHO)
            KWA(8)  = KWA(8) + 1
            KWA(12) = 0
            DO 3040 I = 1, N
               X(I,1) = XOLD(I)
 3040       CONTINUE
            F(1) = SC(5)
            DO 3050 J = 1, M
               G(J,1) = GOLD(J)
 3050       CONTINUE
            KWA(3) = KWA(3) - 1
            GOTO 2000
         ENDIF
         IFAIL = 4
         GOTO 8000
      ENDIF
C
      IF (NP.EQ.1) THEN
         ALPHA = SC(1)
         PHIA  = PHI(1)
         T     = ALPHA*SC(3) - PHIA + SC(2)
         IF (ABS(T).GT.ZERO) THEN
            ABAR = HALF*ALPHA*ALPHA*SC(3)/T
         ELSE
            ABAR = BETA*ALPHA
         ENDIF
         ALPHA = MAX(BETA*ALPHA,ABAR)
         ALPHA = MIN(ALPHA,0.9D0*SC(1))
         SC(1) = ALPHA
      ELSE
         SC(1) = SC(1)*SC(9)**NP
      ENDIF
      IF (SC(1).LT.EPSM) THEN
         IFAIL = 4
         GOTO 8000
      ENDIF
      GOTO 2800
C
C   Acceptance of the test point number IACC.
C
 3600 CONTINUE
      KWA(15) = IACC
      ALPHA   = ALP(IACC)
      SC(1)   = ALPHA
      IF (IPRINT.GT.3) CALL NLPRNT (5, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
      IF (IACC.NE.1) THEN
         DO 3610 I = 1, N
            X(I,1) = X(I,IACC)
 3610    CONTINUE
         F(1) = F(IACC)
         DO 3620 J = 1, M
            G(J,1) = G(J,IACC)
 3620    CONTINUE
      ENDIF
C
C   New multiplier estimate and gradient of the Lagrangian function at
C   the previous iterate.
C
      DO 3630 J = 1, MNN2
         V(J) = V(J) + ALPHA*(U(J)-V(J))
         U(J) = V(J)
 3630 CONTINUE
      DO 3650 I = 1, N
         S = DF(I)
         DO 3640 J = 1, M
            S = S - V(J)*DG(J,I)
 3640    CONTINUE
         QK(I) = -S
 3650 CONTINUE
      KWA(2) = KWA(2) + 1
      IFAIL  = -2
      RETURN
C
C   ------------------------------------------------------------------
C   New gradient values, update of the quasi-Newton matrix.
C   ------------------------------------------------------------------
C
 4000 CONTINUE
      DO 4020 I = 1, N
         S = DF(I)
         DO 4010 J = 1, M
            S = S - V(J)*DG(J,I)
 4010    CONTINUE
         QV(I) = S + QK(I)
         CP(I) = X(I,1) - XOLD(I)
 4020 CONTINUE
      PTQ = ZERO
      PTP = ZERO
      DO 4030 I = 1, N
         PTQ = PTQ + CP(I)*QV(I)
         PTP = PTP + CP(I)*CP(I)
 4030 CONTINUE
      GAM = ONE
      IF (PTP.GT.ZERO) GAM = PTQ/PTP
      SC(14) = GAM
      KWA(12) = KWA(12) + 1
C
C   Scaling and periodic restarts of the BFGS matrix.
C
      IDUM = 0
      IF (MODE.EQ.2 .AND. KWA(3).EQ.1) IDUM = 1
      IF (MODE.EQ.3 .AND. GAM.LE.SQRT(ACC)) IDUM = 1
      IF (MODE.GT.3) THEN
         IF (KWA(12).GE.MODE) IDUM = 1
      ENDIF
      IF (IDUM.EQ.1 .AND. GAM.GT.ZERO) THEN
         DO 4050 J = 1, N
            DO 4040 I = 1, N
               C(I,J) = ZERO
 4040       CONTINUE
            C(J,J) = GAM
 4050    CONTINUE
         KWA(12) = 0
      ENDIF
C
      CALL NLBFGS (N, NMAX, C, CP, QV, QD, EPSM, IFQP)
      IF (IFQP.NE.0) THEN
         IF (RHO.GT.ZERO .AND. KWA(8).LT.MAXFUN) THEN
            CALL NLREST (N, NMAX, M, MNN2, C, U, V, R, RHO)
            KWA(8)  = KWA(8) + 1
            KWA(12) = 0
         ELSE
            IFAIL = 3
            GOTO 8000
         ENDIF
      ENDIF
C
      DO 4060 I = 1, N
         XOLD(I) = X(I,1)
 4060 CONTINUE
      GOTO 2000
C
C   ------------------------------------------------------------------
C   Final output.
C   ------------------------------------------------------------------
C
 8000 CONTINUE
      IF (IFAIL.GT.0 .AND. BEST(N+1).LT.F(1)) THEN
         SCVN = ZERO
         DO 8010 J = 1, ME
            SCVN = SCVN + ABS(G(J,1))
 8010    CONTINUE
         DO 8020 J = ME+1, M
            SCVN = SCVN + MAX(ZERO,-G(J,1))
 8020    CONTINUE
         IF (SCVN.GT.SQRT(ACC)) THEN
            DO 8030 I = 1, N
               X(I,1) = BEST(I)
 8030       CONTINUE
            F(1) = BEST(N+1)
            DO 8040 J = 1, M
               G(J,1) = BEST(N+1+J)
 8040       CONTINUE
         ENDIF
      ENDIF
      IF (.NOT.LQL) CALL NLLDLF (N, NMAX, C, D)
 9000 CONTINUE
      IF (IPRINT.GT.0) CALL NLPRNT (2, N, M, ME, MMAX, NMAX, MNN2,
     /     MODE, ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM, MAXIT,
     /     IPRINT, IOUT, X, F, G, U, XL, XU, R, KWA, SC, IFAIL)
      RETURN
      END
