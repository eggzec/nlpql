C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C ======================================================================
C
      SUBROUTINE QLDUAL (M, ME, MMAX, N, NMAX, MNN, C, D, A, B,
     /                   XL, XU, X, U, EPS, MODE, IFAIL,
     /                   RJ, R, Z, RV, DV, UACT, IACT)
C
C   Dual active set algorithm of Goldfarb and Idnani.  RJ contains the
C   inverse J = R^(-1) of the upper triangular Cholesky factor of C,
C   R the upper triangular factor of J^T N, N being the matrix of the
C   normals of the constraints in the active set.
C
      IMPLICIT NONE
      INTEGER M, ME, MMAX, N, NMAX, MNN, MODE, IFAIL
      INTEGER IACT(N)
      DOUBLE PRECISION C(NMAX,*), D(*), A(MMAX,*), B(*), XL(*), XU(*),
     /                 X(*), U(*), EPS
      DOUBLE PRECISION RJ(N,N), R((N*(N+1))/2), Z(N), RV(N), DV(N),
     /                 UACT(N+1)
C
      INTEGER I, J, K, L, NACT, ITER, MAXIT, IP, K1, IB, NA
      DOUBLE PRECISION S, SP, SGN, SMIN, T, T1, T2, ZTN, ZERO, ONE,
     /                 BIG, TOL, TOLV, HUGEB, SUM
      LOGICAL ACTIVE
      DOUBLE PRECISION QLSLK
      EXTERNAL QLSLK
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, BIG=1.0D30, HUGEB=1.0D30)
C
      IFAIL = 0
      TOL   = MAX(EPS,1.0D-16)
      MAXIT = 40*(N+M) + 40
C
C   Cholesky factorization of C and inversion of the factor.  On exit
C   RJ is the upper triangular matrix J with J*J^T = C^(-1).
C
      CALL QLCHOL (N, NMAX, C, RJ, MODE, TOL, IFAIL)
      IF (IFAIL.NE.0) RETURN
C
C   Unconstrained minimum  x = -C^(-1) d = -J J^T d.
C
      DO 20 I = 1, N
         SUM = ZERO
         DO 10 K = 1, I
            SUM = SUM + RJ(K,I)*D(K)
   10    CONTINUE
         Z(I) = SUM
   20 CONTINUE
      DO 40 I = 1, N
         SUM = ZERO
         DO 30 K = I, N
            SUM = SUM + RJ(I,K)*Z(K)
   30    CONTINUE
         X(I) = -SUM
   40 CONTINUE
C
      NACT = 0
      ITER = 0
      DO 50 I = 1, MNN
         U(I) = ZERO
   50 CONTINUE
C
C   ------------------------------------------------------------------
C   Outer loop: determine a constraint violated at the actual iterate.
C   ------------------------------------------------------------------
C
  100 CONTINUE
      ITER = ITER + 1
      IF (ITER.GT.MAXIT) THEN
         IFAIL = 1
         GOTO 800
      ENDIF
C
C   Equality constraints are inserted first and never removed again.
C
      IP  = 0
      SGN = ONE
      DO 120 K = 1, ME
         ACTIVE = .FALSE.
         DO 110 I = 1, NACT
            IF (IABS(IACT(I)).EQ.K) ACTIVE = .TRUE.
  110    CONTINUE
         IF (.NOT.ACTIVE) THEN
            S  = QLSLK (K, M, MMAX, N, A, B, XL, XU, X)
            IP = K
            IF (S.GT.ZERO) SGN = -ONE
            SP = SGN*S
            GOTO 200
         ENDIF
  120 CONTINUE
C
C   Inequality constraints and bounds, most violated one is selected.
C
      TOLV = TOL*(ONE + ABS(X(1)))
      DO 130 I = 2, N
         TOLV = MAX(TOLV,TOL*ABS(X(I)))
  130 CONTINUE
      SMIN = -TOLV
      DO 150 K = ME+1, MNN
         IF (K.GT.M .AND. K.LE.M+N) THEN
            IF (XL(K-M).LE.-HUGEB) GOTO 150
         ENDIF
         IF (K.GT.M+N) THEN
            IF (XU(K-M-N).GE.HUGEB) GOTO 150
         ENDIF
         ACTIVE = .FALSE.
         DO 140 I = 1, NACT
            IF (IABS(IACT(I)).EQ.K) ACTIVE = .TRUE.
  140    CONTINUE
         IF (.NOT.ACTIVE) THEN
            S = QLSLK (K, M, MMAX, N, A, B, XL, XU, X)
            IF (S.LT.SMIN) THEN
               SMIN = S
               IP   = K
            ENDIF
         ENDIF
  150 CONTINUE
      IF (IP.EQ.0) GOTO 800
      SGN = ONE
      SP  = SMIN
C
C   ------------------------------------------------------------------
C   Inner loop: try to add constraint IP to the active set.
C   ------------------------------------------------------------------
C
C   UACT(NACT+1) accumulates the multiplier of the constraint IP which
C   is about to enter the active set.
C
  200 CONTINUE
      UACT(NACT+1) = ZERO
C
  220 CONTINUE
      ITER = ITER + 1
      IF (ITER.GT.MAXIT) THEN
         IFAIL = 1
         GOTO 800
      ENDIF
C
C   Search direction in the primal and in the dual space.
C
      CALL QLNORM (IP, M, MMAX, N, A, SGN, RJ, DV)
      DO 230 I = 1, N
         Z(I) = ZERO
  230 CONTINUE
      ZTN = ZERO
      DO 250 J = NACT+1, N
         ZTN = ZTN + DV(J)*DV(J)
         DO 240 I = 1, N
            Z(I) = Z(I) + RJ(I,J)*DV(J)
  240    CONTINUE
  250 CONTINUE
      IF (NACT.GT.0) THEN
         DO 270 I = NACT, 1, -1
            SUM = DV(I)
            DO 260 J = I+1, NACT
               SUM = SUM - R((J*(J-1))/2+I)*RV(J)
  260       CONTINUE
            IB = (I*(I-1))/2 + I
            IF (ABS(R(IB)).LE.ZERO) THEN
               IFAIL = 4
               GOTO 800
            ENDIF
            RV(I) = SUM/R(IB)
  270    CONTINUE
      ENDIF
C
C   Maximum step in the dual space without loosing dual feasibility.
C
      T1 = BIG
      K1 = 0
      DO 280 I = 1, NACT
         IF (IABS(IACT(I)).GT.ME .AND. RV(I).GT.TOL) THEN
            IF (UACT(I)/RV(I) .LT. T1) THEN
               T1 = UACT(I)/RV(I)
               K1 = I
            ENDIF
         ENDIF
  280 CONTINUE
C
C   Minimum step in the primal space to satisfy constraint IP.
C
      IF (ZTN.GT.TOL*TOL) THEN
         T2 = -SP/ZTN
      ELSE
         T2 = BIG
      ENDIF
      T = MIN(T1,T2)
      IF (T.GE.BIG) THEN
         IFAIL = 100 + IP
         GOTO 800
      ENDIF
C
      IF (T2.GE.BIG) THEN
C
C   Step in the dual space only, a constraint has to leave the set.
C
         DO 290 I = 1, NACT
            UACT(I) = UACT(I) - T*RV(I)
  290    CONTINUE
         UACT(NACT+1) = UACT(NACT+1) + T
         CALL QLDROP (K1, N, NACT, RJ, R, IACT, UACT)
         GOTO 220
      ENDIF
C
      DO 300 I = 1, N
         X(I) = X(I) + T*Z(I)
  300 CONTINUE
      DO 310 I = 1, NACT
         UACT(I) = UACT(I) - T*RV(I)
  310 CONTINUE
      UACT(NACT+1) = UACT(NACT+1) + T
      SP = SP + T*ZTN
C
      IF (T2.LE.T1) THEN
C
C   Full step, constraint IP becomes a member of the active set.
C
         IF (NACT.GE.N) THEN
            IFAIL = 4
            GOTO 800
         ENDIF
         S = UACT(NACT+1)
         CALL QLADDC (N, NACT, RJ, R, DV, IFAIL)
         IF (IFAIL.NE.0) GOTO 800
         IF (SGN.GT.ZERO) THEN
            IACT(NACT) =  IP
         ELSE
            IACT(NACT) = -IP
         ENDIF
         UACT(NACT) = S
         GOTO 100
      ENDIF
C
C   Partial step, the blocking constraint is dropped.
C
      CALL QLDROP (K1, N, NACT, RJ, R, IACT, UACT)
      GOTO 220
C
C   ------------------------------------------------------------------
C   Termination.
C   ------------------------------------------------------------------
C
  800 CONTINUE
      DO 810 I = 1, MNN
         U(I) = ZERO
  810 CONTINUE
      NA = NACT
      DO 820 I = 1, NA
         L = IABS(IACT(I))
         IF (IACT(I).LT.0) THEN
            U(L) = -UACT(I)
         ELSE
            U(L) =  UACT(I)
         ENDIF
  820 CONTINUE
      IF (IFAIL.EQ.0) THEN
         DO 830 I = 1, N
            IF (X(I).LT.XL(I)) X(I) = XL(I)
            IF (X(I).GT.XU(I)) X(I) = XU(I)
  830    CONTINUE
      ENDIF
      RETURN
      END
