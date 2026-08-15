C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C ======================================================================
C
      SUBROUTINE QLCHOL (N, NMAX, C, RJ, MODE, TOL, IFAIL)
C
C   Cholesky decomposition C = R^T R with an upper triangular matrix R
C   followed by the inversion of R.  On return RJ = R^(-1), so that
C   RJ*RJ^T = C^(-1).  If C is not sufficiently positive definite, a
C   multiple of the unit matrix is added.
C
      IMPLICIT NONE
      INTEGER N, NMAX, MODE, IFAIL
      DOUBLE PRECISION C(NMAX,*), RJ(N,N), TOL
      INTEGER I, J, K, ITRY
      DOUBLE PRECISION S, DIAG, ADD, TMAX
C
      IFAIL = 0
      IF (MODE.EQ.0) THEN
         DO 20 J = 1, N
            DO 10 I = 1, N
               RJ(I,J) = 0.0D0
               IF (I.LE.J) RJ(I,J) = C(I,J)
   10       CONTINUE
   20    CONTINUE
         GOTO 200
      ENDIF
C
      TMAX = 0.0D0
      DO 30 I = 1, N
         TMAX = MAX(TMAX,ABS(C(I,I)))
   30 CONTINUE
      IF (TMAX.LE.0.0D0) TMAX = 1.0D0
      ADD = 0.0D0
C
      DO 130 ITRY = 1, 30
         DO 50 J = 1, N
            DO 40 I = 1, N
               RJ(I,J) = 0.0D0
               IF (I.LT.J) RJ(I,J) = 0.5D0*(C(I,J) + C(J,I))
               IF (I.EQ.J) RJ(I,J) = C(I,I) + ADD
   40       CONTINUE
   50    CONTINUE
         DO 100 J = 1, N
            S = RJ(J,J)
            DO 60 K = 1, J-1
               S = S - RJ(K,J)*RJ(K,J)
   60       CONTINUE
            IF (S.LE.TMAX*1.0D-14) GOTO 120
            RJ(J,J) = SQRT(S)
            DIAG = RJ(J,J)
            DO 80 I = J+1, N
               S = RJ(J,I)
               DO 70 K = 1, J-1
                  S = S - RJ(K,J)*RJ(K,I)
   70          CONTINUE
               RJ(J,I) = S/DIAG
   80       CONTINUE
  100    CONTINUE
         GOTO 200
  120    CONTINUE
         IF (ADD.LE.0.0D0) THEN
            ADD = TMAX*1.0D-08
         ELSE
            ADD = ADD*1.0D+02
         ENDIF
  130 CONTINUE
      IFAIL = 4
      RETURN
C
C   Inversion of the upper triangular factor.
C
  200 CONTINUE
      DO 260 K = 1, N
         IF (ABS(RJ(K,K)).LE.0.0D0) THEN
            IFAIL = 4
            RETURN
         ENDIF
         RJ(K,K) = 1.0D0/RJ(K,K)
         S = -RJ(K,K)
         DO 210 I = 1, K-1
            RJ(I,K) = S*RJ(I,K)
  210    CONTINUE
         DO 240 J = K+1, N
            S = RJ(K,J)
            RJ(K,J) = 0.0D0
            DO 230 I = 1, K
               RJ(I,J) = RJ(I,J) + S*RJ(I,K)
  230       CONTINUE
  240    CONTINUE
  260 CONTINUE
      RETURN
      END
