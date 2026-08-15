C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C ======================================================================
C
      SUBROUTINE QLDROP (L, N, NACT, RJ, R, IACT, UACT)
C
C   Remove the constraint stored at position L of the active set and
C   restore the upper triangular form of R by Givens rotations.
C
      IMPLICIT NONE
      INTEGER L, N, NACT
      INTEGER IACT(N)
      DOUBLE PRECISION RJ(N,N), R((N*(N+1))/2), UACT(N)
      INTEGER I, J, K, IBS, IBD, IBK
      DOUBLE PRECISION CS, SN, H, T1, T2, A1, A2
C
      IF (L.LT.1 .OR. L.GT.NACT) RETURN
      DO 50 J = L, NACT-1
         IBS = ((J+1)*J)/2
         A1  = R(IBS+J)
         A2  = R(IBS+J+1)
         H   = SQRT(A1*A1 + A2*A2)
         IF (H.GT.0.0D0) THEN
            CS = A1/H
            SN = A2/H
         ELSE
            CS = 1.0D0
            SN = 0.0D0
         ENDIF
         R(IBS+J)   = H
         R(IBS+J+1) = 0.0D0
         DO 20 K = J+2, NACT
            IBK = (K*(K-1))/2
            T1  = R(IBK+J)
            T2  = R(IBK+J+1)
            R(IBK+J)   =  CS*T1 + SN*T2
            R(IBK+J+1) = -SN*T1 + CS*T2
   20    CONTINUE
         DO 30 I = 1, N
            T1 = RJ(I,J)
            T2 = RJ(I,J+1)
            RJ(I,J)   =  CS*T1 + SN*T2
            RJ(I,J+1) = -SN*T1 + CS*T2
   30    CONTINUE
         IBD = (J*(J-1))/2
         DO 40 I = 1, J
            R(IBD+I) = R(IBS+I)
   40    CONTINUE
         IACT(J) = IACT(J+1)
         UACT(J) = UACT(J+1)
   50 CONTINUE
      UACT(NACT) = UACT(NACT+1)
      NACT = NACT - 1
      RETURN
      END
