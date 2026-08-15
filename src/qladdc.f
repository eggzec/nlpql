C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C ======================================================================
C
      SUBROUTINE QLADDC (N, NACT, RJ, R, DV, IFAIL)
C
C   Insert a new constraint into the active set.  DV = J^T n is reduced
C   to upper triangular form by Givens rotations which are applied to
C   the columns of J at the same time.  The first NACT+1 components of
C   the rotated vector form the new column of R.
C
      IMPLICIT NONE
      INTEGER N, NACT, IFAIL
      DOUBLE PRECISION RJ(N,N), R((N*(N+1))/2), DV(N)
      INTEGER I, L, IB
      DOUBLE PRECISION CS, SN, H, T1, T2
C
      IFAIL = 0
      DO 20 I = N, NACT+2, -1
         H = SQRT(DV(I-1)*DV(I-1) + DV(I)*DV(I))
         IF (H.LE.0.0D0) GOTO 20
         CS = DV(I-1)/H
         SN = DV(I)/H
         DV(I-1) = H
         DV(I)   = 0.0D0
         DO 10 L = 1, N
            T1 = RJ(L,I-1)
            T2 = RJ(L,I)
            RJ(L,I-1) =  CS*T1 + SN*T2
            RJ(L,I)   = -SN*T1 + CS*T2
   10    CONTINUE
   20 CONTINUE
C
      IF (DV(NACT+1).LT.0.0D0) THEN
         DV(NACT+1) = -DV(NACT+1)
         DO 30 L = 1, N
            RJ(L,NACT+1) = -RJ(L,NACT+1)
   30    CONTINUE
      ENDIF
      IF (DV(NACT+1).LE.0.0D0) THEN
         IFAIL = 4
         RETURN
      ENDIF
C
      NACT = NACT + 1
      IB   = (NACT*(NACT-1))/2
      DO 40 I = 1, NACT
         R(IB+I) = DV(I)
   40 CONTINUE
      RETURN
      END
