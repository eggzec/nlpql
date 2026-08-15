C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C     QLNORM computes DV = J^T (SGN*n_k) for the constraint number K,
C     where the indices 1,...,M refer to the rows of A, M+1,...,M+N to
C     the lower and M+N+1,...,M+N+N to the upper bounds.  Note that J is
C     upper triangular only before the first constraint enters the
C     active set, afterwards it is a full matrix.
C
C ======================================================================
C
      SUBROUTINE QLNORM (K, M, MMAX, N, A, SGN, RJ, DV)
C
      IMPLICIT NONE
      INTEGER K, M, MMAX, N
      DOUBLE PRECISION A(MMAX,*), SGN, RJ(N,N), DV(N)
      INTEGER I, J
      DOUBLE PRECISION S
C
      IF (K.LE.M) THEN
         DO 20 J = 1, N
            S = 0.0D0
            DO 10 I = 1, N
               S = S + RJ(I,J)*A(K,I)
   10       CONTINUE
            DV(J) = SGN*S
   20    CONTINUE
      ELSE IF (K.LE.M+N) THEN
         I = K - M
         DO 30 J = 1, N
            DV(J) = SGN*RJ(I,J)
   30    CONTINUE
      ELSE
         I = K - M - N
         DO 40 J = 1, N
            DV(J) = -SGN*RJ(I,J)
   40    CONTINUE
      ENDIF
      RETURN
      END
