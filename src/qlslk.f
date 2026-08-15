C ======================================================================
C
C     Part of the NLPQL package.  Auxiliary routine of QL.
C
C ======================================================================
C
      DOUBLE PRECISION FUNCTION QLSLK (K, M, MMAX, N, A, B, XL, XU, X)
C
C   Residual  n_k^T x - r_k  of constraint number K, where the indices
C   1,...,M refer to the rows of A, M+1,...,M+N to the lower bounds and
C   M+N+1,...,M+N+N to the upper bounds.
C
      IMPLICIT NONE
      INTEGER K, M, MMAX, N
      DOUBLE PRECISION A(MMAX,*), B(*), XL(*), XU(*), X(*)
      INTEGER I
      DOUBLE PRECISION S
C
      IF (K.LE.M) THEN
         S = B(K)
         DO 10 I = 1, N
            S = S + A(K,I)*X(I)
   10    CONTINUE
      ELSE IF (K.LE.M+N) THEN
         S = X(K-M) - XL(K-M)
      ELSE
         S = XU(K-M-N) - X(K-M-N)
      ENDIF
      QLSLK = S
      RETURN
      END
