C ======================================================================
C
C     NLBFGS : Modified BFGS update of the approximation C of the
C              Hessian of the Lagrangian function,
C
C          C := C + q q^T / ( p^T q ) - C p p^T C / ( p^T C p )
C
C     with p = x_{k+1} - x_k and q = grad_x L(x_{k+1},u) - grad_x
C     L(x_k,u).  In order to guarantee that all matrices remain
C     positive definite, the modification of Powell is applied, i.e.,
C     q is replaced by theta*q + (1-theta)*C*p with
C
C          theta := 0.8 p^T C p / ( p^T C p - p^T q )
C
C     whenever p^T q < 0.2 p^T C p.
C
C ======================================================================
C
      SUBROUTINE NLBFGS (N, NMAX, C, P, Q, W, EPSM, IERR)
C
      IMPLICIT NONE
      INTEGER N, NMAX, IERR
      DOUBLE PRECISION C(NMAX,*), P(*), Q(*), W(*), EPSM
      INTEGER I, J
      DOUBLE PRECISION PTQ, PTCP, TH, S, ZERO, ONE
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
      IERR = 0
      DO 20 I = 1, N
         S = ZERO
         DO 10 J = 1, N
            S = S + C(I,J)*P(J)
   10    CONTINUE
         W(I) = S
   20 CONTINUE
      PTCP = ZERO
      PTQ  = ZERO
      DO 30 I = 1, N
         PTCP = PTCP + P(I)*W(I)
         PTQ  = PTQ  + P(I)*Q(I)
   30 CONTINUE
      IF (PTCP.LE.ZERO) THEN
         IERR = 1
         RETURN
      ENDIF
      IF (PTQ .LT. 0.2D0*PTCP) THEN
         TH = 0.8D0*PTCP/(PTCP - PTQ)
         DO 40 I = 1, N
            Q(I) = TH*Q(I) + (ONE - TH)*W(I)
   40    CONTINUE
         PTQ = 0.2D0*PTCP
      ENDIF
      IF (ABS(PTQ).LE.EPSM*MAX(ONE,ABS(PTCP))) THEN
         IERR = 1
         RETURN
      ENDIF
      DO 60 J = 1, N
         DO 50 I = 1, N
            C(I,J) = C(I,J) + Q(I)*Q(J)/PTQ - W(I)*W(J)/PTCP
   50    CONTINUE
   60 CONTINUE
      DO 80 J = 1, N
         DO 70 I = 1, J-1
            S = 0.5D0*(C(I,J) + C(J,I))
            C(I,J) = S
            C(J,I) = S
   70    CONTINUE
   80 CONTINUE
      RETURN
      END
