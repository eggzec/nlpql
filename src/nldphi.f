C ======================================================================
C
C     NLDPHI : Directional derivative of the augmented Lagrangian merit
C              function at alpha = 0,
C
C        phi'(0) = grad psi_r(x,v)^T ( d , u - v )^T
C
C     with
C
C        d psi / d x_i = grad f_i - sum_{j in J} ( v_j - r_j g_j ) grad g_ji
C        d psi / d v_j = -g_j          for j in J
C        d psi / d v_j = -v_j / r_j    for j in K
C
C ======================================================================
C
      SUBROUTINE NLDPHI (M, ME, MMAX, N, DF, DG, G, U, V, R, DD, DPHI)
C
      IMPLICIT NONE
      INTEGER M, ME, MMAX, N
      DOUBLE PRECISION DF(*), DG(MMAX,*), G(MMAX,*), U(*), V(*), R(*),
     /                 DD(*), DPHI
      INTEGER I, J
      DOUBLE PRECISION S, AJ, ZERO
      PARAMETER (ZERO=0.0D0)
C
      DPHI = ZERO
      DO 10 I = 1, N
         DPHI = DPHI + DF(I)*DD(I)
   10 CONTINUE
      DO 40 J = 1, M
         IF (R(J).LE.ZERO) GOTO 40
         AJ = ZERO
         DO 20 I = 1, N
            AJ = AJ + DG(J,I)*DD(I)
   20    CONTINUE
         IF (J.LE.ME .OR. G(J,1).LE.V(J)/R(J)) THEN
            S = V(J) - R(J)*G(J,1)
            DPHI = DPHI - S*AJ - G(J,1)*(U(J)-V(J))
         ELSE
            DPHI = DPHI - (V(J)/R(J))*(U(J)-V(J))
         ENDIF
   40 CONTINUE
      RETURN
      END
