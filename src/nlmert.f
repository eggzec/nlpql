C ======================================================================
C
C     NLMERT : Augmented Lagrangian merit function of the NLPQL family,
C
C        psi_r(x,v) = f(x) - sum_{j in J} ( v_j g_j(x) - 1/2 r_j g_j(x)^2 )
C                          - 1/2 sum_{j in K} v_j^2 / r_j
C
C     with  J = {1,...,me} u {j : me < j <= m, g_j(x) <= v_j/r_j}  and
C     K = {1,...,m} \ J.  The multiplier vector is evaluated along the
C     search direction of the multipliers, i.e., v is replaced by
C     v + alpha*(u - v).
C
C ======================================================================
C
      DOUBLE PRECISION FUNCTION NLMERT (M, ME, F, G, V, U, R, ALPHA)
C
      IMPLICIT NONE
      INTEGER M, ME
      DOUBLE PRECISION F, G(*), V(*), U(*), R(*), ALPHA
      INTEGER J
      DOUBLE PRECISION PSI, VJ, ZERO, HALF
      PARAMETER (ZERO=0.0D0, HALF=0.5D0)
C
      PSI = F
      DO 10 J = 1, M
         VJ = V(J) + ALPHA*(U(J) - V(J))
         IF (R(J).LE.ZERO) GOTO 10
         IF (J.LE.ME .OR. G(J).LE.VJ/R(J)) THEN
            PSI = PSI - (VJ*G(J) - HALF*R(J)*G(J)*G(J))
         ELSE
            PSI = PSI - HALF*VJ*VJ/R(J)
         ENDIF
   10 CONTINUE
      NLMERT = PSI
      RETURN
      END
