C ======================================================================
C
C     NLPQLY : AN EASY-TO-USE VERSION OF THE SQP CODE NLPQLP
C
C     NLPQLY simplifies the numerical solution of the nonlinear
C     programming problem
C
C          min    f(x)
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     by calling the standard SQP code NLPQLP, where the calling
C     sequence is simplified as much as possible.  A user has to
C     provide objective and constraint function values in the same code
C     which calls NLPQLY (reverse communication).  All derivatives are
C     internally approximated by forward differences
C
C          d f / d x_i  =  ( f(x + eta_i e_i) - f(x) ) / eta_i
C
C     with eta_i = eta * max(1.0D-5, |x_i|) and eta the square root of
C     the machine precision.  Most tolerances are set to reasonable
C     default values, i.e., ACCQP = 0, STPMIN = 0, MAXFUN = 20,
C     MAXNM = 10, RHO = 100 and MODE = 0.
C
C   PARAMETERS:
C
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      N       : Number of optimization variables.
C      X(N)    : Starting values, on return the current iterate.  The
C                objective and the constraint function values have to
C                be evaluated at X whenever NLPQLY returns a negative
C                value of IFAIL.
C      F       : Objective function value at X.
C      G(M)    : Constraint function values at X.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      ACC     : Desired final accuracy, should not be smaller than
C                1.0D-7 since forward differences are applied.
C      MAXIT   : Maximum number of iterations.
C      IPRINT  : Output level, 0 <= IPRINT <= 4.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason.  Must be set to zero when calling
C                NLPQLY the first time.  A negative value requests new
C                function values, IFAIL = 0 indicates a successful
C                return and IFAIL > 0 an error, see NLPQLP.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least
C                3*N*N + M*N + 45*N + 14*M + 200.
C      KWA(LKWA) : Integer working array.
C      LKWA    : Length of KWA, at least N + 27.
C      ACT(LACT) : Logical array indicating active constraints.
C      LACT    : Length of ACT, at least 2*M + 10.
C
C ======================================================================
C
      SUBROUTINE NLPQLY (M, ME, N, X, F, G, XL, XU, ACC, MAXIT,
     /                   IPRINT, IOUT, IFAIL, WA, LWA, KWA, LKWA,
     /                   ACT, LACT)
C
      IMPLICIT NONE
      INTEGER M, ME, N, MAXIT, IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), XL(*), XU(*), WA(LWA), ACC
      LOGICAL ACT(LACT)
C
      INTEGER NQ, MG, MNN2, LXW, LDF, LDG, LU, LC, LD, LG, LXB, LFB,
     /        LGB, LST, LW, LWW, LNEED
C
      NQ   = N + 1
      MG   = MAX(M,1)
      MNN2 = M + N + N + 2
      LXW  = 1
      LDF  = LXW + NQ
      LDG  = LDF + NQ
      LU   = LDG + MG*NQ
      LC   = LU  + MNN2
      LD   = LC  + NQ*NQ
      LG   = LD  + NQ
      LXB  = LG  + MG
      LFB  = LXB + N
      LGB  = LFB + 1
      LST  = LGB + MG
      LW   = LST + 6
      LWW  = LWA - LW + 1
      LNEED = LW + 23*N + 4*M + 3*MG + (N+M+1) + 150
     /            + (3*NQ*NQ)/2 + 10*NQ + 2*MG + 2
      IF (LWA.LT.LNEED .OR. LKWA.LT.N+27 .OR. LACT.LT.2*M+10) THEN
         IFAIL = 5
         RETURN
      ENDIF
C
      CALL NLPQLYW (M, ME, N, NQ, MG, MNN2, X, F, G, XL, XU, ACC,
     /              MAXIT, IPRINT, IOUT, IFAIL, WA(LXW), WA(LDF),
     /              WA(LDG), WA(LU), WA(LC), WA(LD), WA(LG), WA(LXB),
     /              WA(LFB), WA(LGB), WA(LST), WA(LW), LWW, KWA, LKWA,
     /              ACT, LACT)
      RETURN
      END
