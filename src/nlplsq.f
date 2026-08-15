C ======================================================================
C
C     NLPLSQ : CONSTRAINED NONLINEAR LEAST SQUARES
C
C     NLPLSQ solves the constrained nonlinear least squares
C     problem
C
C          min    sum f_i(x)^2 ,  i = 1,...,L
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     L additional variables and L additional equality
C     constraints are introduced, so that typical features of a
C     Gauss-Newton type method are retained.
C
C     The problem is transformed into a smooth nonlinear program which
C     is solved by the SQP code NLPQLP, see NLDFTW for the details of
C     the transformation.  Model functions are provided by reverse
C     communication.
C
C   PARAMETERS:
C
C      L       : Number of individual functions f_1,...,f_L.
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      LMMAX   : Dimension of G and row dimension of DG, at least
C                M + 2*L and at least one.
C      N       : Number of optimization variables.
C      LNMAX   : Dimension of X, DF, XL, XU and row dimension of C, at
C                least N + L + 1.
C      LMNN2   : Dimension of U, must be equal to M + 3*L + 2*N + 2.
C      X(LNMAX): The first N positions contain the starting values, on
C                return the final iterate.  The remaining positions are
C                used for the additional variables.
C      F       : On return the final objective function value.
C      G(LMMAX): The first M positions contain the constraint values,
C                the subsequent L positions the values of the
C                individual functions f_1(x),...,f_L(x).  The array is
C                used internally for the transformed constraints.
C      DF(LNMAX) : Internally used for the gradient of the transformed
C                objective function.
C      DG(LMMAX,LNMAX) : The first M rows contain the gradients of the
C                constraints, the subsequent L rows the gradients of
C                the individual functions.  The array is used
C                internally for the transformed Jacobian.
C      U(LMNN2): On return the multipliers of the transformed program.
C      XL(LNMAX),XU(LNMAX) : The first N positions contain the bounds
C                of the variables, the remaining ones are set
C                internally.
C      C(LNMAX,LNMAX) : Quasi-Newton matrix of the transformed program.
C      D(LNMAX): Auxiliary array.
C      ACC     : Desired final accuracy.
C      ACCQP   : Tolerance of the QP solver.
C      MAXFUN  : Upper bound for the number of function calls during
C                the line search.
C      MAXIT   : Maximum number of iterations.
C      MAXNM   : Stack size of the non-monotone line search.
C      RHO     : Restart parameter.
C      IPRINT  : Output level.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason.  IFAIL = -1 requests new values of
C                the constraints and of the individual functions,
C                IFAIL = -2 their gradients, both subject to the first
C                N positions of X.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least
C                3*LNMAX*LNMAX/2 + 34*LNMAX + 6*LMMAX + 4*L + 160.
C      KWA(LKWA) : Integer working array.
C      LKWA    : Length of KWA, at least LNMAX + 26.
C      ACT(LACT) : Logical array of active constraints.
C      LACT    : Length of ACT, at least 2*(M + 2*L) + 10.
C      QPSLVE  : External subroutine solving the QP subproblem.
C
C ======================================================================
C
      SUBROUTINE NLPLSQ (L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G,
     /                   DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN,
     /                   MAXIT, MAXNM, RHO, IPRINT, IOUT, IFAIL, WA,
     /                   LWA, KWA, LKWA, ACT, LACT, QPSLVE)
C
      IMPLICIT NONE
      INTEGER L, M, ME, LMMAX, N, LNMAX, LMNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(LMMAX,*), U(*), XL(*),
     /                 XU(*), C(LNMAX,*), D(*), WA(LWA), ACC, ACCQP,
     /                 RHO
      LOGICAL ACT(LACT)
      EXTERNAL QPSLVE
C
      INTEGER LFV, LST, LW, LWW
C
      IF (L.LT.1 .OR. M.LT.0 .OR. ME.LT.0 .OR. ME.GT.M .OR. N.LT.1)
     /   THEN
         IFAIL = 6
         RETURN
      ENDIF
      LFV = 1
      LST = LFV + 1
      LW  = LST + 6
      LWW = LWA - LW + 1
      IF (LWW.LT.1) THEN
         IFAIL = 5
         RETURN
      ENDIF
      CALL NLDFTW (1, L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G,
     /             DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN, MAXIT,
     /             MAXNM, RHO, IPRINT, IOUT, IFAIL, WA(LFV), WA(LST),
     /             WA(LW), LWW, KWA, LKWA, ACT, LACT, QPSLVE)
      RETURN
      END
