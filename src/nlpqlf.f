C ======================================================================
C
C     NLPQLF : A FEASIBLE SQP METHOD FOR PROBLEMS WHERE THE MODEL
C              FUNCTIONS CAN ONLY BE EVALUATED ON A CONVEX SUBSET
C
C     NLPQLF solves the nonlinear programming problem
C
C          min    f(x)
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,mef
C                      g_j(x) >= 0 ,  j = mef+1,...,mf
C                      e_i(x) >= 0 ,  i = 1,...,m
C                      xl <= x <= xu
C
C     where the objective function f and the constraints g_1, ..., g_mf
C     can be evaluated only for argument values within the convex set
C
C          F := { x in R^n : e_i(x) >= 0 , i = 1,...,m }
C
C     described by the so-called feasibility constraints e_1, ..., e_m,
C     which are assumed to be concave and continuously differentiable.
C     It is implicitly assumed that the evaluation of the feasibility
C     constraints is much less expensive than an evaluation of f or of
C     the remaining constraints.
C
C     The starting point must satisfy the feasibility constraints.  The
C     iteration is driven by the SQP code NLPQLP, which takes all
C     constraints into account.  Before the objective function and the
C     remaining constraints are requested at a test point of the line
C     search, only the feasibility constraints are evaluated.  If one of
C     them is violated, the steplength is reduced by the factor one half
C     and a new test point is generated, so that f and g are never
C     evaluated outside of F.  Since F is convex and since the actual
C     iterate belongs to F, the reduction terminates.
C
C     Note that the two-stage search arc of Schittkowski (2013), where
C     the search direction is obtained from a quadratic program extended
C     by the nonlinear feasibility constraints, is not implemented.  The
C     feasibility of all arguments at which f and g are evaluated is
C     guaranteed by the steplength reduction described above.
C
C   PARAMETERS:
C
C      MF      : Number of constraints g_1,...,g_mf.
C      MEF     : Number of equality constraints among them.
C      M       : Number of feasibility constraints e_1,...,e_m.
C      MMAX    : Dimension of G and row dimension of DG, at least
C                MF + M and at least one.
C      N       : Number of optimization variables.
C      NMAX    : Row dimension of C, NMAX > N.
C      MNN2    : Must be equal to MF + M + N + N + 2.
C      X(NMAX) : Starting values, on return the final iterate.  The
C                starting point must satisfy the feasibility
C                constraints.
C      F       : Objective function value at X.
C      G(MMAX) : The first MF positions contain the values of the
C                constraints g_j, the subsequent M positions the values
C                of the feasibility constraints e_i.
C      DF(NMAX): Gradient of the objective function.
C      DG(MMAX,NMAX) : The first MF rows contain the gradients of the
C                constraints g_j, the subsequent M rows the gradients
C                of the feasibility constraints e_i.
C      U(MNN2) : Multipliers.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      C(NMAX,NMAX) : Quasi-Newton matrix.
C      D(NMAX) : Auxiliary array.
C      ACC     : Desired final accuracy.
C      ACCF    : Tolerance by which the feasibility constraints must be
C                satisfied, a small non-negative number.
C      ACCQP   : Tolerance of the QP solver.
C      MAXFUN  : Upper bound for the number of function calls during
C                the line search.
C      MAXIT   : Maximum number of iterations.
C      MAXNM   : Stack size of the non-monotone line search.
C      RHO     : Restart parameter.
C      IPRINT  : Output level.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason.  In addition to the flags of
C                NLPQLP the following requests are used:
C                -1 - compute F and the values of the constraints
C                     g_1,...,g_mf, to be stored in G(1),...,G(MF).
C                     The actual iterate satisfies all feasibility
C                     constraints.
C                -2 - compute DF and the gradients of all constraints,
C                     to be stored in the first MF + M rows of DG.
C                -3 - compute only the values of the feasibility
C                     constraints e_1,...,e_m, to be stored in
C                     G(MF+1),...,G(MF+M).  The actual iterate may
C                     violate them.
C                 8 - the starting point is not feasible.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least 24*N + 5*(MF+M) + 4*MMAX + 160
C                plus the memory needed by the QP solver.
C      KWA(LKWA) : Integer working array.
C      LKWA    : Length of KWA, at least NMAX + 26.
C      ACT(LACT) : Logical array of active constraints.
C      LACT    : Length of ACT, at least 2*(MF+M) + 10.
C      QPSLVE  : External subroutine solving the QP subproblem.
C
C ======================================================================
C
      SUBROUTINE NLPQLF (MF, MEF, M, MMAX, N, NMAX, MNN2, X, F, G, DF,
     /                   DG, U, XL, XU, C, D, ACC, ACCF, ACCQP, MAXFUN,
     /                   MAXIT, MAXNM, RHO, IPRINT, IOUT, IFAIL, WA,
     /                   LWA, KWA, LKWA, ACT, LACT, QPSLVE)
C
      IMPLICIT NONE
      INTEGER MF, MEF, M, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(LWA), ACC, ACCF,
     /                 ACCQP, RHO
      LOGICAL ACT(LACT)
      EXTERNAL QPSLVE
C
      INTEGER LFV, LST, LW, LWW
C
      IF (MF.LT.0 .OR. MEF.LT.0 .OR. MEF.GT.MF .OR. M.LT.0
     /    .OR. N.LT.1 .OR. N.GE.NMAX .OR. MMAX.LT.MAX(MF+M,1)
     /    .OR. MNN2.NE.MF+M+N+N+2) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
      LFV = 1
      LST = LFV + 1
      LW  = LST + 6
      LWW = LWA - LW + 1
      IF (LWW.LT.1) THEN
         IFAIL = 5
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
      CALL NLPQLFW (MF, MEF, M, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /              U, XL, XU, C, D, ACC, ACCF, ACCQP, MAXFUN, MAXIT,
     /              MAXNM, RHO, IPRINT, IOUT, IFAIL, WA(LFV), WA(LST),
     /              WA(LW), LWW, KWA, LKWA, ACT, LACT, QPSLVE)
      RETURN
 1000 FORMAT (' *** ERROR IN NLPQLF: IFAIL = ',I5)
      END
