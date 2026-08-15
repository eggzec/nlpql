C ======================================================================
C
C     NLPQLB : AN SQP ALGORITHM WITH ACTIVE SET STRATEGY FOR PROBLEMS
C              WITH A VERY LARGE NUMBER OF CONSTRAINTS
C
C     NLPQLB solves the nonlinear programming problem
C
C          min    f(x)
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     where m is very large compared to n and where the Jacobian of the
C     constraints does not possess any sparsity pattern that could be
C     exploited numerically.  The algorithm proceeds from a user
C     provided bound mw for the maximum number of expected active
C     constraints, n <= mw <= m, and generates quadratic programming
C     subproblems with mw linear constraints only, the so-called
C     working set
C
C          W = J* u Kbar*  ,  |W| = mw ,
C
C     where the index set of active constraints
C
C          J* = {1,...,me} u {j : me < j <= m, g_j(x) < eps or u_j > 0}
C
C     is always contained in W.  Only for the constraints of the working
C     set new gradient values must be computed.  Since all constraints
C     outside the working set are inactive, i.e., satisfy g_j(x) > eps,
C     the convergence conditions of the reduced problem are applicable
C     for the original one as well.
C
C     One iteration of the reduced problem is performed by the SQP code
C     NLPQLP.  The working set is checked at each new iterate.  As long
C     as no constraint outside the working set becomes active, the
C     working set remains unchanged and NLPQLP proceeds without any
C     interruption.  Otherwise the working set is rearranged and the SQP
C     iteration is continued from the actual iterate, retaining the
C     quasi-Newton matrix that has been accumulated so far.
C
C     Reference:
C
C        Schittkowski K. (2010): NLPQLB: A Fortran implementation of an
C        SQP algorithm with active set strategy for solving optimization
C        problems with a very large number of nonlinear constraints,
C        User's guide, University of Bayreuth
C
C   PARAMETERS:
C
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      MW      : Size of the working set, N <= MW <= M.
C      MWMAX   : Row dimension of DG, MWMAX >= MW.
C      N       : Number of optimization variables.
C      NMAX    : Row dimension of C, NMAX > N.
C      MNN2    : Must be equal to MW + N + N + 2.
C      X(NMAX) : Starting values, on return the final iterate.
C      F       : Objective function value at X.
C      G(M)    : Values of all constraints at X.
C      DF(NMAX): Gradient of the objective function.
C      DG(MWMAX,NMAX) : Gradients of the constraints of the working set,
C                the J-th row belongs to the constraint KWA(J).
C      U(MNN2) : Multipliers of the working set and of the bounds.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      C(NMAX,NMAX) : Quasi-Newton matrix.
C      D(NMAX) : Auxiliary array.
C      ACC     : Desired final accuracy.
C      ACCQP   : Tolerance of the QP solver.
C      MAXFUN  : Upper bound for the number of function calls during the
C                line search.
C      MAXIT   : Maximum number of iterations.
C      MAXNM   : Stack size of the non-monotone line search.
C      RHOB    : Restart parameter, non-negative.
C      IPRINT  : Output level.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason.  IFAIL = 11 indicates that there
C                are too many active constraints, i.e., MW has to be
C                increased.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least 24*N + 9*MWMAX + 160 plus the
C                memory needed by the QP solver.
C      KWA(LKWA) : Integer working array.  The first MW positions
C                contain the working set.
C      LKWA    : Length of KWA, at least 2*MW + MAX(N+1,MW/NMAX) + 26.
C      ACT(LACT) : Logical array, ACT(KWA(J)) indicates whether a new
C                gradient of the KWA(J)-th constraint is needed.
C      LACT    : Length of ACT, at least 2*M + 2*MW + 11.
C      QPSLVE  : External subroutine solving the QP subproblem.
C
C ======================================================================
C
      SUBROUTINE NLPQLB (M, ME, MW, MWMAX, N, NMAX, MNN2, X, F, G,
     /                   DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN,
     /                   MAXIT, MAXNM, RHOB, IPRINT, IOUT, IFAIL,
     /                   WA, LWA, KWA, LKWA, ACT, LACT, QPSLVE)
C
      IMPLICIT NONE
      INTEGER M, ME, MW, MWMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MWMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(LWA), ACC, ACCQP,
     /                 RHOB
      LOGICAL ACT(LACT)
      EXTERNAL QPSLVE
C
      INTEGER MWG, LGW, LFV, LST, LW, LWW, LNEED, LKP, LKW, LAP, LAW
C
      IF (M.LT.1 .OR. ME.LT.0 .OR. ME.GT.M .OR. MW.LT.1 .OR. MW.GT.M
     /    .OR. MWMAX.LT.MW .OR. N.GE.NMAX .OR. N.LT.1
     /    .OR. MNN2.NE.MW+N+N+2) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
      IF (M.GT.N .AND. MW.LE.N) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
C
      MWG = MAX(MWMAX,1)
      LGW = 1
      LFV = LGW + MWG
      LST = LFV + 1
      LW  = LST + 6
      LWW = LWA - LW + 1
      LNEED = LW + 23*N + 4*MW + 3*MWG + (N+MW+1) + 150
     /           + (3*NMAX*NMAX)/2 + 10*NMAX + 2*MWG + 2
C
C   The first MW positions of KWA contain the working set, the integer
C   working array of the SQP code starts behind it.
C
      LKP = 2*MW + 1
      LKW = LKWA - LKP + 1
      LAP = M + 1
      LAW = LACT - M
      IF (LWA.LT.LNEED .OR. LKW.LT.26+N+1 .OR. LAW.LT.2*MW+10) THEN
         IFAIL = 5
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
C
      CALL NLPQLBW (M, ME, MW, MWMAX, MWG, N, NMAX, MNN2, X, F, G,
     /              DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN,
     /              MAXIT, MAXNM, RHOB, IPRINT, IOUT, IFAIL,
     /              WA(LGW), WA(LFV), WA(LST), WA(LW), LWW,
     /              KWA, KWA(LKP), LKW, ACT, ACT(LAP), LAW, QPSLVE)
      RETURN
 1000 FORMAT (' *** ERROR IN NLPQLB: IFAIL = ',I5)
      END
