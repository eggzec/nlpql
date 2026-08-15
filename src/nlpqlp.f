C ======================================================================
C
C     NLPQLP : A SEQUENTIAL QUADRATIC PROGRAMMING ALGORITHM WITH
C              DISTRIBUTED AND NON-MONOTONE LINE SEARCH
C
C     The subroutine solves the smooth nonlinear programming problem
C
C          min    f(x)
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     by a sequential quadratic programming (SQP) method.  Proceeding
C     from a quadratic approximation of the Lagrangian function
C
C          L(x,u) := f(x) - sum_{j=1}^m u_j g_j(x)
C
C     and a linearization of the constraints, the quadratic programming
C     subproblem
C
C          min  1/2 d^T C_k d + grad f(x_k)^T d + 1/2 sigma delta^2
C          d in R^n, delta in R :
C               grad g_j(x_k)^T d + (1-delta) g_j(x_k)  = 0 , j<=me
C               grad g_j(x_k)^T d + (1-delta) g_j(x_k) >= 0 , j> me
C               xl - x_k <= d <= xu - x_k ,  0 <= delta <= 1
C
C     is formulated and solved by the QP code passed in QPSLVE.  The
C     additional variable delta prevents inconsistent linearizations and
C     is fixed at zero as long as the linearized constraints possess a
C     feasible solution.  A new iterate is obtained by
C
C          x_{k+1} := x_k + alpha_k d_k ,
C          v_{k+1} := v_k + alpha_k (u_k - v_k)
C
C     where the steplength alpha_k is computed by a line search with
C     respect to the augmented Lagrangian merit function
C
C          psi_r(x,v) := f(x) - sum_{j in J} (v_j g_j(x)
C                             - 1/2 r_j g_j(x)^2)
C                             - 1/2 sum_{j in K} v_j^2/r_j
C
C     with J := {1,...,me} u {j : me < j <= m, g_j(x) <= v_j/r_j} and
C     K := {1,...,m} \ J.  Depending on the number NP of nodes of a
C     distributed system, the objective and constraint functions can be
C     evaluated simultaneously at NP predetermined test points along the
C     search direction.  In error situations, where the line search
C     cannot be terminated within MAXFUN steps, a non-monotone line
C     search is performed, i.e., the reference value psi_r(x_k,v_k) of
C     the sufficient descent test is replaced by
C
C          max { psi_{r_j}(x_j,v_j) : j = k-p(k),...,k } ,
C
C     p(k) = min(k,MAXNM).  The Hessian approximation C_k is updated by
C     the modified BFGS formula of Powell, and various restart options
C     with a scaled identity matrix are implemented.
C
C     Model functions and gradients must be provided by the calling
C     program through reverse communication.
C
C     References:
C
C        Schittkowski K. (1985/86): NLPQL: A Fortran subroutine solving
C        constrained nonlinear programming problems, Annals of
C        Operations Research, Vol. 5, 485-500
C
C        Dai Y.H., Schittkowski K. (2008): A sequential quadratic
C        programming algorithm with non-monotone line search, Pacific
C        Journal of Optimization, Vol. 4, 335-351
C
C     The code was written from scratch from the published user's guides
C     and papers of the NLPQL family of codes.  The names of the routine
C     and of its arguments follow that documentation.
C
C ======================================================================
C
      SUBROUTINE NLPQLP (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G,
     /                   DF, DG, U, XL, XU, C, D, ACC, ACCQP, STPMIN,
     /                   MAXFUN, MAXIT, MAXNM, RHO, IPRINT, MODE, IOUT,
     /                   IFAIL, WA, LWA, KWA, LKWA, ACT, LACT, LQL,
     /                   QPSLVE)
C
C   PARAMETERS:
C
C      NP      : Number of parallel function evaluations provided by
C                the calling program, 1 <= NP <= 50.
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      MMAX    : Row dimension of G and DG, MMAX >= max(1,M).
C      N       : Number of optimization variables.
C      NMAX    : Row dimension of C, NMAX > N and NMAX >= 2.
C      MNN2    : Must be equal to M + N + N + 2.
C      X(NMAX,NP) : Starting values in the first column, on return the
C                current iterate.  Internally used to store NP
C                different arguments for which function values have to
C                be provided by the calling program.
C      F(NP)   : Objective function values, F(1) on return.
C      G(MMAX,NP) : Constraint function values, first column on return.
C      DF(NMAX): Gradient of the objective function at X(.,1).
C      DG(MMAX,NMAX) : Jacobian of the constraints at X(.,1), the J-th
C                row contains the gradient of the J-th constraint.
C                Column N+1 is used internally as scratch space.
C      U(MNN2) : Multipliers, first M for the constraints, then N+1 for
C                the lower and N+1 for the upper bounds.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      C(NMAX,NMAX) : Approximation of the Hessian of the Lagrangian.
C      D(NMAX) : Diagonal of the LDL decomposition of C if LQL is false.
C      ACC     : Desired final accuracy.
C      ACCQP   : Tolerance for the QP solver.
C      STPMIN  : Minimum steplength in case of NP > 1.
C      MAXFUN  : Upper bound for the number of function calls during the
C                line search, MAXFUN <= 50.
C      MAXIT   : Maximum number of outer iterations.
C      MAXNM   : Stack size for the non-monotone line search, <= 50.
C                MAXNM = 0 gives a monotone line search.
C      RHO     : Restart parameter, the BFGS matrix is set to RHO*I in
C                case of an uphill search direction.
C      IPRINT  : Output level, 0 <= IPRINT <= 4.
C      MODE    : 0 - normal execution
C                1 - initial multipliers in U and Hessian in C, D
C                2 - initial scaling after the first step
C                3 - scaled resets if the scaling parameter is less
C                    than the square root of ACC
C               >3 - initial and repeated resets every MODE steps
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason, see the user's guide.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least
C                23*N + 4*M + 3*MMAX + NP*(N+M+1) + 150 plus the
C                memory required by the QP solver.
C      KWA(LKWA) : Integer working array, first 5 positions contain the
C                number of function evaluations, gradient evaluations,
C                iterations, QP solutions and a flag for a better
C                feasible but non-stationary iterate.
C      LKWA    : Length of KWA, at least 25 plus the memory required by
C                the QP solver.
C      ACT(LACT) : Logical array indicating active constraints.
C      LACT    : Length of ACT, at least 2*M+10.
C      LQL     : If true, the QP subproblem is solved proceeding from a
C                full positive definite quasi-Newton matrix, otherwise
C                C and D return an LDL decomposition of it.
C      QPSLVE  : External subroutine solving the QP subproblem.
C
      IMPLICIT NONE
      INTEGER NP, M, ME, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, MODE, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), DF(*), DG(MMAX,*),
     /                 U(*), XL(*), XU(*), C(NMAX,*), D(*), WA(LWA),
     /                 ACC, ACCQP, STPMIN, RHO
      LOGICAL ACT(LACT), LQL
      EXTERNAL QPSLVE
C
      INTEGER LBEST, LXOLD, LQK, LGOLD, LDD, LV, LR, LNM, LALP, LPHI,
     /        LQD, LQXL, LQXU, LCP, LQV, LSC, LQLWA, NQLWA, LNEED,
     /        MM, MG
C
C   Check of the dimensions and partition of the working arrays.
C
      MM = MAX(M,1)
      MG = MAX(MMAX,1)
      IF (M.GT.MMAX .OR. N.GE.NMAX .OR. MNN2.NE.M+N+N+2) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
      IF (NP.LT.1 .OR. NP.GT.50 .OR. IPRINT.LT.0 .OR. IPRINT.GT.4
     /    .OR. MODE.LT.0 .OR. IOUT.LT.1 .OR. MAXNM.LT.0
     /    .OR. MAXNM.GT.50 .OR. MAXFUN.LT.1 .OR. MAXFUN.GT.50
     /    .OR. MAXIT.LT.1 .OR. ME.LT.0 .OR. ME.GT.M .OR. N.LT.1) THEN
         IFAIL = 9
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
C
      LBEST = 1
      LXOLD = LBEST + N + 1 + MM
      LQK   = LXOLD + N
      LGOLD = LQK   + N
      LDD   = LGOLD + MM
      LV    = LDD   + N + 1
      LR    = LV    + MNN2
      LNM   = LR    + MM
      LALP  = LNM   + 50
      LPHI  = LALP  + NP
      LQD   = LPHI  + NP
      LQXL  = LQD   + N + 1
      LQXU  = LQXL  + N + 1
      LCP   = LQXU  + N + 1
      LQV   = LCP   + N
      LSC   = LQV   + N
      LQLWA = LSC   + 20
      NQLWA = (3*NMAX*NMAX)/2 + 10*NMAX + 2*MG + 2
      LNEED = LQLWA + NQLWA - 1
      IF (LWA.LT.LNEED .OR. LKWA.LT.25+N+1 .OR. LACT.LT.2*M+10) THEN
         IFAIL = 5
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
C
      CALL NLPQLW (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, STPMIN, MAXFUN, MAXIT,
     /             MAXNM, RHO, IPRINT, MODE, IOUT, IFAIL,
     /             WA(LBEST), WA(LXOLD), WA(LQK), WA(LGOLD), WA(LDD),
     /             WA(LV), WA(LR), WA(LNM), WA(LALP), WA(LPHI),
     /             WA(LQD), WA(LQXL), WA(LQXU), WA(LCP), WA(LQV),
     /             WA(LSC), WA(LQLWA), NQLWA, KWA, KWA(26), LKWA-25,
     /             ACT, LACT, LQL, QPSLVE)
      RETURN
 1000 FORMAT (' *** ERROR IN NLPQLP: IFAIL = ',I5)
      END
