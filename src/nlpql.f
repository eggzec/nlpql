C ======================================================================
C
C     NLPQL : THE ORIGINAL SEQUENTIAL QUADRATIC PROGRAMMING CODE
C
C     NLPQL solves the general nonlinear programming problem
C
C          min    f(x)
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     by the sequential quadratic programming method described in
C
C        Schittkowski K. (1985/86): NLPQL: A Fortran subroutine solving
C        constrained nonlinear programming problems, Annals of
C        Operations Research, Vol. 5, 485-500
C
C     The routine is the serial predecessor of NLPQLP and is obtained
C     from it by fixing the number of parallel function evaluations to
C     one, by performing a monotone line search and by suppressing the
C     internal restarts, i.e., by NP = 1, MAXNM = 0 and RHO = 0.
C     Model functions and gradients are provided by reverse
C     communication exactly as for NLPQLP.
C
C   PARAMETERS:
C
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      MMAX    : Row dimension of G and DG, MMAX >= max(1,M).
C      N       : Number of optimization variables.
C      NMAX    : Row dimension of C, NMAX > N.
C      MNN     : Must be equal to M + N + N.
C      X(NMAX) : Starting value, on return the final iterate.
C      F       : Objective function value at X.
C      G(MMAX) : Constraint function values at X.
C      DF(NMAX): Gradient of the objective function.
C      DG(MMAX,NMAX) : Jacobian of the constraints, column N+1 is used
C                internally as scratch space.
C      U(MNN)  : Multipliers, first M for the constraints, then N for
C                the lower and N for the upper bounds.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      C(NMAX,NMAX) : Quasi-Newton matrix.
C      D(NMAX) : Diagonal of the LDL decomposition of C.
C      ACC     : Desired final accuracy.
C      MAXFUN  : Upper bound for the number of function calls during the
C                line search.
C      MAXIT   : Maximum number of outer iterations.
C      IPRINT  : Output level.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason, see NLPQLP.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least
C                3*NMAX*NMAX/2 + 34*NMAX + 11*MMAX + 4*M + 160.
C      KWA(LKWA) : Integer working array.
C      LKWA    : Length of KWA, at least NMAX + 26.
C      ACTIVE(LACT) : Logical array of active constraints.
C      LACT    : Length of ACTIVE, at least 2*M+10.
C
C ======================================================================
C
      SUBROUTINE NLPQL (M, ME, MMAX, N, NMAX, MNN, X, F, G, DF, DG,
     /                  U, XL, XU, C, D, ACC, MAXFUN, MAXIT, IPRINT,
     /                  IOUT, IFAIL, WA, LWA, KWA, LKWA, ACTIVE, LACT)
C
      IMPLICIT NONE
      INTEGER M, ME, MMAX, N, NMAX, MNN, MAXFUN, MAXIT, IPRINT, IOUT,
     /        IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(*), ACC
      LOGICAL ACTIVE(LACT)
C
      INTEGER I, MNN2, LU, LW, LWW, MODE
      DOUBLE PRECISION FV(1), ZERO
      PARAMETER (ZERO=0.0D0)
      EXTERNAL QL
C
      IF (MNN.NE.M+N+N) THEN
         IFAIL = 6
         RETURN
      ENDIF
      MNN2 = M + N + N + 2
      LU   = 1
      LW   = LU + MNN2
      LWW  = LWA - LW + 1
      IF (LWW.LT.1) THEN
         IFAIL = 5
         RETURN
      ENDIF
C
C   The original code does not provide the two additional multipliers
C   belonging to the variable which prevents inconsistent linearized
C   constraints, so that a separate multiplier vector is used.
C
      IF (IFAIL.EQ.0) THEN
         DO 10 I = 1, MNN2
            WA(LU+I-1) = ZERO
   10    CONTINUE
      ENDIF
      MODE  = 0
      FV(1) = F
      CALL NLPQLP (1, M, ME, MMAX, N, NMAX, MNN2, X, FV, G, DF, DG,
     /             WA(LU), XL, XU, C, D, ACC, ZERO, ZERO, MAXFUN,
     /             MAXIT, 0, ZERO, IPRINT, MODE, IOUT, IFAIL,
     /             WA(LW), LWW, KWA, LKWA, ACTIVE, LACT, .TRUE., QL)
      F = FV(1)
      DO 20 I = 1, M
         U(I) = WA(LU+I-1)
   20 CONTINUE
      DO 30 I = 1, N
         U(M+I)   = WA(LU+M+I-1)
         U(M+N+I) = WA(LU+M+N+I)
   30 CONTINUE
      RETURN
      END
