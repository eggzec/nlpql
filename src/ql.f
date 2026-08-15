C ======================================================================
C
C     QL : A FORTRAN CODE FOR CONVEX QUADRATIC PROGRAMMING
C
C     An implementation of the primal-dual method of Goldfarb and Idnani
C     (1983) for the strictly convex quadratic program
C
C          min    1/2 x^T C x + d^T x
C          x in R^n :  a_j^T x + b_j  = 0 ,  j = 1,...,me
C                      a_j^T x + b_j >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     Lower and upper bounds are handled separately, i.e., they are
C     never stored as rows of the constraint matrix A.  The successive
C     minimizers subject to the working set are computed by orthogonal
C     Givens rotations applied to the inverse Cholesky factor of C,
C     so that the whole procedure is numerically stable.
C
C     The calling sequence and the meaning of all arguments follow the
C     published user's guide of the code QL of K. Schittkowski.  The
C     source code below was written from scratch from that documentation
C     and from the algorithm of Goldfarb and Idnani.
C
C     Reference:
C
C        Goldfarb D., Idnani A. (1983): A numerically stable method for
C        solving strictly convex quadratic programs, Mathematical
C        Programming, Vol. 27, 1-33
C
C ======================================================================
C
      SUBROUTINE QL (M, ME, MMAX, N, NMAX, MNN, C, D, A, B,
     /               XL, XU, X, U, EPS, MODE, IOUT, IFAIL, IPRINT,
     /               WAR, LWAR, IWAR, LIWAR)
C
C   PURPOSE:
C
C      Solve the strictly convex quadratic program stated above.
C
C   PARAMETERS:
C
C      M       : Number of constraints.
C      ME      : Number of equality constraints.
C      MMAX    : Row dimension of A, MMAX >= max(1,M).
C      N       : Number of optimization variables.
C      NMAX    : Row dimension of C, NMAX >= N.
C      MNN     : Must be equal to M+N+N, dimension of U.
C      C(NMAX,N)  : Objective function matrix.  For MODE=0 the upper
C                triangular Cholesky factor R with C = R^T R.
C      D(N)    : Constant vector of the quadratic objective function.
C      A(MMAX,N) : Matrix of the linear constraints, first ME rows for
C                equality, then M-ME rows for inequality constraints.
C      B(M)    : Constant values of the linear constraints.
C      XL(N),XU(N) : Lower and upper bounds of the variables.
C      X(N)    : On return the optimal solution.
C      U(MNN)  : On return the multipliers, first M for the linear
C                constraints, then N for the lower and N for the upper
C                bounds.
C      EPS     : Final termination accuracy.
C      MODE    : 0 - Cholesky factor of C provided in upper part of C.
C                1 - Cholesky decomposition internally computed.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason, see below.
C      IPRINT  : 0 - no output, 1 - final error message.
C      WAR(LWAR)  : Real working array.
C      LWAR    : Length of WAR, at least
C                3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX + 1.
C      IWAR(LIWAR): Integer working array.
C      LIWAR   : Length of IWAR, at least N.
C
C   IFAIL:
C
C      0   : Optimality conditions satisfied.
C      1   : Termination after too many iterations (40*(N+M)).
C      2   : Termination accuracy insufficient.
C      3   : Inconsistency, division by zero.
C      4   : Numerical instabilities.
C      5   : LWAR, LIWAR, MNN or EPS incorrect.
C      >100: Constraint number IFAIL-100 causes an inconsistency.
C
      IMPLICIT NONE
      INTEGER M, ME, MMAX, N, NMAX, MNN, MODE, IOUT, IFAIL, IPRINT,
     /        LWAR, LIWAR
      INTEGER IWAR(LIWAR)
      DOUBLE PRECISION C(NMAX,*), D(*), A(MMAX,*), B(*), XL(*), XU(*),
     /                 X(*), U(*), EPS, WAR(LWAR)
C
      INTEGER LJ, LR, LZ, LRV, LDV, LUA, LNEED, NRT
C
      IFAIL = 0
      IF (N.LT.1 .OR. M.LT.0 .OR. ME.LT.0 .OR. ME.GT.M
     /    .OR. MMAX.LT.1 .OR. MMAX.LT.M .OR. NMAX.LT.N
     /    .OR. MNN.NE.M+N+N .OR. EPS.LE.0.0D0
     /    .OR. (MODE.NE.0 .AND. MODE.NE.1)) THEN
         IFAIL = 5
         GOTO 900
      ENDIF
C
C   Partition of the real working array.
C
      NRT = (N*(N+1))/2
      LJ  = 1
      LR  = LJ  + N*N
      LZ  = LR  + NRT
      LRV = LZ  + N
      LDV = LRV + N
      LUA = LDV + N
      LNEED = LUA + N
      IF (LWAR.LT.LNEED .OR. LIWAR.LT.N) THEN
         IFAIL = 5
         GOTO 900
      ENDIF
C
      CALL QLDUAL (M, ME, MMAX, N, NMAX, MNN, C, D, A, B, XL, XU,
     /             X, U, EPS, MODE, IFAIL,
     /             WAR(LJ), WAR(LR), WAR(LZ), WAR(LRV), WAR(LDV),
     /             WAR(LUA), IWAR)
C
  900 CONTINUE
      IF (IPRINT.GT.0 .AND. IFAIL.NE.0) THEN
         WRITE (IOUT,1000) IFAIL
      ENDIF
      RETURN
 1000 FORMAT (' *** ERROR IN QL: TERMINATION WITH IFAIL = ',I5)
      END
