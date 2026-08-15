C ======================================================================
C
C     NLPJOB : MULTICRITERIA OPTIMIZATION
C
C     NLPJOB solves the smooth multicriteria or multiobjective problem
C
C          min  ( f_1(x), ..., f_L(x) )
C          x in R^n :  g_j(x)  = 0 ,  j = 1,...,me
C                      g_j(x) >= 0 ,  j = me+1,...,m
C                      xl <= x <= xu
C
C     by a transformation into a scalar nonlinear program which is then
C     solved by the SQP code NLPQLP.  Sixteen different transformations
C     are provided, selected by MODEL:
C
C      0  individual minimum,        f := f_i , i = IMIN
C      1  weighted sum,              f := sum w_i f_i
C      2  hierarchical optimization, f := f_i , i = IMIN, subject to the
C                                    additional constraints
C                                    f_j <= (1 + w_j/100) f_j* ,
C                                    j = 1,...,IMIN-1
C      3  trade-off method,          f := f_i , i = IMIN, subject to the
C                                    additional constraints
C                                    f_j <= w_j , j /= IMIN
C      4  distance function, L1,     f := sum |f_i - y_i|
C      5  distance function, L2,     f := sum (f_i - y_i)^2
C      6  global criterion,          f := sum (f_i - f_i*)/|f_i*|
C      7  global criterion, L2,      f := sum ((f_i - f_i*)/f_i*)^2
C      8  min-max no. 1,             f := max |f_i|
C      9  min-max no. 2,             f := max f_i
C     10  min-max no. 3,             f := max |f_i - y_i|
C     11  min-max no. 4,             f := max (f_i - f_i*)/|f_i*|
C     12  min-max no. 5,             f := max w_i (f_i - f_i*)/|f_i*|
C     13  min-max no. 6,             f := max w_i f_i
C     14  weighted global criterion, f := sum w_i (f_i - y_i)/y_i
C     15  weighted global criterion, L2,
C                                    f := sum w_i ((f_i - y_i)/y_i)^2
C
C     Here w_1, ..., w_L are weights or bounds passed in W, y_1, ...,
C     y_L are goal values and f_1*, ..., f_L* the individual minima,
C     both passed in FK.  The individual minima are obtained by running
C     NLPJOB with MODEL = 0 for i = 1, ..., L.
C
C     Whenever the resulting scalar objective function is not
C     differentiable, i.e., for the L1, the maximum norm and the
C     min-max models, or possesses a structure that should not be
C     passed to a general purpose solver directly, as for sums of
C     squares, additional variables and constraints are introduced, see
C     NLDFTW.
C
C   PARAMETERS:
C
C      L       : Number of objective functions.
C      M       : Total number of constraints.
C      ME      : Number of equality constraints.
C      LMMAX   : Dimension of G and row dimension of DG, must be
C                greater or equal to M + L + L.
C      N       : Number of optimization variables.
C      LNMAX   : Dimension of X, DF, XL, XU and row dimension of C,
C                must be at least two and greater than N + L.
C      LMNN2   : Dimension of U, at least 4*L + 2*N + M + 2.
C      MODEL   : Desired scalar transformation, 0 <= MODEL <= 15.
C      IMIN    : Index of the objective function taken into account for
C                MODEL = 0, 2, 3.
C      X(LNMAX): The first N positions contain the starting values, on
C                return the final iterate.
C      F       : On return the objective function value of the scalar
C                program.
C      G(LMMAX): The first M positions contain the constraint values,
C                the subsequent L positions the objective function
C                values, evaluated at the first N positions of X.
C      DF(LNMAX) : Internally used for the gradient of the scalar
C                objective function.
C      DG(LMMAX,LNMAX) : The first M rows contain the gradients of the
C                constraints, the subsequent L rows the gradients of
C                the objective functions.
C      U(LMNN2): On return the multipliers of the scalar program.
C      XL(LNMAX),XU(LNMAX) : The first N positions contain the bounds
C                of the variables, the remaining ones are set
C                internally.
C      W(L)    : Weights, bounds or goal values, depending on MODEL.
C      FK(L)   : Individual minima or goal values, depending on MODEL.
C                The values must be different from zero for
C                MODEL = 6, 7, 11, 12, 14, 15.
C      FW(L)   : On return the objective function values subject to the
C                final iterate.
C      ACC     : Desired final accuracy.
C      ACCQP   : Tolerance of the QP solver.
C      MAXFUN  : Upper bound for the number of function calls during
C                the line search.
C      MAXIT   : Maximum number of iterations.
C      IPRINT  : Output level.
C      IOUT    : Output unit number.
C      IFAIL   : Termination reason.  IFAIL = -1 requests new values of
C                the constraints and of the objective functions,
C                IFAIL = -2 their gradients, both subject to the first
C                N positions of X.  IFAIL = 11 indicates zero values in
C                FK.
C      WA(LWA) : Real working array.
C      LWA     : Length of WA, at least
C                5*LNMAX*LNMAX/2 + 34*LNMAX + 9*LMMAX + 150.
C      KWA(LKWA) : Integer working array.
C      LKWA    : Length of KWA, at least LNMAX + 25.
C      LOGWA(LLOGWA) : Logical working array.
C      LLOGWA  : Length of LOGWA, at least 2*LMMAX + 10.
C
C ======================================================================
C
      SUBROUTINE NLPJOB (L, M, ME, LMMAX, N, LNMAX, LMNN2, MODEL, IMIN,
     /                   X, F, G, DF, DG, U, XL, XU, W, FK, FW, ACC,
     /                   ACCQP, MAXFUN, MAXIT, IPRINT, IOUT, IFAIL, WA,
     /                   LWA, KWA, LKWA, LOGWA, LLOGWA)
C
      IMPLICIT NONE
      INTEGER L, M, ME, LMMAX, N, LNMAX, LMNN2, MODEL, IMIN, MAXFUN,
     /        MAXIT, IPRINT, IOUT, IFAIL, LWA, LKWA, LLOGWA
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(LMMAX,*), U(*), XL(*),
     /                 XU(*), W(*), FK(*), FW(*), WA(LWA), ACC, ACCQP
      LOGICAL LOGWA(LLOGWA)
C
      INTEGER LA, LB, LFV, LST, LC, LD, LW, LWW, NT
C
      IF (L.LT.1 .OR. M.LT.0 .OR. ME.LT.0 .OR. ME.GT.M .OR. N.LT.1
     /    .OR. MODEL.LT.0 .OR. MODEL.GT.15 .OR. LMMAX.LT.M+L+L
     /    .OR. LNMAX.LE.N+L) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) 6
         RETURN
      ENDIF
C
C   Partition of the real working array.  The quasi-Newton matrix of
C   the scalar program is kept in WA as well, since NLPJOB does not
C   possess corresponding formal arguments.
C
      NT  = LNMAX
      LA  = 1
      LB  = LA  + L
      LFV = LB  + L
      LST = LFV + 1
      LC  = LST + 6
      LD  = LC  + NT*NT
      LW  = LD  + NT
      LWW = LWA - LW + 1
      IF (LWW.LT.1 .OR. LLOGWA.LT.2*LMMAX+10 .OR. LKWA.LT.LNMAX+25)
     /   THEN
         IFAIL = 5
         IF (IPRINT.GT.0) WRITE (IOUT,1000) 5
         RETURN
      ENDIF
C
      CALL NLPJBW (L, M, ME, LMMAX, N, LNMAX, LMNN2, MODEL, IMIN, X,
     /             F, G, DF, DG, U, XL, XU, W, FK, FW, ACC, ACCQP,
     /             MAXFUN, MAXIT, IPRINT, IOUT, IFAIL, WA(LA), WA(LB),
     /             WA(LFV), WA(LST), WA(LC), WA(LD), WA(LW), LWW, KWA,
     /             LKWA, LOGWA, LLOGWA)
      RETURN
 1000 FORMAT (' *** ERROR IN NLPJOB: IFAIL = ',I5)
      END
