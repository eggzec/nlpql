C ======================================================================
C
C     NLPRNT : Output of the SQP algorithm.
C
C        MODEP = 0 : headline and input parameters
C        MODEP = 1 : one line of intermediate results (IPRINT = 2)
C        MODEP = 2 : final convergence analysis
C        MODEP = 3 : detailed information for one iteration (IPRINT>=3)
C        MODEP = 4 : one line search step (IPRINT = 4)
C        MODEP = 5 : successful termination of a line search (IPRINT=4)
C
C ======================================================================
C
      SUBROUTINE NLPRNT (MODEP, N, M, ME, MMAX, NMAX, MNN2, MODE,
     /                   ACC, ACCQP, STPMIN, RHO, MAXFUN, MAXNM,
     /                   MAXIT, IPRINT, IOUT, X, F, G, U, XL, XU, R,
     /                   KWA, SC, IFAIL)
C
      IMPLICIT NONE
      INTEGER MODEP, N, M, ME, MMAX, NMAX, MNN2, MODE, MAXFUN, MAXNM,
     /        MAXIT, IPRINT, IOUT, IFAIL
      INTEGER KWA(25)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), U(*), XL(*), XU(*),
     /                 R(*), SC(20), ACC, ACCQP, STPMIN, RHO
      INTEGER I, J, NQ
C
      NQ = N + 1
      GOTO (100, 200, 300, 400, 500, 600, 700), MODEP+1
      RETURN
C
C   Headline and parameters.
C
  100 CONTINUE
      WRITE (IOUT,1000)
      WRITE (IOUT,1010) N, M, ME, MODE, ACC, ACCQP, STPMIN, RHO,
     /                  MAXFUN, MAXNM, MAXIT, IPRINT
      IF (IPRINT.EQ.2) WRITE (IOUT,1020)
      RETURN
C
C   One line of intermediate results.
C
  200 CONTINUE
      WRITE (IOUT,1030) KWA(3), F(1), SC(6), KWA(11), KWA(7),
     /                  SC(1), SC(7), SC(11)
      RETURN
C
C   Final convergence analysis.
C
  300 CONTINUE
      IF (IPRINT.LE.0) RETURN
      WRITE (IOUT,1040)
      IF (IFAIL.NE.0) WRITE (IOUT,1050) IFAIL
      WRITE (IOUT,1060) KWA(3)
      WRITE (IOUT,1070) F(1)
      WRITE (IOUT,1080)
      WRITE (IOUT,1200) (X(I,1),I=1,N)
      IF (N.LE.1000) THEN
         WRITE (IOUT,1090)
         WRITE (IOUT,1200) (X(I,1)-XL(I),I=1,N)
         WRITE (IOUT,1100)
         WRITE (IOUT,1200) (XU(I)-X(I,1),I=1,N)
         WRITE (IOUT,1110)
         WRITE (IOUT,1200) (U(M+I),I=1,N)
         WRITE (IOUT,1120)
         WRITE (IOUT,1200) (U(M+NQ+I),I=1,N)
      ENDIF
      IF (M.GT.0 .AND. M.LE.1000) THEN
         WRITE (IOUT,1130)
         WRITE (IOUT,1200) (G(J,1),J=1,M)
         WRITE (IOUT,1140)
         WRITE (IOUT,1200) (U(J),J=1,M)
      ENDIF
      WRITE (IOUT,1150) KWA(1)
      WRITE (IOUT,1160) KWA(2)
      WRITE (IOUT,1170) KWA(4)
      RETURN
C
C   Detailed information for one iteration, first part.
C
  400 CONTINUE
      WRITE (IOUT,2000) KWA(3)
      WRITE (IOUT,2010) F(1)
      WRITE (IOUT,2020)
      WRITE (IOUT,2200) (X(I,1),I=1,MIN(N,1000))
      IF (MNN2.LE.1000) THEN
         WRITE (IOUT,2030)
         WRITE (IOUT,2200) (U(J),J=1,MNN2)
      ENDIF
      IF (M.GT.0 .AND. M.LE.1000) THEN
         WRITE (IOUT,2040)
         WRITE (IOUT,2200) (G(J,1),J=1,M)
      ENDIF
      RETURN
C
C   Detailed information for one iteration, second part.
C
  700 CONTINUE
      WRITE (IOUT,2050) SC(6)
      WRITE (IOUT,2060) KWA(11)
      WRITE (IOUT,2070) SC(11)
      WRITE (IOUT,2080) SC(15)
      WRITE (IOUT,2090) SC(4)
      IF (M.GT.0 .AND. M.LE.1000) THEN
         WRITE (IOUT,2100)
         WRITE (IOUT,2200) (R(J),J=1,M)
      ENDIF
      WRITE (IOUT,2110) SC(16)
      RETURN
C
C   Line search protocol.
C
  500 CONTINUE
      WRITE (IOUT,2120) KWA(7), SC(1), SC(17)
      RETURN
C
  600 CONTINUE
      IF (KWA(7).EQ.1) THEN
         WRITE (IOUT,2130)
      ELSE
         WRITE (IOUT,2140) KWA(7), SC(1)
      ENDIF
      RETURN
C
 1000 FORMAT (/,4X,68('-'),/,
     /   5X,'START OF THE SEQUENTIAL QUADRATIC PROGRAMMING ALGORITHM',
     /   /,5X,'NLPQLP, VERSION 4.2',/,4X,68('-'),/)
 1010 FORMAT (5X,'Parameters:',/,
     /   8X,'N      = ',I8,/,
     /   8X,'M      = ',I8,/,
     /   8X,'ME     = ',I8,/,
     /   8X,'MODE   = ',I8,/,
     /   8X,'ACC    = ',D12.4,/,
     /   8X,'ACCQP  = ',D12.4,/,
     /   8X,'STPMIN = ',D12.4,/,
     /   8X,'RHO    = ',D12.4,/,
     /   8X,'MAXFUN = ',I8,/,
     /   8X,'MAX_NM = ',I8,/,
     /   8X,'MAXIT  = ',I8,/,
     /   8X,'IPRINT = ',I8,/)
 1020 FORMAT (' Output in the following order:',/,
     /   5X,'IT    - iteration number',/,
     /   5X,'F     - objective function value',/,
     /   5X,'SCV   - sum of constraint violations',/,
     /   5X,'NA    - number of active constraints',/,
     /   5X,'I     - number of line search iterations',/,
     /   5X,'ALPHA - steplength parameter',/,
     /   5X,'DELTA - additional variable to prevent inconsistency',/,
     /   5X,'KKT   - Karush-Kuhn-Tucker optimality criterion',//,
     /   '   IT',9X,'F',11X,'SCV',5X,'NA',2X,'I',3X,'ALPHA',5X,
     /   'DELTA',6X,'KKT',/,1X,70('-'))
 1030 FORMAT (I5,1X,D16.8,1X,D9.2,I4,I3,3(1X,D9.2))
 1040 FORMAT (/,6X,'--- Final Convergence Analysis at Best Iterate ---'
     /        ,/)
 1050 FORMAT (8X,'Termination reason:            IFAIL = ',I8)
 1060 FORMAT (8X,'Best result at iteration:      ITER  = ',I8)
 1070 FORMAT (8X,'Objective function value:      F(X)  = ',D16.8)
 1080 FORMAT (8X,'Approximation of solution:     X     = ')
 1090 FORMAT (8X,'Distance from lower bound:     X-XL  = ')
 1100 FORMAT (8X,'Distance from upper bound:     XU-X  = ')
 1110 FORMAT (8X,'Multipliers for lower bounds:  U     = ')
 1120 FORMAT (8X,'Multipliers for upper bounds:  U     = ')
 1130 FORMAT (8X,'Constraint values:             G(X)  = ')
 1140 FORMAT (8X,'Multipliers for constraints:   U     = ')
 1150 FORMAT (8X,'Number of function calls:      NFUNC = ',I8)
 1160 FORMAT (8X,'Number of gradient calls:      NGRAD = ',I8)
 1170 FORMAT (8X,'Number of calls of QP solver:  NQL   = ',I8)
 1200 FORMAT (10X,4D17.8)
C
 2000 FORMAT (//,5X,'Iteration ',I4,/)
 2010 FORMAT (8X,'Function value:  F(X) = ',D16.8)
 2020 FORMAT (8X,'Variable:  X =')
 2030 FORMAT (8X,'Multipliers:  U =')
 2040 FORMAT (8X,'Constraints: G(X) =')
 2050 FORMAT (8X,'Sum of constraint violations:',20X,'SCV = ',D12.4)
 2060 FORMAT (8X,'Number of active constraints:',20X,'NAC = ',I4)
 2070 FORMAT (8X,'Karush-Kuhn-Tucker optimality condition:',9X,
     /        'KKT = ',D12.4)
 2080 FORMAT (8X,'Norm of Lagrangian gradient:',21X,'NLG = ',D12.4)
 2090 FORMAT (8X,'Product of search direction with BFGS matrix:',4X,
     /        'DBD = ',D12.4)
 2100 FORMAT (8X,'Penalty parameter:  R =')
 2110 FORMAT (8X,'Product of Lag-gradient with search direction:',3X,
     /        'DLP = ',D12.4)
 2120 FORMAT (8X,'Line search ',I2,': ALPHA = ',D9.2,
     /        ', merit function FCT = ',D14.6)
 2130 FORMAT (8X,'Line search successful after one step:  ALPHA = 1')
 2140 FORMAT (8X,'Line search successful after ',I2,' steps:  ALPHA = ',
     /        D12.4)
 2200 FORMAT (11X,4D16.8)
      END
