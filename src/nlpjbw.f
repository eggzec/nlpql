C ======================================================================
C
C     NLPJBW : Working routine of NLPJOB.
C
C     Every transformation of the multicriteria problem is written as a
C     residual vector
C
C          r_i(x) := a_i ( f_i(x) - b_i ) ,   i = 1,...,L
C
C     combined by one of the norms provided by NLDFTW, plus at most
C     L-1 additional inequality constraints.  The coefficients a_i and
C     b_i and the kind of the combination follow from MODEL:
C
C        MODEL  kind        a_i                b_i     additional
C        ------------------------------------------------------------
C          0    sum         1 for i = IMIN     0       none
C          1    sum         w_i                0       none
C          2    sum         1 for i = IMIN     0       IMIN-1
C          3    sum         1 for i = IMIN     0       L-1
C          4    L1          1                  w_i     none
C          5    L2          1                  w_i     none
C          6    sum         1/|f_i*|           f_i*    none
C          7    L2          1/f_i*             f_i*    none
C          8    max abs     1                  0       none
C          9    max         1                  0       none
C         10    max abs     1                  w_i     none
C         11    max         1/|f_i*|           f_i*    none
C         12    max         w_i/|f_i*|         f_i*    none
C         13    max         w_i                0       none
C         14    sum         w_i/y_i            y_i     none
C         15    L2          sqrt(w_i)/y_i      y_i     none
C
C     The constraint block handed over to NLDFTW consists of the M
C     original constraints followed by the additional ones, the
C     subsequent L positions hold the residuals.
C
C ======================================================================
C
      SUBROUTINE NLPJBW (L, M, ME, LMMAX, N, LNMAX, LMNN2, MODEL, IMIN,
     /                   X, F, G, DF, DG, U, XL, XU, W, FK, FW, ACC,
     /                   ACCQP, MAXFUN, MAXIT, IPRINT, IOUT, IFAIL, A,
     /                   B, FV, ST, C, D, WA, LWA, KWA, LKWA, ACT,
     /                   LACT)
C
      IMPLICIT NONE
      INTEGER L, M, ME, LMMAX, N, LNMAX, LMNN2, MODEL, IMIN, MAXFUN,
     /        MAXIT, IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(LNMAX), F, G(LMMAX), DF(LNMAX),
     /                 DG(LMMAX,LNMAX), U(*), XL(LNMAX), XU(LNMAX),
     /                 W(L), FK(L), FW(L), WA(LWA), ACC, ACCQP
      DOUBLE PRECISION A(L), B(L), FV(1), ST(6), C(LNMAX,LNMAX),
     /                 D(LNMAX)
      LOGICAL ACT(LACT)
      EXTERNAL QL
C
      INTEGER I, J, K, KIND, NEX, MEX, MT, NT, MNN2, IERR
      DOUBLE PRECISION S, ZERO, ONE, RHO
      PARAMETER (ZERO=0.0D0, ONE=1.0D0)
C
C   Coefficients of the residuals and number of additional constraints.
C
      CALL NLPJBC (L, MODEL, IMIN, W, FK, A, B, KIND, NEX, IERR)
      IF (IERR.NE.0) THEN
         IFAIL = IERR
         IF (IPRINT.GT.0) WRITE (IOUT,1000) IFAIL
         RETURN
      ENDIF
      MEX = M + NEX
      IF (KIND.EQ.1) THEN
         MT = MEX + L
         NT = N + L
      ELSE IF (KIND.EQ.2) THEN
         MT = MEX + 2*L
         NT = N + L
      ELSE IF (KIND.EQ.3) THEN
         MT = MEX + 2*L
         NT = N + 1
      ELSE IF (KIND.EQ.4) THEN
         MT = MEX + L
         NT = N + 1
      ELSE
         MT = MEX
         NT = N
      ENDIF
      MNN2 = MT + NT + NT + 2
      IF (LMNN2.LT.MNN2) THEN
         IFAIL = 6
         IF (IPRINT.GT.0) WRITE (IOUT,1000) 6
         RETURN
      ENDIF
      RHO = 1.0D+2
C
C   Rearrangement of the function and gradient values into the layout
C   expected by NLDFTW.  On entry with IFAIL = 0 both are available, on
C   entry with IFAIL = -1 only the function values and on entry with
C   IFAIL = -2 only the gradients have been recomputed.
C
      IF (IFAIL.EQ.0 .OR. IFAIL.EQ.-2) THEN
         CALL NLPJBG (L, M, LMMAX, N, LNMAX, NEX, MODEL, IMIN, A, DG)
      ENDIF
      IF (IFAIL.EQ.0 .OR. IFAIL.EQ.-1) THEN
         CALL NLPJBF (L, M, LMMAX, NEX, MODEL, IMIN, A, B, W, FK, G,
     /                FW)
      ENDIF
C
      CALL NLDFTW (KIND, L, MEX, ME, LMMAX, N, LNMAX, MNN2, X, F, G,
     /             DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN, MAXIT,
     /             10, RHO, IPRINT, IOUT, IFAIL, FV, ST, WA, LWA, KWA,
     /             LKWA, ACT, LACT, QL)
      RETURN
 1000 FORMAT (' *** ERROR IN NLPJOB: IFAIL = ',I5)
      END
