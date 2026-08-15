C ======================================================================
C
C     NLDFTW : Common working routine of the data fitting codes NLPLSQ,
C              NLPLSX, NLPL1, NLPINF and NLPMMX.
C
C     All of them proceed from L individual functions f_1(x), ...,
C     f_L(x) and a set of constraints
C
C          g_j(x)  = 0 ,  j = 1,...,me
C          g_j(x) >= 0 ,  j = me+1,...,m
C          xl <= x <= xu
C
C     and differ only in the norm by which the individual functions are
C     combined.  Since the resulting objective functions are either
C     non-differentiable or possess a structure that must not be passed
C     to a general purpose solver directly, the problems are transformed
C     into smooth nonlinear programs by additional variables and
C     constraints.  MODEL selects the transformation:
C
C     MODEL = 1 (NLPLSQ), least squares, L additional variables z and L
C     additional equality constraints,
C
C          min  sum z_i^2
C               f_i(x) - z_i  = 0 ,  i = 1,...,L
C               g_j(x)  = 0 ,  j = 1,...,me
C               g_j(x) >= 0 ,  j = me+1,...,m
C
C     MODEL = 2 (NLPL1), sum of absolute values, L additional variables
C     and 2L additional inequality constraints,
C
C          min  sum z_i
C               -f_i(x) + z_i >= 0 ,  i = 1,...,L
C                f_i(x) + z_i >= 0 ,  i = 1,...,L
C               0 <= z
C
C     MODEL = 3 (NLPINF), maximum of absolute values, one additional
C     variable and 2L additional inequality constraints,
C
C          min  z
C               -f_i(x) + z >= 0 ,  i = 1,...,L
C                f_i(x) + z >= 0 ,  i = 1,...,L
C               0 <= z
C
C     MODEL = 4 (NLPMMX), maximum of the functions, one additional
C     variable and L additional inequality constraints,
C
C          min  z
C               -f_i(x) + z >= 0 ,  i = 1,...,L
C
C     MODEL = 6, a plain linear combination of the individual functions,
C
C          min  sum f_i(x)
C
C     which is already smooth, so that neither additional variables nor
C     additional constraints are needed.  The model is used by NLPJOB
C     for those scalar transformations of a multicriteria problem that
C     are differentiable.
C
C     MODEL = 5 (NLPLSX), least squares with very many terms.  No
C     additional variables are introduced, since their number would be
C     prohibitive.  The sum of squares is minimized directly, the
C     gradient being assembled from the individual gradients,
C
C          min  sum f_i(x)^2 ,   grad f = 2 sum f_i(x) grad f_i(x) .
C
C     The transformed program is solved by the SQP code NLPQLP.  Model
C     functions are provided by reverse communication, the routine
C     returns IFAIL = -1 to request the L function values and the M
C     constraint values, and IFAIL = -2 to request their gradients.
C
C        ST(1) - phase, 1 = function values, 2 = gradient values
C        ST(2) - termination flag of NLPQLP
C
C ======================================================================
C
      SUBROUTINE NLDFTW (MODEL, L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F,
     /                   G, DF, DG, U, XL, XU, C, D, ACC, ACCQP,
     /                   MAXFUN, MAXIT, MAXNM, RHO, IPRINT, IOUT,
     /                   IFAIL, FV, ST, WA, LWA, KWA, LKWA, ACT, LACT,
     /                   QPSLVE)
C
      IMPLICIT NONE
      INTEGER MODEL, L, M, ME, LMMAX, N, LNMAX, LMNN2, MAXFUN, MAXIT,
     /        MAXNM, IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(LNMAX), F, G(LMMAX), DF(LNMAX),
     /                 DG(LMMAX,LNMAX), U(LMNN2), XL(LNMAX), XU(LNMAX),
     /                 C(LNMAX,LNMAX), D(LNMAX), WA(LWA), ACC, ACCQP,
     /                 RHO
      DOUBLE PRECISION FV(1), ST(6)
      LOGICAL ACT(LACT)
      EXTERNAL QPSLVE
C
      INTEGER I, J, K, LZ, NT, MT, MET, IPH, IFP, MODEQ
      DOUBLE PRECISION S, T, ZERO, ONE, TWO, BIG
      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0, BIG=1.0D+30)
C
C   Dimensions of the transformed program.
C
      IF (MODEL.EQ.1) THEN
         LZ  = L
         MT  = M + L
         MET = ME + L
      ELSE IF (MODEL.EQ.2) THEN
         LZ  = L
         MT  = M + 2*L
         MET = ME
      ELSE IF (MODEL.EQ.3) THEN
         LZ  = 1
         MT  = M + 2*L
         MET = ME
      ELSE IF (MODEL.EQ.4) THEN
         LZ  = 1
         MT  = M + L
         MET = ME
      ELSE
         LZ  = 0
         MT  = M
         MET = ME
      ENDIF
      IF (MODEL.LT.1 .OR. MODEL.GT.6) THEN
         IFAIL = 9
         RETURN
      ENDIF
      NT = N + LZ
      IF (LNMAX.LT.NT+1 .OR. LMMAX.LT.MAX(MT,1)
     /    .OR. LMNN2.NE.MT+NT+NT+2) THEN
         IFAIL = 6
         RETURN
      ENDIF
C
      IF (IFAIL.EQ.0) GOTO 100
      IPH = INT(ST(1))
      IF (IPH.EQ.1) GOTO 300
      IF (IPH.EQ.2) GOTO 400
      IFAIL = 9
      RETURN
C
C   Initialization.  Starting values of the additional variables are
C   chosen so that the additional constraints are satisfied at the
C   starting point.
C
  100 CONTINUE
      ST(2) = ZERO
      IF (MODEL.EQ.1) THEN
         DO 110 I = 1, L
            X(N+I)  = G(M+I)
            XL(N+I) = -BIG
            XU(N+I) =  BIG
  110    CONTINUE
      ELSE IF (MODEL.EQ.2) THEN
         DO 120 I = 1, L
            X(N+I)  = ABS(G(M+I))
            XL(N+I) = ZERO
            XU(N+I) = BIG
  120    CONTINUE
      ELSE IF (MODEL.EQ.3) THEN
         S = ZERO
         DO 130 I = 1, L
            S = MAX(S,ABS(G(M+I)))
  130    CONTINUE
         X(N+1)  = S
         XL(N+1) = ZERO
         XU(N+1) = BIG
      ELSE IF (MODEL.EQ.4) THEN
         S = -BIG
         DO 140 I = 1, L
            S = MAX(S,G(M+I))
  140    CONTINUE
         X(N+1)  = S
         XL(N+1) = -BIG
         XU(N+1) =  BIG
      ENDIF
C
C   When starting, the calling program has provided function values as
C   well as gradients, so that both have to be transformed.  Note that
C   the gradients must be transformed first, since for MODEL = 5 the
C   gradient of the objective function depends on the untransformed
C   function values.
C
      CALL NLDFTG (MODEL, L, M, ME, LMMAX, N, LNMAX, NT, X, G, DF, DG)
      CALL NLDFTF (MODEL, L, M, ME, LMMAX, N, LNMAX, X, G, FV)
      GOTO 700
C
C   Re-entry with new function values.
C
  300 CONTINUE
      CALL NLDFTF (MODEL, L, M, ME, LMMAX, N, LNMAX, X, G, FV)
      GOTO 700
C
C   Re-entry with new gradient values.
C
  400 CONTINUE
      CALL NLDFTG (MODEL, L, M, ME, LMMAX, N, LNMAX, NT, X, G, DF, DG)
      GOTO 700
C
C   One call of the SQP algorithm.
C
  700 CONTINUE
      IFP   = INT(ST(2))
      MODEQ = 0
      CALL NLPQLP (1, MT, MET, LMMAX, NT, LNMAX, LMNN2, X, FV, G, DF,
     /             DG, U, XL, XU, C, D, ACC, ACCQP, ZERO, MAXFUN,
     /             MAXIT, MAXNM, RHO, IPRINT, MODEQ, IOUT, IFP, WA,
     /             LWA, KWA, LKWA, ACT, LACT, .TRUE., QPSLVE)
      ST(2) = DBLE(IFP)
      F     = FV(1)
      IF (IFP.EQ.-1) THEN
         ST(1) = ONE
         IFAIL = -1
         RETURN
      ENDIF
      IF (IFP.EQ.-2) THEN
         ST(1) = TWO
         IFAIL = -2
         RETURN
      ENDIF
      IFAIL = IFP
      RETURN
      END
