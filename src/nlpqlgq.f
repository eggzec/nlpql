C ======================================================================
C
C     NLPQLGQ : Convenience wrapper which executes NLPQLG with the
C               quadratic programming code QL that comes with the
C               package.
C
C ======================================================================
C
      SUBROUTINE NLPQLGQ (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G,
     /                    DF, DG, U, XL, XU, C, D, ACC, ACCQP, STPMIN,
     /                    MAXFUN, MAXIT, MAXNM, RHO, NCYCLE, IPRINT,
     /                    MODE, IOUT, IFAIL, WA, LWA, KWA, LKWA, ACT,
     /                    LACT, LQL)
C
      IMPLICIT NONE
      INTEGER NP, M, ME, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        NCYCLE, IPRINT, MODE, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), DF(*), DG(MMAX,*),
     /                 U(*), XL(*), XU(*), C(NMAX,*), D(*), WA(LWA),
     /                 ACC, ACCQP, STPMIN, RHO
      LOGICAL ACT(LACT), LQL
      EXTERNAL QL
C
      CALL NLPQLG (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, STPMIN, MAXFUN,
     /             MAXIT, MAXNM, RHO, NCYCLE, IPRINT, MODE, IOUT,
     /             IFAIL, WA, LWA, KWA, LKWA, ACT, LACT, LQL, QL)
      RETURN
      END
