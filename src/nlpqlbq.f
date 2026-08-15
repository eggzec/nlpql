C ======================================================================
C
C     NLPQLBQ : Convenience wrapper which executes NLPQLB with the
C               quadratic programming code QL that comes with the
C               package.
C
C ======================================================================
C
      SUBROUTINE NLPQLBQ (M, ME, MW, MWMAX, N, NMAX, MNN2, X, F, G,
     /                    DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN,
     /                    MAXIT, MAXNM, RHOB, IPRINT, IOUT, IFAIL,
     /                    WA, LWA, KWA, LKWA, ACT, LACT)
C
      IMPLICIT NONE
      INTEGER M, ME, MW, MWMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MWMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(LWA), ACC, ACCQP,
     /                 RHOB
      LOGICAL ACT(LACT)
      EXTERNAL QL
C
      CALL NLPQLB (M, ME, MW, MWMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, MAXFUN, MAXIT, MAXNM,
     /             RHOB, IPRINT, IOUT, IFAIL, WA, LWA, KWA, LKWA,
     /             ACT, LACT, QL)
      RETURN
      END
