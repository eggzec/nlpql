C ======================================================================
C
C     NLPQLFQ : Convenience wrapper which executes NLPQLF with the
C     quadratic programming code QL that comes with the package.
C
C ======================================================================
C
      SUBROUTINE NLPQLFQ (MF, MEF, M, MMAX, N, NMAX, MNN2, X, F, G, DF,
     /                    DG, U, XL, XU, C, D, ACC, ACCF, ACCQP,
     /                    MAXFUN, MAXIT, MAXNM, RHO, IPRINT, IOUT,
     /                    IFAIL, WA, LWA, KWA, LKWA, ACT, LACT)
C
      IMPLICIT NONE
      INTEGER MF, MEF, M, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(MMAX,*), U(*), XL(*),
     /                 XU(*), C(NMAX,*), D(*), WA(LWA), ACC, ACCF,
     /                 ACCQP, RHO
      LOGICAL ACT(LACT)
      EXTERNAL QL
C
      CALL NLPQLF (MF, MEF, M, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCF, ACCQP, MAXFUN, MAXIT,
     /             MAXNM, RHO, IPRINT, IOUT, IFAIL, WA, LWA, KWA,
     /             LKWA, ACT, LACT, QL)
      RETURN
      END
