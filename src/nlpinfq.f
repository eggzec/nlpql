C ======================================================================
C
C     NLPINFQ : Convenience wrapper which executes NLPINF with the
C     quadratic programming code QL that comes with the package.  The
C     only difference to NLPINF is the missing argument QPSLVE, so that
C     the routine can be called from programming languages that are not
C     able to pass a Fortran subroutine as an argument.
C
C ======================================================================
C
      SUBROUTINE NLPINFQ (L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G,
     /                    DF, DG, U, XL, XU, C, D, ACC, ACCQP, MAXFUN,
     /                    MAXIT, MAXNM, RHO, IPRINT, IOUT, IFAIL, WA,
     /                    LWA, KWA, LKWA, ACT, LACT)
C
      IMPLICIT NONE
      INTEGER L, M, ME, LMMAX, N, LNMAX, LMNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(*), F, G(*), DF(*), DG(LMMAX,*), U(*), XL(*),
     /                 XU(*), C(LNMAX,*), D(*), WA(LWA), ACC, ACCQP,
     /                 RHO
      LOGICAL ACT(LACT)
      EXTERNAL QL
C
      CALL NLPINF (L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, MAXFUN, MAXIT, MAXNM,
     /             RHO, IPRINT, IOUT, IFAIL, WA, LWA, KWA, LKWA, ACT,
     /             LACT, QL)
      RETURN
      END
