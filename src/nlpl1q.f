C ======================================================================
C
C     NLPL1Q : Convenience wrapper which executes NLPL1 with the
C     quadratic programming code QL that comes with the package.  The
C     only difference to NLPL1 is the missing argument QPSLVE, so that
C     the routine can be called from programming languages that are not
C     able to pass a Fortran subroutine as an argument.
C
C ======================================================================
C
      SUBROUTINE NLPL1Q (L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G,
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
      CALL NLPL1 (L, M, ME, LMMAX, N, LNMAX, LMNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, MAXFUN, MAXIT, MAXNM,
     /             RHO, IPRINT, IOUT, IFAIL, WA, LWA, KWA, LKWA, ACT,
     /             LACT, QL)
      RETURN
      END
