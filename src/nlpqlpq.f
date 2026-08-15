C ======================================================================
C
C     NLPQLPQ : Convenience wrapper which executes NLPQLP with the
C               quadratic programming code QL that comes with the
C               package.  The only difference to NLPQLP is the missing
C               argument QPSLVE, so that the routine can be called from
C               programming languages that are not able to pass a
C               Fortran subroutine as an argument.
C
C ======================================================================
C
      SUBROUTINE NLPQLPQ (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G,
     /                    DF, DG, U, XL, XU, C, D, ACC, ACCQP, STPMIN,
     /                    MAXFUN, MAXIT, MAXNM, RHO, IPRINT, MODE,
     /                    IOUT, IFAIL, WA, LWA, KWA, LKWA, ACT, LACT,
     /                    LQL)
C
      IMPLICIT NONE
      INTEGER NP, M, ME, MMAX, N, NMAX, MNN2, MAXFUN, MAXIT, MAXNM,
     /        IPRINT, MODE, IOUT, IFAIL, LWA, LKWA, LACT
      INTEGER KWA(LKWA)
      DOUBLE PRECISION X(NMAX,*), F(*), G(MMAX,*), DF(*), DG(MMAX,*),
     /                 U(*), XL(*), XU(*), C(NMAX,*), D(*), WA(LWA),
     /                 ACC, ACCQP, STPMIN, RHO
      LOGICAL ACT(LACT), LQL
      EXTERNAL QL
C
      CALL NLPQLP (NP, M, ME, MMAX, N, NMAX, MNN2, X, F, G, DF, DG,
     /             U, XL, XU, C, D, ACC, ACCQP, STPMIN, MAXFUN,
     /             MAXIT, MAXNM, RHO, IPRINT, MODE, IOUT, IFAIL,
     /             WA, LWA, KWA, LKWA, ACT, LACT, LQL, QL)
      RETURN
      END
