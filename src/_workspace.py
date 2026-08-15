"""Exact working array sizes required by the Fortran routines.

The Fortran codes never allocate memory.  Every routine checks the
length of the working arrays that it receives and returns ``IFAIL = 5``
if one of them is too short.  The functions below reproduce the
partitioning of the Fortran sources so that the Python layer always
passes arrays of exactly the right size.
"""

from __future__ import annotations


#: Transformation used by the data fitting codes, see ``NLDFTW``.
FIT_MODELS = {"nlplsq": 1, "nlpl1": 2, "nlpinf": 3, "nlpmmx": 4, "nlplsx": 5}


def _ql_real(nmax: int, m: int) -> int:
    """Return the real workspace consumed by the QP solver QL.

    Args:
        nmax: Row dimension of the objective function matrix.
        m: Number of linear constraints of the subproblem.

    Returns:
        Number of double precision numbers needed by QL.
    """
    return (3 * nmax * nmax) // 2 + 10 * nmax + 2 * max(m, 1) + 2


def nlpqlp_real(n: int, m: int, nmax: int, mmax: int, nproc: int) -> int:
    """Return the real workspace consumed by ``NLPQLP``.

    The expression reproduces the partitioning of ``nlpqlp.f`` exactly.

    Args:
        n: Number of optimization variables of the program solved.
        m: Number of constraints of the program solved.
        nmax: Row dimension of the quasi-Newton matrix.
        mmax: Row dimension of the constraint Jacobian.
        nproc: Number of simultaneous function evaluations.

    Returns:
        Number of double precision numbers needed by NLPQLP.
    """
    mm = max(m, 1)
    mg = max(mmax, 1)
    mnn2 = m + n + n + 2
    off = 1 + (n + 1 + mm) + n + n + mm + (n + 1) + mnn2 + mm + 50
    off += 2 * nproc + 3 * (n + 1) + n + n + 20
    return off + _ql_real(nmax, mg) - 1


def nlpqlp_bound(n: int, m: int, nmax: int, mmax: int, nproc: int) -> int:
    """Return the documented upper bound of the workspace of ``NLPQLP``.

    The entry routines of ``NLPQLY`` and ``NLPQLB`` check the length of
    their working array against this bound instead of the exact
    partitioning, therefore the Python layer has to reproduce it.

    Args:
        n: Number of optimization variables of the program solved.
        m: Number of constraints of the program solved.
        nmax: Row dimension of the quasi-Newton matrix.
        mmax: Row dimension of the constraint Jacobian.
        nproc: Number of simultaneous function evaluations.

    Returns:
        Number of double precision numbers demanded by the check.
    """
    mg = max(mmax, 1)
    return (
        23 * n
        + 4 * m
        + 3 * mg
        + nproc * (n + m + 1)
        + 150
        + (3 * nmax * nmax) // 2
        + 10 * nmax
        + 2 * mg
        + 2
    )


def nlpqlp_sizes(n: int, m: int, nproc: int = 1) -> dict[str, int]:
    """Return the array sizes required by ``NLPQLP``.

    Args:
        n: Number of optimization variables.
        m: Total number of constraints.
        nproc: Number of simultaneous function evaluations.

    Returns:
        Mapping with the keys ``nmax``, ``mmax``, ``mnn2``, ``lwa``,
        ``lkwa`` and ``lact``.
    """
    nmax = n + 1
    mmax = max(m, 1)
    return {
        "nmax": nmax,
        "mmax": mmax,
        "mnn2": m + n + n + 2,
        "lwa": nlpqlp_real(n, m, nmax, mmax, nproc),
        "lkwa": 26 + nmax,
        "lact": 2 * m + 10,
    }


def nlpql_sizes(n: int, m: int) -> dict[str, int]:
    """Return the array sizes required by ``NLPQL``.

    Args:
        n: Number of optimization variables.
        m: Total number of constraints.

    Returns:
        Mapping with the keys ``nmax``, ``mmax``, ``mnn``, ``lwa``,
        ``lkwa`` and ``lact``.
    """
    base = nlpqlp_sizes(n, m, 1)
    return {
        "nmax": base["nmax"],
        "mmax": base["mmax"],
        "mnn": m + n + n,
        "lwa": base["lwa"] + base["mnn2"],
        "lkwa": base["lkwa"],
        "lact": base["lact"],
    }


def nlpqly_sizes(n: int, m: int) -> dict[str, int]:
    """Return the array sizes required by ``NLPQLY``.

    Args:
        n: Number of optimization variables.
        m: Total number of constraints.

    Returns:
        Mapping with the keys ``lwa``, ``lkwa`` and ``lact``.
    """
    nq = n + 1
    mg = max(m, 1)
    mnn2 = m + n + n + 2
    off = 1 + nq + nq + mg * nq + mnn2 + nq * nq + nq + mg + n + 1 + mg + 6
    lwa = off + nlpqlp_bound(n, m, nq, mg, 1)
    return {"lwa": lwa, "lkwa": n + 27, "lact": 2 * m + 10}


def nlpqlb_sizes(n: int, m: int, mw: int) -> dict[str, int]:
    """Return the array sizes required by ``NLPQLB``.

    Args:
        n: Number of optimization variables.
        m: Total number of constraints.
        mw: Size of the working set.

    Returns:
        Mapping with the keys ``nmax``, ``mwmax``, ``mnn2``, ``lwa``,
        ``lkwa`` and ``lact``.
    """
    nmax = n + 1
    mwmax = max(mw, 1)
    off = 1 + mwmax + 1 + 6
    lwa = off + nlpqlp_bound(n, mw, nmax, mwmax, 1)
    lkwa = 2 * mw + 26 + nmax + max(nmax + 1, mw // nmax)
    return {
        "nmax": nmax,
        "mwmax": mwmax,
        "mnn2": mw + n + n + 2,
        "lwa": lwa,
        "lkwa": lkwa,
        "lact": 2 * m + 2 * mw + 11,
    }


def nlpqlg_sizes(n: int, m: int, nproc: int = 1) -> dict[str, int]:
    """Return the array sizes required by ``NLPQLG``.

    Args:
        n: Number of optimization variables.
        m: Total number of constraints.
        nproc: Number of simultaneous function evaluations.

    Returns:
        Mapping with the keys ``nmax``, ``mmax``, ``mnn2``, ``lwa``,
        ``lkwa`` and ``lact``.
    """
    base = nlpqlp_sizes(n, m, nproc)
    base["lwa"] += 2 * n + max(m, 1) + 12
    return base


def fit_dimensions(model: int, n: int, m: int, ell: int) -> tuple[int, int]:
    """Return the size of the program generated by a fitting transform.

    Args:
        model: Transformation number of ``NLDFTW``.
        n: Number of optimization variables.
        m: Number of constraints.
        ell: Number of individual functions.

    Returns:
        The number of variables and of constraints of the transformed
        program.
    """
    if model == 1:
        return n + ell, m + ell
    if model == 2:  # ruff:ignore[magic-value-comparison]
        return n + ell, m + 2 * ell
    if model == 3:  # ruff:ignore[magic-value-comparison]
        return n + 1, m + 2 * ell
    if model == 4:  # ruff:ignore[magic-value-comparison]
        return n + 1, m + ell
    return n, m


def fit_sizes(method: str, n: int, m: int, ell: int) -> dict[str, int]:
    """Return the array sizes required by a data fitting code.

    Args:
        method: Name of the code, lower case.
        n: Number of optimization variables.
        m: Number of constraints.
        ell: Number of individual functions.

    Returns:
        Mapping with the keys ``model``, ``lmmax``, ``lnmax``,
        ``lmnn2``, ``lwa``, ``lkwa`` and ``lact``.
    """
    model = FIT_MODELS[method]
    nt, mt = fit_dimensions(model, n, m, ell)
    lmmax = max(m + 2 * ell, 1)
    lnmax = n + ell + 1
    lwa = 7 + nlpqlp_real(nt, mt, lnmax, lmmax, 1)
    return {
        "model": model,
        "lmmax": lmmax,
        "lnmax": lnmax,
        "lmnn2": mt + nt + nt + 2,
        "lwa": lwa,
        "lkwa": lnmax + 26,
        "lact": 2 * mt + 10,
    }


def nlpjob_sizes(n: int, m: int, ell: int) -> dict[str, int]:
    """Return the array sizes required by ``NLPJOB``.

    The sizes cover every one of the sixteen transformations, so that
    the model may be changed without reallocating.

    Args:
        n: Number of optimization variables.
        m: Number of constraints.
        ell: Number of objective functions.

    Returns:
        Mapping with the keys ``lmmax``, ``lnmax``, ``lmnn2``, ``lwa``,
        ``lkwa`` and ``llogwa``.
    """
    lmmax = max(m + 2 * ell, 1)
    lnmax = n + ell + 1
    nt, mt = n + ell, m + 2 * ell
    lwa = 2 * ell + 7 + lnmax * lnmax + lnmax
    lwa += nlpqlp_real(nt, mt, lnmax, lmmax, 1)
    return {
        "lmmax": lmmax,
        "lnmax": lnmax,
        "lmnn2": 4 * ell + 2 * n + m + 2,
        "lwa": lwa,
        "lkwa": lnmax + 26,
        "llogwa": 2 * lmmax + 10,
    }


def nlpqlf_sizes(n: int, mf: int, m: int) -> dict[str, int]:
    """Return the array sizes required by ``NLPQLF``.

    Args:
        n: Number of optimization variables.
        mf: Number of ordinary constraints.
        m: Number of feasibility constraints.

    Returns:
        Mapping with the keys ``nmax``, ``mmax``, ``mnn2``, ``lwa``,
        ``lkwa`` and ``lact``.
    """
    nmax = n + 1
    mmax = max(mf + m, 1)
    return {
        "nmax": nmax,
        "mmax": mmax,
        "mnn2": mf + m + n + n + 2,
        "lwa": 7 + nlpqlp_real(n, mf + m, nmax, mmax, 1),
        "lkwa": nmax + 26,
        "lact": 2 * (mf + m) + 10,
    }
