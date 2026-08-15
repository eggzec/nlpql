"""Termination flags of the NLPQL family of codes."""

from __future__ import annotations


#: Termination flags above this value are reported by the QP solver.
QP_ERROR_OFFSET = 100

IFAIL_MESSAGES: dict[int, str] = {
    0: "Optimality conditions satisfied.",
    1: "Maximum number of iterations exceeded.",
    2: "Uphill search direction.",
    3: "Underflow when computing the new BFGS update matrix.",
    4: "Line search exceeded the maximum number of function calls.",
    5: "Length of a working array too short.",
    6: "False dimensions.",
    7: "Search direction close to zero at an infeasible iterate.",
    8: "Starting point violates a lower or upper bound.",
    9: "Wrong input parameter.",
    10: "Inconsistency in the quadratic programming subproblem.",
    11: "Too many successive non-evaluable function calls, or too many "
    "active constraints for the given working set.",
}


def message(ifail: int) -> str:
    """Return a human readable description of a termination flag.

    Args:
        ifail: Termination flag returned by one of the Fortran routines.

    Returns:
        A short description of the termination reason.
    """
    if ifail < 0:
        return "Reverse communication request pending."
    if ifail > QP_ERROR_OFFSET:
        return (
            f"Error {ifail - QP_ERROR_OFFSET} of the quadratic "
            "programming solver."
        )
    return IFAIL_MESSAGES.get(ifail, f"Unknown termination flag {ifail}.")


def success(ifail: int) -> bool:
    """Return whether a termination flag indicates a successful run.

    Args:
        ifail: Termination flag returned by one of the Fortran routines.

    Returns:
        ``True`` if the optimality conditions have been satisfied.
    """
    return ifail == 0
