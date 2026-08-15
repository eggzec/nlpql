"""NLPQL - Non-Linear Programming by Quadratic Lagrangian (NLPQL) for Python.

The package provides a Python front end and a complete Fortran
implementation of the NLPQL family of sequential quadratic programming
codes:

=========  ==================================================
NLPQL      the original SQP method
NLPQLP     distributed and non-monotone line search
NLPQLY     easy-to-use version with internal differences
NLPQLB     active set strategy for very many constraints
NLPQLG     successive restarts for better local minima
NLPQLF     model functions evaluable on a convex subset only
NLPJOB     multicriteria optimization, 16 transformations
NLPLSQ     constrained nonlinear least squares
NLPLSX     least squares with very many terms
NLPL1      sum of absolute values
NLPINF     maximum norm data fitting
NLPMMX     min-max optimization
QL         convex quadratic programming (Goldfarb-Idnani)
=========  ==================================================

The high level entry point is ``minimize``::

    from nlpql import minimize

    res = minimize(
        fun, x0, jac=grad, bounds=bnds, constraints=[{"type": "ineq", "fun": g}]
    )

The Fortran routines themselves are available in ``nlpql.raw`` and use
the reverse communication protocol of the original codes.
"""

# ruff:file-ignore[non-empty-init-module]

from __future__ import annotations

from . import _nlpql as raw
from ._optimize import (
    FIT_METHODS,
    JOB_MODELS,
    OptimizeResult,
    fit,
    minimize,
    minimize_multi,
)
from ._problem import Problem, VectorProblem
from ._qp import solve_qp
from ._status import IFAIL_MESSAGES
from ._workspace import (
    fit_sizes,
    nlpjob_sizes,
    nlpql_sizes,
    nlpqlb_sizes,
    nlpqlf_sizes,
    nlpqlg_sizes,
    nlpqlp_sizes,
    nlpqly_sizes,
)


__all__ = [
    "FIT_METHODS",
    "IFAIL_MESSAGES",
    "JOB_MODELS",
    "METHODS",
    "OptimizeResult",
    "Problem",
    "VectorProblem",
    "fit",
    "fit_sizes",
    "minimize",
    "minimize_multi",
    "nlpjob_sizes",
    "nlpql_sizes",
    "nlpqlb_sizes",
    "nlpqlf_sizes",
    "nlpqlg_sizes",
    "nlpqlp_sizes",
    "nlpqly_sizes",
    "raw",
    "solve_qp",
]

__version__ = "0.1.0"

METHODS = ("NLPQLP", "NLPQL", "NLPQLY", "NLPQLB", "NLPQLG", "NLPQLF")
