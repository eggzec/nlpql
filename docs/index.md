# NLPQL

**Non-Linear Programming by Quadratic Lagrangian (NLPQL) for Python**

`nlpql` provides an open source Python front end and a complete Fortran
implementation of the NLPQL family of sequential quadratic programming
codes for the smooth nonlinear program

```text
min  f(x)
     g_j(x)  = 0 ,  j = 1, ..., me
     g_j(x) >= 0 ,  j = me+1, ..., m
     xl <= x <= xu
```

## Overview

### Nonlinear programming

| code     | purpose                                                          |
| -------- | ---------------------------------------------------------------- |
| `NLPQLP` | distributed and non-monotone line search, internal restarts       |
| `NLPQL`  | the original SQP method                                           |
| `NLPQLY` | easy-to-use, all derivatives approximated internally              |
| `NLPQLB` | active set strategy for a very large number of constraints        |
| `NLPQLG` | successive restarts for a stepwise improvement of local minima    |
| `NLPQLF` | model functions evaluable only on a convex subset                 |

### Data fitting and multicriteria optimization

Every code of this group transforms its problem into a nonlinear
program of the form above and solves it with `NLPQLP`.

| code     | purpose                                                          |
| -------- | ---------------------------------------------------------------- |
| `NLPLSQ` | constrained nonlinear least squares                               |
| `NLPLSX` | least squares with a very large number of terms                   |
| `NLPL1`  | sum of absolute values                                            |
| `NLPINF` | maximum norm data fitting                                        |
| `NLPMMX` | min-max optimization                                              |
| `NLPJOB` | multicriteria optimization, sixteen scalar transformations        |

### Quadratic programming

| code     | purpose                                                          |
| -------- | ---------------------------------------------------------------- |
| `QL`     | convex quadratic programming, solves the subproblem of every SQP step |

## Example

```python
import numpy as np
from nlpql import minimize

res = minimize(
    fun=lambda x: -x[0] * x[1] * x[2],
    x0=[10.0, 10.0, 10.0],
    jac=lambda x: np.array([-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]]),
    bounds=[(0.0, 42.0)] * 3,
    constraints={
        "type": "ineq",
        "fun": lambda x: np.array([
            x[0] + 2 * x[1] + 2 * x[2],
            72 - x[0] - 2 * x[1] - 2 * x[2],
        ]),
    },
)
print(res.x, res.fun)  # [24. 12. 12.] -3456.0
```

## Properties

- upper and lower bounds are handled separately and are satisfied by
  every iterate
- initial multipliers and an initial quasi-Newton matrix may be provided
- reverse communication, the model functions are evaluated by the
  calling program
- an additional variable prevents inconsistent linearized constraints
- extremely robust in the presence of noisy function and derivative
  values, thanks to the non-monotone line search
- several restart options in case of uphill search directions caused by
  inaccurate derivatives
- initial and periodic restarts with a scaled identity matrix
- no `COMMON` blocks, no `SAVE`, no memory allocation, therefore
  thread-safe and re-entrant
- objective and constraint values may be evaluated simultaneously at
  several test points of the line search

## Provenance

The algorithms were developed by Prof. Dr. K. Schittkowski and
co-authors. The Fortran sources in this repository were written from
scratch from the published user's guides and papers, no source code of
the original implementations was used. The names of the subroutines and
of their arguments follow the published documentation, so that existing
calling programs can be adapted easily.

## Documentation

- [Installation](installation.md) - installation guide
- [Quickstart](quickstart.md) - runnable examples
- [Theory](theory.md) - the mathematics of the implemented algorithms
- [API Reference](api.md) - functions, options and termination flags
- [References](references.md) - literature citations
