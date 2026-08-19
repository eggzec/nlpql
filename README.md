![NLPQL](https://raw.githubusercontent.com/eggzec/nlpql/master/docs/assets/nlpql-banner.png)

# NLPQL

**Non-Linear Programming by Quadratic Lagrangian (NLPQL) for Python**

[![Tests](https://github.com/eggzec/nlpql/actions/workflows/test.yml/badge.svg)](https://github.com/eggzec/nlpql/actions/workflows/test.yml)
[![Documentation](https://github.com/eggzec/nlpql/actions/workflows/docs.yml/badge.svg)](https://github.com/eggzec/nlpql/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/github/eggzec/nlpql/graph/badge.svg)](https://codecov.io/github/eggzec/nlpql)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=eggzec_nlpql&metric=alert_status)](https://sonarcloud.io/project/overview?id=eggzec_nlpql)
[![License](https://img.shields.io/badge/license-GPL%203.0-blue.svg)](./LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/nlpql.svg?label=PyPI%20downloads)](https://pypi.org/project/nlpql/)
[![Python versions](https://img.shields.io/pypi/pyversions/nlpql.svg)](https://pypi.org/project/nlpql/)

`nlpql` solves the smooth nonlinear programming problem

```text
min  f(x)
     g_j(x)  = 0 ,  j = 1, ..., me
     g_j(x) >= 0 ,  j = me+1, ..., m
     xl <= x <= xu
```

by a sequential quadratic programming method. A quadratic subproblem is
formulated from a quadratic approximation of the Lagrangian and a
linearization of the constraints, and the steplength is determined by a
line search with respect to an augmented Lagrangian merit function.
Data fitting, min-max and multicriteria problems are transformed into
programs of this form and solved by the same algorithm.

## Available codes

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

## Quick example

Rosenbrock's post office problem, test problem TP37 of Hock and
Schittkowski:

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
        "jac": lambda x: np.array([[1.0, 2.0, 2.0], [-1.0, -2.0, -2.0]]),
    },
)

print(res.x)  # [24. 12. 12.]
print(res.fun)  # -3456.0
```

Problems with a very large number of constraints are handled by the
active set code, which only needs the gradients of a working set:

```python
m = 200_000
y = np.arange(m) / (m - 1.0)

res = minimize(
    fun=lambda x: float(np.sum(np.exp(x))),
    x0=[1.0, 0.5, 0.0],
    method="NLPQLB",
    jac=np.exp,
    bounds=[(-100.0, 100.0)] * 3,
    constraints={
        "type": "ineq",
        "fun": lambda x: x[0] + x[1] * y + x[2] * y**2 - 1.0 / (1.0 + y**2),
        "jac": lambda x: np.column_stack([np.ones_like(y), y, y**2]),
    },
    options={"mw": 420, "acc": 1e-10, "rho": 0.1},
)
print(res.fun)  # 4.3011838
```

## Features

- upper and lower bounds are handled separately and stay satisfied
- an additional variable prevents inconsistent linearized constraints
- non-monotone line search and automatic restarts make the method very
  stable for noisy function and derivative values
- objective and constraints may be evaluated simultaneously at several
  test points of the line search
- initial multipliers and an initial quasi-Newton matrix may be provided
- reverse communication, so the model functions can be supplied by an
  external simulation
- no `COMMON` blocks, no `SAVE`, no memory allocation, therefore
  thread-safe and re-entrant
- the detailed iteration protocol of the original codes is available
  through `options={"iprint": 4}`

## Installation

```bash
pip install nlpql
```

Requires Python 3.10+ and NumPy. No external runtime dependencies. See
the [full installation guide](https://eggzec.github.io/nlpql/installation/)
for uv, poetry and source builds.

## Documentation

- [Theory](https://eggzec.github.io/nlpql/theory/) — the mathematics of the implemented algorithms
- [Quickstart](https://eggzec.github.io/nlpql/quickstart/) — runnable examples
- [API Reference](https://eggzec.github.io/nlpql/api/) — functions, options and termination flags
- [References](https://eggzec.github.io/nlpql/references/) — literature citations

## Provenance

The algorithms were developed by Prof. Dr. K. Schittkowski and
co-authors. The sources in this repository were written from scratch
from the published user's guides and papers; no source code of the
original implementations was used. Subroutine and argument names follow
the published documentation, so existing calling programs can be adapted
easily. Reimplementation from the documentation was explicitly permitted
by the copyright holder.

## License

GNU General Public License v3 (GPLv3) — see [LICENSE](LICENSE).
