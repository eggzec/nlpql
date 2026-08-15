# Quickstart

## A first example

`minimize` takes the objective, a starting point, optional bounds and a
list of constraints. The example is Rosenbrock's post office problem,
test problem TP37 of Hock and Schittkowski, which is also the
demonstration example of the NLPQLP user's guide.

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
print(res.multipliers)  # first m entries: [0., 144., ...]
```

## Problem formulation

The problem solved is

```text
min  f(x)
     g_j(x)  = 0 ,  j = 1, ..., me
     g_j(x) >= 0 ,  j = me+1, ..., m
     xl <= x <= xu
```

Constraints are given as a dictionary or as a sequence of
dictionaries:

| key    | meaning                                                     |
| ------ | ----------------------------------------------------------- |
| `type` | `"eq"` for `g(x) = 0`, `"ineq"` for `g(x) >= 0`              |
| `fun`  | callable returning a scalar or a vector of constraint values |
| `jac`  | optional callable returning the Jacobian of `fun`            |
| `args` | optional extra arguments of `fun` and `jac`                  |

Equality constraints are sorted to the front internally, exactly as the
Fortran routines expect them. If no `jac` is given, derivatives are
approximated by forward differences.

## Choosing a code

```python
res = minimize(fun, x0, method="NLPQLP", ...)
```

| method   | use it for                                                    |
| -------- | ------------------------------------------------------------- |
| `NLPQLP` | the default, non-monotone line search and internal restarts    |
| `NLPQL`  | the original code, monotone line search, no restarts           |
| `NLPQLY` | easy-to-use, all derivatives approximated inside Fortran       |
| `NLPQLB` | very many constraints, active set strategy, needs `mw`         |
| `NLPQLG` | successive restarts to improve a poor local minimum            |
| `NLPQLF` | objective evaluable only on a convex subset, needs `feasibility` |

Data fitting, min-max and multicriteria problems use the separate entry
points [`fit`](#data-fitting) and
[`minimize_multi`](#multicriteria-optimization).

## Options

```python
res = minimize(fun, x0, options={"acc": 1e-11, "maxiter": 200})
```

| option    | default | meaning                                                  |
| --------- | ------: | -------------------------------------------------------- |
| `acc`     |  `1e-9` | desired final accuracy                                    |
| `accqp`   |   `0.0` | tolerance of the QP solver, `0` selects `10 * eps`        |
| `stpmin`  |   `0.0` | minimum steplength of the distributed line search         |
| `maxiter` |   `100` | maximum number of outer iterations                        |
| `maxfun`  |    `20` | maximum number of function calls per line search          |
| `maxnm`   |    `10` | queue length of the non-monotone line search, `0` = off   |
| `rho`     | `100.0` | restart parameter, the BFGS matrix is reset to `rho * I`  |
| `mode`    |     `0` | scaling and restart strategy of the quasi-Newton matrix   |
| `nproc`   |     `1` | number of simultaneous function evaluations               |
| `lql`     |  `True` | solve the QP from a full positive definite matrix         |
| `ncycle`  |     `1` | number of cycles of `NLPQLG`                              |
| `mw`      |  `None` | size of the working set of `NLPQLB`                       |
| `eps`     |  `None` | increment of the forward difference formula               |
| `iprint`  |     `0` | output level, `0` to `4`                                  |
| `disp`    | `False` | shortcut for `iprint = 2`                                 |

## Verbose output

`iprint = 2` prints one line per iteration, `iprint = 3` a detailed
block and `iprint = 4` in addition the whole line search protocol:

```python
res = minimize(
    fun, x0, jac=jac, bounds=bnds, constraints=cons, options={"iprint": 4}
)
```

```text
     Iteration    1

        Function value:  F(X) =  -0.10000000D+04
        Variable:  X =
             0.10000000D+02  0.10000000D+02  0.10000000D+02
        Constraints: G(X) =
             0.50000000D+02  0.22000000D+02
        Sum of constraint violations:                    SCV =   0.0000D+00
        Number of active constraints:                    NAC =    1
        Karush-Kuhn-Tucker optimality condition:         KKT =   0.4364D+04
        Norm of Lagrangian gradient:                     NLG =   0.3219D+02
        Product of search direction with BFGS matrix:    DBD =   0.1036D+04
        Product of Lag-gradient with search direction:   DLP =  -0.2700D+04
        Line search  1: ALPHA =  0.10D+01, merit function FCT =  -0.236250D+04
        Line search successful after one step:  ALPHA = 1
```

## Equality constraints

```python
res = minimize(
    fun=lambda x: x[0] ** 2 + x[1] ** 2,
    x0=[3.0, 1.0],
    jac=lambda x: 2.0 * x,
    constraints=[
        {"type": "eq", "fun": lambda x: np.array([x[0] + x[1] - 3.0])},
        {"type": "ineq", "fun": lambda x: np.array([x[0] - 1.0])},
    ],
)
```

## Very many constraints

A discretized semi-infinite problem with 200 000 constraints and only
three variables:

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
print(res.working_set)  # indices of the constraints in the working set
```

`mw` is the size of the working set. It has to be large enough to hold
all constraints that are active at any iterate; `res.status == 11`
indicates that it has to be increased.

## Improving a local minimum

```python
rastrigin = lambda x: float(np.sum(x**2 - 5.0 * np.cos(2.0 * np.pi * x)))

local = minimize(rastrigin, [2.7, 3.2], bounds=[(-5.0, 5.0)] * 2)
best = minimize(
    rastrigin,
    [2.7, 3.2],
    bounds=[(-5.0, 5.0)] * 2,
    method="NLPQLG",
    options={"ncycle": 25},
)

print(local.fun)  # 2.869...
print(best.fun)  # -10.0
```

## Data fitting

`fit` combines a vector of individual functions by one of four norms.
The residuals are returned by a single callable, the transformation
into a smooth nonlinear program is done inside the Fortran layer.

```python
import numpy as np
from nlpql import fit

# Rosenbrock's function written as a residual vector
residual = lambda x: np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
jacobian = lambda x: np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])

res = fit(residual, [-1.2, 1.0], method="NLPLSQ", jac=jacobian)
print(res.fun)  # 0.0, the sum of squares
print(res.residuals)  # the individual values at the solution
```

| method   | objective                        | transformation             |
| -------- | -------------------------------- | -------------------------- |
| `NLPLSQ` | `sum f_i(x)^2`                   | L variables, L equalities   |
| `NLPLSX` | `sum f_i(x)^2`, very many terms  | none, gradient assembled    |
| `NLPL1`  | `sum abs(f_i(x))`                | L variables, 2L inequalities|
| `NLPINF` | `max abs(f_i(x))`                | 1 variable, 2L inequalities |
| `NLPMMX` | `max f_i(x)`                     | 1 variable, L inequalities  |

Constraints and bounds are given exactly as for `minimize`:

```python
res = fit(
    residual,
    [-1.2, 1.0],
    method="NLPINF",
    jac=jacobian,
    bounds=[(-10.0, 10.0)] * 2,
    constraints={"type": "ineq", "fun": lambda x: np.array([2.0 - x.sum()])},
)
```

Use `NLPLSX` instead of `NLPLSQ` as soon as the number of terms becomes
large, since `NLPLSQ` introduces one additional variable per term.

## Multicriteria optimization

`minimize_multi` reduces a vector of criteria to a scalar objective by
one of sixteen transformations and solves the resulting program.

```python
import numpy as np
from nlpql import minimize_multi, JOB_MODELS

objectives = lambda x: np.array([(x[0] + 3.0) ** 2 + 1.0, x[1]])
circle = {"type": "ineq", "fun": lambda x: np.array([9.0 - x @ x])}

# the individual minima are needed by some of the transformations
ideal = [
    minimize_multi(
        objectives,
        [0.0, 0.0],
        model=0,
        imin=i + 1,
        bounds=[(-10.0, 10.0)] * 2,
        constraints=circle,
    ).fun
    for i in range(2)
]

res = minimize_multi(
    objectives,
    [0.0, 0.0],
    model=12,
    weights=[1.0, 1.0],
    fk=ideal,
    bounds=[(-10.0, 10.0)] * 2,
    constraints=circle,
)
print(res.x, res.objectives)
```

| model | transformation                                            |
| ----: | --------------------------------------------------------- |
|     0 | individual minimum of criterion `imin`                     |
|     1 | weighted sum                                               |
|     2 | hierarchical optimization                                  |
|     3 | trade-off method                                           |
|     4 | distance from the goals in the L1 norm                     |
|     5 | distance from the goals in the L2 norm                     |
|     6 | global criterion                                           |
|     7 | global criterion in the L2 norm                            |
|     8 | min-max of the absolute values                             |
|     9 | min-max of the criteria                                    |
|    10 | min-max of the absolute distances from the goals           |
|    11 | min-max of the relative distances from the ideal values    |
|    12 | min-max of the weighted relative distances                 |
|    13 | min-max of the weighted criteria                           |
|    14 | weighted global criterion                                  |
|    15 | weighted global criterion in the L2 norm                   |

`weights` holds the weights, the bounds or the goal values and `fk` the
individual minima or the goal values, depending on the model. The
entries of `fk` must be different from zero for the models 6, 7, 11, 12,
14 and 15, otherwise `status` is 11. The names are available in
`nlpql.JOB_MODELS`.

## Functions that can only be evaluated on a subset

`NLPQLF` is meant for problems where the objective and some of the
constraints are defined only on a convex set described by additional,
cheap feasibility constraints. Every argument at which the objective is
requested satisfies them.

```python
import numpy as np
from nlpql import minimize


def fun(x):
    # the square root exists only inside the unit disc
    return -np.sqrt(1.0 - x[0] ** 2 - x[1] ** 2) + ((x - 2.0) ** 2).sum()


res = minimize(
    fun,
    [0.0, 0.0],
    method="NLPQLF",
    bounds=[(-2.0, 2.0)] * 2,
    options={
        "acc": 1e-10,
        "feasibility": {
            "type": "ineq",
            "fun": lambda x: np.array([1.0 - x[0] ** 2 - x[1] ** 2]),
        },
    },
)
print(res.x, res.feasibility)
```

The starting point has to satisfy the feasibility constraints, otherwise
`status` is 8. Their values at the solution are returned in
`res.feasibility`.

## Quadratic programming

The QP solver is available on its own:

```python
from nlpql import solve_qp

res = solve_qp(
    c=np.eye(5),
    d=np.array([-21.98, -1.26, 61.39, 5.3, 101.3]),
    a=np.array([[-7.56, 0.0, 0.0, 0.0, 0.5]]),
    b=np.array([39.1]),
    xl=np.full(5, -100.0),
    xu=np.full(5, 100.0),
)
print(res.fun)  # -6996.50559772314
```

## Reverse communication

The Fortran routines themselves are reachable through `nlpql.raw` and
use the reverse communication protocol of the original codes, i.e. they
return `IFAIL = -1` to request function values and `IFAIL = -2` to
request gradients. This is useful when the model functions are provided
by an external simulation.

```python
import nlpql

print(nlpql.raw.nlpqlpq.__doc__)
```
